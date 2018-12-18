#include "THC.h"
#include <algorithm>
#include <cfloat>

#include "common.h"
#include "gpu_util.cuh"

using std::max;
using std::min;

template <typename Dtype>
__global__ void PSROIPoolingForward(
    const int nthreads,
    const Dtype* bottom_data,
    const Dtype spatial_scale,
    const int channels, // total number of channels for one image, e.g. C*N*N
    const int height, const int width,
    const int pooled_height, const int pooled_width, 
    const Dtype* bottom_rois,
    const int output_dim, // the number of class, e.g. C
    Dtype* top_data,
    int* mapping_channel) {

    // DEBUG
    //printf("[INIT c1=%.2f,c2=%.2f,c3=%.2f,c4=%.2f,c5=%.2f]\n", bottom_rois[0], bottom_rois[1], bottom_rois[2], bottom_rois[3], bottom_rois[4]);

    // DEBUG
    //printf("[INIT-DATA c1=%.2f,c2=%.2f,c3=%.2f,c4=%.2f,c5=%.2f]\n", bottom_data[0], bottom_data[1], bottom_data[2], bottom_data[3], bottom_data[4]);

    CUDA_KERNEL_LOOP(index, nthreads){
      // The output is in order (n, ctop, ph, pw)
      int pw = index % pooled_width;
      int ph = (index / pooled_width) % pooled_height;
      int ctop = (index / pooled_width / pooled_height) % output_dim;
      int n = index / pooled_width / pooled_height / output_dim;

      // [start, end) interval for spatial sampling
      bottom_rois += n * 5;
      
      // DEBUG
      //printf("[c1=%.2f,c2=%.2f,c3=%.2f,c4=%.2f,c5=%.2f]\n", bottom_rois[0], bottom_rois[1], bottom_rois[2], bottom_rois[3], bottom_rois[4]);

      // DEBUG
      //printf("spatial_scale=%.3f\n", spatial_scale);

      int roi_batch_ind = bottom_rois[0] - 1; // -1 is due to the Lua/C conversion
      //Dtype roi_start_w = static_cast<Dtype>(round(bottom_rois[1])) * spatial_scale;
      //Dtype roi_start_h = static_cast<Dtype>(round(bottom_rois[2])) * spatial_scale;
      //Dtype roi_end_w = static_cast<Dtype>(round(bottom_rois[3]) + 1.) * spatial_scale;
      //Dtype roi_end_h = static_cast<Dtype>(round(bottom_rois[4]) + 1.) * spatial_scale;
      Dtype roi_start_w = bottom_rois[1] * spatial_scale;
      Dtype roi_start_h = bottom_rois[2] * spatial_scale;
      Dtype roi_end_w = static_cast<Dtype>(bottom_rois[3] + 1.) * spatial_scale;
      Dtype roi_end_h = static_cast<Dtype>(bottom_rois[4] + 1.) * spatial_scale;
      bool roi_is_empty = (roi_end_h <= roi_start_h) || (roi_end_w <= roi_start_w);

      // DEBUG
      //printf("[hs=%.2f,ws=%.2f,he=%.2f,we=%.2f]\n", roi_start_h, roi_start_w, roi_end_h, roi_end_w);

      // Force too small ROIs to be 1x1
      Dtype roi_width = max(roi_end_w - roi_start_w, 0.1); //avoid 0
      Dtype roi_height = max(roi_end_h - roi_start_h, 0.1);

      // Compute w and h at bottom 
      Dtype bin_size_h = roi_height / static_cast<Dtype>(pooled_height);
      Dtype bin_size_w = roi_width / static_cast<Dtype>(pooled_width);
      int hstart = floor(static_cast<Dtype>(ph) * bin_size_h
                          + roi_start_h);
      int wstart = floor(static_cast<Dtype>(pw)* bin_size_w
                          + roi_start_w);
      int hend = ceil(static_cast<Dtype>(ph + 1) * bin_size_h
                        + roi_start_h);
      int wend = ceil(static_cast<Dtype>(pw + 1) * bin_size_w
                        + roi_start_w);
      // Add roi offsets and clip to input boundaries
      hstart = min(max(hstart, 0), height);
      hend = min(max(hend, 0), height);
      wstart = min(max(wstart, 0),width);
      wend = min(max(wend, 0), width);
      bool is_empty = roi_is_empty || (hend <= hstart) || (wend <= wstart);


      // DEBUG
      //printf("[%d,%d,%d,%d]\n", wstart+1, hstart+1, wend, hend);

      int gw = pw;
      int gh = ph;
      int c = ctop * pooled_width * pooled_height + gh * pooled_width + gw; 

      bottom_data += (roi_batch_ind * channels + c) * height * width;
      Dtype out_sum = 0;
      for (int h = hstart; h < hend; ++h){
        for (int w = wstart; w < wend; ++w){
          int bottom_index = h*width + w;
          out_sum += bottom_data[bottom_index];
        }
      }

      // DEBUG
      //if (is_empty) {
      //  printf("empty\n");
      //} else {
      //  printf("non-empty\n");
      //}

      Dtype bin_area = (hend - hstart)*(wend - wstart);
      top_data[index] = is_empty? 0. : out_sum/bin_area;
      mapping_channel[index] = c;
    }
}



extern "C"
void PSROIPooling_updateOutput(THCState *state, THCudaTensor *output, THCudaTensor *indices, THCudaTensor *data, THCudaTensor* rois, int height, int width, int pooled_height, int pooled_width, int output_dim, double spatial_scale)
{
  
  // DEBUG
  //printf("PSROIPooling_updateOutput, spatial_scale=%.3f\n", spatial_scale);
  //printf("PSROIPooling_updateOutput, height=%d\n", height);
  //printf("PSROIPooling_updateOutput, width=%d\n", width);
  //printf("PSROIPooling_updateOutput, pooled_height=%d\n", pooled_height);
  //printf("PSROIPooling_updateOutput, pooled_width=%d\n", pooled_width);
  //printf("PSROIPooling_updateOutput, output_dim=%d\n", output_dim);


  THAssert(THCudaTensor_nDimension(state, data) == 4);
  THAssert(THCudaTensor_nDimension(state, rois) == 2 && rois->size[1] == 5);
  THAssert(THCudaTensor_isContiguous(state, data));
  THAssert(THCudaTensor_isContiguous(state, rois));
  long num_rois = rois->size[0];
  int channels = pooled_height * pooled_width * output_dim;
  THCudaTensor_zero(state, output);
  THCudaTensor_zero(state, indices);
  THCudaTensor_resize4d(state, output, num_rois, output_dim, pooled_height, pooled_width);
  THCudaTensor_resize4d(state, indices, num_rois, output_dim, pooled_height, pooled_width);
  long count = THCudaTensor_nElement(state, output);

  PSROIPoolingForward<float> << <GET_BLOCKS(count), CUDA_NUM_THREADS, 0, THCState_getCurrentStream(state) >> >(count, THCudaTensor_data(state, data), spatial_scale, channels, height, width, pooled_height, pooled_width, THCudaTensor_data(state, rois), output_dim, THCudaTensor_data(state, output), (int*)THCudaTensor_data(state, indices));

  // check for errors
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("error in PSROIPooling_updateOutput: %s\n", cudaGetErrorString(err));
    THError("aborting");
  }
}


template <typename Dtype>
__global__ void PSROIPoolingBackwardAtomic(
  const int nthreads,
  const Dtype* top_diff,
  const int* mapping_channel,
  const int num_rois,
  const Dtype spatial_scale,
  const int channels,
  const int height, const int width,
  const int pooled_height, const int pooled_width,
  const int output_dim, 
  Dtype* bottom_diff,
  const Dtype* bottom_rois) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // The output is in order (n, ctop, ph, pw)
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int n = index / pooled_width / pooled_height / output_dim;

    // [start, end) interval for spatial sampling
    bottom_rois += n * 5;
    int roi_batch_ind = bottom_rois[0] - 1; // -1 is due to the Lua/C conversion
    //Dtype roi_start_w = static_cast<Dtype>(round(bottom_rois[1])) * spatial_scale;
    //Dtype roi_start_h = static_cast<Dtype>(round(bottom_rois[2])) * spatial_scale;
    //Dtype roi_end_w = static_cast<Dtype>(round(bottom_rois[3]) + 1.) * spatial_scale;
    //Dtype roi_end_h = static_cast<Dtype>(round(bottom_rois[4]) + 1.) * spatial_scale;
    Dtype roi_start_w = bottom_rois[1] * spatial_scale;
    Dtype roi_start_h = bottom_rois[2] * spatial_scale;
    Dtype roi_end_w = static_cast<Dtype>(bottom_rois[3] + 1.) * spatial_scale;
    Dtype roi_end_h = static_cast<Dtype>(bottom_rois[4] + 1.) * spatial_scale;
    bool roi_is_empty = (roi_end_h <= roi_start_h) || (roi_end_w <= roi_start_w);

    // Force too small ROIs to be 1x1
    Dtype roi_width = max(roi_end_w - roi_start_w, 0.1); //avoid 0
    Dtype roi_height = max(roi_end_h - roi_start_h, 0.1);

    // Compute w and h at bottom 
    Dtype bin_size_h = roi_height / static_cast<Dtype>(pooled_height);
    Dtype bin_size_w = roi_width / static_cast<Dtype>(pooled_width);

    int hstart = floor(static_cast<Dtype>(ph)* bin_size_h
      + roi_start_h);
    int wstart = floor(static_cast<Dtype>(pw)* bin_size_w
      + roi_start_w);
    int hend = ceil(static_cast<Dtype>(ph + 1) * bin_size_h
      + roi_start_h);
    int wend = ceil(static_cast<Dtype>(pw + 1) * bin_size_w
      + roi_start_w);
    // Add roi offsets and clip to input boundaries
    hstart = min(max(hstart, 0), height);
    hend = min(max(hend, 0), height);
    wstart = min(max(wstart, 0), width);
    wend = min(max(wend, 0), width);
    bool is_empty = roi_is_empty || (hend <= hstart) || (wend <= wstart);

    // Compute c at bottom
    int c = mapping_channel[index];
    Dtype* offset_bottom_diff = bottom_diff + (roi_batch_ind * channels + c) * height * width;
    Dtype bin_area = (hend - hstart)*(wend - wstart);
    Dtype diff_val = is_empty ? 0. : top_diff[index] / bin_area;
    for (int h = hstart; h < hend; ++h){
      for (int w = wstart; w < wend; ++w){
        int bottom_index = h*width + w;
        caffe_gpu_atomic_add(diff_val, offset_bottom_diff + bottom_index);
      }
    }
  }
}


extern "C"
void PSROIPooling_updateGradInputAtomic(THCState *state, THCudaTensor *gradInput, THCudaTensor *gradOutput, THCudaTensor *data, THCudaTensor* rois, THCudaTensor *indices, int height, int width, int pooled_height, int pooled_width, int output_dim, double spatial_scale)
{
  THAssert(THCudaTensor_nDimension(state, data) == 4);
  THAssert(THCudaTensor_nDimension(state, rois) == 2 && rois->size[1] == 5);
  THAssert(THCudaTensor_isContiguous(state, data));
  THAssert(THCudaTensor_isContiguous(state, rois));
  long num_rois = rois->size[0];
  int channels = pooled_height * pooled_width * output_dim;
  THCudaTensor_resizeAs(state, gradInput, data);
  THCudaTensor_zero(state, gradInput);

  long count = THCudaTensor_nElement(state, gradOutput);

  PSROIPoolingBackwardAtomic<float> << <GET_BLOCKS(count), CUDA_NUM_THREADS, 0, THCState_getCurrentStream(state) >> >(count, THCudaTensor_data(state, gradOutput), (int*)THCudaTensor_data(state, indices), num_rois, spatial_scale, channels, height, width, pooled_height, pooled_width, output_dim, THCudaTensor_data(state, gradInput), THCudaTensor_data(state, rois));

  // check for errors
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("error in PSROIPooling_updateGradInputAtomic: %s\n", cudaGetErrorString(err));
    THError("aborting");
  }
}