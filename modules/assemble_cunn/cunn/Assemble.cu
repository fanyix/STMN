#include "THCUNN.h"
#include "THCHalf.h"
#include "THCHalfAutoNumerics.cuh"
#include "common.h"

template <typename Dtype>
__global__ void AssembleKernel(
            int n, int H, int W, int D, int pad,
            Dtype *cur_prev_aff,
            Dtype *feat,
            Dtype *output,
            Dtype *masked_cpa) 
  {
    // n = D*H*W
    CUDA_KERNEL_LOOP(index, n) {
      int HW = H*W;
      int d = index / HW;
      int loc = index % HW;
      int y = loc / W;
      int x = loc % W;

      // Init a mass counter
      Dtype mass = ScalarConvert<float, Dtype>::to(0);
      for (int i = -pad; i <= pad; i++){
        for (int j = -pad; j <= pad; j++){
          int prev_y = y + i;
          int prev_x = x + j;
          if (prev_y >= 0 && prev_y < H && prev_x >= 0 && prev_x < W)
          {
            int flat_idx = y * W * HW + x * HW + prev_y * W + prev_x; 
            Dtype coef = cur_prev_aff[flat_idx];
            // mass += coef * coef;
            if (coef > 0) {
              mass += ScalarConvert<float, Dtype>::to(1.0);
            }
          }
        }
      }
      // mass = sqrt(mass);

      // Avoid divied-by-0
      if (mass > -1e-8 && mass < 1e-8) {
        mass = ScalarConvert<float, Dtype>::to(1e-8);
      }

      // Looping the local region
      Dtype val = ScalarConvert<float, Dtype>::to(0);
      for (int i = -pad; i <= pad; i++){
        for (int j = -pad; j <= pad; j++){
          int prev_y = y + i;
          int prev_x = x + j;
          if (prev_y >= 0 && prev_y < H && prev_x >= 0 && prev_x < W)
          {
            // Update output
            int flat_idx = y * W * HW + x * HW + prev_y * W + prev_x; 
            Dtype a = cur_prev_aff[flat_idx];
            a = a / mass;
            int feat_flat_idx = d * HW + prev_y * W + prev_x;
            Dtype fc = feat[feat_flat_idx];
            val += a * fc;
            // Update gradient
            if (d == 0) { // The thread for the first dim is responsible for this
            	masked_cpa[flat_idx] += a;
            }
          }
        }
      }

      // Get the right cell in the output
      int output_idx = d * HW + y * W + x;
      output[output_idx] = val;
    }
}

#include "generic/Assemble.cu"
#include "THCGenerateFloatTypes.h"
