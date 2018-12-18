#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/Assemble.cu"
#else

#include "../common.h"

void THNN_(Assemble_updateOutput)(
            THCState *state,
            THCTensor *cur_prev_aff,
            THCTensor *feat,
            THCTensor *output,
            THCTensor *masked_cpa,
            int pad
          )
{
	bool DEBUG = false;

	int H = cur_prev_aff->size[0];
	int W = cur_prev_aff->size[1];
	int K = 2 * pad + 1; // K is the local region window size
	long D = feat->size[0];
	long N = feat->size[1];

	// cur_prev_aff = THCTensor_(newContiguous)(state, cur_prev_aff);
	// feat = THCTensor_(newContiguous)(state, feat);
	output = THCTensor_(newContiguous)(state, output);
	masked_cpa = THCTensor_(newContiguous)(state, masked_cpa);

	// Zero out value 
	THCTensor_(zero)(state, output);
	THCTensor_(zero)(state, masked_cpa);

	// Get total count
	int count = H * W * D;

	// Launch CUDA kernel
	AssembleKernel<real>
	  <<<GET_BLOCKS(count), CUDA_NUM_THREADS, 0, THCState_getCurrentStream(state) >>>
	  	(count, H, W, D, pad, THCTensor_(data)(state, cur_prev_aff), THCTensor_(data)(state, feat), THCTensor_(data)(state, output), THCTensor_(data)(state, masked_cpa));

	// Catch any error
	THCudaCheck(cudaGetLastError());
}

#endif



