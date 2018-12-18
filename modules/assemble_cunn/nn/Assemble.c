#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/Assemble.c"
#else

void THNN_(Assemble_updateOutput)(
            THNNState *state,
            THTensor *cur_prev_aff,
            THTensor *feat,
            THTensor *output,
            THTensor *masked_cpa,
            int pad
          )
{
  bool DEBUG = false;

  int H = cur_prev_aff->size[0];
  int W = cur_prev_aff->size[1];
  int K = 2 * pad + 1; // K is the local region window size
  long D = feat->size[0];
  long N = feat->size[1];

  cur_prev_aff = THTensor_(newContiguous)(cur_prev_aff);
  // real *cur_prev_aff_data = THTensor_(data)(cur_prev_aff);
  feat = THTensor_(newContiguous)(feat);
  // real *feat_data = THTensor_(data)(feat);
  output = THTensor_(newContiguous)(output);
  // real *output_data = THTensor_(data)(output);
  masked_cpa = THTensor_(newContiguous)(masked_cpa);

  // Zero out value 
  THTensor_(zero)(output);
  THTensor_(zero)(masked_cpa);

  // Initialize a temporary tensor 
  //THTensor* temp = THTensor_(new)();
  //THTensor_(resize1d)(temp, D);
  real mass;

  // Loop over x and y
  for (int y = 0; y < H; y++)
  {
    for (int x = 0; x < W; x++)
    {
      // Get the right cell in the output
      long output_cell_idx = y * W + x;
      THTensor *output_cell = THTensor_(newSelect)(output, 1, output_cell_idx);

      // Zero out output_cell
      // THTensor_(zero)(output_cell);

      // Init a mass counter
      mass = 0.0;

      for (int i = -pad; i <= pad; i++){
        for (int j = -pad; j <= pad; j++){
          int prev_y = y + i;
          int prev_x = x + j;
          if (prev_y >= 0 && prev_y < H && prev_x >= 0 && prev_x < W)
          {
            real a = THTensor_(get4d)(cur_prev_aff, (long)y, (long)x, (long)prev_y, (long)prev_x);
            mass += a * a;
          }
        }
      }
      mass = sqrt(mass);

      // Avoid divied-by-0
      if (mass > -1e-8 && mass < 1e-8) {
        mass = 1e-8;
      }

      // Looping the local region
      for (int i = -pad; i <= pad; i++){
        for (int j = -pad; j <= pad; j++){
          int prev_y = y + i;
          int prev_x = x + j;
          if (prev_y >= 0 && prev_y < H && prev_x >= 0 && prev_x < W)
          {
            if (DEBUG)
            {
              printf("cur_prev_aff=[%d %d %d %d], y,x,prev_y,prev_x = [%d %d %d %d]\n", 
                cur_prev_aff->size[0], cur_prev_aff->size[1], cur_prev_aff->size[2], 
                cur_prev_aff->size[3], y,x,prev_y,prev_x);
            }

            // Update output
            real a = THTensor_(get4d)(cur_prev_aff, (long)y, (long)x, (long)prev_y, (long)prev_x);
            a = a / mass;
            long sliceIndex = prev_y * W + prev_x;
            real *fc = THTensor_(data)(THTensor_(newSelect)(feat, 1, sliceIndex));
            THBlas_(axpy)(D, a, fc, N, THTensor_(data)(output_cell), N);

            if (DEBUG)
            {
              printf("masked_cpa=[%d %d %d %d], y,x,prev_y,prev_x = [%d %d %d %d]\n", 
              masked_cpa->size[0], masked_cpa->size[1], masked_cpa->size[2], 
              masked_cpa->size[3], y,x,prev_y,prev_x);
            }
            
            // Update gradient
            real b = THTensor_(get4d)(masked_cpa, (long)y, (long)x, (long)prev_y, (long)prev_x);
            THTensor_(set4d)(masked_cpa, (long)y, (long)x, (long)prev_y, (long)prev_x, b + a);
          }
        }
      }
    }
  }
}

#endif
