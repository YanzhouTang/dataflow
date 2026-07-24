// 3DGS Sorting stage (M-CGRA): one compare-and-swap stage of a sorting network
// that orders gaussians by depth and reorders their indices along with them
// (RAICHU §4.3.3: "generate gaussian-centric job lists" = depth-ordered data
// reorganization). Even-odd stage over disjoint pairs (2i, 2i+1) => no
// loop-carried dependency, so it maps cleanly. The compare-swap uses the
// FVCU-style min/max + select primitives.
void sortStage(float *depth_in, float *depth_out, int *idx_in, int *idx_out) {
#pragma clang loop vectorize(disable) interleave(disable)
  for (int i = 0; i < 16; i++) {
    float a = depth_in[2 * i];
    float b = depth_in[2 * i + 1];
    int ia = idx_in[2 * i];
    int ib = idx_in[2 * i + 1];
    // Ascending compare-and-swap on the pair.
    int gt = a > b;
    depth_out[2 * i] = a < b ? a : b;     // min
    depth_out[2 * i + 1] = a < b ? b : a; // max
    idx_out[2 * i] = gt ? ib : ia;        // index follows the smaller depth
    idx_out[2 * i + 1] = gt ? ia : ib;
  }
}
