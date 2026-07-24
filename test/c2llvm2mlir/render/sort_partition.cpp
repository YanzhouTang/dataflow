// 3DGS Sorting - Stage B: approximate sort via pivot partition (M-CGRA).
// This is GSCore's hierarchical-sorting Stage-1 (§4.2, Figure 7/8): within a
// tile's Gaussian group, a pivot depth splits the group into a "lower" chunk
// and a "higher" chunk. Each Gaussian is compared to the pivot and scattered
// to the corresponding chunk using a running write cursor (lo/hi), which is the
// loop-carried recurrence. Precise sorting later runs only on the lower chunk
// (enabling early-termination skips), so this partition is the workload-
// reducing preprocessing step. The data-dependent scatter is exactly the kind
// of irregular memory orchestration the M-CGRA targets.
void sortPartition(float *depth_in, int *gid_in, float pivot,
                   float *lower_d, int *lower_g,
                   float *higher_d, int *higher_g) {
  int lo = 0;
  int hi = 0;
#pragma clang loop vectorize(disable) interleave(disable)
  for (int i = 0; i < 32; i++) {
    float d = depth_in[i];
    int g = gid_in[i];
    if (d < pivot) {
      lower_d[lo] = d;
      lower_g[lo] = g;
      lo = lo + 1;
    } else {
      higher_d[hi] = d;
      higher_g[hi] = g;
      hi = hi + 1;
    }
  }
}
