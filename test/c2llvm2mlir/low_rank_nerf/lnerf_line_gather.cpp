// Low-Rank NeRF (TensoRF) - factor gather + trilinear line sampling (M-CGRA),
// RAICHU §4.3.2 ("coherent embedding generation ... extensive tile reuse").
// For each sample the density/appearance factors live on per-axis 1D line grids
// and 2D plane grids. Sampling a line factor at a continuous coordinate is a
// gather of the two neighboring grid entries + a linear interpolation. This
// kernel gathers, per rank, the two neighbors of each axis line and lerps them
// -- the irregular indexed loads make it an M-CGRA (gather) task.
void lnerfLineGather(float *line_x, int *idx0, int *idx1, float *frac,
                     float *out_v) {
#pragma clang loop vectorize(disable) interleave(disable)
  for (int i = 0; i < 32; i++) {
    int i0 = idx0[i];
    int i1 = idx1[i];
    float f = frac[i];
    float a = line_x[i0]; // gather neighbor 0
    float b = line_x[i1]; // gather neighbor 1
    out_v[i] = a + f * (b - a); // linear interpolation along the line
  }
}
