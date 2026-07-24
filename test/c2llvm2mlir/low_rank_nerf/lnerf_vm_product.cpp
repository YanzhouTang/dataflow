// Low-Rank NeRF (TensoRF) - per-rank VM product stage (C-CGRA), RAICHU §4.3.2.
// Forms the per-rank vector-matrix contribution term_r = vx_r·Myz_r +
// vy_r·Mxz_r + vz_r·Mxy_r (the FMA / G1 "partial_dot" stage) for the R=6 rank
// components and materializes them for the reduce stage (lnerf_vm_reduce). One
// sample's six rank terms are produced per loop iteration.
void lnerfVMProduct(float *vx, float *vy, float *vz,
                    float *Myz, float *Mxz, float *Mxy,
                    float *out_term) {
#pragma clang loop vectorize(disable) interleave(disable)
  for (int r = 0; r < 32; r++) {
    // Each work item = one (sample,rank) VM term = a 3-axis multiply-add.
    out_term[r] = vx[r] * Myz[r] + vy[r] * Mxz[r] + vz[r] * Mxy[r];
  }
}
