// MLP NeRF (classic NeRF) - ray point sampling (C-CGRA), RAICHU §RT/Fig 1
// "ray marching". For each sample distance t along a ray, the 3D sample
// position is p = o + t*d. Recurrence-free per-sample compute -> C-CGRA.
void mnerfSample(float ox, float oy, float oz,
                 float dx, float dy, float dz,
                 float *t_in,
                 float *out_px, float *out_py, float *out_pz) {
#pragma clang loop vectorize(disable) interleave(disable)
  for (int i = 0; i < 32; i++) {
    float t = t_in[i];
    out_px[i] = ox + t * dx;
    out_py[i] = oy + t * dy;
    out_pz[i] = oz + t * dz;
  }
}
