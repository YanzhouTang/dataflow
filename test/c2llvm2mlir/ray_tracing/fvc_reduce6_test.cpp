// G2 fusion probe: a 6-element sum (left-associative) should collapse into one
// neura.fvc_reduce6 (the FVCU reduce stage used in NeRF / low-rank NeRF dot
// products).
void reduce6(float *a, float *out) {
#pragma clang loop vectorize(disable) interleave(disable)
  for (int i = 0; i < 32; i++) {
    out[i] = a[6 * i + 0] + a[6 * i + 1] + a[6 * i + 2] + a[6 * i + 3] +
             a[6 * i + 4] + a[6 * i + 5];
  }
}
