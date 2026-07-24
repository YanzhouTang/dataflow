// 3DGS Feature Computation - bounding-box radius via eigenvalues (C-CGRA).
// GSCore Eq.3: r_major = 3*sqrt(max(lambda1,lambda2)),
//              r_minor = 3*sqrt(min(lambda1,lambda2)),
// where lambda1,2 are eigenvalues of the 2D covariance [a b; b c]:
//   mid = (a+c)/2, disc = sqrt(mid^2 - det), lambda = mid +/- disc.
// Uses the real sqrt (lowered to neura.fsqrt).
#include <math.h>
void bboxRadius(float *cov_a, float *cov_b, float *cov_c,
                float *out_rmajor, float *out_rminor) {
#pragma clang loop vectorize(disable) interleave(disable)
  for (int i = 0; i < 32; i++) {
    float a = cov_a[i];
    float b = cov_b[i];
    float c = cov_c[i];
    float mid = 0.5f * (a + c);
    float det = a * c - b * b;
    float disc = sqrtf(mid * mid - det);
    float lambda1 = mid + disc;
    float lambda2 = mid - disc;
    float lmax = fmaxf(lambda1, lambda2);
    float lmin = fminf(lambda1, lambda2);
    out_rmajor[i] = 3.0f * sqrtf(lmax);
    out_rminor[i] = 3.0f * sqrtf(lmin);
  }
}
