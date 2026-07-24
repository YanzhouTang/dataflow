// 3DGS Shape-Aware Intersection Test (C-CGRA), GSCore §4.1 (Fig 8).
// For each Gaussian, GSCore selects the tighter bounding volume - an axis-
// aligned box (AABB) for near-circular splats or an oriented box (OBB) for
// elongated ones - then tests overlap with a tile. This kernel: derives the
// major/minor extents from the 2D covariance eigenvalues, selects AABB vs OBB
// by eccentricity, runs both overlap tests against the given tile, and returns
// the selected result (1 = intersect).
#include <math.h>
void intersectShape(float *cx, float *cy, float *cov_a, float *cov_b,
                    float *cov_c, float *axis_x, float *axis_y,
                    float tile_cx, float tile_cy, float tile_half,
                    float ecc_thresh, int *out_hit) {
#pragma clang loop vectorize(disable) interleave(disable)
  for (int i = 0; i < 32; i++) {
    float a = cov_a[i], b = cov_b[i], c = cov_c[i];
    float mid = 0.5f * (a + c);
    float disc = sqrtf(mid * mid - (a * c - b * b));
    float lmax = mid + disc;
    float lmin = mid - disc;
    float rmajor = 3.0f * sqrtf(lmax);
    float rminor = 3.0f * sqrtf(lmin);

    // Vector from tile center to splat center.
    float dx = cx[i] - tile_cx;
    float dy = cy[i] - tile_cy;

    // AABB overlap test (circle of radius rmajor vs tile box).
    float adx = dx > 0.0f ? dx : -dx;
    float ady = dy > 0.0f ? dy : -dy;
    int aabb_hit = (adx < tile_half + rmajor) & (ady < tile_half + rmajor);

    // OBB test: project the center offset onto the splat principal axes and
    // compare against the per-axis extents (rmajor along axis, rminor across).
    float ax = axis_x[i], ay = axis_y[i];
    float proj_major = dx * ax + dy * ay;    // along major axis
    float proj_minor = -dx * ay + dy * ax;   // along minor axis
    float pmaj = proj_major > 0.0f ? proj_major : -proj_major;
    float pmin = proj_minor > 0.0f ? proj_minor : -proj_minor;
    int obb_hit = (pmaj < tile_half + rmajor) & (pmin < tile_half + rminor);

    // Shape-aware selection: elongated splats use the OBB, round ones the AABB.
    float ecc = rmajor - ecc_thresh * rminor;
    out_hit[i] = ecc > 0.0f ? obb_hit : aabb_hit;
  }
}
