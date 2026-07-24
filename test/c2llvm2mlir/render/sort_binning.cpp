// 3DGS Sorting - Stage A: tile binning + sort-key generation (M-CGRA).
// GSCore §3.1/§4.2: after feature computation, each Gaussian's 2D bounding box
// (screen center +/- radius, radius from Eq.3) determines the tile it falls in.
// The standard 3DGS sort forms a combined key = (tile_id in the high bits,
// depth in the low bits) so that a single radix sort orders Gaussians by tile
// first and by depth within a tile. This is the memory-address / key
// generation step, so it runs on the M-CGRA.
#include <math.h>
void sortBinning(float *screen_x, float *screen_y, float *radius, float *depth,
                 float inv_tile, float grid_w, float *out_key, int *out_gid) {
#pragma clang loop vectorize(disable) interleave(disable)
  for (int i = 0; i < 32; i++) {
    float rx = radius[i];
    // Tile of the bounding-box min corner (tile intersection).
    float tx = (screen_x[i] - rx) * inv_tile;
    float ty = (screen_y[i] - rx) * inv_tile;
    float tile_id = ty * grid_w + tx;

    // Combined radix key: tile_id in high magnitude, depth in low magnitude.
    // (kept in f32 arithmetic to stay mapper-tractable; a real impl packs bits)
    float depth_q = depth[i];
    out_key[i] = tile_id * 65536.0f + depth_q;
    out_gid[i] = i;
  }
}
