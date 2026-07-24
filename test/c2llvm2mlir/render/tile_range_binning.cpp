// 3DGS Sorting - tile-RANGE coverage count (M-CGRA).
// The original 3DGS duplicates a Gaussian once per overlapped tile: a Gaussian
// whose 2D bounding box spans a rectangle of tiles emits one (tile_id, depth)
// key per covered tile. On a CGRA the duplication is a three-step primitive:
//   (1) per-Gaussian coverage count  = (tx1-tx0+1)*(ty1-ty0+1)   <-- THIS kernel
//   (2) prefix-sum of counts -> per-Gaussian write offset (scan)
//   (3) scatter one key per covered tile at that offset (see sort_partition's
//       cursor-driven scatter).
// This kernel is step (1): from the screen center +/- radius it computes the
// covered tile rectangle, the tiles-touched count (the duplication factor), and
// the first covered tile id. Single loop => mapper-tractable; the dynamic
// per-Gaussian duplication is realized by (2)+(3).
void tileCoverage(float *screen_x, float *screen_y, float *radius,
                  float inv_tile, int grid_w,
                  int *out_ntiles, int *out_tile0, int *out_tilex_span) {
#pragma clang loop vectorize(disable) interleave(disable)
  for (int i = 0; i < 32; i++) {
    float rx = radius[i];
    int tx0 = (int)((screen_x[i] - rx) * inv_tile);
    int tx1 = (int)((screen_x[i] + rx) * inv_tile);
    int ty0 = (int)((screen_y[i] - rx) * inv_tile);
    int ty1 = (int)((screen_y[i] + rx) * inv_tile);
    int nx = tx1 - tx0 + 1;
    int ny = ty1 - ty0 + 1;
    out_ntiles[i] = nx * ny;             // duplication factor (tiles touched)
    out_tile0[i] = ty0 * grid_w + tx0;   // first covered tile id
    out_tilex_span[i] = nx;              // row stride for the scatter walk
  }
}
