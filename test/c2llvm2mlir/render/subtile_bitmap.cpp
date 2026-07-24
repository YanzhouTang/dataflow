// 3DGS Sub-tile Skipping Bitmap (C-CGRA), GSCore §4.3 (Fig 8).
// A tile is split into subtiles; for each Gaussian, GSCore encodes which
// subtiles the Gaussian's bounding box overlaps as a bitmap, so rasterization
// can skip the subtiles whose bit is 0. This kernel computes the 2x2 = 4-bit
// bitmap by intersection-testing the splat (center +/- radius) against each of
// the four subtile boxes and packing the results (bit i for subtile i).
void subtileBitmap(float *cx, float *cy, float *radius,
                   float tile_x0, float tile_y0, float subtile,
                   int *out_bitmap) {
#pragma clang loop vectorize(disable) interleave(disable)
  for (int i = 0; i < 32; i++) {
    float r = radius[i];
    float lo_x = cx[i] - r, hi_x = cx[i] + r;
    float lo_y = cy[i] - r, hi_y = cy[i] + r;

    // Subtile boundaries (2x2 grid within the tile).
    float mx = tile_x0 + subtile;       // vertical split
    float my = tile_y0 + subtile;       // horizontal split
    float ex = tile_x0 + 2.0f * subtile;
    float ey = tile_y0 + 2.0f * subtile;

    // Overlap of the splat bbox with each subtile's x/y span.
    int ovl_left = (lo_x < mx) & (hi_x > tile_x0);
    int ovl_right = (lo_x < ex) & (hi_x > mx);
    int ovl_bot = (lo_y < my) & (hi_y > tile_y0);
    int ovl_top = (lo_y < ey) & (hi_y > my);

    // Subtile order: 0=bot-left,1=bot-right,2=top-left,3=top-right.
    int b0 = ovl_bot & ovl_left;
    int b1 = ovl_bot & ovl_right;
    int b2 = ovl_top & ovl_left;
    int b3 = ovl_top & ovl_right;
    out_bitmap[i] = b0 + 2 * b1 + 4 * b2 + 8 * b3;
  }
}
