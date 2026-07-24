// Ray Tracing - hit-node stream compaction (M-CGRA), RAICHU §RT.
// BVH traversal produces a sparse hit mask over candidate nodes; compaction
// packs the hit node ids into a dense task list for the next traversal/leaf
// step. Each element conditionally scatters to out_list[cursor] and advances a
// running write cursor (the loop-carried recurrence). Data-dependent scatter =
// M-CGRA memory orchestration (same primitive as the 3DGS sort partition).
void rtCompact(int *hit_mask, int *node_id, int *out_list, int *out_count) {
  int cursor = 0;
#pragma clang loop vectorize(disable) interleave(disable)
  for (int i = 0; i < 32; i++) {
    if (hit_mask[i]) {
      out_list[cursor] = node_id[i];
      cursor = cursor + 1;
    }
  }
  out_count[0] = cursor;
}
