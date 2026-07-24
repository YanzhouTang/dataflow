// phase2_compact_simple.cpp - 简化版的压缩与分派 kernel
#include <cstdio>

// Phase 2 kernel：把命中节点分流到"下一轮内部节点队列"与"三角任务队列"
// 输入：
//   - bvh_left/right: BVH 树的左右子节点数组
//   - bvh_tri_idx: 叶子节点对应的三角形索引
//   - bvh_is_leaf: 节点是否为叶子节点（1=叶子，0=内部）
//   - in_node_indices: 输入的命中节点索引数组
//   - num_hits: 命中节点数量
// 输出：
//   - out_next_nodes: 下一轮要测试的内部节点
//   - out_tri_indices: 需要进行三角形相交测试的三角形索引
//   - out_next_count: 下一轮节点数量（通过指针返回）
//   - out_tri_count: 三角形任务数量（通过指针返回）
__attribute__((noinline))
void kernel_phase2_compact_dispatch(
    const int* bvh_left,
    const int* bvh_right,
    const int* bvh_tri_idx,
    const int* bvh_is_leaf,
    const int* in_node_indices,
    int num_hits,
    int* out_next_nodes,
    int* out_tri_indices,
    int* out_next_count,
    int* out_tri_count)
{
    int next_count = 0;
    int tri_count = 0;
    
    // 循环遍历所有命中的节点
    for (int i = 0; i < num_hits; i++) {
        int node_idx = in_node_indices[i];
        
        // 判断是叶子节点还是内部节点
        int is_leaf = bvh_is_leaf[node_idx];
        
        if (is_leaf) {
            // 叶子节点：输出三角形索引
            int tri_idx = bvh_tri_idx[node_idx];
            if (tri_idx >= 0) {
                out_tri_indices[tri_count] = tri_idx;
                tri_count++;
            }
        } else {
            // 内部节点：输出左右子节点
            int left_child = bvh_left[node_idx];
            int right_child = bvh_right[node_idx];
            
            if (left_child >= 0) {
                out_next_nodes[next_count] = left_child;
                next_count++;
            }
            
            if (right_child >= 0) {
                out_next_nodes[next_count] = right_child;
                next_count++;
            }
        }
    }
    
    // 输出计数
    *out_next_count = next_count;
    *out_tri_count = tri_count;
}

int main() {
    // 最小 BVH 关系（仅 Phase2 需要的字段）
    int bvh_left[8]  = {1, 3, 5, -1, -1, -1, -1, -1};
    int bvh_right[8] = {2, 4, 6, -1, -1, -1, -1, -1};
    int bvh_tri[8]   = {-1, -1, -1, 0, 1, 2, 3, -1};
    int bvh_leaf[8]  = {0, 0, 0, 1, 1, 1, 1, 0};
    
    // 构造一批"命中节点"（来自 Phase1 的输出）
    int in_nodes[5] = {0, 1, 3, 4, 5};
    int num_hits = 5;
    
    // 输出缓冲区
    int out_next_nodes[16];  // 最多可能有 num_hits * 2 个节点
    int out_tri_indices[16]; // 最多可能有 num_hits 个三角形
    int next_count = 0;
    int tri_count = 0;
    
    // 执行 kernel
    kernel_phase2_compact_dispatch(
        bvh_left, bvh_right, bvh_tri, bvh_leaf,
        in_nodes, num_hits,
        out_next_nodes, out_tri_indices,
        &next_count, &tri_count
    );
    
    // 输出结果
    printf("[Phase2] in_hits=%d | out_next_nodes=%d | out_tri_tasks=%d\n",
           num_hits, next_count, tri_count);
    
    if (next_count > 0) {
        printf("  next_nodes: ");
        for (int i = 0; i < next_count; i++) {
            printf("%d ", out_next_nodes[i]);
        }
        printf("\n");
    }
    
    if (tri_count > 0) {
        printf("  tri_tasks: ");
        for (int i = 0; i < tri_count; i++) {
            printf("%d ", out_tri_indices[i]);
        }
        printf("\n");
    }
    
    return 0;
}

