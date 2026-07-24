// phase2_compact.cpp
#include <cstdio>
#include <vector>

struct BVHSoA {
    const int* left; const int* right;
    const int* tri_idx;
    const int* is_leaf; // 1=leaf
};

struct NodeHit { int node_idx; float t_near, t_far; };
struct TriTask { int tri_idx; };

// Phase 2 kernel：把命中节点分流到“下一轮内部节点队列”与“三角任务队列”
static inline void kernel_phase2_compact_dispatch(const BVHSoA& bvh,
                                                  const std::vector<NodeHit>& in_hits,
                                                  std::vector<int>& out_next_nodes,
                                                  std::vector<TriTask>& out_tri_tasks)
{
    out_next_nodes.clear();
    out_tri_tasks.clear();
    out_next_nodes.reserve(in_hits.size()*2);
    out_tri_tasks.reserve(in_hits.size());

    for (const auto& h : in_hits) {
        int n = h.node_idx;
        if (bvh.is_leaf[n]) {
            int t = bvh.tri_idx[n];
            if (t >= 0) out_tri_tasks.push_back(TriTask{t});
        } else {
            int L = bvh.left[n], R = bvh.right[n];
            if (L >= 0) out_next_nodes.push_back(L);
            if (R >= 0) out_next_nodes.push_back(R);
        }
    }
}

int main() {
    // 最小 BVH 关系（仅 Phase2 需的字段）
    int left [8] = {1,3,5,-1,-1,-1,-1,-1};
    int right[8] = {2,4,6,-1,-1,-1,-1,-1};
    int tri  [8] = {-1,-1,-1, 0, 1, 2, 3,-1};
    int leaf [8] = {0, 0, 0, 1, 1, 1, 1, 0};

    BVHSoA bvh{ left, right, tri, leaf };

    // 构造一批“命中节点”（一般来自 Phase1 的输出，这里手工给一个集合）
    std::vector<NodeHit> hits = {
        {0,0.0f,1e9f}, // internal
        {1,0.0f,1e9f}, // internal
        {3,0.0f,1e9f}, // leaf → tri 0
        {4,0.0f,1e9f}, // leaf → tri 1
        {5,0.0f,1e9f}, // leaf → tri 2
    };

    std::vector<int>     next_nodes;
    std::vector<TriTask> tri_tasks;

    kernel_phase2_compact_dispatch(bvh, hits, next_nodes, tri_tasks);

    printf("[Phase2] in_hits=%zu | out_next_nodes=%zu | out_tri_tasks=%zu\n",
           hits.size(), next_nodes.size(), tri_tasks.size());

    if (!next_nodes.empty()) {
        printf("  next_nodes: ");
        for (int n : next_nodes) printf("%d ", n);
        printf("\n");
    }
    if (!tri_tasks.empty()) {
        printf("  tri_tasks: ");
        for (auto& t : tri_tasks) printf("%d ", t.tri_idx);
        printf("\n");
    }
    return 0;
}
