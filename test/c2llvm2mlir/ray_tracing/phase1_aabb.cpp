// phase1_aabb.cpp
#include <cstdio>
#include <vector>
#include <cmath>
#include <cfloat>
#include <algorithm>

struct BVHSoA {
    const float* min_x; const float* min_y; const float* min_z;
    const float* max_x; const float* max_y; const float* max_z;
    const int*   left;  const int*   right;
    const int*   tri_idx;
    const int*   is_leaf; // 1=leaf, 0=internal
};

struct Ray { float ox, oy, oz, dx, dy, dz; };

struct NodeHit {
    int   node_idx;
    float t_near;
    float t_far;
};

// ====== 与原代码一致的 AABB 相交（返回 0/1）======
int intersectAABB(float rox, float roy, float roz,
                  float rdx, float rdy, float rdz,
                  float minx, float miny, float minz,
                  float maxx, float maxy, float maxz,
                  float t_min, float t_max) {
    float invx = 1.0f / rdx;
    float invy = 1.0f / rdy;
    float invz = 1.0f / rdz;

    float t1x = (minx - rox) * invx;
    float t2x = (maxx - rox) * invx;
    float tminx = fminf(t1x, t2x);
    float tmaxx = fmaxf(t1x, t2x);

    float t1y = (miny - roy) * invy;
    float t2y = (maxy - roy) * invy;
    float tminy = fminf(t1y, t2y);
    float tmaxy = fmaxf(t1y, t2y);

    float t1z = (minz - roz) * invz;
    float t2z = (maxz - roz) * invz;
    float tminz = fminf(t1z, t2z);
    float tmaxz = fmaxf(t1z, t2z);

    float tnear = fmaxf(fmaxf(tminx, tminy), tminz);
    float tfar  = fminf(fminf(tmaxx, tmaxy), tmaxz);

    tnear = fmaxf(tnear, t_min);
    tfar  = fminf(tfar,  t_max);

    return (tnear <= tfar) ? 1 : 0;
}

// Phase 1 kernel：对一批节点做 AABB 判盒，输出命中节点（t_near/t_far 简化占位）
__attribute__((noinline))
static void kernel_phase1_aabb(const Ray& ray, const BVHSoA& bvh,
                               const std::vector<int>& in_nodes,
                               float t_max,
                               std::vector<NodeHit>& out_hits)
{
    out_hits.clear();
    out_hits.reserve(in_nodes.size());
    for (int idx : in_nodes) {
        int hit = intersectAABB(ray.ox, ray.oy, ray.oz,
                                ray.dx, ray.dy, ray.dz,
                                bvh.min_x[idx], bvh.min_y[idx], bvh.min_z[idx],
                                bvh.max_x[idx], bvh.max_y[idx], bvh.max_z[idx],
                                0.0f, t_max);
        if (hit) {
            NodeHit nh{idx, 0.0f, t_max}; // 需要严格 t_near/t_far 可在此内联 slab 取真值
            out_hits.push_back(nh);
        }
    }
}

int main() {
    // ====== 用与你原代码一致的最小场景 ======
    float bvh_min_x[8] = {-6.0f, -6.0f, 0.5f, -3.0f, -6.0f, 0.5f, 3.0f, 0.0f};
    float bvh_min_y[8] = {-6.0f, -6.0f, -6.0f, -3.0f, -6.0f, -3.0f, -6.0f, 0.0f};
    float bvh_min_z[8] = {-1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 0.0f};

    float bvh_max_x[8] = {6.0f, 0.5f, 6.0f, 0.5f, -3.0f, 3.0f, 6.0f, 0.0f};
    float bvh_max_y[8] = {6.0f, 6.0f, 6.0f, 6.0f, -3.0f, 6.0f, -3.0f, 0.0f};
    float bvh_max_z[8] = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 0.0f};

    int bvh_left_child[8]  = {1, 3, 5, -1, -1, -1, -1, -1};
    int bvh_right_child[8] = {2, 4, 6, -1, -1, -1, -1, -1};
    int bvh_tri_idx[8]     = {-1, -1, -1, 0, 1, 2, 3, -1};
    int bvh_is_leaf[8]     = {0, 0, 0, 1, 1, 1, 1, 0};

    BVHSoA bvh{ bvh_min_x, bvh_min_y, bvh_min_z,
                bvh_max_x, bvh_max_y, bvh_max_z,
                bvh_left_child, bvh_right_child,
                bvh_tri_idx, bvh_is_leaf };

    // 单条 ray
    Ray ray{0.f,0.f,10.f, 0.f,0.f,-1.f};

    // 构造一批节点（可替换成更大批量做 profile）
    std::vector<int> in_nodes = {0,1,2,3,4,5,6};
    std::vector<NodeHit> hits;

    float t_max = FLT_MAX;
    kernel_phase1_aabb(ray, bvh, in_nodes, t_max, hits);

    // 输出统计结果（用于 sanity check / profile）
    printf("[Phase1] in=%zu, hit=%zu\n", in_nodes.size(), hits.size());
    for (auto& h : hits) {
        printf("  node=%d t_near=%.3f t_far=%.3f\n", h.node_idx, h.t_near, h.t_far);
    }
    return 0;
}
