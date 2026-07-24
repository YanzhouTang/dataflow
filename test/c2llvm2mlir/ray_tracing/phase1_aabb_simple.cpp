// phase1_aabb_simple.cpp - 简化版 AABB 测试，保留循环结构
#include <cmath>
#include <cstdio>

// 简化的光线结构
struct Ray {
    float ox, oy, oz;  // 原点
    float dx, dy, dz;  // 方向
};

// Phase 1 kernel：对一批节点做 AABB 判盒（循环版本，固定数组）
// 输入：
//   - rox, roy, roz: 光线原点
//   - rdx, rdy, rdz: 光线方向
//   - bvh_min_x/y/z, bvh_max_x/y/z: BVH 包围盒数组
//   - node_indices: 要测试的节点索引数组
//   - num_nodes: 节点数量
//   - hit_results: 输出命中结果（1=命中，0=未命中）
__attribute__((noinline))
void kernel_phase1_aabb(
    float rox, float roy, float roz,
    float rdx, float rdy, float rdz,
    const float* bvh_min_x,
    const float* bvh_min_y,
    const float* bvh_min_z,
    const float* bvh_max_x,
    const float* bvh_max_y,
    const float* bvh_max_z,
    const int* node_indices,
    int* hit_results,
    int num_nodes)
{
    // 预计算倒数
    float invx = 1.0f / rdx;
    float invy = 1.0f / rdy;
    float invz = 1.0f / rdz;
    
    // 循环遍历所有节点（每次迭代做一个 AABB slab test）
    for (int i = 0; i < num_nodes; i++) {
        int idx = node_indices[i];
        
        // 从 BVH 数组加载当前节点的包围盒
        float minx = bvh_min_x[idx];
        float miny = bvh_min_y[idx];
        float minz = bvh_min_z[idx];
        float maxx = bvh_max_x[idx];
        float maxy = bvh_max_y[idx];
        float maxz = bvh_max_z[idx];
        
        // X 轴 slab test
        float t1x = (minx - rox) * invx;
        float t2x = (maxx - rox) * invx;
        float tminx = fminf(t1x, t2x);
        float tmaxx = fmaxf(t1x, t2x);
        
        // Y 轴 slab test
        float t1y = (miny - roy) * invy;
        float t2y = (maxy - roy) * invy;
        float tminy = fminf(t1y, t2y);
        float tmaxy = fmaxf(t1y, t2y);
        
        // Z 轴 slab test
        float t1z = (minz - roz) * invz;
        float t2z = (maxz - roz) * invz;
        float tminz = fminf(t1z, t2z);
        float tmaxz = fmaxf(t1z, t2z);
        
        // 合并三个轴的区间
        float tnear = fmaxf(fmaxf(tminx, tminy), tminz);
        float tfar = fminf(fminf(tmaxx, tmaxy), tmaxz);
        
        // 判断相交
        hit_results[i] = (tnear <= tfar) ? 1 : 0;
    }
}

int main() {
    // 测试数据：与原版一致的 BVH
    float bvh_min_x[8] = {-6.0f, -6.0f, 0.5f, -3.0f, -6.0f, 0.5f, 3.0f, 0.0f};
    float bvh_min_y[8] = {-6.0f, -6.0f, -6.0f, -3.0f, -6.0f, -3.0f, -6.0f, 0.0f};
    float bvh_min_z[8] = {-1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 0.0f};
    
    float bvh_max_x[8] = {6.0f, 0.5f, 6.0f, 0.5f, -3.0f, 3.0f, 6.0f, 0.0f};
    float bvh_max_y[8] = {6.0f, 6.0f, 6.0f, 6.0f, -3.0f, 6.0f, -3.0f, 0.0f};
    float bvh_max_z[8] = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 0.0f};
    
    // 光线：从 (0,0,10) 沿 -z 方向
    Ray ray = {0.0f, 0.0f, 10.0f, 0.0f, 0.0f, -1.0f};
    
    // 要测试的节点
    int node_indices[7] = {0, 1, 2, 3, 4, 5, 6};
    int hit_results[7];
    int num_nodes = 7;
    
    // 执行 kernel
    kernel_phase1_aabb(ray.ox, ray.oy, ray.oz,
                       ray.dx, ray.dy, ray.dz,
                       bvh_min_x, bvh_min_y, bvh_min_z,
                       bvh_max_x, bvh_max_y, bvh_max_z,
                       node_indices, hit_results, num_nodes);
    
    // 输出结果
    int hit_count = 0;
    for (int i = 0; i < num_nodes; i++) {
        if (hit_results[i]) {
            hit_count++;
            printf("  node=%d hit\n", node_indices[i]);
        }
    }
    printf("[Phase1] in=%d, hit=%d\n", num_nodes, hit_count);
    
    return 0;
}

