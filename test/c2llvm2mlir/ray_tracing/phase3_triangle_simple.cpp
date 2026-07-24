// phase3_triangle_simple.cpp - 简化版的光线-三角形相交测试 kernel
#include <cstdio>
#include <cfloat>

// Phase 3 kernel：对一批三角形任务执行 Möller-Trumbore 相交测试
// 输入：
//   - ray_ox/oy/oz: 光线原点
//   - ray_dx/dy/dz: 光线方向
//   - tri_v0x/v0y/v0z, v1x/v1y/v1z, v2x/v2y/v2z: 三角形顶点数据 (SoA 格式)
//   - tri_indices: 要测试的三角形索引数组
//   - num_tasks: 三角形任务数量
//   - t_max_in: 当前最大 t 值
// 输出：
//   - out_best_t: 最近的相交距离
//   - out_best_tri: 最近的三角形索引
__attribute__((noinline))
void kernel_phase3_triangle_intersect(
    float ray_ox, float ray_oy, float ray_oz,
    float ray_dx, float ray_dy, float ray_dz,
    const float* tri_v0x, const float* tri_v0y, const float* tri_v0z,
    const float* tri_v1x, const float* tri_v1y, const float* tri_v1z,
    const float* tri_v2x, const float* tri_v2y, const float* tri_v2z,
    const int* tri_indices,
    int num_tasks,
    float t_max_in,
    float* out_best_t,
    int* out_best_tri)
{
    const float EPSILON = 1e-7f;
    float best_t = t_max_in;
    int best_tri = -1;
    
    // 遍历所有三角形任务
    for (int i = 0; i < num_tasks; i++) {
        int idx = tri_indices[i];
        
        // 加载三角形顶点
        float v0x = tri_v0x[idx], v0y = tri_v0y[idx], v0z = tri_v0z[idx];
        float v1x = tri_v1x[idx], v1y = tri_v1y[idx], v1z = tri_v1z[idx];
        float v2x = tri_v2x[idx], v2y = tri_v2y[idx], v2z = tri_v2z[idx];
        
        // Möller-Trumbore 算法
        // 计算边向量
        float e1x = v1x - v0x;
        float e1y = v1y - v0y;
        float e1z = v1z - v0z;
        
        float e2x = v2x - v0x;
        float e2y = v2y - v0y;
        float e2z = v2z - v0z;
        
        // h = ray_d × e2
        float hx = ray_dy * e2z - ray_dz * e2y;
        float hy = ray_dz * e2x - ray_dx * e2z;
        float hz = ray_dx * e2y - ray_dy * e2x;
        
        // a = e1 · h
        float a = e1x * hx + e1y * hy + e1z * hz;
        
        // 计算 f = 1/a（避免除以零在实际中通过 EPSILON 处理）
        float f = 1.0f / a;
        
        // s = ray_o - v0
        float sx = ray_ox - v0x;
        float sy = ray_oy - v0y;
        float sz = ray_oz - v0z;
        
        // u = f * (s · h)
        float u = f * (sx * hx + sy * hy + sz * hz);
        
        // 检查 u 是否在 [0, 1] 范围内
        if (u < 0.0f || u > 1.0f) {
            continue;
        }
        
        // q = s × e1
        float qx = sy * e1z - sz * e1y;
        float qy = sz * e1x - sx * e1z;
        float qz = sx * e1y - sy * e1x;
        
        // v = f * (ray_d · q)
        float v = f * (ray_dx * qx + ray_dy * qy + ray_dz * qz);
        
        // 检查 v 和 u+v 是否在有效范围内
        if (v < 0.0f || u + v > 1.0f) {
            continue;
        }
        
        // t = f * (e2 · q)
        float t = f * (e2x * qx + e2y * qy + e2z * qz);
        
        // 检查 t 是否有效且比当前最优解更近
        if (t > EPSILON && t < best_t) {
            best_t = t;
            best_tri = idx;
        }
    }
    
    // 输出结果
    *out_best_t = best_t;
    *out_best_tri = best_tri;
}

int main() {
    // 4 个三角形（在 XY 平面上，Z=0）
    float v0x[4] = {-2.0f, -5.0f, 1.0f, 4.0f};
    float v0y[4] = {-2.0f, -5.0f, -2.0f, -5.0f};
    float v0z[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    
    float v1x[4] = {2.0f, -4.0f, 2.5f, 5.5f};
    float v1y[4] = {-2.0f, -4.0f, -2.0f, -4.0f};
    float v1z[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    
    float v2x[4] = {0.0f, -4.5f, 1.5f, 4.5f};
    float v2y[4] = {2.0f, -4.5f, 2.0f, -4.5f};
    float v2z[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    
    // 光线：从 (0,0,10) 沿 -Z 方向发射
    float ray_ox = 0.0f, ray_oy = 0.0f, ray_oz = 10.0f;
    float ray_dx = 0.0f, ray_dy = 0.0f, ray_dz = -1.0f;
    
    // 三角形任务列表（测试所有 4 个三角形）
    int tri_indices[4] = {0, 1, 2, 3};
    int num_tasks = 4;
    
    // 初始 t_max
    float t_max = FLT_MAX;
    float best_t;
    int best_tri;
    
    // 执行 kernel
    kernel_phase3_triangle_intersect(
        ray_ox, ray_oy, ray_oz,
        ray_dx, ray_dy, ray_dz,
        v0x, v0y, v0z,
        v1x, v1y, v1z,
        v2x, v2y, v2z,
        tri_indices,
        num_tasks,
        t_max,
        &best_t,
        &best_tri
    );
    
    // 输出结果
    if (best_tri >= 0) {
        printf("[Phase3] best_tri=%d best_t=%.6f\n", best_tri, best_t);
    } else {
        printf("[Phase3] no hit\n");
    }
    
    return 0;
}

