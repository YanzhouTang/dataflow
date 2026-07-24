// phase3_triangle.cpp
#include <cstdio>
#include <vector>
#include <cfloat>
#include <cmath>

struct TriSoA {
    const float* v0x; const float* v0y; const float* v0z;
    const float* v1x; const float* v1y; const float* v1z;
    const float* v2x; const float* v2y; const float* v2z;
};

struct Ray { float ox, oy, oz, dx, dy, dz; };
struct TriTask { int tri_idx; };

// 与原代码一致的 MT 相交（返回 0/1，并写出 t）
int intersectTriangle(float rox, float roy, float roz,
                      float rdx, float rdy, float rdz,
                      float v0x, float v0y, float v0z,
                      float v1x, float v1y, float v1z,
                      float v2x, float v2y, float v2z,
                      float* out_t) {
    const float EPSILON = 1e-7f;

    float e1x=v1x-v0x, e1y=v1y-v0y, e1z=v1z-v0z;
    float e2x=v2x-v0x, e2y=v2y-v0y, e2z=v2z-v0z;

    float hx = rdy*e2z - rdz*e2y;
    float hy = rdz*e2x - rdx*e2z;
    float hz = rdx*e2y - rdy*e2x;

    float a = e1x*hx + e1y*hy + e1z*hz;
    float f = 1.0f / a;

    float sx = rox - v0x, sy = roy - v0y, sz = roz - v0z;
    float u = f * (sx*hx + sy*hy + sz*hz);
    if (u < 0.0f || u > 1.0f) return 0;

    float qx = sy*e1z - sz*e1y;
    float qy = sz*e1x - sx*e1z;
    float qz = sx*e1y - sy*e1x;

    float v = f * (rdx*qx + rdy*qy + rdz*qz);
    if (v < 0.0f || u + v > 1.0f) return 0;

    float t = f * (e2x*qx + e2y*qy + e2z*qz);
    if (t > EPSILON) { *out_t = t; return 1; }
    return 0;
}

// Phase 3 kernel：对一批三角任务，输出该批的 best_t/best_tri
static inline void kernel_phase3_tri(const Ray& ray, const TriSoA& tri,
                                     const std::vector<TriTask>& tasks,
                                     float t_max_in,
                                     float& best_t, int& best_tri)
{
    best_t = t_max_in; best_tri = -1;
    for (auto& tk : tasks) {
        int id = tk.tri_idx;
        float t;
        int hit = intersectTriangle(ray.ox, ray.oy, ray.oz,
                                    ray.dx, ray.dy, ray.dz,
                                    tri.v0x[id], tri.v0y[id], tri.v0z[id],
                                    tri.v1x[id], tri.v1y[id], tri.v1z[id],
                                    tri.v2x[id], tri.v2y[id], tri.v2z[id],
                                    &t);
        if (hit && t < best_t) { best_t = t; best_tri = id; }
    }
}

int main() {
    // 与你原代码一致的 4 个三角形
    float v0x[4] = {-2.0f, -5.0f, 1.0f, 4.0f};
    float v0y[4] = {-2.0f, -5.0f, -2.0f, -5.0f};
    float v0z[4] = {0,0,0,0};
    float v1x[4] = { 2.0f, -4.0f, 2.5f, 5.5f};
    float v1y[4] = {-2.0f, -4.0f,-2.0f,-4.0f};
    float v1z[4] = {0,0,0,0};
    float v2x[4] = { 0.0f, -4.5f, 1.5f, 4.5f};
    float v2y[4] = { 2.0f, -4.5f, 2.0f, -4.5f};
    float v2z[4] = {0,0,0,0};

    TriSoA tri{ v0x,v0y,v0z, v1x,v1y,v1z, v2x,v2y,v2z };

    // 单条 ray
    Ray ray{0.f,0.f,10.f, 0.f,0.f,-1.f};

    // 一批三角任务（可改大批量做 profile）
    std::vector<TriTask> tasks = {{0},{1},{2},{3}};

    float best_t, t_max = FLT_MAX;
    int best_tri;
    kernel_phase3_tri(ray, tri, tasks, t_max, best_t, best_tri);

    if (best_tri >= 0) {
        printf("[Phase3] best_tri=%d best_t=%.6f\n", best_tri, best_t);
    } else {
        printf("[Phase3] no hit\n");
    }
    return 0;
}
