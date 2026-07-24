#include <stdio.h>
#include <string.h>
#include <math.h>
#include <float.h>

// BVH节点结构（使用结构数组SoA风格存储）
// 每个节点包含：AABB包围盒、左右子节点索引、三角形索引

// 函数声明
void rayTraceBVH(float ray_origin_x, float ray_origin_y, float ray_origin_z,
                 float ray_dir_x, float ray_dir_y, float ray_dir_z,
                 float* bvh_min_x, float* bvh_min_y, float* bvh_min_z,
                 float* bvh_max_x, float* bvh_max_y, float* bvh_max_z,
                 int* bvh_left_child, int* bvh_right_child,
                 int* bvh_tri_idx, int* bvh_is_leaf,
                 float* tri_v0_x, float* tri_v0_y, float* tri_v0_z,
                 float* tri_v1_x, float* tri_v1_y, float* tri_v1_z,
                 float* tri_v2_x, float* tri_v2_y, float* tri_v2_z,
                 int* stack,
                 float* hit_t, int* hit_tri_id);

// AABB包围盒相交测试
int intersectAABB(float ray_orig_x, float ray_orig_y, float ray_orig_z,
                  float ray_dir_x, float ray_dir_y, float ray_dir_z,
                  float box_min_x, float box_min_y, float box_min_z,
                  float box_max_x, float box_max_y, float box_max_z,
                  float t_min, float t_max) {
    // 计算光线的倒数方向
    float inv_dir_x = 1.0f / ray_dir_x;
    float inv_dir_y = 1.0f / ray_dir_y;
    float inv_dir_z = 1.0f / ray_dir_z;
    
    // 计算与x平面的交点
    float t1_x = (box_min_x - ray_orig_x) * inv_dir_x;
    float t2_x = (box_max_x - ray_orig_x) * inv_dir_x;
    
    float tmin_x = fminf(t1_x, t2_x);
    float tmax_x = fmaxf(t1_x, t2_x);
    
    // 计算与y平面的交点
    float t1_y = (box_min_y - ray_orig_y) * inv_dir_y;
    float t2_y = (box_max_y - ray_orig_y) * inv_dir_y;
    
    float tmin_y = fminf(t1_y, t2_y);
    float tmax_y = fmaxf(t1_y, t2_y);
    
    // 计算与z平面的交点
    float t1_z = (box_min_z - ray_orig_z) * inv_dir_z;
    float t2_z = (box_max_z - ray_orig_z) * inv_dir_z;
    
    float tmin_z = fminf(t1_z, t2_z);
    float tmax_z = fmaxf(t1_z, t2_z);
    
    // 计算最终的交点范围
    float t_near = fmaxf(fmaxf(tmin_x, tmin_y), tmin_z);
    float t_far = fminf(fminf(tmax_x, tmax_y), tmax_z);
    
    t_near = fmaxf(t_near, t_min);
    t_far = fminf(t_far, t_max);
    
    // 判断是否相交
    return (t_near <= t_far) ? 1 : 0;
}

// 三角形相交测试 (Möller-Trumbore算法)
int intersectTriangle(float ray_orig_x, float ray_orig_y, float ray_orig_z,
                      float ray_dir_x, float ray_dir_y, float ray_dir_z,
                      float v0_x, float v0_y, float v0_z,
                      float v1_x, float v1_y, float v1_z,
                      float v2_x, float v2_y, float v2_z,
                      float* out_t) {
    const float EPSILON = 0.0000001f;
    
    // 计算边向量
    float edge1_x = v1_x - v0_x;
    float edge1_y = v1_y - v0_y;
    float edge1_z = v1_z - v0_z;
    
    float edge2_x = v2_x - v0_x;
    float edge2_y = v2_y - v0_y;
    float edge2_z = v2_z - v0_z;
    
    // 计算叉积 ray_dir × edge2
    float h_x = ray_dir_y * edge2_z - ray_dir_z * edge2_y;
    float h_y = ray_dir_z * edge2_x - ray_dir_x * edge2_z;
    float h_z = ray_dir_x * edge2_y - ray_dir_y * edge2_x;
    
    // 计算行列式
    float a = edge1_x * h_x + edge1_y * h_y + edge1_z * h_z;
    
    // 如果行列式接近0，光线与三角形平行，f会非常大
    // u和v会超出[0,1]范围，会被后续的边界检查自然拒绝
    // 这样可以避免额外的分支判断
    float f = 1.0f / a;
    
    // 计算从v0到ray_origin的向量
    float s_x = ray_orig_x - v0_x;
    float s_y = ray_orig_y - v0_y;
    float s_z = ray_orig_z - v0_z;
    
    // 计算u参数
    float u = f * (s_x * h_x + s_y * h_y + s_z * h_z);
    
    if (u < 0.0f || u > 1.0f) {
        return 0;
    }
    
    // 计算叉积 s × edge1
    float q_x = s_y * edge1_z - s_z * edge1_y;
    float q_y = s_z * edge1_x - s_x * edge1_z;
    float q_z = s_x * edge1_y - s_y * edge1_x;
    
    // 计算v参数
    float v = f * (ray_dir_x * q_x + ray_dir_y * q_y + ray_dir_z * q_z);
    
    if (v < 0.0f || u + v > 1.0f) {
        return 0;
    }
    
    // 计算t参数
    float t = f * (edge2_x * q_x + edge2_y * q_y + edge2_z * q_z);
    
    if (t > EPSILON) {
        *out_t = t;
        return 1;
    }
    
    return 0;
}

int main() {
    // 初始化BVH树数据（完整的3层树结构）
    // 节点0: 根节点，包含整个场景
    // 节点1: 左子树（包含三角形0和1）
    // 节点2: 右子树（包含三角形2和3）
    // 节点3: 叶子节点（三角形0）
    // 节点4: 叶子节点（三角形1）
    // 节点5: 叶子节点（三角形2）
    // 节点6: 叶子节点（三角形3）
    
    float bvh_min_x[8] = {-6.0f, -6.0f, 0.5f, -3.0f, -6.0f, 0.5f, 3.0f, 0.0f};
    float bvh_min_y[8] = {-6.0f, -6.0f, -6.0f, -3.0f, -6.0f, -3.0f, -6.0f, 0.0f};
    float bvh_min_z[8] = {-1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 0.0f};
    
    float bvh_max_x[8] = {6.0f, 0.5f, 6.0f, 0.5f, -3.0f, 3.0f, 6.0f, 0.0f};
    float bvh_max_y[8] = {6.0f, 6.0f, 6.0f, 6.0f, -3.0f, 6.0f, -3.0f, 0.0f};
    float bvh_max_z[8] = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 0.0f};
    
    int bvh_left_child[8] = {1, 3, 5, -1, -1, -1, -1, -1};
    int bvh_right_child[8] = {2, 4, 6, -1, -1, -1, -1, -1};
    int bvh_tri_idx[8] = {-1, -1, -1, 0, 1, 2, 3, -1};
    int bvh_is_leaf[8] = {0, 0, 0, 1, 1, 1, 1, 0};
    
    // 初始化三角形数据（4个三角形）
    // 三角形0: z=0平面，原点附近
    // 三角形1: z=0平面，左下区域
    // 三角形2: z=0平面，右上区域
    // 三角形3: z=0平面，右下区域
    float tri_v0_x[4] = {-2.0f, -5.0f, 1.0f, 4.0f};
    float tri_v0_y[4] = {-2.0f, -5.0f, -2.0f, -5.0f};
    float tri_v0_z[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    
    float tri_v1_x[4] = {2.0f, -4.0f, 2.5f, 5.5f};
    float tri_v1_y[4] = {-2.0f, -4.0f, -2.0f, -4.0f};
    float tri_v1_z[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    
    float tri_v2_x[4] = {0.0f, -4.5f, 1.5f, 4.5f};
    float tri_v2_y[4] = {2.0f, -4.5f, 2.0f, -4.5f};
    float tri_v2_z[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    
    // 光线数据 - 从上方沿着-Z方向发射，瞄准原点附近
    float ray_origin_x = 0.0f;
    float ray_origin_y = 0.0f;
    float ray_origin_z = 10.0f;
    
    float ray_dir_x = 0.0f;
    float ray_dir_y = 0.0f;
    float ray_dir_z = -1.0f;
    
    // 归一化光线方向
    float dir_len = sqrtf(ray_dir_x * ray_dir_x + 
                          ray_dir_y * ray_dir_y + 
                          ray_dir_z * ray_dir_z);
    ray_dir_x /= dir_len;
    ray_dir_y /= dir_len;
    ray_dir_z /= dir_len;
    
    // BVH遍历栈
    int stack[16];
    
    // 输出数据
    float hit_t = FLT_MAX;
    int hit_tri_id = -1;
    
    // 调用ray tracing函数
    rayTraceBVH(ray_origin_x, ray_origin_y, ray_origin_z,
                ray_dir_x, ray_dir_y, ray_dir_z,
                bvh_min_x, bvh_min_y, bvh_min_z,
                bvh_max_x, bvh_max_y, bvh_max_z,
                bvh_left_child, bvh_right_child,
                bvh_tri_idx, bvh_is_leaf,
                tri_v0_x, tri_v0_y, tri_v0_z,
                tri_v1_x, tri_v1_y, tri_v1_z,
                tri_v2_x, tri_v2_y, tri_v2_z,
                stack,
                &hit_t, &hit_tri_id);
    
    // 输出结果
    if (hit_tri_id >= 0) {
        printf("Hit triangle %d at distance %f\n", hit_tri_id, hit_t);
        float hit_x = ray_origin_x + ray_dir_x * hit_t;
        float hit_y = ray_origin_y + ray_dir_y * hit_t;
        float hit_z = ray_origin_z + ray_dir_z * hit_t;
        printf("Hit point: (%f, %f, %f)\n", hit_x, hit_y, hit_z);
    } else {
        printf("No hit\n");
    }
    
    return 0;
}

// BVH树遍历主函数
void rayTraceBVH(float ray_origin_x, float ray_origin_y, float ray_origin_z,
                 float ray_dir_x, float ray_dir_y, float ray_dir_z,
                 float* bvh_min_x, float* bvh_min_y, float* bvh_min_z,
                 float* bvh_max_x, float* bvh_max_y, float* bvh_max_z,
                 int* bvh_left_child, int* bvh_right_child,
                 int* bvh_tri_idx, int* bvh_is_leaf,
                 float* tri_v0_x, float* tri_v0_y, float* tri_v0_z,
                 float* tri_v1_x, float* tri_v1_y, float* tri_v1_z,
                 float* tri_v2_x, float* tri_v2_y, float* tri_v2_z,
                 int* stack,
                 float* hit_t, int* hit_tri_id) {
    
    // 使用传入的栈进行BVH遍历（最大深度16）
    int stack_ptr = 0;
    
    // 初始化
    *hit_t = FLT_MAX;
    *hit_tri_id = -1;
    
    // 将根节点压栈
    stack[stack_ptr] = 0;
    stack_ptr++;
    
    // 迭代遍历BVH树
#pragma clang loop vectorize(disable) interleave(disable)
    while (stack_ptr > 0) {
        // 弹出节点
        stack_ptr--;
        int node_idx = stack[stack_ptr];
        
        // 检查AABB包围盒相交
        int hit_box = intersectAABB(
            ray_origin_x, ray_origin_y, ray_origin_z,
            ray_dir_x, ray_dir_y, ray_dir_z,
            bvh_min_x[node_idx], bvh_min_y[node_idx], bvh_min_z[node_idx],
            bvh_max_x[node_idx], bvh_max_y[node_idx], bvh_max_z[node_idx],
            0.0f, *hit_t
        );
        
        if (hit_box == 0) {
            continue;
        }
        
        // 如果是叶子节点，进行三角形相交测试
        if (bvh_is_leaf[node_idx] == 1) {
            int tri_idx = bvh_tri_idx[node_idx];
            
            if (tri_idx >= 0) {
                float t;
                int hit = intersectTriangle(
                    ray_origin_x, ray_origin_y, ray_origin_z,
                    ray_dir_x, ray_dir_y, ray_dir_z,
                    tri_v0_x[tri_idx], tri_v0_y[tri_idx], tri_v0_z[tri_idx],
                    tri_v1_x[tri_idx], tri_v1_y[tri_idx], tri_v1_z[tri_idx],
                    tri_v2_x[tri_idx], tri_v2_y[tri_idx], tri_v2_z[tri_idx],
                    &t
                );
                
                if (hit == 1 && t < *hit_t) {
                    *hit_t = t;
                    *hit_tri_id = tri_idx;
                }
            }
        } else {
            // 内部节点，将子节点压栈
            int left = bvh_left_child[node_idx];
            int right = bvh_right_child[node_idx];
            
            if (right >= 0 && stack_ptr < 16) {
                stack[stack_ptr] = right;
                stack_ptr++;
            }
            
            if (left >= 0 && stack_ptr < 16) {
                stack[stack_ptr] = left;
                stack_ptr++;
            }
        }
    }
}

