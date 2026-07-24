#include <stdio.h>
#include <string.h>
#include <math.h>

// 函数声明
void renderPixel(float pixel_x, float pixel_y, int W, int H,
                 float* out_color,
                 float* points_x, float* points_y,
                 float* features,
                 float* conic_x, float* conic_y, float* conic_z,
                 float* opacity,
                 float* depths,
                 int* point_indices);

int main() {
    // 初始化输入数据
    float points_x[32] = {
        100.5, 120.7, 240.8, 245.8, 260.8, 247.6, 230.7, 182.1,
        151.6, 215.6, 226.8, 240.4, 191.4, 218.8, 235.6, 247.8,
        410.5, 383.6, 233.6, 204.8, 170.9, 245.8, 280.5, 213.6,
        377.3, 245.7, 218.9, 196.6, 420.9, 383.6, 225.8, 245.7
    };
    
    float points_y[32] = {
        151.6, 196.6, 300.5, 227.6, 151.6, 182.4, 215.2, 240.7,
        218.8, 170.5, 247.9, 196.6, 230.2, 151.6, 250.5, 235.3,
        226.8, 215.8, 182.1, 245.6, 200.5, 160.6, 397.7, 218.6,
        245.8, 230.1, 205.8, 191.6, 375.6, 310.5, 227.9, 204.8
    };
    
    float features[96] = {
        0.8, 0.2, 0.3, 0.7, 0.9, 0.4, 0.5, 0.5, 0.8, 0.2, 0.3, 0.7,
        0.65, 0.8, 0.1, 0.3, 0.2, 0.9, 0.5, 0.65, 0.7, 0.8, 0.3, 0.2,
        0.1, 0.9, 0.8, 0.3, 0.7, 0.4, 0.5, 0.65, 0.2, 0.8, 0.9, 0.1,
        0.7, 0.3, 0.65, 0.2, 0.8, 0.5, 0.4, 0.9, 0.1, 0.65, 0.2, 0.8,
        0.3, 0.7, 0.4, 0.9, 0.2, 0.5, 0.8, 0.1, 0.65, 0.3, 0.8, 0.7,
        0.4, 0.2, 0.9, 0.5, 0.1, 0.7, 0.8, 0.4, 0.3, 0.65, 0.2, 0.5,
        0.9, 0.1, 0.7, 0.4, 0.8, 0.3, 0.65, 0.2, 0.9, 0.5, 0.7, 0.1,
        0.8, 0.65, 0.2, 0.4, 0.9, 0.3, 0.5, 0.7, 0.1, 0.4, 0.65
    };
    
    float conic_x[32] = {
        0.5, 0.7, 0.65, 0.8, 0.5, 0.9, 0.7, 0.65, 0.8, 0.5, 0.65, 0.7,
        0.9, 0.65, 0.5, 0.8, 0.7, 0.65, 0.5, 0.9, 0.8, 0.65, 0.7, 0.5,
        0.9, 0.8, 0.65, 0.7, 0.5, 0.9, 0.8, 0.65
    };
    
    float conic_y[32] = {
        0.1, 0.2, -0.1, 0.3, -0.2, 0.1, -0.3, 0.2, -0.1, 0.3, -0.2, 0.1,
        -0.3, 0.2, -0.1, 0.3, -0.2, 0.1, -0.3, 0.2, -0.1, 0.3, -0.2, 0.1,
        -0.3, 0.2, -0.1, 0.3, -0.2, 0.1, -0.3, 0.2
    };
    
    float conic_z[32] = {
        0.5, 0.65, 0.8, 0.7, 0.5, 0.9, 0.65, 0.7, 0.8, 0.5, 0.65, 0.7,
        0.9, 0.65, 0.5, 0.8, 0.7, 0.65, 0.5, 0.9, 0.8, 0.65, 0.7, 0.5,
        0.9, 0.8, 0.65, 0.7, 0.5, 0.9, 0.8, 0.65
    };
    
    float opacity[32] = {
        0.7, 0.8, 0.9, 0.65, 0.8, 0.7, 0.9, 0.8, 0.65, 0.9, 0.7, 0.8,
        0.65, 0.9, 0.7, 0.8, 0.9, 0.7, 0.8, 0.65, 0.9, 0.7, 0.8, 0.9,
        0.7, 0.8, 0.9, 0.65, 0.8, 0.7, 0.9, 0.8
    };
    
    float depths[32] = {
        10.5, 15.2, 20.7, 18.6, 25.6, 30.2, 12.6, 22.2, 14.8, 28.5, 16.7,
        24.2, 19.8, 27.6, 13.8, 21.5, 17.6, 26.8, 11.8, 29.7, 23.3, 15.8,
        20.6, 13.5, 25.3, 19.4, 12.9, 27.8, 14.6, 22.8, 18.8, 26.2
    };
    
    int point_indices[32] = {
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
        16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31
    };
    
    float output_color[3];
    
    // 调用渲染函数
    renderPixel(250.0f, 250.0f, 800, 600,
                output_color,
                points_x, points_y,
                features,
                conic_x, conic_y, conic_z,
                opacity,
                depths,
                point_indices);
    
    // 输出结果
    printf("Color: R=%f, G=%f, B=%f\n", 
           output_color[0], output_color[1], output_color[2]);
    
    return 0;
}

void renderPixel(float pixel_x, float pixel_y, int W, int H,
                 float* out_color,
                 float* points_x, float* points_y,
                 float* features,
                 float* conic_x, float* conic_y, float* conic_z,
                 float* opacity,
                 float* depths,
                 int* point_indices) {
    float T = 1.0f;
    float C0 = 0.0f;
    float C1 = 0.0f;
    float C2 = 0.0f;
    
#pragma clang loop vectorize(disable) interleave(disable)
    for (int i = 0; i < 32; i++) {
        int idx = point_indices[i];
        
        // Load and compute distance
        float d_x = points_x[idx] - pixel_x;
        float d_y = points_y[idx] - pixel_y;
        
        // Load conic parameters
        float con_x = conic_x[idx];
        float con_y = conic_y[idx];
        float con_z = conic_z[idx];
        
        // Calculate power value
        float power = -0.5f * (con_x * d_x * d_x + con_z * d_y * d_y) - con_y * d_x * d_y;
        
        // Calculate alpha (clamp to prevent numerical issues)
        float alpha_raw = opacity[idx] * (1.0f + power + 0.5f * power * power);
        // Inline clamp: clamp alpha_raw to [0, 0.99]
        float alpha_temp = fmaxf(0.0f, alpha_raw);
        float alpha = fminf(0.99f, alpha_temp);
        
        // Accumulate color (weighted by alpha * T)
        float weight = alpha * T;
        C0 += features[idx * 3 + 0] * weight;
        C1 += features[idx * 3 + 1] * weight;
        C2 += features[idx * 3 + 2] * weight;
        
        // Update transparency
        T = T * (1.0f - alpha);
    }
    
    // Output final color
    out_color[0] = C0;
    out_color[1] = C1;
    out_color[2] = C2;
}

