// phase4_feedback.cpp
// Phase 4: 测试更新/广播延迟（控制主导）
// 核心：归约操作 - 从多个候选中找最小值并更新

// 简化的 Phase4 kernel：归约更新操作
// n 作为参数传入，使循环不可展开
static inline void kernel_phase4_feedback(
    float  t_batch[],      // 输入：一批候选 t 值
    int    n,              // 输入：批次大小（不可展开）
    float& best_t,         // 输出：最优 t 值
    int&   best_idx)       // 输出：最优索引
{
    for (int i = 0; i < n; i++) {
        // 核心操作：比较 + 条件更新（测试控制依赖和反馈延迟）
        if (t_batch[i] < best_t) {
            best_t   = t_batch[i];
            best_idx = i;
        }
    }
}

int main(int argc, char* argv[]) {
    // 模拟来自 Phase3 的一批候选距离
    float t_batch[] = {8.2f, 3.7f, 5.4f, 2.9f, 9.1f};
    int n = (argc > 1) ? 5 : 5;  // 使用 argc 让 n 运行时确定
    
    float best_t = 99999.0f;
    int   best_idx = -1;
    
    kernel_phase4_feedback(t_batch, n, best_t, best_idx);
    
    return best_idx;
}
