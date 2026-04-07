# 第6章 GGML后端系统 —— 异构计算的"多面手"

## 学习目标
1. 理解ggml_backend的抽象架构设计
2. 掌握CPU后端的SIMD优化原理
3. 了解GPU后端（CUDA/Metal/Vulkan）的架构差异
4. 理解异构计算的任务分发机制

---

## 生活类比：跨国公司的多地区运营

想象GGML是一家**跨国制造集团**：

- **总部（Backend接口）** = 制定统一的运营标准和管理流程
- **各地工厂（具体后端）** = 根据不同地区的资源特色本地化生产
  - **CPU工厂** = 勤劳全能型，任何地方都能开工（通用性最强）
  - **CUDA工厂** = NVIDIA特区的超级流水线（吞吐量最大）
  - **Metal工厂** = Apple生态的精品车间（能效比最优）
- **物流调度（Backend调度器）** = 根据订单特点分配到最合适的工厂
- **标准集装箱（ggml_tensor）** = 统一的货物包装，各地都能识别

就像跨国公司需要统一标准又能因地制宜，GGML的后端系统也需要统一接口又能发挥各平台优势。

---

## 源码地图

```
ggml/include/ggml-backend.h
├── ggml_backend_t         # 后端句柄
├── ggml_backend_buffer_t  # 缓冲区句柄
├── ggml_backend_i         # 后端接口定义（C++虚表风格）
└── 核心API声明

ggml/src/ggml-backend.cpp
├── 后端注册与管理（第100-500行）
├── 缓冲区管理（第500-1000行）
├── 后端调度器（第1000-2000行）
└── 多后端协调（第2000-3000行）

ggml/src/ggml-cpu/
├── ggml-cpu.c            # CPU后端实现
├── ops.h/ops.cpp         # CPU算子
└── 各种ISA优化（AVX/AVX2/AVX512/NEON）

ggml/src/ggml-cuda/
├── ggml-cuda.cu          # CUDA后端主文件
├── 各种算子.cu文件       # CUDA kernels
└── 设备管理

ggml/src/ggml-metal/
├── ggml-metal.m          # Metal后端（Objective-C）
├── ggml-metal.metal      # Metal shaders
└── 缓冲区管理
```

---

## 6.1 后端抽象架构（ggml_backend）

### 6.1.1 后端接口设计 —— 统一标准的制定

**源码位置**：`ggml/include/ggml-backend.h` (第100-200行)

```c
// 后端接口（类似C++虚函数表）
struct ggml_backend_i {
    const char * (*get_name)(ggml_backend_t backend);

    // 内存分配
    ggml_backend_buffer_t (*alloc_buffer)(
        ggml_backend_t backend,
        size_t size);

    // 数据传输
    void (*set_tensor_async)(ggml_backend_t backend,
        struct ggml_tensor * tensor,
        const void * data,
        size_t offset,
        size_t size,
        void * stream);

    void (*get_tensor_async)(ggml_backend_t backend,
        const struct ggml_tensor * tensor,
        void * data,
        size_t offset,
        size_t size,
        void * stream);

    // 同步
    void (*synchronize)(ggml_backend_t backend);

    // 计算图执行
    void (*graph_compute)(ggml_backend_t backend,
        struct ggml_cgraph * cgraph);

    // 能力查询
    bool (*supports_op)(ggml_backend_t backend,
        const struct ggml_tensor * op);
};

// 后端实例
struct ggml_backend {
    struct ggml_backend_i iface;
    void * context;  // 后端私有数据
};
```

**设计模式分析**：
- **C语言模拟虚函数表**：通过函数指针实现多态
- **统一接口，各异实现**：所有后端暴露相同API
- **私有上下文**：`context`字段允许各后端存储私有数据

### 6.1.2 后端能力查询 —— 知己知彼

**源码位置**：`ggml/src/ggml-backend.cpp` (第500-600行)

```cpp
// 检查后端是否支持某算子
bool ggml_backend_supports_op(
        ggml_backend_t backend,
        const struct ggml_tensor * op) {
    return backend->iface.supports_op(backend, op);
}

// 使用示例：选择最佳后端
for (auto &backend : available_backends) {
    if (ggml_backend_supports_op(backend, tensor)) {
        // 使用该后端执行
        break;
    }
}
```

### 6.1.3 缓冲区抽象 —— 统一内存视图

**源码位置**：`ggml/include/ggml-backend.h` (第200-300行)

```c
struct ggml_backend_buffer_i {
    void (*free_buffer)(ggml_backend_buffer_t buffer);
    void * (*get_base)(ggml_backend_buffer_t buffer);
    size_t (*get_size)(ggml_backend_buffer_t buffer);

    // 内存同步
    void (*memset_tensor)(ggml_backend_buffer_t buffer,
        struct ggml_tensor * tensor,
        uint8_t value);
    void (*set_tensor)(ggml_backend_buffer_t buffer,
        struct ggml_tensor * tensor,
        const void * data,
        size_t offset,
        size_t size);
};

struct ggml_backend_buffer {
    struct ggml_backend_buffer_i iface;
    void * context;
    size_t size;
    enum ggml_backend_type type;
};
```

---

## 6.2 CPU后端优化

### 6.2.1 SIMD指令集利用 —— 并行加工的威力

**什么是SIMD？**
- **S**ingle **I**nstruction **M**ultiple **D**ata
- 一条指令同时处理多个数据
- x86: SSE(128bit) → AVX(256bit) → AVX512(512bit)
- ARM: NEON(128bit)

**源码位置**：`ggml/src/ggml-cpu/ggml-cpu.c` (第1000-2000行)

```c
// FP32向量加法 - AVX2版本（一次处理8个float）
void ggml_vec_add_f32_avx2(const int n, float * z, const float * x, const float * y) {
    const int np = (n & ~(8-1));  // 对齐到8的倍数

    // ① AVX2批量处理（每次8个）
    for (int i = 0; i < np; i += 8) {
        __m256 vx = _mm256_loadu_ps(x + i);  // 加载8个float
        __m256 vy = _mm256_loadu_ps(y + i);
        __m256 vz = _mm256_add_ps(vx, vy);    // 一次加8个
        _mm256_storeu_ps(z + i, vz);          // 存储结果
    }

    // ② 剩余元素逐个处理
    for (int i = np; i < n; i++) {
        z[i] = x[i] + y[i];
    }
}
```

### 6.2.2 多线程并行策略 —— 人多力量大

**源码位置**：`ggml/src/ggml-cpu/ggml-cpu.c` (第2000-3000行)

```c
// 线程池任务结构
struct ggml_compute_params {
    int ith;           // 当前线程ID
    int nth;           // 总线程数
    size_t wsize;      // 工作缓冲区大小
    void * wdata;      // 工作缓冲区

    // 线程同步
    atomic_int * shared_n_done;
    bool shared_abort;
};

// 矩阵乘法并行化
void ggml_compute_forward_mul_mat_f32(
    const struct ggml_compute_params * params,
    struct ggml_tensor * dst) {

    const int ith = params->ith;  // 当前线程编号
    const int nth = params->nth;  // 总线程数

    // 将M维度分割给不同线程
    const int64_t m = dst->ne[1];
    const int64_t m_per_thread = (m + nth - 1) / nth;
    const int64_t m_start = ith * m_per_thread;
    const int64_t m_end = MIN(m_start + m_per_thread, m);

    // 每个线程处理自己的行范围
    for (int64_t i = m_start; i < m_end; i++) {
        // 计算第i行结果
    }
}
```

---

## 6.3 GPU后端实现

### 6.3.1 CUDA后端架构 —— NVIDIA特区的超级工厂

**源码位置**：`ggml/src/ggml-cuda/ggml-cuda.cu` (第1-500行)

```cuda
// CUDA后端上下文
struct ggml_cuda_context {
    cudaStream_t stream;           // CUDA流（异步执行）
    cublasHandle_t cublas_handle;  // cuBLAS句柄（GEMM优化）
    int device;                    // GPU设备ID

    // 内存池
    std::unordered_map<void*, cudaMemoryType> tensor_cache;
};

// CUDA后端实现的关键函数
static void ggml_cuda_set_tensor(...)
static void ggml_cuda_get_tensor(...)
static void ggml_cuda_graph_compute(...)
```

**CUDA后端特点**：
1. **cuBLAS加速**：大矩阵乘法调用高度优化的cuBLAS库
2. **Kernel融合**：小算子合并为单个CUDA kernel，减少启动开销
3. **异步执行**：CPU提交任务后立即返回，GPU并行计算
4. **多流并行**：不同计算图可在不同CUDA流并发执行

### 6.3.2 Metal后端原理 —— Apple生态的精品车间

**源码位置**：`ggml/src/ggml-metal/ggml-metal.m` (第1-500行)

```objc
// Metal后端上下文
@interface ggml_metal_context {
    id<MTLDevice> device;           // Metal设备
    id<MTLCommandQueue> queue;      // 命令队列
    id<MTLLibrary> library;         // Metal shader库

    // 算子pipeline缓存
    NSMutableDictionary * pipelines;
}

// 执行计算图
- (void) graph_compute:(struct ggml_cgraph *)cgraph {
    id<MTLCommandBuffer> cmd_buffer = [queue commandBuffer];

    for (int i = 0; i < cgraph->n_nodes; i++) {
        struct ggml_tensor * node = cgraph->nodes[i];
        id<MTLComputePipelineState> pipeline = [self getPipeline:node->op];

        id<MTLComputeCommandEncoder> encoder = [cmd_buffer computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        // 设置参数、调度执行...
        [encoder endEncoding];
    }

    [cmd_buffer commit];
    [cmd_buffer waitUntilCompleted];
}
@end
```

**Metal后端特点**：
1. **Unified Memory**：CPU/GPU共享内存，零拷贝数据传输
2. **Shader预编译**：Metal shader可预编译，启动更快
3. **能效优化**：Apple Silicon的GPU能效比优异

### 6.3.3 Vulkan/SYCL后端 —— 跨平台的选择

**Vulkan后端**：
- 目标：跨平台GPU（Windows/Linux/Android）
- 技术：SPIR-V中间表示，Vulkan compute shaders
- 状态：相对较新，持续优化中

**SYCL后端**：
- 目标：Intel GPU/Xeon Phi
- 技术：基于oneAPI的SYCL标准
- 优势：代码可移植性好

---

## 6.4 异构计算与任务分发

### 6.4.1 后端调度器 —— 物流调度中心

**源码位置**：`ggml/src/ggml-backend.cpp` (第1000-1500行)

```cpp
// 后端调度器
struct ggml_backend_sched {
    // 注册的后端
    std::vector<ggml_backend_t> backends;

    // 张量分配策略
    std::unordered_map<ggml_tensor*, ggml_backend_t> tensor_backend;

    // 分割点（某些算子在CPU执行更快）
    std::vector<int> split_points;
};

// 调度决策流程
void ggml_backend_sched_alloc_splits(...) {
    for (int i = 0; i < graph->n_nodes; i++) {
        struct ggml_tensor * node = graph->nodes[i];

        // ① 检查各后端是否支持该算子
        for (auto &backend : sched->backends) {
            if (ggml_backend_supports_op(backend, node)) {
                // ② 评估在该后端执行的成本
                float cost = estimate_cost(backend, node);
                // ③ 选择成本最低的后端
                // ...
            }
        }
    }
}
```

### 6.4.2 CPU-GPU协作模式

**场景1：纯GPU执行**
```
输入 -> [GPU] -> 算子1 -> 算子2 -> 算子3 -> 输出
         ↑ 所有计算在GPU完成
```

**场景2：混合执行**
```
输入 -> [GPU] -> 矩阵乘 -> [CPU] -> 特殊算子 -> [GPU] -> 输出
         ↑ 大部分计算      ↑ CPU更适合某些算子
```

**场景3：模型分层**
```
模型有40层，设置n_gpu_layers=35

Layer 0-34: 在GPU执行（速度快）
Layer 35-39: 在CPU执行（显存不足）
```

**源码位置**：`src/llama-model.cpp` (模型加载相关代码)

---

## 设计中的取舍

### 为什么后端接口用C而不是C++？

| 方案 | 优点 | 缺点 | GGML选择 |
|-----|------|------|---------|
| 纯C接口 | ABI稳定，易绑定 | 需手动实现多态 | **是** |
| C++虚函数 | 天然多态 | ABI不稳定，难绑定 | 否 |
| COM风格 | 微软生态友好 | 复杂度高 | 否 |

**GGML的权衡**：
- 核心接口用C，保证最大兼容性
- 内部实现可以用C++（如CUDA后端）
- 通过函数指针表模拟虚函数

### 为什么小算子不都放到GPU执行？

**GPU的开销**：
1. **Kernel启动开销**：~10-50μs
2. **数据传输开销**：PCIe带宽限制
3. **内存分配开销**：GPU显存管理

**决策阈值**：
```
if (算子计算量 > GPU开销) {
    使用GPU
} else {
    使用CPU（避免来回传输）
}
```

**实际策略**：
- 大批量矩阵乘 → GPU
- 小向量操作 → CPU
- 量化反量化 → GPU（和矩阵乘融合）

---

## 动手练习

### 练习1：检测可用后端
编写程序列出系统上所有可用的GGML后端：
```c
// 提示：使用 ggml_backend_t backends[GGML_MAX_BACKENDS];
// 和 ggml_backend_get_count() 等API
```

### 练习2：对比CPU vs CUDA性能
使用llama-bench测试同一模型在CPU和CUDA后端下的性能：
```bash
# CPU版本
./llama-bench -m model.gguf -t 4

# CUDA版本
./llama-bench -m model.gguf -ngl 99
```
分析：
- 哪种后端的tokens/s更高？
- 内存占用差异？
- 不同batch size下的表现？

### 练习3：分析后端调度
阅读 `ggml/src/ggml-backend.cpp` 第1500-2000行，回答：
1. `ggml_backend_sched`如何决定张量分配到哪个后端？
2. 何时会发生CPU-GPU之间的数据传输？
3. 如何减少不必要的数据拷贝？

---

## 本课小结

| 概念 | 一句话总结 |
|------|-----------|
| ggml_backend_i | 后端接口定义，C语言模拟虚函数表 |
| supports_op | 查询后端是否支持某算子 |
| CPU后端 | SIMD+多线程，通用但非最优 |
| CUDA后端 | cuBLAS+异步，吞吐量最高 |
| Metal后端 | Unified Memory，能效比最优 |
| 后端调度器 | 自动选择最佳后端执行任务 |

---

*本章对应源码版本：master (2026-04-07)*
