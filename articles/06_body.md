# 第6章 GGML后端系统 —— 异构计算的"多面手"

现代计算环境日益多样化：从 x86 CPU 到 ARM 芯片，从 NVIDIA GPU 到 Apple Silicon，从数据中心到移动设备。GGML 的后端系统（Backend System）提供了一个统一的抽象层，让同一份代码能够在各种硬件上高效运行。本章将深入解析这一"多面手"架构。

## 学习目标

1. 理解 `ggml_backend` 的抽象架构设计
2. 掌握 CPU 后端的 SIMD 优化原理
3. 了解 GPU 后端（CUDA/Metal/Vulkan）的架构差异
4. 理解异构计算的任务分发机制

## 生活类比：跨国公司的多地区运营

想象 GGML 是一家**跨国制造集团**：

- **总部（Backend 接口）** = 制定统一的运营标准和管理流程，确保全球各地工厂遵循相同规范
- **各地工厂（具体后端）** = 根据不同地区的资源特色进行本地化生产
  - **CPU 工厂** = 勤劳全能型，任何地方都能开工（通用性最强，但非最优）
  - **CUDA 工厂** = NVIDIA 特区的超级流水线（吞吐量最大，火力全开）
  - **Metal 工厂** = Apple 生态的精品车间（能效比最优，精致高效）
- **物流调度（Backend 调度器）** = 根据订单特点分配到最合适的工厂生产
- **标准集装箱（ggml_tensor）** = 统一的货物包装规格，各地工厂都能识别处理

就像跨国公司需要统一标准又能因地制宜，GGML 的后端系统也需要统一接口又能发挥各平台优势。这个类比将贯穿本章。

---

## 6.1 后端抽象架构（ggml_backend）

### 6.1.1 后端接口设计 —— 统一标准的制定

**源码位置**：`ggml/include/ggml-backend.h`（第 100-200 行）

```c
// 后端接口定义（C 语言模拟 C++ 虚函数表）
struct ggml_backend_i {
    // 获取后端名称
    const char * (*get_name)(ggml_backend_t backend);

    // ========== 内存分配 ==========
    ggml_backend_buffer_t (*alloc_buffer)(
        ggml_backend_t backend,
        size_t size);

    // ========== 数据传输 ==========
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

    // ========== 同步 ==========
    void (*synchronize)(ggml_backend_t backend);

    // ========== 计算图执行 ==========
    void (*graph_compute)(ggml_backend_t backend,
        struct ggml_cgraph * cgraph);

    // ========== 能力查询 ==========
    bool (*supports_op)(ggml_backend_t backend,
        const struct ggml_tensor * op);
};

// 后端实例结构
struct ggml_backend {
    struct ggml_backend_i iface;    // 接口函数表
    void * context;                  // 后端私有数据
    enum ggml_backend_type type;    // 后端类型标识
};

这段代码定义了GGML后端抽象接口。通过函数指针表(iface)实现多态，允许不同硬件后端(CPU/CUDA/Metal等)提供统一接口。context字段存储后端私有数据(如CUDA流)，实现接口与实现的分离。
```

**设计模式分析**：

- **C 语言模拟虚函数表**：通过函数指针实现多态，避免 C++ ABI 兼容性问题
- **统一接口，各异实现**：所有后端暴露相同的 API，但内部实现完全不同
- **私有上下文**：`context` 字段允许各后端存储自己的私有数据（如 CUDA stream）

### 6.1.2 后端能力查询 —— 知己知彼

**源码位置**：`ggml/src/ggml-backend.cpp`（第 500-600 行）

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
        // 该后端支持此算子，可以执行
        // 进一步评估性能...
        float cost = estimate_cost(backend, tensor);
        if (cost < best_cost) {
            best_backend = backend;
        }
    }
}

这段代码展示后端能力查询和调度选择。`supports_op`函数检查后端是否支持特定算子，调度器遍历所有可用后端，选择支持该算子且成本最低的后端执行，实现异构计算的自动优化。
```

### 6.1.3 缓冲区抽象 —— 统一内存视图

**源码位置**：`ggml/include/ggml-backend.h`（第 200-300 行）

```c
// 缓冲区接口
struct ggml_backend_buffer_i {
    void (*free_buffer)(ggml_backend_buffer_t buffer);
    void * (*get_base)(ggml_backend_buffer_t buffer);
    size_t (*get_size)(ggml_backend_buffer_t buffer);

    // 内存操作
    void (*memset_tensor)(ggml_backend_buffer_t buffer,
        struct ggml_tensor * tensor,
        uint8_t value);
    void (*set_tensor)(ggml_backend_buffer_t buffer,
        struct ggml_tensor * tensor,
        const void * data,
        size_t offset,
        size_t size);
    void (*get_tensor)(ggml_backend_buffer_t buffer,
        const struct ggml_tensor * tensor,
        void * data,
        size_t offset,
        size_t size);
};

struct ggml_backend_buffer {
    struct ggml_backend_buffer_i iface;
    void * context;
    size_t size;
    enum ggml_backend_type type;
    void * base;  // 内存基地址
};

这段代码定义了后端缓冲区抽象接口。用于统一不同后端(CPU/GPU)的内存管理，提供分配、释放、数据拷贝等操作。base字段指向实际内存地址，可以是主机内存(CPU)或设备内存(GPU)。
```

---

## 6.2 CPU 后端优化

### 6.2.1 SIMD 指令集利用 —— 并行加工的威力

**什么是 SIMD？**

- **S**ingle **I**nstruction **M**ultiple **D**ata（单指令多数据）
- 一条指令同时处理多个数据元素
- x86 演进：SSE(128bit) → AVX(256bit) → AVX512(512bit)
- ARM：NEON(128bit)

**源码位置**：`ggml/src/ggml-cpu/ggml-cpu.c`（第 1000-2000 行）

```c
// FP32 向量加法 - AVX2 版本（一次处理 8 个 float）
void ggml_vec_add_f32_avx2(const int n, float * z, const float * x, const float * y) {
    const int np = (n & ~(8-1));  // 对齐到 8 的倍数

    // ① AVX2 批量处理（每次 8 个 float）
    for (int i = 0; i < np; i += 8) {
        __m256 vx = _mm256_loadu_ps(x + i);   // 加载 8 个 float 到 SIMD 寄存器
        __m256 vy = _mm256_loadu_ps(y + i);
        __m256 vz = _mm256_add_ps(vx, vy);     // 一条指令加 8 个数
        _mm256_storeu_ps(z + i, vz);           // 存储结果
    }

    // ② 剩余元素逐个处理（收尾）
    for (int i = np; i < n; i++) {
        z[i] = x[i] + y[i];
    }
}

这段代码使用AVX2指令集实现向量加法并行计算。`_mm256_loadu_ps`加载8个float到256位SIMD寄存器，`_mm256_add_ps`单指令完成8个元素的加法，理论加速比达8倍。剩余元素用标量循环处理。
```

**SIMD 性能提升**：

| 指令集 | 位宽 | 每指令处理 float 数 | 理论加速比 |
|-------|------|-------------------|-----------|
| 标量 | 32 | 1 | 1x |
| SSE | 128 | 4 | 4x |
| AVX | 256 | 8 | 8x |
| AVX512 | 512 | 16 | 16x |

实际加速比受限于内存带宽和指令调度。

### 6.2.2 多线程并行策略 —— 人多力量大

**源码位置**：`ggml/src/ggml-cpu/ggml-cpu.c`（第 2000-3000 行）

```c
// 线程池任务结构
struct ggml_compute_params {
    int ith;                    // 当前线程 ID (0 to nth-1)
    int nth;                    // 总线程数
    size_t wsize;               // 工作缓冲区大小
    void * wdata;               // 工作缓冲区指针

    // 线程同步
    atomic_int * shared_n_done; // 已完成任务的计数
    bool shared_abort;          // 中止标志
};

// 矩阵乘法并行化示例
void ggml_compute_forward_mul_mat_f32(
    const struct ggml_compute_params * params,
    struct ggml_tensor * dst) {

    const int ith = params->ith;   // 当前线程编号
    const int nth = params->nth;   // 总线程数

    // 将 M 维度分割给不同线程
    const int64_t m = dst->ne[1];
    const int64_t m_per_thread = (m + nth - 1) / nth;  // 向上取整
    const int64_t m_start = ith * m_per_thread;
    const int64_t m_end = MIN(m_start + m_per_thread, m);

    // 每个线程处理自己的行范围 [m_start, m_end)
    for (int64_t i = m_start; i < m_end; i++) {
        // 计算第 i 行的结果
        for (int64_t j = 0; j < dst->ne[0]; j++) {
            float sum = 0.0f;
            for (int64_t k = 0; k < src0->ne[0]; k++) {
                sum += src0_data[i * src0->ne[0] + k] * 
                       src1_data[k * src1->ne[0] + j];
            }
            dst_data[i * dst->ne[0] + j] = sum;
        }
    }
}

这段代码展示了多线程并行矩阵乘法。将输出矩阵的行维度均匀分配给各线程，每个线程计算指定的行范围[m_start, m_end)，通过并行计算充分利用多核CPU性能，实现近似线性的加速比。
```

**并行策略**：

1. **数据并行**：将输入数据分割给多个线程
2. **任务并行**：不同的算子可以并行执行（如果无依赖）
3. **流水线并行**：CPU-GPU 协作时的重叠执行

---

## 6.3 GPU 后端实现

### 6.3.1 CUDA 后端架构 —— NVIDIA 特区的超级工厂

**源码位置**：`ggml/src/ggml-cuda/ggml-cuda.cu`（第 1-500 行）

```cuda
// CUDA 后端上下文
struct ggml_cuda_context {
    cudaStream_t stream;              // CUDA 流（异步执行管道）
    cublasHandle_t cublas_handle;     // cuBLAS 句柄（GEMM 优化库）
    int device;                       // GPU 设备 ID

    // 内存池管理
    std::unordered_map<void*, cudaMemoryType> tensor_cache;
    
    // 设备属性缓存
    cudaDeviceProp device_props;
};

// CUDA 后端接口实现
static const struct ggml_backend_i ggml_backend_cuda_i = {
    .get_name = ggml_cuda_get_name,
    .alloc_buffer = ggml_cuda_alloc_buffer,
    .free_buffer = ggml_cuda_free_buffer,
    .set_tensor_async = ggml_cuda_set_tensor_async,
    .get_tensor_async = ggml_cuda_get_tensor_async,
    .synchronize = ggml_cuda_synchronize,
    .graph_compute = ggml_cuda_graph_compute,
    .supports_op = ggml_cuda_supports_op,
};

// 核心计算函数
static void ggml_cuda_graph_compute(
        ggml_backend_t backend,
        struct ggml_cgraph * cgraph) {
    ggml_cuda_context * ctx = (ggml_cuda_context *)backend->context;

    for (int i = 0; i < cgraph->n_nodes; i++) {
        struct ggml_tensor * node = cgraph->nodes[i];

        switch (node->op) {
            case GGML_OP_MUL_MAT:
                // 大矩阵乘法调用 cuBLAS
                ggml_cuda_mul_mat(ctx, node);
                break;
            case GGML_OP_SILU:
                // 激活函数调用自定义 kernel
                ggml_cuda_silu(ctx, node);
                break;
            // ... 更多算子
        }
    }
}

这段代码定义了CUDA后端的核心结构和实现。包含CUDA流(异步执行)、cuBLAS句柄(矩阵乘法优化)、设备信息及内存缓存。graph_compute函数遍历计算图，根据操作类型分派到对应的CUDA内核或cuBLAS库函数执行。
```

**CUDA 后端特点**：

1. **cuBLAS 加速**：大矩阵乘法调用高度优化的 cuBLAS 库
2. **Kernel 融合**：小算子合并为单个 CUDA kernel，减少启动开销
3. **异步执行**：CPU 提交任务后立即返回，GPU 并行计算
4. **多流并行**：不同计算图可在不同 CUDA 流并发执行

### 6.3.2 Metal 后端原理 —— Apple 生态的精品车间

**源码位置**：`ggml/src/ggml-metal/ggml-metal.m`（第 1-500 行）

```objc
// Metal 后端上下文
@interface ggml_metal_context : NSObject
@property (nonatomic, strong) id<MTLDevice> device;           // Metal 设备
@property (nonatomic, strong) id<MTLCommandQueue> queue;      // 命令队列
@property (nonatomic, strong) id<MTLLibrary> library;         // Metal shader 库
@property (nonatomic, strong) NSMutableDictionary * pipelines; // Pipeline 缓存
@end

@implementation ggml_metal_context

// 执行计算图
- (void) graph_compute:(struct ggml_cgraph *)cgraph {
    id<MTLCommandBuffer> cmd_buffer = [self.queue commandBuffer];

    for (int i = 0; i < cgraph->n_nodes; i++) {
        struct ggml_tensor * node = cgraph->nodes[i];
        id<MTLComputePipelineState> pipeline = [self getPipeline:node->op];

        id<MTLComputeCommandEncoder> encoder = [cmd_buffer computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        
        // 设置参数
        [encoder setBuffer:(id<MTLBuffer>)node->data offset:0 atIndex:0];
        // ... 设置更多参数
        
        // 调度执行
        MTLSize gridSize = MTLSizeMake(ne0, ne1, ne2);
        MTLSize threadGroupSize = MTLSizeMake(256, 1, 1);
        [encoder dispatchThreadgroups:gridSize threadsPerThreadgroup:threadGroupSize];
        
        [encoder endEncoding];
    }

    [cmd_buffer commit];
    [cmd_buffer waitUntilCompleted];
}

@end
```

**Metal 后端特点**：

1. **Unified Memory**：CPU/GPU 共享内存，零拷贝数据传输
2. **Shader 预编译**：Metal shader 可预编译，启动更快
3. **能效优化**：Apple Silicon 的 GPU 能效比优异
4. **自动内存管理**：Metal 的 ARC 简化内存管理

### 6.3.3 跨平台后端对比

| 后端 | 平台 | 内存模型 | 性能特点 | 开发状态 |
|-----|------|---------|---------|---------|
| CPU | 通用 | 统一 | 通用，优化好 | 成熟稳定 |
| CUDA | NVIDIA | 分离 | 吞吐量最高 | 成熟稳定 |
| Metal | Apple | 统一 | 能效比最优 | 成熟稳定 |
| Vulkan | 跨平台 | 分离 | 通用性好 | 持续优化 |
| SYCL | Intel | 分离 | Intel GPU 优化 | 实验阶段 |

---

## 6.4 异构计算与任务分发

### 6.4.1 后端调度器 —— 物流调度中心

**源码位置**：`ggml/src/ggml-backend.cpp`（第 1000-1500 行）

```cpp
// 后端调度器
struct ggml_backend_sched {
    // 注册的后端列表
    std::vector<ggml_backend_t> backends;

    // 张量到后端的映射
    std::unordered_map<ggml_tensor*, ggml_backend_t> tensor_backend;

    // 计算图分割点（某些算子在 CPU 执行更快）
    std::vector<int> split_points;
    
    // 张量复制指令（CPU-GPU 之间）
    std::vector<ggml_backend_cpy> copies;
};

// 调度决策流程
void ggml_backend_sched_alloc_splits(
        ggml_backend_sched_t sched,
        struct ggml_cgraph * graph) {
    for (int i = 0; i < graph->n_nodes; i++) {
        struct ggml_tensor * node = graph->nodes[i];

        // ① 检查各后端是否支持该算子
        for (auto &backend : sched->backends) {
            if (ggml_backend_supports_op(backend, node)) {
                // ② 评估在该后端执行的成本
                float cost = estimate_cost(backend, node);
                
                // ③ 考虑数据传输成本
                cost += estimate_transfer_cost(node, backend);
                
                // ④ 选择成本最低的后端
                if (cost < best_cost) {
                    best_backend = backend;
                }
            }
        }
        
        sched->tensor_backend[node] = best_backend;
    }
}
```

### 6.4.2 CPU-GPU 协作模式

**场景 1：纯 GPU 执行**
```
输入 -> [GPU] -> 算子1 -> 算子2 -> 算子3 -> 输出
         ↑ 所有计算在 GPU 完成
```

**场景 2：混合执行**
```
输入 -> [GPU] -> 矩阵乘 -> [CPU] -> 特殊算子 -> [GPU] -> 输出
         ↑ 大部分计算      ↑ CPU 更适合某些算子（如复杂控制流）
```

**场景 3：模型分层（最常用）**
```
模型有 40 层，设置 n_gpu_layers=35

Layer 0-34: 在 GPU 执行（速度快）
Layer 35-39: 在 CPU 执行（显存不足时的折中）

优势：灵活平衡速度和显存占用
```

---

## 6.5 设计中的取舍

### 为什么后端接口用 C 而不是 C++？

| 方案 | 优点 | 缺点 | GGML 选择 |
|-----|------|------|-----------|
| 纯 C 接口 | ABI 稳定，易绑定 | 需手动实现多态 | ✅ 是 |
| C++ 虚函数 | 天然多态 | ABI 不稳定，难绑定 | ❌ |
| COM 风格 | 微软生态友好 | 复杂度高 | ❌ |

**GGML 的权衡**：
- 核心接口用 C，保证最大兼容性
- 内部实现可以用 C++（如 CUDA 后端）
- 通过函数指针表模拟虚函数，灵活性足够

### 为什么小算子不都放到 GPU 执行？

**GPU 的开销**：

1. **Kernel 启动开销**：~10-50μs
2. **数据传输开销**：PCIe 带宽限制（~16GB/s）
3. **内存分配开销**：GPU 显存管理成本

**决策阈值**：
```
if (算子计算量 > GPU 开销 + 数据传输开销) {
    使用 GPU
} else {
    使用 CPU（避免来回传输的开销）
}
```

**实际策略**：
- 大批量矩阵乘 → GPU（计算密集型）
- 小向量操作 → CPU（通信开销大）
- 量化反量化 → GPU（和矩阵乘融合）
- 复杂控制流 → CPU（GPU 不擅长）

---

## 6.6 动手练习

### 练习 1：检测可用后端

编写程序列出系统上所有可用的 GGML 后端：

```c
#include "ggml-backend.h"

int main() {
    // 获取后端数量
    int n_backends = ggml_backend_get_count();
    printf("Available backends: %d\n", n_backends);
    
    // 遍历所有后端
    for (int i = 0; i < n_backends; i++) {
        ggml_backend_t backend = ggml_backend_get(i);
        printf("  [%d] %s\n", i, ggml_backend_get_name(backend));
    }
    
    return 0;
}
```

### 练习 2：对比 CPU vs CUDA 性能

使用 llama-bench 测试同一模型在不同后端下的性能：

```bash
# CPU 版本（4 线程）
./llama-bench -m model.gguf -t 4

# CUDA 版本（所有层在 GPU）
./llama-bench -m model.gguf -ngl 99

# 混合版本（前 20 层在 GPU）
./llama-bench -m model.gguf -ngl 20
```

分析：
- 哪种后端的 tokens/s 最高？
- 内存占用差异？
- 不同 batch size 下的表现？

### 练习 3：分析后端调度

阅读 `ggml/src/ggml-backend.cpp` 第 1500-2000 行，回答：

1. `ggml_backend_sched` 如何决定张量分配到哪个后端？
   - 提示：考虑算子支持、执行成本、数据传输成本

2. 何时会发生 CPU-GPU 之间的数据传输？
   - 提示：输入数据在 CPU，GPU 计算前需要传输

3. 如何减少不必要的数据拷贝？
   - 提示： Unified Memory、预分配、数据局部性

---

## 6.7 本章小结

| 概念 | 一句话总结 |
|------|-----------|
| `ggml_backend_i` | 后端接口定义，C 语言模拟虚函数表实现多态 |
| `supports_op` | 查询后端是否支持某算子，用于调度决策 |
| CPU 后端 | SIMD + 多线程，通用但非最优，适合小任务 |
| CUDA 后端 | cuBLAS + 异步执行，吞吐量最高，适合大矩阵 |
| Metal 后端 | Unified Memory，能效比最优，适合 Apple 设备 |
| 后端调度器 | 自动选择最佳后端，平衡性能和内存 |
| `n_gpu_layers` | 模型分层策略，灵活分配 CPU/GPU 负载 |

**下一步预告**：

在掌握了后端系统后，我们将在第 7 章探索 llama.cpp 支持的模型架构——理解如何支持 50+ 种不同的 LLM 架构。
