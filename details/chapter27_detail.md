# 第27章 性能优化技巧 —— 让模型"跑得更快的秘籍"

## 1. 学习目标

- 掌握CMake编译优化选项的配置方法
- 理解运行时批处理、线程、缓存参数调优策略
- 学习内存映射和权重加载优化技术
- 了解移动设备和边缘设备的能耗优化方法
- 掌握性能测试与瓶颈分析技巧

## 2. 生活类比：赛车调校的艺术

想象F1赛车比赛——同样的引擎，通过调校悬挂、轮胎、空气动力学套件，可以在不同赛道上获得最佳表现。llama.cpp的性能优化也是如此：编译优化像是引擎调校（榨取硬件极限），运行时优化像是轮胎和悬挂选择（适应不同场景），内存优化像是减轻车重（提升加速性能）。而`llama-bench`就是我们的测功机和计时器。

## 3. 源码地图

| 文件路径 | 职责 | 核心内容 |
|---------|------|---------|
| `CMakeLists.txt` | 编译配置 | 优化标志、后端选择、功能开关 |
| `src/llama-context.cpp` | 上下文管理 | 批处理大小、线程数配置 |
| `src/llama-kv-cache.cpp` | KV缓存 | 缓存大小、滑动窗口策略 |
| `src/llama-mmap.cpp` | 内存映射 | 文件映射、预读策略 |
| `examples/llama-bench/llama-bench.cpp` | 基准测试 | 性能测试、结果分析 |
| `common/log.cpp` | 日志系统 | 性能日志、调试信息 |

## 4. 详细章节内容

### 4.1 编译优化

#### 4.1.1 CMake优化选项概览

```cmake
# CMakeLists.txt 关键优化选项
option(GGML_NATIVE "Enable native CPU optimizations" ON)
option(GGML_LTO "Enable Link Time Optimization" OFF)
option(GGML_AVX "Enable AVX" ON)
option(GGML_AVX2 "Enable AVX2" ON)
option(GGML_AVX512 "Enable AVX512" OFF)
option(GGML_FMA "Enable FMA" ON)
option(GGML_OPENMP "Enable OpenMP" ON)
```

**图解：编译优化层次**

```
编译优化金字塔

        ┌─────────────┐
        │   LTO/PGO   │  ← 链接时优化/配置文件引导优化
        │  (最高级)   │     提升5-15%
        ├─────────────┤
        │  SIMD指令集 │  ← AVX/AVX2/AVX512/NEON
        │  (架构级)   │     提升2-10x
        ├─────────────┤
        │  编译器优化 │  ← -O3 -march=native
        │  (通用级)   │     提升10-30%
        ├─────────────┤
        │  基础编译   │  ← -O2
        │  (默认级)   │
        └─────────────┘
```

#### 4.1.2 SIMD指令集选择

```cmake
# CPU特性检测与自动启用
include(CheckCXXCompilerFlag)

check_cxx_compiler_flag(-mavx COMPILER_SUPPORTS_AVX)
check_cxx_compiler_flag(-mavx2 COMPILER_SUPPORTS_AVX2)
check_cxx_compiler_flag(-mavx512f COMPILER_SUPPORTS_AVX512)

if(COMPILER_SUPPORTS_AVX)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx")
endif()
```

**各平台SIMD支持：**

| 平台 | 指令集 | CMake选项 | 典型加速 |
|-----|--------|----------|---------|
| x86_64 (Intel/AMD) | AVX | GGML_AVX=ON | 2-3x |
| x86_64 (现代) | AVX2 | GGML_AVX2=ON | 3-5x |
| x86_64 (服务器) | AVX512 | GGML_AVX512=ON | 5-10x |
| ARM64 (Apple) | NEON | 自动检测 | 2-4x |
| ARM64 (Android) | NEON | 自动检测 | 2-3x |

#### 4.1.3 Link Time Optimization (LTO)

```cmake
# 启用LTO
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)

# 或按编译器类型
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -flto")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -flto")
elseif(MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /GL")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /LTCG")
endif()
```

**LTO效果：**
- 跨模块函数内联
- 死代码消除
- 通常提升5-15%性能
- 编译时间增加2-5倍

#### 4.1.4 Profile Guided Optimization (PGO)

```bash
# 两阶段编译流程
# 阶段1：编译带检测的版本
cmake -B build -DGGML_PGO_GENERATE=ON
make -C build

# 运行典型工作负载收集性能数据
./build/bin/llama-cli -m model.gguf -p "典型提示词" -n 100

# 阶段2：使用性能数据重新编译
cmake -B build -DGGML_PGO_USE=ON
make -C build
```

### 4.2 运行时优化

#### 4.2.1 批处理策略

```cpp
// src/llama-context.cpp
struct llama_context_params {
    uint32_t n_batch;     // 最大批处理token数
    uint32_t n_ubatch;    // 物理批次大小（用于并行）
    uint32_t n_seq_max;   // 最大序列数
    // ...
};
```

**批处理参数调优：**

| 场景 | n_batch | n_ubatch | 说明 |
|-----|---------|----------|------|
| 单用户交互 | 512 | 512 | 低延迟优先 |
| 高并发服务 | 2048 | 512 | 吞吐量优先 |
| 长文档处理 | 4096 | 1024 | 大上下文 |
| 嵌入生成 | 8192 | 2048 | 批量处理 |

#### 4.2.2 线程配置

```cpp
// 线程数计算逻辑
int n_threads = std::max(2, std::min(4, 
    (int)sysconf(_SC_NPROCESSORS_ONLN) - 2));

// llama.cpp参数
struct llama_context_params {
    uint32_t n_threads;       // 提示处理线程数
    uint32_t n_threads_batch; // 批处理线程数
};
```

**线程配置建议：**

```
CPU核心数: 8
场景1: 单用户本地使用
  n_threads = 6, n_threads_batch = 6
  原因: 留2核心给系统和UI

场景2: 服务端多并发
  n_threads = 2, n_threads_batch = 8
  原因: 每个请求少量线程，批处理用满核心

场景3: 纯批处理任务
  n_threads = 8, n_threads_batch = 8
  原因: 最大化吞吐量
```

#### 4.2.3 KV缓存优化

```cpp
// src/llama-kv-cache.cpp
struct llama_kv_cache {
    uint32_t size;           // 缓存容量（token数）
    uint32_t size_k;         // K缓存每token大小
    uint32_t size_v;         // V缓存每token大小
    // ...
};

// 缓存类型选择
enum llama_pooling_type {
    LLAMA_POOLING_TYPE_NONE = 0,
    LLAMA_POOLING_TYPE_MEAN = 1,
    LLAMA_POOLING_TYPE_CLS  = 2,
};
```

**KV缓存内存占用计算：**

```
缓存大小 = n_layer × n_ctx × n_embd × 2(K+V) × sizeof(float16)

示例: 7B模型, 32层, 4096上下文, 4096维度
  = 32 × 4096 × 4096 × 2 × 2 bytes
  = 2,147,483,648 bytes
  = 2 GB
```

### 4.3 内存优化

#### 4.3.1 内存映射策略

```cpp
// src/llama-mmap.cpp
struct llama_mmap {
    void* addr;           // 映射地址
    size_t size;          // 映射大小
    bool prefetch;        // 是否预读
    // ...
};

// 使用mmap加载模型
llama_mmap* llama_mmap_init(const char* path, bool prefetch) {
    int fd = open(path, O_RDONLY);
    void* addr = mmap(NULL, size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (prefetch) {
        madvise(addr, size, MADV_WILLNEED);
    }
    // ...
}
```

**内存映射优势：**
- 按需加载：只加载实际使用的部分到内存
- 共享内存：多个进程可共享同一份物理内存
- 快速启动：无需等待整个文件读取

#### 4.3.2 权重加载优化

```cpp
// 模型加载参数
struct llama_model_params {
    uint32_t n_gpu_layers;     // 加载到GPU的层数
    llama_split_mode split_mode;  // 多GPU分割模式
    bool use_mmap;             // 使用内存映射
    bool use_mlock;            // 锁定内存（防止交换）
    // ...
};
```

**GPU层卸载策略：**

```
场景: 7B模型, 24GB VRAM GPU

方案1: 全GPU (-ngl 999)
  - 速度最快
  - 需要约14GB VRAM

方案2: 部分GPU (-ngl 20)
  - 20层在GPU, 12层在CPU
  - 平衡速度和显存

方案3: 仅注意力层 (-ngl 10)
  - 注意力计算在GPU
  - FFN在CPU
  - 显存受限时的选择
```

#### 4.3.3 内存碎片整理

```cpp
// 上下文重置（不重新加载模型）
void llama_kv_cache_clear(struct llama_kv_cache* cache) {
    // 清空KV缓存但保留分配
    memset(cache->k, 0, cache->size * cache->size_k);
    memset(cache->v, 0, cache->size * cache->size_v);
}

// 状态保存/恢复
void llama_state_save_file(...);
void llama_state_load_file(...);
```

### 4.4 能耗优化

#### 4.4.1 移动设备优化

```cpp
// examples/llama.android/lib/src/main/cpp/ai_chat.cpp
constexpr int N_THREADS_MIN = 2;
constexpr int N_THREADS_MAX = 4;
constexpr int N_THREADS_HEADROOM = 2;

// 根据CPU核心数动态调整
int n_threads = std::max(N_THREADS_MIN, 
    std::min(N_THREADS_MAX,
        (int)sysconf(_SC_NPROCESSORS_ONLN) - N_THREADS_HEADROOM));
```

**移动端优化策略：**

| 策略 | 实现 | 效果 |
|-----|------|------|
| 动态线程数 | 根据CPU核心数限制线程 | 减少发热 |
| 小批次处理 | n_batch=512 | 快速响应 |
| 量化模型 | Q4_K_M或更低 | 减少计算 |
| 上下文限制 | n_ctx=2048 | 节省内存 |

#### 4.4.2 功耗与性能平衡

```cpp
// 性能模式选择
enum class PerformanceMode {
    POWER_SAVE,     // 省电模式：限制线程、降频
    BALANCED,       // 平衡模式：默认配置
    PERFORMANCE,    // 性能模式：最大线程、高性能核心
};

void set_performance_mode(PerformanceMode mode) {
    switch(mode) {
        case POWER_SAVE:
            n_threads = 2;
            // 使用小核心（ARM big.LITTLE）
            break;
        case PERFORMANCE:
            n_threads = std::thread::hardware_concurrency();
            // 使用大核心
            break;
    }
}
```

## 5. 设计中的取舍

### 5.1 编译时 vs 运行时优化

| 维度 | 编译时优化 | 运行时优化 |
|-----|-----------|-----------|
| 灵活性 | 低（需重新编译） | 高（参数可调） |
| 效果 | 更高（可达30%） | 中等（10-20%） |
| 复杂度 | 高（需了解编译器） | 低（改参数即可） |
| 适用场景 | 生产部署 | 开发调试 |

### 5.2 内存vs速度权衡

```
选项A: 全内存加载
  - 启动慢（加载全部权重）
  - 运行快（无IO等待）
  - 内存占用高

选项B: 内存映射
  - 启动快（按需加载）
  - 首次推理慢（页错误）
  - 内存占用低

llama.cpp默认: 选项B，可通过--mlock强制全内存
```

### 5.3 精度vs速度权衡

| 量化类型 | 精度损失 | 速度提升 | 内存节省 | 适用场景 |
|---------|---------|---------|---------|---------|
| F16 | 极小 | 1x | 50% | 精度敏感 |
| Q8_0 | 很小 | 1.5x | 75% | 通用 |
| Q4_K_M | 小 | 2-3x | 87.5% | 推荐 |
| Q3_K_M | 中等 | 3-4x | 90% | 资源受限 |
| Q2_K | 较大 | 4-5x | 93% | 实验性 |

## 6. 动手练习

### 练习1：编译优化对比

```bash
# 基准编译
cmake -B build_baseline
make -C build_baseline -j

# 优化编译
cmake -B build_optimized -DGGML_NATIVE=ON -DGGML_LTO=ON
make -C build_optimized -j

# 对比测试
./build_baseline/bin/llama-bench -m model.gguf
./build_optimized/bin/llama-bench -m model.gguf
```

### 练习2：批处理参数调优

```bash
# 测试不同批处理大小
for batch in 128 256 512 1024 2048; do
    echo "Testing batch size: $batch"
    ./llama-bench -m model.gguf -p 512 -n 128 -b $batch
done
```

### 练习3：GPU层卸载优化

```bash
# 找到最佳GPU层数
for ngl in 0 10 20 30 40; do
    echo "GPU layers: $ngl"
    ./llama-cli -m model.gguf -ngl $ngl -p "Hello" -n 100 --timing
done
```

### 练习4：内存使用监控

```python
#!/usr/bin/env python3
"""监控llama.cpp内存使用"""
import subprocess
import psutil
import time

def monitor_memory(pid, interval=1.0):
    process = psutil.Process(pid)
    max_rss = 0
    
    while process.is_running():
        mem = process.memory_info()
        max_rss = max(max_rss, mem.rss)
        print(f"RSS: {mem.rss / 1024 / 1024:.1f} MB")
        time.sleep(interval)
    
    print(f"Peak RSS: {max_rss / 1024 / 1024:.1f} MB")

# 运行并监控
cmd = ["./llama-cli", "-m", "model.gguf", "-p", "Hello", "-n", "100"]
proc = subprocess.Popen(cmd)
monitor_memory(proc.pid)
```

## 7. 本课小结

- **编译优化**：使用`-DGGML_NATIVE=ON -DGGML_LTO=ON`获得最大性能
- **SIMD指令**：AVX2是现代x86 CPU的甜点，AVX512需权衡功耗
- **批处理调优**：n_batch影响吞吐量，n_ubatch影响延迟
- **线程配置**：n_threads根据核心数调整，留余量给系统
- **内存优化**：mmap加速启动，mlock保证稳定性
- **GPU卸载**：根据显存大小选择合适的n_gpu_layers

**性能优化检查清单：**
- [ ] 编译时启用了Native优化
- [ ] 使用了合适的SIMD指令集
- [ ] 批处理大小适合使用场景
- [ ] 线程数不超过物理核心
- [ ] GPU层数充分利用显存
- [ ] 使用了内存映射加速启动
