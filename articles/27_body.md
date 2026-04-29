# 第27章 性能优化技巧 —— 让模型"跑得更快的秘籍"

## 学习目标

1. 掌握CMake编译优化选项的配置方法
2. 理解运行时批处理、线程、缓存参数调优策略
3. 学习内存映射和权重加载优化技术
4. 了解移动设备和边缘设备的能耗优化方法
5. 掌握性能测试与瓶颈分析技巧

---

## 生活类比：赛车调校的艺术

想象你是一位F1车队的工程师，面前是一辆顶级赛车。同样的引擎、同样的底盘，通过精密的调校——悬挂硬度、轮胎选择、空气动力学套件角度——可以让赛车在不同赛道上发挥出截然不同的性能。摩纳哥的街道赛需要高下压力和敏捷转向，而蒙扎的高速赛道则需要低阻力和直线速度。

llama.cpp的性能优化就像是这场赛车调校的艺术：

**编译优化 = 引擎调校**

就像是调整引擎的进气、点火时机、涡轮增压压力，编译优化通过SIMD指令集、链接时优化（LTO）、配置文件引导优化（PGO），从底层榨取硬件的每一分性能。这是"一次调校，持续受益"的投资。

**运行时优化 = 轮胎和悬挂选择**

比赛日根据天气和赛道状况选择干胎还是雨胎、软胎还是硬胎，就像是根据使用场景调整批处理大小、线程数、GPU层数。不同的选择会带来延迟、吞吐量、内存占用之间的不同权衡。

**内存优化 = 减轻车重**

F1赛车每减轻1公斤，圈速就能提升约0.03秒。内存优化也是如此——使用内存映射（mmap）避免不必要的拷贝、将部分层卸载到GPU、量化减小模型体积，都能让系统"轻装上阵"。

**能耗优化 = 燃油管理**

在长距离比赛中，燃油管理至关重要。移动设备上的能耗优化类似——动态调整线程数、限制上下文长度、使用激进量化，在性能和电池寿命之间找到平衡。

而`llama-bench`就是我们的测功机和计时器，客观测量每一种调校方案的效果。

---

## 27.1 编译优化 —— 榨取硬件极限

### 27.1.1 CMake优化选项概览

llama.cpp的CMakeLists.txt提供了丰富的优化选项，让你可以针对特定硬件进行精细调校。

**源码位置**：`CMakeLists.txt` (第1-200行)

```cmake
# CMakeLists.txt - 性能优化选项

# ========== 基础优化 ==========
# 使用本地CPU指令集（-march=native）
# 这会针对编译机器的CPU生成最优代码
option(GGML_NATIVE "Enable native CPU optimizations" ON)

# 链接时优化（Link Time Optimization）
# 跨模块进行优化，通常提升5-15%性能
option(GGML_LTO "Enable Link Time Optimization" OFF)

# ========== SIMD指令集选项 ==========
# x86/x64平台的向量化指令集
option(GGML_AVX "Enable AVX" ON)           # 2011年Sandy Bridge+
option(GGML_AVX2 "Enable AVX2" ON)         # 2013年Haswell+
option(GGML_AVX512 "Enable AVX512" OFF)    # 2017年Skylake Xeon+
option(GGML_FMA "Enable FMA" ON)           # 融合乘加指令
option(GGML_F16C "Enable F16C" ON)         # 半精度转换

# ARM平台
# NEON指令集在ARM64上自动检测和启用

# ========== 并行化选项 ==========
option(GGML_OPENMP "Enable OpenMP" ON)     # 多线程并行

# ========== GPU后端选项 ==========
option(GGML_CUDA "Enable CUDA" OFF)        # NVIDIA GPU
option(GGML_METAL "Enable Metal" OFF)      # Apple Silicon
option(GGML_VULKAN "Enable Vulkan" OFF)    # 跨平台GPU
option(GGML_SYCL "Enable SYCL" OFF)        # Intel GPU
```

**编译优化层次金字塔**

```
                    ┌─────────────┐
                    │   LTO/PGO   │  ← 链接时/配置文件引导优化
                    │  (最高级)   │     提升5-15%，编译时间增加
                    ├─────────────┤
                    │  SIMD指令集 │  ← AVX/AVX2/AVX512/NEON
                    │  (架构级)   │     提升2-10x，依赖硬件
                    ├─────────────┤
                    │  编译器优化 │  ← -O3 -march=native
                    │  (通用级)   │     提升10-30%
                    ├─────────────┤
                    │  基础编译   │  ← -O2
                    │  (默认级)   │
                    └─────────────┘
                    
性能提升从上到下递减，但编译时间和复杂度递增
```

### 27.1.2 SIMD指令集选择

SIMD（Single Instruction Multiple Data，单指令多数据）是现代CPU性能的关键。它允许一条指令同时处理多个数据元素。

**源码位置**：`CMakeLists.txt` (第200-400行)

```cmake
# CPU特性检测与自动启用
include(CheckCXXCompilerFlag)
include(CheckCXXSourceRuns)

# 检测AVX支持
check_cxx_compiler_flag(-mavx COMPILER_SUPPORTS_AVX)
if(COMPILER_SUPPORTS_AVX AND GGML_AVX)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx")
    add_compile_definitions(GGML_USE_AVX)
endif()

# 检测AVX2支持
check_cxx_compiler_flag(-mavx2 COMPILER_SUPPORTS_AVX2)
if(COMPILER_SUPPORTS_AVX2 AND GGML_AVX2)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx2")
    add_compile_definitions(GGML_USE_AVX2)
endif()

# 检测AVX512支持（更谨慎，可能降低频率）
check_cxx_compiler_flag(-mavx512f COMPILER_SUPPORTS_AVX512)
if(COMPILER_SUPPORTS_AVX512 AND GGML_AVX512)
    # AVX512可能导致CPU降频，需要权衡
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx512f")
    add_compile_definitions(GGML_USE_AVX512)
endif()

# 检测FMA（融合乘加）
check_cxx_compiler_flag(-mfma COMPILER_SUPPORTS_FMA)
if(COMPILER_SUPPORTS_FMA AND GGML_FMA)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mfma")
    add_compile_definitions(GGML_USE_FMA)
endif()
```

**各平台SIMD支持对比**

| 平台 | 指令集 | CMake选项 | 典型加速 | 适用CPU |
|-----|--------|----------|---------|---------|
| x86_64 (基础) | SSE4.2 | 自动 | 1x (基准) | 所有x64 CPU |
| x86_64 (主流) | AVX | GGML_AVX=ON | 2-3x | Sandy Bridge+ |
| x86_64 (现代) | AVX2 | GGML_AVX2=ON | 3-5x | Haswell+ |
| x86_64 (高端) | AVX512 | GGML_AVX512=ON | 5-10x* | Skylake Xeon+ |
| ARM64 (Apple) | NEON | 自动检测 | 2-4x | M1/M2/M3 |
| ARM64 (移动) | NEON | 自动检测 | 2-3x | Snapdragon/Tensor |

*注意：AVX512在某些工作负载下可能因CPU降频而不如AVX2

**AVX512的权衡**

```cpp
/**
 * AVX512是双刃剑：
 * 
 * 优势：
 * - 512-bit寄存器，理论性能翻倍
 * - 新指令如VNNI加速AI推理
 * 
 * 劣势：
 * - 可能导致CPU降频（thermal throttling）
 * - 功耗增加
 * - 编译代码体积增大
 * 
 * llama.cpp默认关闭AVX512，需要显式启用
 */

// 实际测试示例（Llama-2-7B）
// CPU: Intel Xeon Gold 6248
// 
// AVX2:  45 tokens/sec
// AVX512: 52 tokens/sec (+15%)
// 
// 但在长时间运行后，AVX512可能因降频反而更慢
```

AVX512 需要根据具体硬件和场景权衡。对于大多数用户，AVX2 是当前的最佳甜点——它在不降频的前提下提供了显著的加速效果。

### 27.1.3 Link Time Optimization (LTO)

LTO允许编译器在链接阶段跨模块进行优化，效果类似于把所有代码放入一个文件编译。

**源码位置**：`CMakeLists.txt` (第400-500行)

```cmake
# 启用LTO
option(GGML_LTO "Enable Link Time Optimization" OFF)

if(GGML_LTO)
    include(CheckIPOSupported)
    check_ipo_supported(RESULT ipo_supported OUTPUT ipo_error)
    
    if(ipo_supported)
        set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
        message(STATUS "LTO enabled")
    else()
        message(WARNING "LTO not supported: ${ipo_error}")
    endif()
endif()

# 或者按编译器类型手动设置
if(GGML_LTO)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -flto")
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -flto")
    elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -flto=thin")
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -flto=thin")
    elseif(MSVC)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /GL")
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /LTCG")
    endif()
endif()
```

**LTO的效果与代价**

```
收益：
- 跨模块函数内联（最重要的优化）
- 死代码消除更彻底
- 更好的常量传播
- 通常提升5-15%性能

代价：
- 编译时间增加2-5倍
- 内存使用增加
- 调试信息可能受影响
- 增量编译效果降低

适用场景：
- 生产部署前的最终构建
- 不适用于频繁修改的开发循环
```

### 27.1.4 Profile Guided Optimization (PGO)

PGO是最激进的优化方式——先用典型工作负载运行程序收集性能数据，然后用这些数据指导编译器优化。

```bash
#!/bin/bash
# PGO两阶段编译流程

# ========== 阶段1：编译带检测的版本 ==========
mkdir -p build_pgo
cd build_pgo

# GCC/Clang PGO生成模式
cmake .. \
    -DCMAKE_CXX_FLAGS="-fprofile-generate" \
    -DCMAKE_EXE_LINKER_FLAGS="-fprofile-generate"

make -j$(nproc)

# ========== 收集性能数据 ==========
# 运行典型工作负载，生成.profraw文件
./bin/llama-cli -m model.gguf -p "典型提示词" -n 1000
./bin/llama-bench -m model.gguf

# 合并性能数据（LLVM）
llvm-profdata merge -output=default.profdata *.profraw
# 或GCC：无需合并

# ========== 阶段2：使用性能数据重新编译 ==========
make clean

cmake .. \
    -DCMAKE_CXX_FLAGS="-fprofile-use=default.profdata" \
    -DCMAKE_EXE_LINKER_FLAGS="-fprofile-use=default.profdata"

make -j$(nproc)

echo "PGO优化完成！"
```

**PGO的典型效果**

```
基准：     100 tokens/sec
-O3优化：   115 tokens/sec (+15%)
-O3 + LTO： 125 tokens/sec (+25%)
-O3 + PGO： 135 tokens/sec (+35%)

注意：PGO效果取决于训练工作负载的代表性
如果训练数据与实际使用差异大，效果会打折扣
```

---

## 27.2 运行时优化 —— 适应不同场景

### 27.2.1 批处理策略

批处理是提升吞吐量的关键。llama.cpp支持两级批处理：逻辑批次（n_batch）和物理批次（n_ubatch）。

**源码位置**：`src/llama-context.cpp` (第1-200行)

```cpp
/**
 * 上下文参数结构
 * 
 * 批处理参数影响推理的吞吐量和延迟。
 */
struct llama_context_params {
    // 最大逻辑批处理token数
    // 这决定了单次decode最多处理多少token
    uint32_t n_batch = 512;
    
    // 物理批处理大小（用于并行计算）
    // 通常 <= n_batch，用于控制内存和计算粒度
    uint32_t n_ubatch = 512;
    
    // 最大序列数（用于并行生成多个序列）
    uint32_t n_seq_max = 1;
    
    // 线程配置
    uint32_t n_threads = 0;        // 提示处理线程数
    uint32_t n_threads_batch = 0;  // 批处理线程数
    
    // ... 其他参数
};

/**
 * 批处理参数选择指南
 * 
 * 场景1：单用户交互（低延迟优先）
 *   n_batch = 512, n_ubatch = 512
 *   原因：快速响应用户输入，不需要超大批次
 * 
 * 场景2：高并发服务端（吞吐量优先）
 *   n_batch = 2048, n_ubatch = 512
 *   原因：聚合多个请求，提高GPU利用率
 * 
 * 场景3：长文档处理
 *   n_batch = 4096, n_ubatch = 1024
 *   原因：一次性处理长提示，减少批次数量
 * 
 * 场景4：批量嵌入生成
 *   n_batch = 8192, n_ubatch = 2048
 *   原因：最大化吞吐量，延迟不敏感
 */
```

**批处理性能模型**

```
处理时间 = 固定开销 + k × token数

小批次：固定开销占比高，效率低
        [固定开销████████] [token处理█]

大批次：固定开销被摊薄，效率高
        [固定开销█] [token处理████████████████]

但批次过大会：
1. 增加首次token延迟（用户等待时间）
2. 增加内存占用
3. 可能超出上下文限制
```

**参数调优示例**

```bash
# 单用户交互（如llama-cli）
./llama-cli -m model.gguf --batch-size 512 -p "Hello"

# 服务端（如llama-server）
./llama-server -m model.gguf --batch-size 2048 --ubatch-size 512

# 批量嵌入
./llama-embedding -m model.gguf --batch-size 8192 -f texts.txt
```

选择合适的批处理大小是运行时优化的第一步。交互场景优先低延迟（512），批处理场景优先高吞吐量（2048+）。

### 27.2.2 线程配置

合理的线程配置对性能至关重要。线程过多会导致上下文切换开销，过少则无法充分利用CPU。

**源码位置**：`src/llama-context.cpp` (第200-400行)

```cpp
/**
 * 线程数计算逻辑
 * 
 * 默认策略：保留一些核心给系统和其他任务
 */
int get_default_n_threads() {
    int n_cpus = std::thread::hardware_concurrency();
    
    if (n_cpus <= 4) {
        // 小系统：使用所有核心
        return n_cpus;
    } else {
        // 大系统：留1-2核心给系统
        return std::max(2, n_cpus - 2);
    }
}

/**
 * 线程配置策略
 * 
 * 场景1：单用户本地使用
 *   n_threads = n_cpus - 2
 *   n_threads_batch = n_cpus - 2
 *   原因：留2核心给系统和UI，避免卡顿
 * 
 * 场景2：服务端多并发
 *   n_threads = 2
 *   n_threads_batch = n_cpus
 *   原因：每个请求少量线程，批处理用满核心
 * 
 * 场景3：纯批处理任务
 *   n_threads = n_cpus
 *   n_threads_batch = n_cpus
 *   原因：最大化吞吐量，独占机器
 */

// 配置示例
struct llama_context_params params = llama_context_default_params();

// 根据场景调整
params.n_threads = 6;        // 提示处理线程
params.n_threads_batch = 8;  // 批处理可用更多线程

struct llama_context * ctx = llama_new_context_with_model(model, params);
```

**线程数与性能关系**

```
性能
 ↑
 │     ╭──────╮
 │    ╱        ╲
 │   ╱          ╲
 │  ╱            ╲
 │ ╱              ╲
 │╱                ╲
 └──────────────────→ 线程数
  0    4    8    12   16
 
 假设：8核心16线程CPU
 
 最优线程数通常在物理核心数附近（8）
 超过物理核心后，超线程的收益递减
 过多线程会导致调度开销
```

### 27.2.3 KV缓存优化

KV缓存是大模型推理中内存占用的主要部分。合理配置缓存策略可以显著降低内存使用。

**源码位置**：`src/llama-kv-cache.cpp` (第1-300行)

```cpp
/**
 * KV缓存结构
 * 
 * 对于每个token，我们存储：
 * - Key向量：用于注意力计算
 * - Value向量：用于注意力计算
 */
struct llama_kv_cache {
    // 缓存容量（最大token数）
    uint32_t size = 0;
    
    // 每个token的K/V大小（字节）
    size_t size_k = 0;
    size_t size_v = 0;
    
    // K/V缓存数据
    struct ggml_tensor * k = nullptr;
    struct ggml_tensor * v = nullptr;
    
    // 缓存单元格状态（是否被占用）
    std::vector<llama_kv_cell> cells;
    
    // ... 其他字段
};

/**
 * 计算KV缓存内存占用
 * 
 * 公式：2 × n_layer × n_ctx × n_embd × sizeof(dtype)
 * 
 * 示例：7B模型，32层，4096上下文，4096维度，FP16
 *   = 2 × 32 × 4096 × 4096 × 2 bytes
 *   = 2,147,483,648 bytes
 *   = 2 GB
 * 
 * 如果扩展到32K上下文：
 *   = 16 GB（需要量化或窗口策略）
 */
size_t llama_kv_cache_size(const struct llama_kv_cache & cache) {
    return cache.size * (cache.size_k + cache.size_v);
}

/**
 * KV缓存优化策略
 * 
 * 策略1：滑动窗口（Window Attention）
 * - 只缓存最近的W个token
 * - 内存复杂度从O(n_ctx)降到O(W)
 * - 适合长文本生成
 * 
 * 策略2：KV缓存量化
 * - 将KV缓存从FP16量化到8-bit或4-bit
 * - 内存减半或更多
 * - 略有精度损失
 * 
 * 策略3：动态分配
 * - 按需分配缓存单元格
 * - 空闲时释放内存
 * - 适合多会话场景
 */
```

**KV缓存内存占用计算表**

| 模型 | 层数 | 上下文 | 维度 | 数据类型 | 内存占用 |
|-----|------|--------|------|---------|---------|
| 7B | 32 | 4096 | 4096 | FP16 | 2.0 GB |
| 7B | 32 | 8192 | 4096 | FP16 | 4.0 GB |
| 7B | 32 | 32768 | 4096 | FP16 | 16.0 GB |
| 13B | 40 | 4096 | 5120 | FP16 | 3.1 GB |
| 70B | 80 | 4096 | 8192 | FP16 | 10.0 GB |
| 70B | 80 | 4096 | 8192 | Q8_0 | 5.0 GB |

---

## 27.3 内存优化 —— 提升启动和运行效率

### 27.3.1 内存映射策略

内存映射（mmap）是一种让操作系统按需加载文件内容的技术，可以显著加速大模型启动。

**源码位置**：`src/llama-mmap.cpp` (第1-300行)

```cpp
/**
 * 内存映射实现
 * 
 * 使用POSIX mmap API将文件映射到虚拟地址空间。
 * 数据只在实际访问时才会从磁盘加载到物理内存。
 */
struct llama_mmap {
    void* addr;           // 映射的起始地址
    size_t size;          // 映射的大小
    int fd;               // 文件描述符
    bool prefetch;        // 是否预读
    
    // 构造函数
    llama_mmap(const char* path, bool prefetch = true) {
        // 打开文件
        fd = open(path, O_RDONLY);
        if (fd < 0) {
            throw std::runtime_error("Failed to open file");
        }
        
        // 获取文件大小
        struct stat st;
        if (fstat(fd, &st) < 0) {
            close(fd);
            throw std::runtime_error("Failed to stat file");
        }
        size = st.st_size;
        
        // 创建内存映射
        addr = mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
        if (addr == MAP_FAILED) {
            close(fd);
            throw std::runtime_error("Failed to mmap file");
        }
        
        // 预读（可选）
        if (prefetch) {
            // 告诉内核这些页面很快会被访问
            madvise(addr, size, MADV_WILLNEED);
            
            // 对于大文件，使用顺序访问模式
            madvise(addr, size, MADV_SEQUENTIAL);
        }
    }
    
    // 析构函数
    ~llama_mmap() {
        if (addr != MAP_FAILED) {
            munmap(addr, size);
        }
        if (fd >= 0) {
            close(fd);
        }
    }
    
    // 禁止拷贝
    llama_mmap(const llama_mmap&) = delete;
    llama_mmap& operator=(const llama_mmap&) = delete;
};

/**
 * 内存映射 vs 普通读取
 * 
 * 普通读取（传统方式）：
 * 1. 分配内存（malloc）
 * 2. 读取文件到内存（read/fread）
 * 3. 等待整个文件读取完成
 * 4. 启动时间 = 文件大小 / 磁盘速度
 *    （7GB模型在SSD上约需10秒）
 * 
 * 内存映射（mmap方式）：
 * 1. 创建内存映射（mmap）
 * 2. 立即返回，无需等待读取
 * 3. 按需加载页面（page fault时）
 * 4. 启动时间 ≈ 0（立即启动）
 * 5. 首次推理时按需加载权重
 * 
 * 优势：
 * - 启动时间极短
 * - 多个进程共享物理内存
 * - 操作系统自动管理缓存
 * 
 * 劣势：
 * - 首次推理可能有延迟（页错误）
 * - 需要处理文件可能被修改的情况
 */
```

**启动时间对比**

```bash
# 传统加载
$ time ./llama-cli -m model-7b.gguf --no-mmap -p "Hello"
real    0m12.345s  # 需要读取7GB到内存
user    0m0.567s
sys     0m2.123s

# 内存映射
$ time ./llama-cli -m model-7b.gguf --mmap -p "Hello"
real    0m0.523s   # 立即启动
user    0m0.234s
sys     0m0.089s
```

### 27.3.2 GPU层卸载策略

对于拥有GPU的系统，合理配置层卸载可以在显存和性能之间找到最佳平衡。

**源码位置**：`src/llama-model.cpp` (第1-400行)

```cpp
/**
 * 模型加载参数
 * 
 * n_gpu_layers控制多少层加载到GPU。
 * 合理的配置可以最大化利用显存而不溢出。
 */
struct llama_model_params {
    // 加载到GPU的层数
    // 0 = 纯CPU
    // 1 = 只加载嵌入层
    // 999 = 加载所有层
    uint32_t n_gpu_layers = 0;
    
    // 分割模式（多GPU）
    enum llama_split_mode split_mode = LLAMA_SPLIT_MODE_LAYER;
    
    // 主GPU（用于输出层）
    int main_gpu = 0;
    
    // 张量分割（按行或按层）
    const float* tensor_split = nullptr;
    
    // 是否使用内存映射
    bool use_mmap = true;
    
    // 是否锁定内存（防止交换到磁盘）
    bool use_mlock = false;
    
    // ... 其他参数
};

/**
 * GPU层卸载策略
 * 
 * 场景1：全GPU（显存充足）
 *   n_gpu_layers = 999
 *   效果：速度最快
 *   需求：约14GB VRAM（7B模型F16）
 * 
 * 场景2：部分GPU（平衡方案）
 *   n_gpu_layers = 20
 *   效果：20层在GPU，12层在CPU
 *   优势：平衡速度和显存
 *   需求：约8GB VRAM
 * 
 * 场景3：仅注意力层（显存受限）
 *   n_gpu_layers = 10
 *   效果：注意力计算在GPU，FFN在CPU
 *   优势：注意力是计算瓶颈
 *   需求：约4GB VRAM
 */

// 自动计算最佳层数（CUDA示例）
int get_optimal_n_gpu_layers(const llama_model* model, int device) {
    size_t free_vram, total_vram;
    cudaMemGetInfo(&free_vram, &total_vram);
    
    // 计算每层需要的显存
    size_t layer_size = estimate_layer_size(model);
    
    // 留20%余量给其他用途
    int n_layers = (free_vram * 0.8) / layer_size;
    
    // 不超过模型总层数
    return std::min(n_layers, llama_model_n_layers(model));
}
```

**GPU层配置速查表**

| 模型 | 量化 | 总层数 | 全GPU需显存 | 半GPU(50%) | 1/4GPU(25%) |
|-----|------|--------|-----------|-----------|------------|
| 7B | F16 | 32 | 14 GB | 7 GB | 3.5 GB |
| 7B | Q4_K_M | 32 | 5 GB | 2.5 GB | 1.25 GB |
| 13B | F16 | 40 | 26 GB | 13 GB | 6.5 GB |
| 13B | Q4_K_M | 40 | 9 GB | 4.5 GB | 2.25 GB |
| 70B | Q4_K_M | 80 | 40 GB | 20 GB | 10 GB |

**自动调优命令**

```bash
# 找到最佳GPU层数
for ngl in 0 10 20 30 40; do
    echo "Testing ngl=$ngl"
    ./llama-bench -m model.gguf -ngl $ngl -p 512 -n 128
done

# 一般规律：
# - 每增加一层，速度提升约2-3%
# - CPU和GPU之间传输数据是瓶颈
# - 建议尽可能多地将层放到GPU
```

### 27.3.3 内存碎片整理

长时间运行后，内存碎片可能影响性能。llama.cpp提供了上下文重置功能。

**源码位置**：`src/llama-kv-cache.cpp` (第300-500行)

```cpp
/**
 * 清空KV缓存
 * 
 * 这会保留分配的内存，只清空数据。
 * 适合在同一模型上开始新的对话。
 */
void llama_kv_cache_clear(struct llama_kv_cache* cache) {
    if (!cache) return;
    
    // 清空K缓存
    if (cache->k) {
        size_t k_size = cache->size * cache->size_k;
        memset(cache->k->data, 0, k_size);
    }
    
    // 清空V缓存
    if (cache->v) {
        size_t v_size = cache->size * cache->size_v;
        memset(cache->v->data, 0, v_size);
    }
    
    // 重置所有单元格状态
    for (auto& cell : cache->cells) {
        cell.pos = -1;
        cell.seq_id.clear();
    }
}

/**
 * 状态保存/恢复
 * 
 * 用于实现会话持久化和多轮对话。
 */
size_t llama_state_get_size(const struct llama_context* ctx);
int llama_state_get_data(struct llama_context* ctx, uint8_t* dst, size_t size);
int llama_state_set_data(struct llama_context* ctx, const uint8_t* src, size_t size);

// 便捷的文件接口
bool llama_state_save_file(struct llama_context* ctx, const char* path);
bool llama_state_load_file(struct llama_context* ctx, const char* path);
```

---

## 27.4 能耗优化 —— 移动设备上的平衡艺术

### 27.4.1 移动设备优化

在移动设备（手机、平板）上运行大模型，能耗优化和性能同样重要。

**源码位置**：`examples/llama.android/lib/src/main/cpp/llama-android.cpp` (第1-200行)

```cpp
/**
 * Android平台的优化配置
 * 
 * 移动设备面临的限制：
 * - 电池容量有限
 * - 散热能力有限（被动散热）
 * - 内存相对较小
 * - 需要响应系统生命周期
 */

// 保守的线程数配置
constexpr int N_THREADS_MIN = 2;
constexpr int N_THREADS_MAX = 4;
constexpr int N_THREADS_HEADROOM = 2;

// 根据设备动态调整线程数
int get_optimal_n_threads() {
    int n_cpus = sysconf(_SC_NPROCESSORS_ONLN);
    
    // 留出头 room给系统和UI
    int n_threads = std::max(N_THREADS_MIN, 
        std::min(N_THREADS_MAX, n_cpus - N_THREADS_HEADROOM));
    
    return n_threads;
}

/**
 * 移动端优化策略
 * 
 * 策略1：小批次处理
 * - n_batch = 256或512
 * - 快速响应，降低延迟
 * 
 * 策略2：激进量化
 * - 使用Q4_K_M或Q3_K_M
 * - 减少内存占用和计算量
 * 
 * 策略3：限制上下文
 * - n_ctx = 1024或2048
 * - 节省KV缓存内存
 * 
 * 策略4：动态性能调整
 * - 电池低时降低线程数
 * - 设备发热时暂停生成
 */

// Android性能模式切换
enum class PerformanceMode {
    POWER_SAVE,     // 省电模式
    BALANCED,       // 平衡模式
    PERFORMANCE,    // 性能模式
};

void set_performance_mode(PerformanceMode mode) {
    struct llama_context_params params = llama_context_default_params();
    
    switch(mode) {
        case PerformanceMode::POWER_SAVE:
            // 使用小核心（ARM big.LITTLE）
            params.n_threads = 2;
            params.n_batch = 128;
            // 通知系统降低CPU频率
            break;
            
        case PerformanceMode::BALANCED:
            params.n_threads = get_optimal_n_threads();
            params.n_batch = 512;
            break;
            
        case PerformanceMode::PERFORMANCE:
            // 使用大核心
            params.n_threads = std::min(4, (int)sysconf(_SC_NPROCESSORS_ONLN));
            params.n_batch = 1024;
            break;
    }
    
    // 应用新配置...
}
```

**移动端优化配置表**

| 策略 | 实现 | 效果 | 适用场景 |
|-----|------|------|---------|
| 动态线程数 | 根据CPU核心限制线程 | 减少发热，延长续航 | 所有移动场景 |
| 小批次处理 | n_batch=256-512 | 快速响应 | 交互式应用 |
| 激进量化 | Q4_K_M或更低 | 减少计算量 | 资源受限设备 |
| 上下文限制 | n_ctx=1024-2048 | 节省内存 | 短对话场景 |
| 后台暂停 | 检测生命周期事件 | 节省电量 | 多任务场景 |

### 27.4.2 功耗与性能平衡

```cpp
/**
 * 温度监控和节流
 * 
 * 在设备过热时降低性能，防止降频。
 */
class ThermalThrottler {
public:
    void update_temperature(float temp_c) {
        current_temp = temp_c;
        
        if (temp_c > THERMAL_LIMIT_CRITICAL) {
            // 临界温度：暂停生成
            state = ThermalState::CRITICAL;
            pause_generation();
        } else if (temp_c > THERMAL_LIMIT_WARNING) {
            // 警告温度：降低性能
            state = ThermalState::THROTTLING;
            reduce_performance();
        } else {
            // 正常温度：恢复全性能
            state = ThermalState::NORMAL;
            restore_performance();
        }
    }
    
private:
    static constexpr float THERMAL_LIMIT_WARNING = 70.0f;   // 摄氏度
    static constexpr float THERMAL_LIMIT_CRITICAL = 85.0f;
    
    float current_temp = 0.0f;
    enum class ThermalState { NORMAL, THROTTLING, CRITICAL } state;
};

// Android温度监控示例（Java/Kotlin）
/*
val thermalManager = getSystemService(Context.THERMAL_SERVICE) as ThermalManager
thermalManager.addThermalStatusListener { status ->
    when (status) {
        THERMAL_STATUS_NONE, THERMAL_STATUS_LIGHT -> {
            nativeSetPerformanceMode(PERFORMANCE_MODE_PERFORMANCE)
        }
        THERMAL_STATUS_MODERATE -> {
            nativeSetPerformanceMode(PERFORMANCE_MODE_BALANCED)
        }
        THERMAL_STATUS_SEVERE, THERMAL_STATUS_CRITICAL -> {
            nativeSetPerformanceMode(PERFORMANCE_MODE_POWER_SAVE)
        }
    }
}
*/
```

---

## 设计中的取舍

### 编译时 vs 运行时优化

| 维度 | 编译时优化 | 运行时优化 |
|-----|-----------|-----------|
| 灵活性 | 低（需重新编译） | 高（参数可调） |
| 性能提升 | 更高（可达30%） | 中等（10-20%） |
| 复杂度 | 高（需了解编译器） | 低（改参数即可） |
| 适用场景 | 生产部署 | 开发调试/多场景 |
| 调试难度 | 难（优化后代码难读） | 易 |

### 内存vs速度权衡

```
选项A: 全内存加载（--no-mmap）
  ┌─────────────────────────────────┐
  │ 启动慢（需读取整个文件）          │
  │ 运行快（无页错误）               │
  │ 内存占用高                       │
  │ 适合：小模型、频繁访问            │
  └─────────────────────────────────┘

选项B: 内存映射（--mmap，默认）
  ┌─────────────────────────────────┐
  │ 启动快（按需加载）               │
  │ 首次推理慢（可能有页错误）        │
  │ 内存占用低                       │
  │ 适合：大模型、内存受限            │
  └─────────────────────────────────┘

选项C: 内存映射 + 锁定（--mmap --mlock）
  ┌─────────────────────────────────┐
  │ 启动后锁定到物理内存             │
  │ 避免交换到磁盘                   │
  │ 需要足够物理内存                 │
  │ 适合：生产部署、性能关键场景      │
  └─────────────────────────────────┘
```

### 精度vs速度权衡

| 量化类型 | 精度损失 | 速度提升 | 内存节省 | 适用场景 |
|---------|---------|---------|---------|---------|
| F16 | 极小 | 1x (基准) | 50% | 精度敏感任务 |
| Q8_0 | 很小 | 1.5x | 75% | 通用场景 |
| Q5_K_M | 小 | 2x | 81% | 高质量要求 |
| **Q4_K_M** | **小** | **2-3x** | **87.5%** | **推荐默认** |
| Q3_K_M | 中等 | 3x | 90% | 资源受限 |
| Q2_K | 较大 | 4x | 93% | 实验性 |
| IQ2_XXS | 较大 | 4-5x | 94% | 极限压缩 |

---

## 动手练习

### 练习1：编译优化对比

```bash
#!/bin/bash
# 编译优化对比测试

# 清理
rm -rf build_baseline build_optimized

# 基准编译
echo "Building baseline..."
cmake -B build_baseline -DGGML_NATIVE=OFF -DGGML_LTO=OFF
make -C build_baseline -j$(nproc)

# 优化编译
echo "Building optimized..."
cmake -B build_optimized \
    -DGGML_NATIVE=ON \
    -DGGML_LTO=ON \
    -DGGML_AVX2=ON
make -C build_optimized -j$(nproc)

# 对比测试
echo ""
echo "Running benchmarks..."
echo "Baseline:"
./build_baseline/bin/llama-bench -m model.gguf -p 512 -n 128

echo ""
echo "Optimized:"
./build_optimized/bin/llama-bench -m model.gguf -p 512 -n 128
```

### 练习2：批处理参数调优

```bash
#!/bin/bash
# 批处理参数调优脚本

MODEL="model.gguf"
OUTPUT="batch_tuning_results.txt"

echo "Batch Size Tuning Results" > $OUTPUT
echo "=========================" >> $OUTPUT
echo "" >> $OUTPUT

# 测试不同批处理大小
for batch in 128 256 512 1024 2048 4096; do
    echo "Testing batch size: $batch"
    echo "Batch size: $batch" >> $OUTPUT
    
    # 测量性能
    result=$(./llama-bench -m $MODEL -p $batch -n 128 -b $batch 2>&1 | grep "t/s")
    echo "$result" >> $OUTPUT
    echo "" >> $OUTPUT
done

echo "Results saved to $OUTPUT"
cat $OUTPUT
```

### 练习3：GPU层卸载优化

```bash
#!/bin/bash
# 找到最佳GPU层数

MODEL="model.gguf"

echo "Finding optimal n_gpu_layers..."
echo "================================"
echo ""

# 获取可用显存
nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits

echo ""
echo "Testing different n_gpu_layers values:"
echo ""

for ngl in 0 5 10 15 20 25 30 35 40; do
    echo -n "ngl=$ngl: "
    
    # 运行benchmark，提取tokens/sec
    result=$(./llama-bench -m $MODEL -ngl $ngl -p 512 -n 128 2>&1 | grep "tg " | awk '{print $NF}')
    
    echo "$result t/s"
done

echo ""
echo "Recommendation: Use the highest ngl that doesn't cause OOM"
```

### 练习4：内存使用监控

```python
#!/usr/bin/env python3
"""
监控llama.cpp内存使用情况
"""
import subprocess
import psutil
import time
import argparse

def monitor_memory(pid, interval=1.0, duration=None):
    """监控进程内存使用"""
    try:
        process = psutil.Process(pid)
    except psutil.NoSuchProcess:
        print(f"Process {pid} not found")
        return
    
    max_rss = 0
    max_vms = 0
    samples = []
    
    print(f"Monitoring PID {pid}...")
    print(f"{'Time':>8} {'RSS (MB)':>12} {'VMS (MB)':>12} {'CPU %':>8}")
    print("-" * 50)
    
    start_time = time.time()
    
    while process.is_running():
        try:
            mem = process.memory_info()
            cpu = process.cpu_percent()
            rss_mb = mem.rss / 1024 / 1024
            vms_mb = mem.vms / 1024 / 1024
            
            max_rss = max(max_rss, rss_mb)
            max_vms = max(max_vms, vms_mb)
            
            elapsed = time.time() - start_time
            samples.append((elapsed, rss_mb, vms_mb))
            
            print(f"{elapsed:>8.1f} {rss_mb:>12.1f} {vms_mb:>12.1f} {cpu:>8.1f}")
            
            # 检查是否达到持续时间
            if duration and elapsed >= duration:
                break
            
            time.sleep(interval)
            
        except psutil.NoSuchProcess:
            break
    
    print("-" * 50)
    print(f"Peak RSS: {max_rss:.1f} MB")
    print(f"Peak VMS: {max_vms:.1f} MB")
    
    # 保存详细数据
    with open("memory_profile.csv", "w") as f:
        f.write("time,rss_mb,vms_mb\n")
        for t, r, v in samples:
            f.write(f"{t},{r},{v}\n")
    
    print("Data saved to memory_profile.csv")

def main():
    parser = argparse.ArgumentParser(description="Monitor llama.cpp memory usage")
    parser.add_argument("--command", "-c", nargs="+", required=True,
                       help="Command to run and monitor")
    parser.add_argument("--interval", "-i", type=float, default=1.0,
                       help="Sampling interval (seconds)")
    parser.add_argument("--duration", "-d", type=float, default=None,
                       help="Monitoring duration (seconds)")
    
    args = parser.parse_args()
    
    # 启动进程
    print(f"Starting: {' '.join(args.command)}")
    proc = subprocess.Popen(args.command)
    
    # 监控
    monitor_memory(proc.pid, args.interval, args.duration)
    
    # 等待进程结束
    proc.wait()
    print(f"Process exited with code {proc.returncode}")

if __name__ == "__main__":
    main()

# 使用示例：
# python monitor_memory.py -c ./llama-cli -m model.gguf -p "Hello" -n 100
```

---

## 本课小结

### 性能优化检查清单

**编译优化**
- [ ] 启用了Native优化（`-DGGML_NATIVE=ON`）
- [ ] 使用了合适的SIMD指令集（AVX2为现代CPU甜点）
- [ ] 考虑启用LTO（生产构建）
- [ ] 考虑PGO（极致性能）

**运行时优化**
- [ ] 批处理大小适合使用场景（交互512，服务端2048+）
- [ ] 线程数不超过物理核心（留余量给系统）
- [ ] GPU层数充分利用显存（避免CPU-GPU传输瓶颈）

**内存优化**
- [ ] 使用内存映射加速启动（`--mmap`，默认）
- [ ] 关键部署考虑内存锁定（`--mlock`）
- [ ] KV缓存大小与上下文匹配

**移动优化**
- [ ] 限制线程数（2-4）
- [ ] 使用激进量化（Q4_K_M或更低）
- [ ] 限制上下文长度（1024-2048）
- [ ] 实现温度监控和节流

### 核心要点

| 优化类型 | 关键参数 | 典型收益 | 注意事项 |
|---------|---------|---------|---------|
| 编译优化 | NATIVE, LTO, AVX2 | 10-30% | 编译时间增加 |
| SIMD指令 | AVX/AVX2/AVX512 | 2-10x | AVX512可能降频 |
| 批处理 | n_batch, n_ubatch | 2-5x | 延迟vs吞吐量权衡 |
| 线程 | n_threads | 线性至核心数 | 超线程收益递减 |
| GPU卸载 | n_gpu_layers | 5-50x | 受显存限制 |
| 内存映射 | use_mmap | 启动加速10x+ | 首次推理延迟 |
| 量化 | ftype | 2-4x速度 | 精度损失 |

本章我们一起学习了以下概念：

| 概念 | 解释 |
|------|------|
| SIMD指令集 | 单指令多数据技术，通过AVX/AVX2/NEON等并行指令加速矩阵运算，是CPU推理性能的关键 |
| 批处理策略 | 通过调整n_batch和n_ubatch参数平衡延迟与吞吐量，小批次优先延迟，大批次优先吞吐量 |
| LTO与PGO | 链接时优化和配置文件引导优化，在编译阶段进一步榨取性能，可提升5-35% |
| 内存映射(mmap) | 让操作系统按需加载模型文件，启动时间从数十秒降至毫秒级 |
| GPU层卸载 | 将Transformer层从CPU迁移到GPU执行，充分利用异构计算能力 |
| KV缓存优化 | 通过滑动窗口、量化缓存等策略降低注意力机制的内存占用 |

---

下一章中，我们将学习调试与问题排查——掌握 llama.cpp 的调试工具和问题定位方法。

## 关联阅读

- **第12章**：GGUF文件格式和内存布局
- **第15章**：后端实现（CUDA、Metal等）
- **第24章**：llama-bench性能测试工具详解
- **第29章**：部署和运维中的性能考虑

---

*本章对应源码版本：master (2026-04-07)*
*本章作者：llama.cpp社区*
*最后更新：2026-04-08*
