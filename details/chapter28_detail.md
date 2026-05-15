# 第28章 调试与问题排查 —— 开发者的"火眼金睛"

## 1. 学习目标

- 掌握llama.cpp的日志系统和调试技巧
- 学习使用GDB/LLDB进行源码级调试
- 理解常见内存问题的排查方法
- 掌握数值稳定性问题的诊断
- 学会使用性能分析工具定位瓶颈

## 2. 生活类比：侦探破案

想象你是一位侦探，面对一个"案件"（程序问题）：日志记录是目击证词，调试器是现场勘查工具，性能分析是时间线重建。好的侦探知道如何收集证据（日志）、分析线索（堆栈跟踪）、重现现场（最小复现），最终锁定真凶（bug根源）。

## 3. 源码地图

| 文件路径 | 职责 | 核心内容 |
|---------|------|---------|
| `common/log.h` | 日志宏定义 | LOG_DBG/LOG_INF/LOG_WRN/LOG_ERR |
| `common/log.cpp` | 日志实现 | 异步日志、级别过滤、彩色输出 |
| `src/llama.cpp` | 主实现 | 核心API实现、状态检查 |
| `src/llama-impl.h` | 内部工具 | 日志宏、计时器、格式化、辅助类 |
| `ggml/src/ggml.c` | GGML核心 | 张量操作、计算图执行 |
| `examples/llama-bench/llama-bench.cpp` | 基准测试 | 性能测试、指标收集 |

## 4. 详细章节内容

### 4.0 llama-impl.h 内部工具

`llama-impl.h` 提供了llama.cpp内部使用的各种**实用工具类**，包括日志宏、时间测量、缓冲区视图、字符串处理等。这些是开发和调试时的基础设施。

#### 4.0.1 日志宏系统

**源码位置**：`src/llama-impl.h` (第19-31行)

```cpp
// 内部日志宏（供llama.cpp内部使用）
#define LLAMA_LOG(...)       llama_log_internal(GGML_LOG_LEVEL_NONE , __VA_ARGS__)
#define LLAMA_LOG_INFO(...)  llama_log_internal(GGML_LOG_LEVEL_INFO , __VA_ARGS__)
#define LLAMA_LOG_WARN(...)  llama_log_internal(GGML_LOG_LEVEL_WARN , __VA_ARGS__)
#define LLAMA_LOG_ERROR(...) llama_log_internal(GGML_LOG_LEVEL_ERROR, __VA_ARGS__)
#define LLAMA_LOG_DEBUG(...) llama_log_internal(GGML_LOG_LEVEL_DEBUG, __VA_ARGS__)
#define LLAMA_LOG_CONT(...)  llama_log_internal(GGML_LOG_LEVEL_CONT , __VA_ARGS__)
```

**与 common/log.h 的区别**：

| 特性 | common/log.h | llama-impl.h |
|------|-------------|--------------|
| 使用场景 | 通用工具库 | llama.cpp内部 |
| 宏名称 | LOG_INF/LOG_ERR | LLAMA_LOG_INFO/LLAMA_LOG_ERROR |
| 目标用户 | 示例程序/工具 | llama.cpp核心 |
| 功能范围 | 完整日志系统 | 简单封装 |

**使用示例**：

```cpp
// src/llama-model.cpp
LLAMA_LOG_INFO("%s: loading model from '%s'\n", __func__, path_model);
LLAMA_LOG_WARN("%s: unknown model architecture: %s\n", __func__, arch_name);
```

#### 4.0.2 no_init - 延迟初始化包装

**源码位置**：`src/llama-impl.h` (第37-41行)

```cpp
template <typename T>
struct no_init {
    T value;
    no_init() = default;  // 不初始化value
};
```

**用途**：
- 在需要**手动控制初始化时机**的场景使用
- 避免默认构造函数的开销
- 常用于性能敏感的代码路径

**使用场景**：

```cpp
// 大数组的延迟初始化
std::vector<no_init<float>> large_buffer(size);
// 稍后手动初始化
for (size_t i = 0; i < size; i++) {
    large_buffer[i].value = compute_initial_value(i);
}
```

**注意事项**：
- 使用`no_init`后必须**手动确保初始化**
- 未初始化的值可能包含垃圾数据
- 仅用于性能关键路径，普通代码不建议使用

#### 4.0.3 time_meas - 时间测量器

**源码位置**：`src/llama-impl.h` (第43-50行)

```cpp
struct time_meas {
    time_meas(int64_t & t_acc, bool disable = false);
    ~time_meas();

    const int64_t t_start_us;
    int64_t & t_acc;
};
```

**工作原理**：

```
┌─────────────────────────────────────┐
│  time_meas tm(t_total, no_perf)      │
│       │                              │
│       ▼                              │
│  t_start_us = ggml_time_us()         │
│       │                              │
│       │  [代码执行中...]              │
│       │                              │
│       ▼                              │
│  ~time_meas()                         │
│       │                              │
│       ▼                              │
│  t_acc += ggml_time_us() - t_start_us │
└─────────────────────────────────────┘
```

**使用示例**：

```cpp
// src/llama-context.cpp
int64_t t_graph_us = 0;

void llama_context::build_graph() {
    time_meas tm(t_graph_us, cparams.no_perf);
    // 自动计时：构造函数开始，析构函数结束
    
    // 构建计算图...
    ggml_build_graph(...);
}  // tm析构时自动累加时间到t_graph_us
```

**性能报告**：

```cpp
// 打印性能统计
void llama_context::perf_print() {
    LLAMA_LOG_INFO("graph compute time: %.3f ms\n", t_graph_us / 1000.0);
}
```

#### 4.0.4 buffer_view - 缓冲区视图

**源码位置**：`src/llama-impl.h` (第52-60行)

```cpp
template <typename T>
struct buffer_view {
    T * data;
    size_t size = 0;

    bool has_data() const {
        return data && size > 0;
    }
};
```

**设计模式**：
- **非拥有式引用**：只存储指针和大小，不管理内存
- **零拷贝**：避免不必要的数据复制
- **安全检查**：`has_data()`验证有效性

**使用场景**：

```cpp
// 引用外部数据而不拷贝
buffer_view<float> get_tensor_data(const ggml_tensor * t) {
    return { (float*)t->data, ggml_nelements(t) };
}

// 传递视图而非整个张量
void process_view(buffer_view<float> view) {
    if (!view.has_data()) return;
    for (size_t i = 0; i < view.size; i++) {
        view.data[i] = transform(view.data[i]);
    }
}
```

**与 std::span 的区别**：
- `buffer_view`更简单，兼容旧C++标准
- 显式的`has_data()`检查
- 专为原始指针设计

#### 4.0.5 字符串处理工具

**replace_all - 全局替换**

```cpp
void replace_all(std::string & s, 
                  const std::string & search, 
                  const std::string & replace);

// 使用示例
std::string name = "layer.0.attn.weight";
replace_all(name, ".", "_");  // -> "layer_0_attn_weight"
```

**format - 格式化字符串**

```cpp
// 类型安全的格式化（类似sprintf但返回std::string）
LLAMA_ATTRIBUTE_FORMAT(1, 2)
std::string format(const char * fmt, ...);

// 使用示例
std::string info = format("layer %d, dim=%d", layer_idx, n_embd);
// -> "layer 0, dim=4096"
```

**llama_format_tensor_shape - 张量形状格式化**

```cpp
std::string llama_format_tensor_shape(const std::vector<int64_t> & ne);
std::string llama_format_tensor_shape(const struct ggml_tensor * t);

// 使用示例
ggml_tensor * t = ...;  // shape = [32000, 4096]
std::string shape = llama_format_tensor_shape(t);  // -> "[32000, 4096]"
```

**gguf_kv_to_str - GGUF元数据转字符串**

```cpp
std::string gguf_kv_to_str(const struct gguf_context * ctx, int i);

// 将GGUF元数据值转换为可读字符串
// 自动处理不同数据类型（int, float, string, array等）
```

#### 4.0.6 调试实用技巧

**条件断点宏**

```cpp
// 在特定条件触发断点
#ifdef LLAMA_DEBUG
#define LLAMA_BREAK_IF(cond) if (cond) __debugbreak()
#else
#define LLAMA_BREAK_IF(cond)
#endif

// 使用
LLAMA_BREAK_IF(isnan(tensor->data[0]));  // 遇到NaN自动断点
```

**张量名称常量**

```cpp
// src/llama-impl.h (第73-76行)
#define LLAMA_TENSOR_NAME_FATTN   "__fattn__"
#define LLAMA_TENSOR_NAME_FGDN_AR   "__fgdn_ar__"
#define LLAMA_TENSOR_NAME_FGDN_CH   "__fgdn_ch__"

// 用于标记特殊张量，便于调试时识别
```

**调试检查清单**：

```cpp
// 在关键位置插入检查
LLAMA_LOG_DEBUG("tensor %s: shape=%s, data[0]=%f\n",
                tensor->name,
                llama_format_tensor_shape(tensor).c_str(),
                ((float*)tensor->data)[0]);

// 性能热点计时
{
    time_meas tm(t_hotspot_us, false);
    critical_code();
}
LLAMA_LOG_INFO("hotspot took %.3f ms\n", t_hotspot_us / 1000.0);
```

---

### 4.1 日志系统详解

#### 4.1.1 日志级别体系

```cpp
// common/log.h
#define LOG_LEVEL_DEBUG  4  // 调试信息
#define LOG_LEVEL_INFO   3  // 一般信息
#define LOG_LEVEL_WARN   2  // 警告
#define LOG_LEVEL_ERROR  1  // 错误
#define LOG_LEVEL_OUTPUT 0  // 工具输出

// 使用宏（避免参数计算开销）
#define LOG_DBG(...) LOG_TMPL(GGML_LOG_LEVEL_DEBUG, LOG_LEVEL_DEBUG, __VA_ARGS__)
#define LOG_INF(...) LOG_TMPL(GGML_LOG_LEVEL_INFO,  LOG_LEVEL_INFO,  __VA_ARGS__)
#define LOG_WRN(...) LOG_TMPL(GGML_LOG_LEVEL_WARN,  LOG_LEVEL_WARN,  __VA_ARGS__)
#define LOG_ERR(...) LOG_TMPL(GGML_LOG_LEVEL_ERROR, LOG_LEVEL_ERROR, __VA_ARGS__)
```

**日志输出示例：**

```
# 默认级别(INFO)
llm_load_tensors: ggml ctx size =    0.27 MiB
llm_load_tensors: offloading 32 repeating layers to GPU

# 调试级别(DEBUG)
0.00.035.060 D ggml_backend_metal_log_allocated_size: allocated buffer
0.00.035.064 I llm_load_tensors: ggml ctx size =    0.27 MiB
```

#### 4.1.2 异步日志实现

```cpp
// common/log.cpp
struct common_log {
    std::mutex mtx;
    std::thread thrd;
    std::condition_variable cv;
    
    // 环形缓冲区
    std::vector<common_log_entry> entries;
    size_t head;
    size_t tail;
    
    void add(enum ggml_log_level level, const char* fmt, va_list args) {
        std::lock_guard<std::mutex> lock(mtx);
        auto& entry = entries[tail];
        
        // 格式化消息
        vsnprintf(entry.msg.data(), entry.msg.size(), fmt, args);
        entry.level = level;
        
        tail = (tail + 1) % entries.size();
        cv.notify_one();
    }
};
```

**异步日志优势：**
- 不阻塞主线程
- 批量写入磁盘
- 自动处理缓冲区溢出

#### 4.1.3 日志配置

```cpp
// 设置日志级别
void common_log_set_verbosity_thold(int verbosity);

// 启用文件日志
void common_log_set_file(struct common_log* log, const char* file);

// 启用颜色
void common_log_set_colors(struct common_log* log, log_colors colors);

// 启用时间戳
void common_log_set_timestamps(struct common_log* log, bool timestamps);
```

**命令行使用：**

```bash
# 启用调试日志
./llama-cli -m model.gguf -p "Hello" --verbose

# 或设置环境变量
export LLAMA_LOG_LEVEL=4
./llama-cli -m model.gguf -p "Hello"
```

### 4.2 调试工具与方法

#### 4.2.1 GDB调试基础

```bash
# 编译调试版本
cmake -B build_debug -DCMAKE_BUILD_TYPE=Debug
make -C build_debug -j

# 启动GDB
gdb ./build_debug/bin/llama-cli

# 常用命令
(gdb) break llama_decode          # 设置断点
(gdb) run -m model.gguf -p "Hi"   # 运行
(gdb) next                        # 单步
(gdb) print ctx->n_tokens         # 查看变量
(gdb) backtrace                   # 查看堆栈
(gdb) continue                    # 继续运行
```

#### 4.2.2 核心数据结构检查

```cpp
// 检查上下文状态
(gdb) p *ctx
$1 = {
  model = 0x5555555f5eb0,
  n_tokens = 512,
  kv_cache = {
    size = 4096,
    used = 512,
    // ...
  }
}

// 检查张量
(gdb) p tensor->ne[0]@4  # 打印前4个维度
$2 = {4096, 11008, 1, 1}

(gdb) p tensor->data
$3 = (void *) 0x7fff5000
```

#### 4.2.3 LLDB调试（macOS）

```bash
# LLDB启动
lldb ./build/bin/llama-cli

# 常用命令
(lldb) breakpoint set --name llama_decode
(lldb) run -m model.gguf -p "Hello"
(lldb) frame variable ctx
(lldb) memory read --size 4 --format f 0x7fff5000 --count 10
```

### 4.3 常见问题排查

#### 4.3.1 内存问题

**症状：段错误(Segmentation Fault)**

```
# 排查步骤
1. 检查模型文件完整性
   sha256sum model.gguf

2. 检查内存限制
   ulimit -v  # 虚拟内存限制
   ulimit -m  # 物理内存限制

3. 使用AddressSanitizer编译
cmake -B build_asan -DCMAKE_CXX_FLAGS="-fsanitize=address -g"
make -C build_asan -j
./build_asan/bin/llama-cli -m model.gguf -p "Test"
```

**症状：内存泄漏**

```bash
# 使用Valgrind
valgrind --leak-check=full \
         --show-leak-kinds=all \
         --track-origins=yes \
         ./llama-cli -m model.gguf -p "Test" -n 10
```

#### 4.3.2 数值稳定性问题

**症状：输出NaN或Inf**

```cpp
// 检查张量数值
bool check_tensor_values(const ggml_tensor* tensor) {
    const float* data = (const float*)tensor->data;
    size_t n = ggml_nelements(tensor);
    
    for (size_t i = 0; i < n; i++) {
        if (isnan(data[i]) || isinf(data[i])) {
            LOG_ERR("Invalid value at index %zu: %f\n", i, data[i]);
            return false;
        }
    }
    return true;
}
```

**常见原因：**
- 量化溢出（使用不兼容的量化类型）
- 学习率过大（微调模型）
- 输入数值范围异常

#### 4.3.3 后端兼容性问题

```cpp
// 检查后端支持
void check_backend_support() {
    for (size_t i = 0; i < ggml_backend_reg_count(); i++) {
        auto* reg = ggml_backend_reg_get(i);
        LOG_INF("Backend: %s\n", ggml_backend_reg_name(reg));
    }
}

// CUDA特定检查
#ifdef GGML_USE_CUDA
    int device_count;
    cudaGetDeviceCount(&device_count);
    if (device_count == 0) {
        LOG_WRN("No CUDA devices found, falling back to CPU\n");
    }
#endif
```

### 4.4 性能分析

#### 4.4.1 内置性能统计

```cpp
// llama.cpp性能上下文
struct llama_perf_context {
    uint64_t t_start_ms;     // 开始时间
    uint64_t t_load_ms;      // 加载时间
    uint64_t t_p_eval_ms;    // 提示处理时间
    uint64_t t_eval_ms;      // 生成时间
    
    uint32_t n_p_eval;       // 处理的提示token数
    uint32_t n_eval;         // 生成的token数
};

// 打印性能统计
void llama_perf_context_print(const struct llama_context* ctx);
```

**输出示例：**

```
llama_perf_context_print:        load time =    1234.56 ms
llama_perf_context_print: prompt eval time =     234.56 ms /   512 tokens (    0.46 ms/token)
llama_perf_context_print:        eval time =    1234.56 ms /   128 tokens (    9.64 ms/token)
llama_perf_context_print:       total time =    2469.12 ms
```

#### 4.4.2 使用perf进行CPU分析

```bash
# 记录性能数据
perf record -g ./llama-cli -m model.gguf -p "Hello" -n 100

# 生成报告
perf report

# 火焰图
perf script | stackcollapse-perf.pl | flamegraph.pl > flamegraph.svg
```

#### 4.4.3 CUDA分析

```bash
# Nsight Systems
nsys profile -o report ./llama-cli -m model.gguf -p "Hello" -n 100
nsys-ui report.nsys-rep

# Nsight Compute (内核级分析)
ncu -o profile_report ./llama-cli -m model.gguf -p "Hello" -n 10
```

## 5. 设计中的取舍

### 5.1 日志级别选择

| 级别 | 使用场景 | 性能影响 | 磁盘占用 |
|-----|---------|---------|---------|
| DEBUG | 开发调试 | 高（大量IO） | 大 |
| INFO | 正常运行 | 中 | 中 |
| WARN | 生产环境 | 低 | 小 |
| ERROR | 关键系统 | 极低 | 极小 |

### 5.2 调试信息权衡

```
编译选项对比:

Release (-O3)
  - 优点: 最快运行速度
  - 缺点: 无法调试
  
RelWithDebInfo (-O2 -g)
  - 优点: 可调试，速度接近Release
  - 缺点: 编译时间增加，二进制变大
  
Debug (-O0 -g)
  - 优点: 最佳调试体验
  - 缺点: 运行慢5-10倍

llama.cpp建议: 开发用RelWithDebInfo，发布用Release
```

### 5.3 断言使用策略

```cpp
// 硬断言 - 不可恢复的错误
GGML_ASSERT(tensor != nullptr);  // 发布版也检查

// 软断言 - 可恢复的错误
if (!tensor) {
    LOG_WRN("Tensor is null, skipping\n");
    return false;
}

// 调试断言 - 仅调试版
#ifndef NDEBUG
    assert(check_invariants(ctx));
#endif
```

## 6. 动手练习

### 练习1：日志级别实验

```cpp
// 创建测试程序
#include "common/log.h"

int main() {
    // 测试各级别日志
    LOG_DBG("Debug message: %d\n", 1);
    LOG_INF("Info message: %d\n", 2);
    LOG_WRN("Warning message: %d\n", 3);
    LOG_ERR("Error message: %d\n", 4);
    
    return 0;
}
```

```bash
# 编译并测试不同级别
./test_log 2>/dev/null  # 只显示INFO及以上
./test_log --verbose    # 显示DEBUG
```

### 练习2：GDB调试会话

```bash
# 1. 编译调试版本
cmake -B build_debug -DCMAKE_BUILD_TYPE=Debug
make -C build_debug -j

# 2. 启动调试会话
gdb ./build_debug/bin/llama-cli

# 3. 设置断点并运行
(gdb) break llama_decode
(gdb) run -m model.gguf -p "Hello" -n 10

# 4. 检查变量
(gdb) print ctx->kv_cache->size
(gdb) print ctx->n_tokens

# 5. 单步跟踪
(gdb) step
(gdb) next
```

### 练习3：内存问题检测

```bash
# 使用AddressSanitizer
cmake -B build_asan \
    -DCMAKE_CXX_FLAGS="-fsanitize=address -fno-omit-frame-pointer -g" \
    -DCMAKE_C_FLAGS="-fsanitize=address -fno-omit-frame-pointer -g"
make -C build_asan -j

# 运行测试
./build_asan/bin/llama-cli -m model.gguf -p "Test"

# 分析输出
# AddressSanitizer会报告内存错误的位置和堆栈
```

### 练习4：性能瓶颈定位

```bash
# 1. 编译性能分析版本
cmake -B build_perf -DCMAKE_BUILD_TYPE=RelWithDebInfo
make -C build_perf -j

# 2. 使用perf记录
perf record -g -- ./build_perf/bin/llama-cli \
    -m model.gguf -p "Hello world, this is a test" -n 100

# 3. 分析报告
perf report --sort=dso,symbol

# 4. 生成火焰图
perf script | ./stackcollapse-perf.pl | ./flamegraph.pl > llama_perf.svg
```

### 练习5：创建调试工具函数

```cpp
// debug_utils.h
#pragma once
#include "llama.h"
#include "ggml.h"

// 检查上下文状态
bool llama_debug_check_context(const llama_context* ctx);

// 打印张量信息
void llama_debug_print_tensor(const ggml_tensor* tensor, const char* name);

// 检查KV缓存一致性
bool llama_debug_check_kv_cache(const llama_context* ctx);

// 性能计时器
class PerfTimer {
    uint64_t t_start;
    const char* name;
public:
    explicit PerfTimer(const char* n) : name(n) {
        t_start = ggml_time_us();
    }
    ~PerfTimer() {
        uint64_t elapsed = ggml_time_us() - t_start;
        LOG_INF("[Timer] %s: %.3f ms\n", name, elapsed / 1000.0);
    }
};

#define PERF_TIMER(name) PerfTimer _timer_##__LINE__(name)
```

## 7. 本课小结

- **日志系统**：使用LOG_DBG/LOG_INF等级别宏，通过--verbose控制输出
- **调试器**：GDB/LLDB用于源码级调试，学会查看变量和堆栈
- **内存问题**：使用AddressSanitizer和Valgrind检测
- **数值稳定性**：检查NaN/Inf，关注量化类型兼容性
- **性能分析**：perf用于CPU分析，Nsight用于GPU分析

**调试检查清单：**
- [ ] 启用了适当的日志级别
- [ ] 模型文件SHA校验通过
- [ ] 内存限制足够
- [ ] 后端设备检测正常
- [ ] 张量数值无异常
- [ ] 性能瓶颈已定位
