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
| `ggml/src/ggml.c` | GGML核心 | 张量操作、计算图执行 |
| `examples/llama-bench/llama-bench.cpp` | 基准测试 | 性能测试、指标收集 |

## 4. 详细章节内容

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
