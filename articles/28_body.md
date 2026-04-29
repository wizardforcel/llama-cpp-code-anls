# 第28章 调试与问题排查 —— 开发者的"火眼金睛"

## 学习目标

1. 掌握llama.cpp的日志系统和调试技巧
2. 学习使用GDB/LLDB进行源码级调试
3. 理解常见内存问题的排查方法
4. 掌握数值稳定性问题的诊断
5. 学会使用性能分析工具定位瓶颈

---

## 生活类比：侦探破案

想象你是一位经验丰富的侦探，正在调查一起复杂的"案件"（程序bug）。你的工具箱里有各种专业设备：日志记录就像是收集目击证词，调试器是现场勘查的精密仪器，性能分析是重建案件时间线，内存检查是法医鉴定。

一个好的侦探知道如何：

**收集证据（日志分析）** —— 从案发现场收集所有可能的线索。日志不会说谎，它记录了程序运行的每一个关键时刻。学会读懂日志，就像学会分析指纹和DNA。

**现场勘查（调试器使用）** —— 在关键位置设置"警戒线"（断点），仔细观察嫌疑人的一举一动（单步执行），检查现场的每一个细节（变量值）。

**重现现场（最小复现）** —— 最好的侦探能够将案件的关键要素剥离出来，在受控环境下重现犯罪过程。编写最小复现案例，是定位bug的最有效方法。

**科学鉴定（专业工具）** —— 面对复杂的案件，需要借助专业实验室：AddressSanitizer检测内存问题就像毒理分析，perf性能分析就像弹道测试，最终锁定真凶（bug根源）。

---

## 28.1 日志系统详解 —— 程序的"黑匣子"

### 28.1.1 日志级别体系

llama.cpp使用分层日志系统，不同级别用于不同场景。

**源码位置**：`common/log.h` (第1-100行)

```cpp
/**
 * 日志级别定义
 * 
 * 级别数值越小，优先级越高。
 * 设置阈值后，只有优先级高于阈值的消息会被输出。
 */
#define LOG_LEVEL_DEBUG  4  // 调试信息：详细的内部状态
#define LOG_LEVEL_INFO   3  // 一般信息：正常运行状态
#define LOG_LEVEL_WARN   2  // 警告：可能的问题，程序继续运行
#define LOG_LEVEL_ERROR  1  // 错误：严重问题，可能无法恢复
#define LOG_LEVEL_OUTPUT 0  // 工具输出：非日志的程序输出

/**
 * 日志宏定义
 * 
 * 使用宏而非函数，避免参数计算开销。
 * 当日志级别被过滤时，参数不会被计算。
 */
#define LOG_DBG(...) \
    LOG_TMPL(GGML_LOG_LEVEL_DEBUG, LOG_LEVEL_DEBUG, __VA_ARGS__)
#define LOG_INF(...) \
    LOG_TMPL(GGML_LOG_LEVEL_INFO,  LOG_LEVEL_INFO,  __VA_ARGS__)
#define LOG_WRN(...) \
    LOG_TMPL(GGML_LOG_LEVEL_WARN,  LOG_LEVEL_WARN,  __VA_ARGS__)
#define LOG_ERR(...) \
    LOG_TMPL(GGML_LOG_LEVEL_ERROR, LOG_LEVEL_ERROR, __VA_ARGS__)

/**
 * 日志宏的实现原理
 * 
 * 双重检查确保最小开销：
 * 1. 编译时检查（如果阈值设为ERROR，DEBUG代码会被编译掉）
 * 2. 运行时检查（动态调整阈值）
 */
#define LOG_TMPL(LEVEL, INTLEVEL, ...)                                 \
    do {                                                               \
        if (INTLEVEL <= LOG_LEVEL_ERROR) { /* 编译时检查 */            \
            if (common_log_get_verbosity() >= INTLEVEL) {              \
                common_log_internal(LEVEL, __FILE__, __LINE__, __VA_ARGS__); \
            }                                                          \
        }                                                              \
    } while(0)

// 使用示例
void example_function() {
    LOG_DBG("Entering function, param=%d\n", 42);     // 只在调试时输出
    LOG_INF("Loading model from %s\n", path);         // 一般信息
    LOG_WRN("Deprecated API used\n");                 // 警告
    LOG_ERR("Failed to open file: %s\n", strerror(errno)); // 错误
}
```

**日志输出示例**

```bash
# 默认级别(INFO)的输出
$ ./llama-cli -m model.gguf -p "Hello"
llm_load_tensors: ggml ctx size =    0.27 MiB
llm_load_tensors: offloading 32 repeating layers to GPU
llm_load_tensors: loaded 291 tensors (4096.50 MB)

# 调试级别(DEBUG)的输出
$ ./llama-cli -m model.gguf -p "Hello" --verbose
0.00.035.060 D ggml_backend_metal_log_allocated_size: allocated buffer, size = 134217728
0.00.035.064 I llm_load_tensors: ggml ctx size =    0.27 MiB
0.00.035.070 D llm_load_tensors: tensor 'token_embd.weight' (f16) -> Metal
0.00.035.075 D llm_load_tensors: tensor 'blk.0.attn_q.weight' (q4_K) -> Metal
...
```

### 28.1.2 异步日志实现

llama.cpp的日志系统是异步的，避免IO阻塞主线程。

**源码位置**：`common/log.cpp` (第1-200行)

```cpp
/**
 * 异步日志实现
 * 
 * 日志消息先写入环形缓冲区，由后台线程批量写入。
 * 这样即使磁盘IO慢，也不会阻塞主线程。
 */
struct common_log {
    std::mutex mtx;                    // 保护缓冲区的锁
    std::thread thrd;                  // 后台写入线程
    std::condition_variable cv;        // 用于唤醒线程
    bool running = true;               // 线程运行标志
    
    // 环形缓冲区
    static constexpr size_t RING_SIZE = 1024;
    std::vector<common_log_entry> entries{RING_SIZE};
    size_t head = 0;   // 读取位置
    size_t tail = 0;   // 写入位置
    
    // 日志级别阈值
    int verbosity_thold = LOG_LEVEL_INFO;
    
    // 文件输出
    FILE* file = nullptr;
    
    // 配置选项
    bool timestamps = true;
    bool colors = true;
};

/**
 * 添加日志条目
 */
void common_log_add(
    struct common_log* log,
    enum ggml_log_level level,
    const char* fmt,
    ...
) {
    va_list args;
    va_start(args, fmt);
    
    {
        std::lock_guard<std::mutex> lock(log->mtx);
        
        // 获取写入位置
        auto& entry = log->entries[log->tail];
        
        // 格式化消息
        vsnprintf(entry.msg.data(), entry.msg.size(), fmt, args);
        entry.level = level;
        entry.timestamp = ggml_time_us();
        
        // 推进写入位置
        log->tail = (log->tail + 1) % log->entries.size();
        
        // 如果缓冲区满，丢弃最旧的消息
        if (log->tail == log->head) {
            log->head = (log->head + 1) % log->entries.size();
        }
    }
    
    // 唤醒后台线程
    log->cv.notify_one();
    va_end(args);
}

/**
 * 后台写入线程
 */
void common_log_thread(struct common_log* log) {
    while (log->running) {
        std::unique_lock<std::mutex> lock(log->mtx);
        
        // 等待有新消息或超时
        log->cv.wait_for(lock, std::chrono::milliseconds(100),
            [&] { return log->head != log->tail || !log->running; }
        );
        
        // 批量写入所有待处理消息
        while (log->head != log->tail) {
            auto& entry = log->entries[log->head];
            
            // 格式化输出
            if (log->timestamps) {
                print_timestamp(entry.timestamp);
            }
            
            if (log->colors) {
                print_colored(entry.level, entry.msg.data());
            } else {
                fprintf(log->file ? log->file : stderr, "%s", entry.msg.data());
            }
            
            log->head = (log->head + 1) % log->entries.size();
        }
        
        fflush(log->file ? log->file : stderr);
    }
}

/**
 * 异步日志优势：
 * 
 * 1. 不阻塞主线程
 *    日志写入磁盘通常需要1-10ms
 *    异步方式下，主线程只需执行几个内存操作（<1μs）
 * 
 * 2. 批量写入
 *    合并多个小写入为一个大写入，减少系统调用开销
 * 
 * 3. 自动处理缓冲区溢出
 *    当生产速度大于消费速度时，旧消息被丢弃
 *    保证程序不会因为日志而崩溃
 */
```

### 28.1.3 日志配置

**源码位置**：`common/log.cpp` (第200-400行)

```cpp
/**
 * 日志配置API
 */

// 设置日志级别阈值
void common_log_set_verbosity_thold(struct common_log* log, int verbosity) {
    log->verbosity_thold = verbosity;
}

// 启用文件日志
void common_log_set_file(struct common_log* log, const char* file) {
    if (log->file) {
        fclose(log->file);
    }
    log->file = fopen(file, "a");
    if (!log->file) {
        LOG_WRN("Failed to open log file: %s\n", file);
    }
}

// 启用/禁用颜色
void common_log_set_colors(struct common_log* log, bool colors) {
    log->colors = colors;
}

// 启用/禁用时间戳
void common_log_set_timestamps(struct common_log* log, bool timestamps) {
    log->timestamps = timestamps;
}

/**
 * 命令行使用示例
 */

// 启用调试日志（详细输出）
$ ./llama-cli -m model.gguf -p "Hello" --verbose

// 或设置环境变量
$ export LLAMA_LOG_LEVEL=4  # DEBUG级别
$ export LLAMA_LOG_LEVEL=3  # INFO级别（默认）
$ export LLAMA_LOG_LEVEL=2  # WARN级别
$ export LLAMA_LOG_LEVEL=1  # ERROR级别

// 输出到文件
$ ./llama-cli -m model.gguf -p "Hello" 2>llama.log

// 同时输出到文件和终端
$ ./llama-cli -m model.gguf -p "Hello" 2> >(tee llama.log)
```

---

## 28.2 调试工具与方法

### 28.2.1 GDB调试基础

GDB是Linux下调试C/C++程序的标准工具。

```bash
# ========== 编译调试版本 ==========
# 注意：llama.cpp默认Release模式优化很强，需要显式启用Debug
cmake -B build_debug -DCMAKE_BUILD_TYPE=Debug
make -C build_debug -j$(nproc)

# 或者使用RelWithDebInfo（推荐，有调试信息但保持一定优化）
cmake -B build_debug -DCMAKE_BUILD_TYPE=RelWithDebInfo
make -C build_debug -j$(nproc)

# ========== 启动GDB ==========
gdb ./build_debug/bin/llama-cli

# ========== 常用GDB命令 ==========

# 设置断点
(gdb) break llama_decode                    # 在函数入口断点
(gdb) break llama.cpp:1234                  # 在指定文件行号断点
(gdb) break llama_decode if ctx->n_tokens > 100  # 条件断点
(gdb) info breakpoints                      # 查看所有断点
(gdb) delete 1                              # 删除断点1

# 运行程序
(gdb) run -m model.gguf -p "Hello" -n 10    # 带参数运行
(gdb) run                                   # 再次运行
(gdb) start                                 # 停在main函数

# 单步执行
(gdb) next          # 单步（不进入函数）
(gdb) step          # 单步（进入函数）
(gdb) finish        # 运行到当前函数返回
(gdb) continue      # 继续运行到下一个断点

# 查看变量
(gdb) print ctx->n_tokens                   # 打印变量
(gdb) print *ctx                            # 打印结构体
(gdb) print ctx->kv_cache->size             # 打印嵌套成员
(gdb) print ctx->n_tokens = 512             # 修改变量值
(gdb) display ctx->n_tokens                 # 每次停顿时自动显示

# 查看内存
(gdb) x/10f tensor->data                    # 以float格式查看10个值
(gdb) x/100bx tensor->data                  # 以hex查看100字节
(gdb) info registers                        # 查看寄存器

# 查看堆栈
(gdb) backtrace                             # 查看调用栈
(gdb) backtrace full                        # 完整堆栈（含局部变量）
(gdb) frame 2                               # 切换到第2帧
(gdb) up                                    # 向上一层
(gdb) down                                  # 向下一层

# 其他
(gdb) list                                  # 显示源代码
(gdb) search printf                         # 搜索代码
(gdb) info locals                           # 显示局部变量
(gdb) info args                             # 显示函数参数
(gdb) quit                                  # 退出GDB
```

### 28.2.2 核心数据结构检查

调试llama.cpp时，经常需要检查这些核心结构：

```cpp
// ========== 检查上下文状态 ==========
(gdb) print *ctx
$1 = {
  model = 0x5555555f5eb0,
  n_tokens = 512,
  n_outputs = 1,
  kv_cache = {
    size = 4096,
    used = 512,
    cells = std::vector of length 4096,
    // ...
  },
  backend = 0x555555600000,
  // ...
}

// ========== 检查模型结构 ==========
(gdb) print *ctx->model
$2 = {
  hparams = {
    n_vocab = 32000,
    n_ctx_train = 4096,
    n_embd = 4096,
    n_layer = 32,
    n_head = 32,
    // ...
  },
  vocab = {
    type = LLAMA_VOCAB_TYPE_SPM,
    tokens = std::vector of length 32000,
    // ...
  },
  // ...
}

// ========== 检查张量 ==========
(gdb) print *tensor
$3 = {
  type = GGML_TYPE_F32,
  ne = {4096, 11008, 1, 1},      // 元素数量（各维度）
  nb = {4, 16384, 180224000, 180224000},  // 步长（字节）
  data = 0x7fff5000,             // 数据指针
  name = "blk.0.ffn_gate.weight"
}

// 查看张量数据
(gdb) x/10f tensor->data
0x7fff5000: 0.001234  -0.005678  0.009012  -0.003456
0x7fff5010: 0.007890  -0.001234  0.005678  -0.009012
0x7fff5020: 0.003456  -0.007890

// ========== 检查KV缓存 ==========
(gdb) print ctx->kv_cache->cells[0]
$4 = {
  pos = 0,
  seq_id = std::unordered_set with 1 element = {0}
}

(gdb) print ctx->kv_cache->cells[512]
$5 = {
  pos = -1,  // -1表示未使用
  seq_id = std::unordered_set with 0 elements
}
```

### 28.2.3 LLDB调试（macOS）

macOS上LLDB是默认调试器，语法与GDB类似但有一些差异。

```bash
# 启动LLDB
lldb ./build/bin/llama-cli

# ========== LLDB命令 ==========

# 设置断点
(lldb) breakpoint set --name llama_decode
(lldb) breakpoint set --file llama.cpp --line 1234
(lldb) breakpoint set --name llama_decode --condition "ctx->n_tokens > 100"
(lldb) breakpoint list
(lldb) breakpoint delete 1

# 运行
(lldb) run -m model.gguf -p "Hello"
(lldb) process launch -- -m model.gguf -p "Hello"

# 单步
(lldb) next
(lldb) step
(lldb) finish
(lldb) continue

# 查看变量
(lldb) frame variable ctx
(lldb) frame variable ctx->n_tokens
(lldb) expr ctx->n_tokens
(lldb) expr ctx->n_tokens = 512

# 查看内存
(lldb) memory read --size 4 --format f 0x7fff5000 --count 10
(lldb) memory read -s4 -f f 0x7fff5000 -c100

# 查看堆栈
(lldb) thread backtrace
(lldb) thread backtrace all
(lldb) frame select 2
(lldb) up
(lldb) down

# 其他
(lldb) source list
(lldb) register read
(lldb) quit
```

---

## 28.3 常见问题排查

### 28.3.1 内存问题

**症状1：段错误(Segmentation Fault)**

```bash
# ========== 排查步骤 ==========

# 1. 检查模型文件完整性
$ sha256sum model.gguf
$ sha256sum -c checksum.txt  # 对比官方校验和

# 2. 检查内存限制
$ ulimit -a
-t: cpu time (seconds)         unlimited
-f: file size (blocks)         unlimited
-d: data seg size (kbytes)     unlimited
-s: stack size (kbytes)        8192
-c: core file size (blocks)    0
-m: resident set size (kbytes) unlimited
-v: virtual memory (kbytes)    unlimited  # 检查这个

# 增加虚拟内存限制
$ ulimit -v unlimited

# 3. 使用AddressSanitizer编译
cmake -B build_asan \
    -DCMAKE_CXX_FLAGS="-fsanitize=address -fno-omit-frame-pointer -g" \
    -DCMAKE_C_FLAGS="-fsanitize=address -fno-omit-frame-pointer -g"
make -C build_asan -j

# 运行测试（ASan会报告详细的内存错误）
$ ./build_asan/bin/llama-cli -m model.gguf -p "Test"

# 典型ASan输出：
==12345==ERROR: AddressSanitizer: heap-buffer-overflow
READ of size 4 at 0x602000123456 thread T0
    #0 in llama_decode llama.cpp:1234
    #1 in main llama-cli.cpp:567
```

**症状2：内存泄漏**

```bash
# 使用Valgrind检测内存泄漏
$ valgrind --leak-check=full \
           --show-leak-kinds=all \
           --track-origins=yes \
           --log-file=valgrind.log \
           ./llama-cli -m model.gguf -p "Test" -n 10

# 查看报告
$ cat valgrind.log

# 典型输出：
==12345== HEAP SUMMARY:
==12345==     in use at exit: 4,096,000 bytes in 100 blocks
==12345==   total heap usage: 1,234 allocs, 1,134 frees
==12345== 
==12345== 4,096,000 bytes in 100 blocks are definitely lost
==12345==    at 0x4C2FB0F: malloc (vg_replace_malloc.c:299)
==12345==    by 0x123456: ggml_new_tensor (ggml.c:1234)
==12345==    by 0x789ABC: llama_build_graph (llama.cpp:567)
```

### 28.3.2 数值稳定性问题

**症状：输出NaN或Inf**

```cpp
// ========== 数值检查工具 ==========

/**
 * 检查张量数值有效性
 */
bool check_tensor_values(const ggml_tensor* tensor, const char* name) {
    if (!tensor || !tensor->data) {
        LOG_ERR("Invalid tensor: %s\n", name);
        return false;
    }
    
    const float* data = (const float*)tensor->data;
    size_t n = ggml_nelements(tensor);
    
    int nan_count = 0;
    int inf_count = 0;
    float min_val = INFINITY;
    float max_val = -INFINITY;
    
    for (size_t i = 0; i < n; i++) {
        float v = data[i];
        
        if (std::isnan(v)) {
            nan_count++;
        } else if (std::isinf(v)) {
            inf_count++;
        } else {
            min_val = std::min(min_val, v);
            max_val = std::max(max_val, v);
        }
    }
    
    if (nan_count > 0 || inf_count > 0) {
        LOG_ERR("Tensor %s: %d NaN, %d Inf (of %zu)\n",
                name, nan_count, inf_count, n);
        LOG_ERR("  Valid range: [%f, %f]\n", min_val, max_val);
        return false;
    }
    
    LOG_DBG("Tensor %s: OK, range [%f, %f]\n", name, min_val, max_val);
    return true;
}

/**
 * 常见数值稳定性问题原因：
 * 
 * 1. 量化溢出
 *    - 使用不兼容的量化类型
 *    - 解决：检查量化参数
 * 
 * 2. 学习率过大（微调模型）
 *    - 权重更新过大导致溢出
 *    - 解决：降低学习率，使用梯度裁剪
 * 
 * 3. 输入数值范围异常
 *    - 输入包含极大或极小的值
 *    - 解决：归一化输入
 * 
 * 4. Softmax数值溢出
 *    - 指数运算导致溢出
 *    - 解决：使用数值稳定的softmax实现
 */

// 在关键位置插入检查
void llama_debug_check_tensors(llama_context* ctx) {
    // 检查输入
    check_tensor_values(ctx->inp_tokens, "inp_tokens");
    
    // 检查各层输出
    for (int i = 0; i < ctx->model->hparams.n_layer; i++) {
        char name[64];
        snprintf(name, sizeof(name), "layer_%d_output", i);
        // 检查中间结果...
    }
    
    // 检查输出
    check_tensor_values(ctx->out_logits, "out_logits");
}
```

数值稳定性检查是排查模型输出异常的第一步。在生产环境中，建议在关键节点加入张量值域检查，及时发现 NaN/Inf 扩散，避免错误在后续计算中放大。

### 28.3.3 后端兼容性问题

```cpp
/**
 * 后端支持检查工具
 */
void check_backend_support() {
    LOG_INF("Available backends:\n");
    
    for (size_t i = 0; i < ggml_backend_reg_count(); i++) {
        auto* reg = ggml_backend_reg_get(i);
        LOG_INF("  [%zu] %s\n", i, ggml_backend_reg_name(reg));
    }
    
    // CUDA特定检查
    #ifdef GGML_USE_CUDA
    {
        int device_count;
        cudaError_t err = cudaGetDeviceCount(&device_count);
        
        if (err != cudaSuccess) {
            LOG_ERR("CUDA error: %s\n", cudaGetErrorString(err));
        } else if (device_count == 0) {
            LOG_WRN("No CUDA devices found, falling back to CPU\n");
        } else {
            LOG_INF("CUDA devices found: %d\n", device_count);
            
            for (int i = 0; i < device_count; i++) {
                cudaDeviceProp prop;
                cudaGetDeviceProperties(&prop, i);
                LOG_INF("  [%d] %s (%d MB)\n", i, prop.name, 
                        prop.totalGlobalMem / 1024 / 1024);
            }
        }
    }
    #endif
    
    // Metal特定检查（macOS）
    #ifdef GGML_USE_METAL
    {
        LOG_INF("Metal backend enabled\n");
        // Metal设备信息在运行时获取
    }
    #endif
    
    // Vulkan检查
    #ifdef GGML_USE_VULKAN
    {
        // Vulkan实例和设备检查
    }
    #endif
}
```

后端兼容性诊断可以快速定位 GPU 不可用等问题的根源。在启动时执行后端检测，能够帮助运维人员及时发现驱动问题、CUDA 版本不匹配等配置错误。

---

## 28.4 性能分析

### 28.4.1 内置性能统计

llama.cpp内置了详细的性能统计功能。

**源码位置**：`src/llama.cpp` (第1-200行)

```cpp
/**
 * 性能上下文结构
 * 
 * 记录各个环节的耗时和处理的token数。
 */
struct llama_perf_context {
    uint64_t t_start_ms;      // 程序开始时间
    uint64_t t_load_ms;       // 模型加载时间
    uint64_t t_p_eval_ms;     // 提示处理时间
    uint64_t t_eval_ms;       // 生成时间
    
    uint32_t n_p_eval;        // 处理的提示token数
    uint32_t n_eval;          // 生成的token数
    
    // 采样时间
    uint64_t t_sample_ms;
    uint32_t n_sample;
};

/**
 * 打印性能统计
 */
void llama_perf_context_print(const struct llama_context* ctx) {
    const auto& p = ctx->perf;
    
    LOG_INF("\n");
    LOG_INF("llama_perf_context_print:        load time = %10.2f ms\n",
            p.t_load_ms / 1000.0);
    LOG_INF("llama_perf_context_print: prompt eval time = %10.2f ms / %5d tokens (%8.3f ms/token)\n",
            p.t_p_eval_ms / 1000.0, p.n_p_eval, 
            p.t_p_eval_ms / 1000.0 / p.n_p_eval);
    LOG_INF("llama_perf_context_print:        eval time = %10.2f ms / %5d tokens (%8.3f ms/token)\n",
            p.t_eval_ms / 1000.0, p.n_eval,
            p.t_eval_ms / 1000.0 / p.n_eval);
    LOG_INF("llama_perf_context_print:       total time = %10.2f ms\n",
            (ggml_time_us() - p.t_start_ms) / 1000.0);
}

// 输出示例：
// llama_perf_context_print:        load time =    1234.56 ms
// llama_perf_context_print: prompt eval time =     234.56 ms /   512 tokens (    0.46 ms/token)
// llama_perf_context_print:        eval time =    1234.56 ms /   128 tokens (    9.64 ms/token)
// llama_perf_context_print:       total time =    2469.12 ms
```

### 28.4.2 使用perf进行CPU分析

```bash
# ========== perf性能分析 ==========

# 1. 编译性能分析版本
cmake -B build_perf -DCMAKE_BUILD_TYPE=RelWithDebInfo
make -C build_perf -j

# 2. 记录性能数据（-g启用调用图）
$ perf record -g -- ./build_perf/bin/llama-cli \
    -m model.gguf -p "Hello world" -n 100

# 3. 生成报告
$ perf report
# 交互式查看热点函数

# 4. 生成火焰图（需要flamegraph.pl脚本）
$ perf script | ./stackcollapse-perf.pl | ./flamegraph.pl > llama_perf.svg

# 5. 查看特定函数
$ perf annotate -s ggml_compute_forward

# 其他常用命令
$ perf stat ./llama-cli -m model.gguf -p "Hello" -n 10  # 统计性能计数器
$ perf top -p $(pgrep llama-server)  # 实时监控运行中的进程
```

### 28.4.3 CUDA分析

```bash
# ========== NVIDIA Nsight工具 ==========

# 1. Nsight Systems（系统级分析）
$ nsys profile -o report ./llama-cli -m model.gguf -p "Hello" -n 100
$ nsys-ui report.nsys-rep  # 打开GUI查看

# 2. Nsight Compute（内核级分析）
$ ncu -o profile_report \
      --metrics gpu__time_duration.sum \
      ./llama-cli -m model.gguf -p "Hello" -n 10
$ ncu-ui profile_report.ncu-rep

# 3. nvprof（旧版工具，仍可工作）
$ nvprof ./llama-cli -m model.gguf -p "Hello" -n 10

# ========== 查看GPU利用率 ==========
$ watch -n 1 nvidia-smi

# 输出解释：
# GPU利用率：计算单元忙碌百分比
# 内存利用率：内存控制器忙碌百分比
# 温度/功耗：散热和电源状态
```

---

## 设计中的取舍

### 28.1 日志级别选择

| 级别 | 使用场景 | 性能影响 | 磁盘占用 | 生产环境 |
|-----|---------|---------|---------|---------|
| DEBUG | 开发调试 | 高（大量IO） | 大 | 否 |
| INFO | 正常运行 | 中 | 中 | 是（默认） |
| WARN | 生产环境 | 低 | 小 | 是 |
| ERROR | 关键系统 | 极低 | 极小 | 是 |

### 28.2 调试信息权衡

```
编译选项对比：

Release (-O3)
  - 优点: 最快运行速度，最小二进制
  - 缺点: 无法调试，堆栈信息不可用
  
RelWithDebInfo (-O2 -g)
  - 优点: 可调试，速度接近Release
  - 缺点: 编译时间增加，二进制变大（2-3x）
  
Debug (-O0 -g)
  - 优点: 最佳调试体验，变量优化最少
  - 缺点: 运行慢5-10倍，不适合大模型

llama.cpp建议: 
- 开发用RelWithDebInfo
- 发布用Release
- 复杂问题用Debug
```

### 28.3 断言使用策略

```cpp
// 硬断言 - 不可恢复的错误，发布版也检查
GGML_ASSERT(tensor != nullptr);
// 如果失败，程序立即终止
// 用于防止后续代码访问无效内存

// 软断言 - 可恢复的错误
if (!tensor) {
    LOG_WRN("Tensor is null, skipping operation\n");
    return false;
}
// 优雅降级，程序继续运行

// 调试断言 - 仅调试版检查
#ifndef NDEBUG
    assert(check_invariants(ctx));
#endif
// 用于开发时检查不变量
// 发布版不执行，零开销
```

---

## 动手练习

### 练习1：日志级别实验

```cpp
// test_log.cpp
#include "common/log.h"
#include <cstdio>

int main() {
    // 测试各级别日志
    LOG_DBG("Debug message: value=%d\n", 1);
    LOG_INF("Info message: value=%d\n", 2);
    LOG_WRN("Warning message: value=%d\n", 3);
    LOG_ERR("Error message: value=%d\n", 4);
    
    // 条件日志
    int verbose = 1;
    if (verbose) {
        LOG_INF("Verbose mode enabled\n");
    }
    
    return 0;
}
```

```bash
# 编译
$ g++ -o test_log test_log.cpp -I. common/log.cpp -lpthread

# 测试不同级别
$ ./test_log 2>/dev/null                    # 默认INFO及以上
$ ./test_log --verbose 2>/dev/null          # DEBUG及以上
$ LLAMA_LOG_LEVEL=1 ./test_log 2>/dev/null  # 只显示ERROR
```

### 练习2：GDB调试会话

```bash
# 1. 编译调试版本
cmake -B build_debug -DCMAKE_BUILD_TYPE=Debug
make -C build_debug -j

# 2. 创建GDB脚本
$ cat > debug.gdb << 'EOF'
set pagination off
set logging on

break llama_decode
run -m model.gguf -p "Hello" -n 10

# 检查上下文
print ctx->n_tokens
print ctx->kv_cache->size

# 继续执行
continue

quit
EOF

# 3. 运行调试会话
$ gdb -x debug.gdb ./build_debug/bin/llama-cli
```

### 练习3：内存问题检测

```bash
#!/bin/bash
# memory_check.sh - 内存检查脚本

# 使用AddressSanitizer
cmake -B build_asan \
    -DCMAKE_CXX_FLAGS="-fsanitize=address -fno-omit-frame-pointer -g" \
    -DCMAKE_C_FLAGS="-fsanitize=address -fno-omit-frame-pointer -g"
make -C build_asan -j

# 运行测试
echo "Running with AddressSanitizer..."
./build_asan/bin/llama-cli \
    -m model.gguf \
    -p "Test prompt for memory checking" \
    -n 10 \
    2>&1 | tee asan_report.txt

# 检查结果
if grep -q "ERROR: AddressSanitizer" asan_report.txt; then
    echo "Memory errors detected! Check asan_report.txt"
    exit 1
else
    echo "No memory errors found."
    exit 0
fi
```

### 练习4：性能瓶颈定位

```bash
#!/bin/bash
# profile.sh - 性能分析脚本

# 编译性能版本
cmake -B build_perf -DCMAKE_BUILD_TYPE=RelWithDebInfo
make -C build_perf -j

# 运行perf
perf record -g -- \
    ./build_perf/bin/llama-cli \
    -m model.gguf \
    -p "Hello world, this is a performance test" \
    -n 100

# 生成报告
echo "Top hotspots:"
perf report --sort=dso,symbol --stdio | head -50

# 生成火焰图（如果有flamegraph工具）
if command -v flamegraph.pl &> /dev/null; then
    perf script | stackcollapse-perf.pl | flamegraph.pl > llama_flamegraph.svg
    echo "Flame graph saved to llama_flamegraph.svg"
fi
```

### 练习5：创建调试工具函数

```cpp
// debug_utils.h - 调试工具头文件
#pragma once

#include "llama.h"
#include "ggml.h"
#include "common/log.h"

// 调试命名空间
namespace llama_debug {

/**
 * 检查上下文状态
 */
bool check_context(const llama_context* ctx);

/**
 * 打印张量信息
 */
void print_tensor(const ggml_tensor* tensor, const char* name);

/**
 * 检查KV缓存一致性
 */
bool check_kv_cache(const llama_context* ctx);

/**
 * 检查模型参数有效性
 */
bool check_model_params(const llama_model* model);

/**
 * 性能计时器
 */
class Timer {
    uint64_t t_start;
    const char* name;
public:
    explicit Timer(const char* n) : name(n) {
        t_start = ggml_time_us();
    }
    ~Timer() {
        uint64_t elapsed = ggml_time_us() - t_start;
        LOG_INF("[Timer] %s: %.3f ms\n", name, elapsed / 1000.0);
    }
};

#define LLAMA_TIMER(name) llama_debug::Timer _timer_##__LINE__(name)

/**
 * 数值检查宏
 */
#define CHECK_TENSOR(t, name) \
    do { \
        if (!llama_debug::check_tensor_values(t, name)) { \
            LOG_ERR("Tensor check failed: %s\n", name); \
            return false; \
        } \
    } while(0)

bool check_tensor_values(const ggml_tensor* tensor, const char* name);

} // namespace llama_debug

// 使用示例
void example_usage(llama_context* ctx) {
    LLAMA_TIMER("example_function");
    
    // 检查上下文
    if (!llama_debug::check_context(ctx)) {
        LOG_ERR("Context check failed\n");
        return;
    }
    
    // 检查KV缓存
    if (!llama_debug::check_kv_cache(ctx)) {
        LOG_WRN("KV cache inconsistency detected\n");
    }
}
```

---

## 本课小结

### 调试工具箱

| 工具 | 用途 | 命令示例 |
|-----|------|---------|
| LOG_DBG/INF/WRN/ERR | 日志输出 | `LOG_INF("msg: %d", val)` |
| GDB | 源码调试 | `gdb ./llama-cli` |
| LLDB | macOS调试 | `lldb ./llama-cli` |
| AddressSanitizer | 内存错误检测 | `-fsanitize=address` |
| Valgrind | 内存泄漏检测 | `valgrind --leak-check=full` |
| perf | CPU性能分析 | `perf record -g` |
| Nsight | GPU分析 | `nsys profile` |

### 调试检查清单

- [ ] 启用了适当的日志级别（--verbose）
- [ ] 模型文件SHA校验通过
- [ ] 内存限制检查（ulimit）
- [ ] 后端设备检测正常
- [ ] 张量数值无NaN/Inf
- [ ] 性能瓶颈已定位（perf）
- [ ] 内存问题已排除（ASan/Valgrind）

### 常见错误速查

| 症状 | 可能原因 | 解决方法 |
|-----|---------|---------|
| Segmentation fault | 模型文件损坏/内存不足 | 校验SHA，检查ulimit |
| NaN输出 | 量化不兼容/数值溢出 | 检查量化类型，使用F16测试 |
| 启动慢 | mmap禁用/磁盘慢 | 启用--mmap，使用SSD |
| GPU不可用 | 驱动问题/CUDA版本不匹配 | nvidia-smi检查，重新编译 |
| 内存泄漏 | 未释放资源 | Valgrind检测，检查释放代码 |

本章我们一起学习了以下概念：

| 概念 | 解释 |
|------|------|
| 日志系统 | llama.cpp的异步分层日志，采用环形缓冲区避免IO阻塞主线程，支持DEBUG/INFO/WARN/ERROR四级 |
| GDB/LLDB调试 | 程序调试的标准工具，通过断点、单步执行、变量查看定位bug，编译时需启用RelWithDebInfo |
| AddressSanitizer | 编译时内存检测工具，可自动发现堆缓冲区溢出、释放后使用等内存错误 |
| 数值稳定性 | 检查张量中是否存在NaN或Inf值，常见于量化溢出、学习率过大、Softmax溢出等场景 |
| 性能分析 | 通过perf/Nsight工具定位CPU/GPU热点函数，生成火焰图可视化性能瓶颈 |

---

下一章中，我们将学习集成与部署案例——掌握 llama.cpp 在嵌入式、服务端和多语言绑定等场景的部署方案。

## 关联阅读

- **第24章**：llama-bench性能测试详解
- **第27章**：性能优化技巧
- **LLaMA.cpp Wiki**：`docs/debugging.md`
- **GDB文档**：https://sourceware.org/gdb/documentation/

---

*本章对应源码版本：master (2026-04-07)*
*本章作者：llama.cpp社区*
*最后更新：2026-04-08*
