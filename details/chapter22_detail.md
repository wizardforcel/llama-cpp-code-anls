# 第22章 通用工具库（common） —— 开发者的"瑞士军刀"

## 学习目标
1. 掌握common库的整体架构和模块划分
2. 深入理解命令行参数解析系统的设计与实现
3. 学会使用网络下载、控制台交互、日志等基础设施
4. 能够基于common库开发自定义工具
5. 理解跨平台兼容性的处理策略

---

## 生活类比：餐厅的后厨支持系统

想象你经营一家餐厅，common库就像后厨的支持系统：

- **命令行解析（arg.cpp）** = 点餐系统（理解顾客的各种特殊要求）
- **网络下载（download.cpp）** = 供应链系统（从仓库自动获取食材）
- **控制台交互（console.cpp）** = 服务员对讲机（实时沟通，处理突发情况）
- **日志系统（log.cpp）** = 后厨监控（记录一切操作，便于排查问题）
- **聊天处理（chat.cpp）** = 菜单翻译（将顾客需求转化为厨师指令）

就像一家高效餐厅需要完善的后厨支持，一个好的LLM工具也需要common库这样的基础设施。

---

## 源码地图

```
common/
├── arg.h/cpp              # 命令行参数解析（3907行）
│   ├── common_arg         # 参数定义结构
│   └── gpt_params         # 全局参数结构
├── common.h/cpp           # 通用工具函数（1924行）
│   ├── common_params      # 参数结构
│   └── common_init        # 初始化函数
├── console.h/cpp          # 控制台交互（1166行）
│   ├── console::readline  # 输入处理
│   └── console::spinner   # 加载动画
├── log.h/cpp              # 日志系统（446行）
│   ├── LOG_INF            # 信息日志宏
│   └── LOG_ERR            # 错误日志宏
├── download.h/cpp         # 网络下载（903行）
│   ├── common_download_model      # 模型下载
│   └── common_download_file_single # 文件下载
├── chat.h/cpp             # 聊天模板处理（2190行）
│   ├── common_chat_format         # 格式化对话
│   └── common_chat_parser_params  # 解析参数
├── sampling.h/cpp         # 采样封装（832行）
│   ├── common_sampler             # 采样器封装
│   └── llama_sampling_params      # 采样参数
├── speculative.h/cpp      # 投机解码（1074行）
├── json-schema-to-grammar.h/cpp  # JSON Schema转换
├── ngram-cache.h/cpp      # N-gram缓存
└── llguidance.cpp         # LLGuidance语法约束
```

---

## 22.1 命令行参数解析（arg.cpp）

### 22.1.1 设计哲学

llama.cpp的参数系统支持**50+个示例工具**，每个工具有不同的参数需求。arg.cpp通过"参数定义+处理器"的模式实现灵活配置。

**源码位置**：`common/arg.h` (第1-131行)

```cpp
// 参数定义结构
struct common_arg {
    // 该参数适用于哪些示例
    std::set<enum llama_example> examples = {LLAMA_EXAMPLE_COMMON};
    
    // 该参数不适用于哪些示例
    std::set<enum llama_example> excludes = {};
    
    // 参数名列表（如 {"-m", "--model"}）
    std::vector<const char *> args;
    std::vector<const char *> args_neg;  // 否定形式（如 --no-mmap）
    
    // 帮助信息
    const char * value_hint = nullptr;   // 值提示（如 "FNAME"）
    const char * env = nullptr;          // 环境变量名
    std::string help;
    
    // 处理器函数
    void (*handler_void)   (common_params & params) = nullptr;
    void (*handler_string) (common_params & params, const std::string & value) = nullptr;
    void (*handler_int)    (common_params & params, int value) = nullptr;
    void (*handler_bool)   (common_params & params, bool value) = nullptr;
    void (*handler_str_str)(common_params & params, const std::string & value1, const std::string & value2) = nullptr;
};
```

### 22.1.2 参数定义示例

**源码位置**：`common/arg.cpp` (第1000-1200行)

```cpp
// 定义 -m / --model 参数
common_arg arg_model() {
    return common_arg(
        {"-m", "--model"},
        "FNAME",
        "模型路径（或从HF下载: repo/model:Q4_K_M）",
        [](common_params & params, const std::string & value) {
            params.model.path = value;
        }
    ).set_examples({
        LLAMA_EXAMPLE_MAIN,
        LLAMA_EXAMPLE_SERVER,
        LLAMA_EXAMPLE_EMBEDDING,
        // ... 其他适用示例
    });
}

// 定义 --ngl / --gpu-layers 参数
common_arg arg_n_gpu_layers() {
    return common_arg(
        {"-ngl", "--gpu-layers", "--n-gpu-layers"},
        "N",
        "GPU层数（-1表示全部）",
        [](common_params & params, int value) {
            params.n_gpu_layers = value;
        }
    );
}

// 定义布尔参数（--mmap / --no-mmap）
common_arg arg_mmap() {
    return common_arg(
        {"--mmap"},
        {"--no-mmap"},
        "启用/禁用内存映射",
        [](common_params & params, bool value) {
            params.use_mmap = value;
        }
    ).set_env("LLAMA_ARG_MMAP");
}
```

### 22.1.3 参数解析流程

**源码位置**：`common/arg.cpp` (第2000-2500行)

```cpp
// 参数解析主函数
bool common_params_parse(int argc, char ** argv, common_params & params, llama_example example) {
    // 1. 收集适用于当前示例的所有参数
    std::vector<common_arg> args = get_args_for_example(example);
    
    // 2. 解析命令行
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        
        // 查找匹配的参数定义
        auto it = find_arg(args, arg);
        if (it != args.end()) {
            // 调用处理器
            if (it->handler_string) {
                it->handler_string(params, argv[++i]);
            } else if (it->handler_int) {
                it->handler_int(params, std::stoi(argv[++i]));
            } else if (it->handler_bool) {
                it->handler_bool(params, true);
            }
            // ...
        }
    }
    
    // 3. 验证参数
    return validate_params(params);
}
```

### 22.1.4 环境变量支持

```cpp
// 参数可从环境变量读取
common_arg & common_arg::set_env(const char * env_name) {
    env = env_name;
    return *this;
}

// 使用示例：
// export LLAMA_ARG_MMAP=0
// ./llama-cli -m model.gguf  # 自动读取环境变量，禁用mmap
```

---

## 22.2 网络下载功能（download.cpp）

### 22.2.1 功能概述

download.cpp实现了从HuggingFace和直接URL下载模型的功能，支持：
- 断点续传
- ETag缓存
- 进度显示
- 多部分GGUF自动检测

**源码位置**：`common/download.h` (第1-92行)

```cpp
// 下载结果
struct common_download_model_result {
    std::string model_path;   // 下载后的本地路径
    std::string mmproj_path;  // 多模态投影模型路径（如有）
};

// 下载选项
struct common_download_model_opts {
    bool download_mmproj = false;  // 是否同时下载mmproj
    bool offline = false;          // 离线模式（仅使用缓存）
};

// 主下载函数
common_download_model_result common_download_model(
    const common_params_model & model,
    const std::string & bearer_token,
    const common_download_model_opts & opts = {},
    const common_header_list & headers = {}
);
```

### 22.2.2 HuggingFace下载

**源码位置**：`common/download.cpp` (第1-300行)

```cpp
// 从HuggingFace仓库下载
common_download_model_result download_from_hf(
    const std::string & repo,
    const std::string & file,
    const std::string & cache_dir
) {
    // 1. 解析repo和tag（如 "ggml-org/models:Q4_K_M"）
    auto [repo_clean, tag] = common_download_split_repo_tag(repo);
    
    // 2. 如果没有指定文件，自动选择最佳GGUF
    std::string target_file = file;
    if (target_file.empty()) {
        target_file = find_best_gguf(repo_clean, tag);  // 按优先级: Q4_K_M > Q4_0 > 第一个
    }
    
    // 3. 检查多部分GGUF（如 model-00001-of-00003.gguf）
    std::vector<std::string> parts = detect_split_files(repo_clean, target_file);
    
    // 4. 下载所有部分
    for (const auto & part : parts) {
        download_file_with_cache(repo_clean, part, cache_dir);
    }
    
    // 5. 如需多模态，下载mmproj
    std::string mmproj = find_mmproj(repo_clean, target_file);
    
    return {local_path, mmproj_path};
}
```

### 22.2.3 断点续传实现

**源码位置**：`common/download.cpp` (第300-600行)

```cpp
// 带断点续传的文件下载
int common_download_file_single(
    const std::string & url,
    const std::string & path,
    const std::string & bearer_token,
    bool offline,
    const common_header_list & headers = {},
    bool skip_etag = false
) {
    // 1. 检查已下载部分
    size_t existing_size = 0;
    if (file_exists(path)) {
        existing_size = file_size(path);
    }
    
    // 2. 发送HTTP请求（带Range头）
    common_remote_params params;
    params.headers = headers;
    
    if (existing_size > 0) {
        params.headers.push_back({"Range", 
            string_format("bytes=%zu-", existing_size)});
    }
    
    // 3. 下载并追加到文件
    auto [http_code, data] = common_remote_get_content(url, params);
    
    if (http_code == 206) {  // Partial Content
        append_to_file(path, data);
    } else if (http_code == 200) {
        write_file(path, data);  // 完整内容
    }
    
    return http_code;
}
```

---

## 22.3 控制台交互（console.cpp）

### 22.3.1 跨平台输入处理

**源码位置**：`common/console.h` (第1-46行)

```cpp
namespace console {
    // 初始化控制台
    void init(bool use_simple_io, bool use_advanced_display);
    void cleanup();
    
    // 设置显示类型（影响颜色）
    enum display_type {
        DISPLAY_TYPE_RESET = 0,       // 重置
        DISPLAY_TYPE_INFO,            // 信息（绿色）
        DISPLAY_TYPE_PROMPT,          // 提示（蓝色）
        DISPLAY_TYPE_REASONING,       // 推理（黄色）
        DISPLAY_TYPE_USER_INPUT,      // 用户输入（白色）
        DISPLAY_TYPE_ERROR            // 错误（红色）
    };
    void set_display(display_type display);
    
    // 读取一行输入（支持多行）
    bool readline(std::string & line, bool multiline_input);
    
    // 加载动画
    namespace spinner {
        void start();
        void stop();
    }
    
    // 日志输出
    void log(const char * fmt, ...);
    void error(const char * fmt, ...);
    void flush();
}
```

### 22.3.2 实现细节

**源码位置**：`common/console.cpp` (第1-300行)

```cpp
// Windows和Unix的不同实现
#ifdef _WIN32
    #include <windows.h>
    #include <conio.h>
#else
    #include <termios.h>
    #include <unistd.h>
    #include <sys/select.h>
#endif

// 读取单行输入（支持退格、光标移动）
bool console::readline(std::string & line, bool multiline_input) {
#ifdef _WIN32
    // Windows: 使用控制台API
    HANDLE hConsole = GetStdHandle(STD_INPUT_HANDLE);
    // ... 处理输入事件
#else
    // Unix: 使用termios设置原始模式
    struct termios old_tio, new_tio;
    tcgetattr(STDIN_FILENO, &old_tio);
    new_tio = old_tio;
    new_tio.c_lflag &= ~(ICANON | ECHO);
    tcsetattr(STDIN_FILENO, TCSANOW, &new_tio);
    
    // 逐字符读取，处理特殊键（方向键、退格等）
    char c;
    while (read(STDIN_FILENO, &c, 1) == 1) {
        if (c == '\n') break;
        if (c == 127) {  // 退格
            if (!line.empty()) {
                line.pop_back();
                printf("\b \b");  // 删除显示
            }
        } else {
            line.push_back(c);
            putchar(c);
        }
    }
    
    tcsetattr(STDIN_FILENO, TCSANOW, &old_tio);
#endif
    return true;
}
```

### 22.3.3 加载动画

```cpp
// Spinner实现（在后台线程运行）
void console::spinner::start() {
    std::thread([]() {
        const char * frames[] = {"⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"};
        int i = 0;
        while (spinner_running) {
            printf("\r%s ", frames[i % 10]);
            fflush(stdout);
            std::this_thread::sleep_for(std::chrono::milliseconds(80));
            i++;
        }
        printf("\r  \r");  // 清除
    }).detach();
}
```

---

## 22.4 日志系统（log.cpp）

### 22.4.1 设计特点

llama.cpp的日志系统支持：
- 多级别日志（DEBUG/INFO/WARN/ERROR）
- 彩色输出
- 异步写入（避免阻塞推理线程）
- 文件日志

**源码位置**：`common/log.h` (第1-119行)

```cpp
// 日志级别
#define LOG_LEVEL_DEBUG  4
#define LOG_LEVEL_INFO   3
#define LOG_LEVEL_WARN   2
#define LOG_LEVEL_ERROR  1
#define LOG_LEVEL_OUTPUT 0

// 彩色输出宏
#define LOG_COL_DEFAULT "\033[0m"
#define LOG_COL_RED     "\033[31m"
#define LOG_COL_GREEN   "\033[32m"
#define LOG_COL_YELLOW  "\033[33m"
#define LOG_COL_BLUE    "\033[34m"

// 便捷宏（自动检查日志级别）
#define LOG_DBG(...) \
    do { if (LOG_DEFAULT_DEBUG <= common_log_verbosity_thold) \
        LOG_TMPL(LOG_LEVEL_DEBUG, LOG_DEFAULT_DEBUG, __VA_ARGS__); } while(0)

#define LOG_INF(...) \
    do { if (LOG_DEFAULT_INFO <= common_log_verbosity_thold) \
        LOG_TMPL(LOG_LEVEL_INFO, LOG_DEFAULT_INFO, __VA_ARGS__); } while(0)

#define LOG_WRN(...) \
    do { if (LOG_DEFAULT_WARN <= common_log_verbosity_thold) \
        LOG_TMPL(LOG_LEVEL_WARN, LOG_DEFAULT_WARN, __VA_ARGS__); } while(0)

#define LOG_ERR(...) \
    do { if (LOG_DEFAULT_ERROR <= common_log_verbosity_thold) \
        LOG_TMPL(LOG_LEVEL_ERROR, LOG_DEFAULT_ERROR, __VA_ARGS__); } while(0)
```

### 22.4.2 异步日志实现

**源码位置**：`common/log.cpp` (第1-200行)

```cpp
struct common_log {
    std::thread worker;              // 后台工作线程
    std::queue<log_entry> queue;    // 日志队列
    std::mutex mutex;
    std::condition_variable cv;
    bool paused = false;
    
    FILE * file = nullptr;           // 日志文件
    bool use_colors = false;
    bool use_prefix = false;
    bool use_timestamps = false;
};

// 添加日志（非阻塞）
void common_log_add(common_log * log, ggml_log_level level, const char * fmt, ...) {
    va_list args;
    va_start(args, fmt);
    
    log_entry entry;
    entry.level = level;
    entry.timestamp = get_time();
    entry.message = string_format(fmt, args);
    
    {
        std::lock_guard<std::mutex> lock(log->mutex);
        log->queue.push(std::move(entry));
    }
    log->cv.notify_one();
    
    va_end(args);
}

// 工作线程：批量写入日志
void log_worker(common_log * log) {
    while (true) {
        std::unique_lock<std::mutex> lock(log->mutex);
        log->cv.wait(lock, [&]() { return !log->queue.empty() || log->paused; });
        
        while (!log->queue.empty()) {
            auto entry = std::move(log->queue.front());
            log->queue.pop();
            lock.unlock();
            
            // 格式化并输出
            write_log_entry(log, entry);
            
            lock.lock();
        }
    }
}
```

---

## 22.5 聊天模板处理（chat.cpp）

### 22.5.1 功能概述

chat.cpp负责将结构化对话格式化为模型输入，支持：
- Jinja2模板
- 多种预设格式（ChatML、Llama-2、Vicuna等）
- 工具调用格式
- 推理内容分离（如DeepSeek的<think>标签）

**源码位置**：`common/chat.h` (第1-100行)

```cpp
// 聊天消息
struct common_chat_msg {
    std::string role;       // "system", "user", "assistant", "tool"
    std::string content;    // 消息内容
    std::string tool_calls; // 工具调用（JSON）
    std::string tool_plan;  // 工具执行计划
};

// 格式化参数
struct common_chat_params {
    std::string prompt;              // 格式化后的提示
    bool add_bos;                    // 是否添加BOS
    bool add_eos;                    // 是否添加EOS
    std::string generation_prompt;   // 生成提示（如"Assistant: "）
    std::string parser;              // 使用的解析器
    std::string thinking_start_tag;  // 推理开始标签
    std::string thinking_end_tag;    // 推理结束标签
};

// 格式化对话
common_chat_params common_chat_format(
    const common_chat_template & tmpl,
    const std::vector<common_chat_msg> & messages,
    const std::string & tools_json,
    const std::string & tool_choice
);
```

### 22.5.2 模板应用示例

```cpp
// ChatML模板示例
// 输入消息：
// [{"role": "user", "content": "Hello!"}]

// 模板：
// {% for message in messages %}
//   {{ '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n' }}
// {% endfor %}

// 输出：
// <|im_start|>user
// Hello!<|im_end|>
// <|im_start|>assistant

// Llama-2模板示例
// <s>[INST] <<SYS>>
// {{ system_prompt }}
// <</SYS>>
// {{ user_message }} [/INST]
```

---

## 设计中的取舍

### 为什么common库使用C++而不是C？

| 特性 | C | C++ | common选择 |
|-----|---|-----|-----------|
| 标准库 | 简单 | 丰富（STL） | C++ |
| 跨平台 | 需手动处理 | 抽象更好 | C++ |
| 与llama.h兼容 | 完美 | 需extern "C" | C++（core保持C） |
| 性能 | 最优 | 接近最优 | C++ |

**架构决策**：
- **core（llama.h）**：C接口，最大化兼容性
- **common**：C++实现，开发效率优先
- **examples**：可使用C或C++

### 为什么参数解析使用"处理器函数"模式？

对比其他模式：
```cpp
// 模式1：直接赋值（简单但不灵活）
params.n_gpu_layers = std::stoi(argv[i]);

// 模式2：反射/元数据（复杂，C++支持差）
set_param("n_gpu_layers", argv[i]);

// 模式3：处理器函数（llama.cpp选择，平衡灵活性和复杂度）
handler_int(params, std::stoi(argv[i]));
```

**处理器函数的优势**：
1. **类型安全**：编译期检查
2. **灵活性**：可在处理器中添加验证逻辑
3. **可组合**：一个参数可影响多个字段

---

## 动手练习

### 练习1：阅读参数解析流程

阅读 `common/arg.cpp` 第2000-2500行，回答：
1. `common_params_parse` 如何处理未知参数？
2. 环境变量优先级如何（相对于命令行）？
3. `examples` 和 `excludes` 如何协同工作？

### 练习2：实现自定义参数

基于以下框架，实现一个 `--custom-seed` 参数：

```cpp
common_arg arg_custom_seed() {
    return common_arg(
        {"--custom-seed"},
        "SEED",
        "设置自定义随机种子（格式: 类型:值）",
        [](common_params & params, const std::string & value) {
            // TODO: 解析 "type:value" 格式
            // 支持 type: random, fixed, timestamp
        }
    );
}
```

### 练习3：日志性能测试

编写程序比较同步日志和异步日志的性能：
```cpp
// 测试场景：10000条日志写入
// 1. 使用 printf（同步）
// 2. 使用 LOG_INF（异步）
// 比较总耗时
```

---

## 本课小结

| 组件 | 功能 | 核心API |
|-----|------|---------|
| arg.cpp | 命令行解析 | `common_params_parse()` |
| download.cpp | 网络下载 | `common_download_model()` |
| console.cpp | 控制台交互 | `console::readline()` |
| log.cpp | 日志系统 | `LOG_INF()`, `LOG_ERR()` |
| chat.cpp | 聊天模板 | `common_chat_format()` |
| sampling.cpp | 采样封装 | `common_sampler_sample()` |

---

*本章对应源码版本：master (2026-04-07)*
