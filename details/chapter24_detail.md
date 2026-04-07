# 第24章 命令行工具详解 —— llama.cpp的"门面担当"

## 学习目标
1. 深入理解llama-cli的交互式和批处理模式实现
2. 掌握llama-server的HTTP API设计和OpenAI兼容接口
3. 了解llama-quantize的量化策略和imatrix使用
4. 学会使用llama-bench进行性能测试和结果分析
5. 能够根据需求选择和配置合适的工具

---

## 生活类比：餐厅的不同服务方式

想象llama.cpp工具集是一家餐厅的不同服务方式：

- **llama-cli** = 堂食服务（面对面交互，即时响应，适合体验）
- **llama-server** = 外卖平台（API接口，批量服务，适合集成）
- **llama-quantize** = 食材加工厂（模型压缩，体积优化，适合分发）
- **llama-bench** = 厨房效率测试（性能评估，瓶颈分析，适合优化）

就像一家成功餐厅需要提供多种服务方式满足不同顾客需求，llama.cpp也通过多种工具服务不同场景。

---

## 源码地图

```
tools/
├── cli/cli.cpp            # 交互式CLI工具（~1500行）
├── server/                # HTTP服务器
│   ├── server.cpp         # 主服务入口
│   ├── server-context.h   # 服务器上下文
│   └── server-task.h      # 任务管理
├── quantize/quantize.cpp  # 量化工具
├── llama-bench/           # 基准测试
│   └── llama-bench.cpp
├── main/                  # 主工具（llama-cli的别名）
└── ...
```

---

## 24.1 llama-cli 交互式对话工具

### 24.1.1 架构概述

**源码位置**：`tools/cli/cli.cpp` (第1-200行)

```cpp
// CLI上下文结构
struct cli_context {
    server_context ctx_server;           // 服务器上下文（复用server代码）
    json messages = json::array();       // 对话历史
    std::vector<raw_buffer> input_files; // 输入文件（多模态）
    task_params defaults;                // 默认参数
    bool verbose_prompt;                 // 是否显示提示
    int reasoning_budget = -1;           // 推理预算
    std::string reasoning_budget_message;
    
    std::atomic<bool> loading_show;      // 加载动画控制

    // 生成完成回复
    std::string generate_completion(result_timings & out_timings);
};
```

### 24.1.2 交互式模式实现

**源码位置**：`tools/cli/cli.cpp` (第200-500行)

```cpp
// 主交互循环
void interactive_loop(cli_context & ctx) {
    // 初始化信号处理（Ctrl+C）
    signal(SIGINT, signal_handler);
    
    // 显示欢迎信息
    console::log("%s\n", LLAMA_ASCII_LOGO);
    console::log("欢迎使用 llama.cpp! 输入 '/help' 查看命令, '/quit' 退出\n\n");
    
    while (!should_stop()) {
        // 显示提示符
        console::set_display(DISPLAY_TYPE_PROMPT);
        console::log("> ");
        console::flush();
        
        // 读取用户输入
        std::string user_input;
        if (!console::readline(user_input, true)) {
            break;
        }
        
        // 处理特殊命令
        if (user_input == "/quit" || user_input == "/q") {
            break;
        }
        if (user_input == "/help" || user_input == "/h") {
            print_help();
            continue;
        }
        if (user_input == "/clear") {
            ctx.messages.clear();
            console::log("对话历史已清空\n");
            continue;
        }
        
        // 添加到对话历史
        ctx.messages.push_back({
            {"role", "user"},
            {"content", user_input}
        });
        
        // 生成回复
        result_timings timings;
        std::string response = ctx.generate_completion(timings);
        
        // 添加助手回复到历史
        ctx.messages.push_back({
            {"role", "assistant"},
            {"content", response}
        });
        
        console::log("\n");
    }
}
```

### 24.1.3 多模态输入处理

**源码位置**：`tools/cli/cli.cpp` (第500-700行)

```cpp
// 加载输入文件（支持文本和图像）
std::string cli_context::load_input_file(const std::string & fname, bool is_media) {
    std::ifstream file(fname, std::ios::binary);
    if (!file) {
        return "";
    }
    
    if (is_media) {
        // 图像/音频文件：读取为二进制，存储在input_files
        raw_buffer buf;
        buf.assign(
            (std::istreambuf_iterator<char>(file)),
            std::istreambuf_iterator<char>()
        );
        input_files.push_back(std::move(buf));
        return mtmd_default_marker();  // 返回 "<__media__>"
    } else {
        // 文本文件：直接读取内容
        std::string content(
            (std::istreambuf_iterator<char>(file)),
            std::istreambuf_iterator<char>()
        );
        return content;
    }
}

// 生成完成回复
std::string cli_context::generate_completion(result_timings & out_timings) {
    server_response_reader rd = ctx_server.get_response_reader();
    
    // 格式化对话
    auto chat_params = format_chat();
    
    // 创建任务
    server_task task = server_task(SERVER_TASK_TYPE_COMPLETION);
    task.id = rd.get_new_id();
    task.params = defaults;
    task.cli_prompt = chat_params.prompt;
    task.cli_files = input_files;
    task.cli = true;
    
    // 发送任务
    rd.post_task({std::move(task)});
    
    // 显示加载动画
    console::spinner::start();
    
    // 流式接收结果
    std::string full_response;
    bool is_thinking = false;
    
    while (true) {
        server_task_result_ptr result = rd.next(should_stop);
        if (!result) break;
        
        console::spinner::stop();
        
        // 处理部分结果（流式输出）
        auto res_partial = dynamic_cast<server_task_result_cmpl_partial *>(result.get());
        if (res_partial) {
            for (const auto & diff : res_partial->oaicompat_msg_diffs) {
                // 处理推理内容（如DeepSeek的<think>标签）
                if (!diff.reasoning_content_delta.empty()) {
                    if (!is_thinking) {
                        console::set_display(DISPLAY_TYPE_REASONING);
                        console::log("[开始思考]\n");
                        is_thinking = true;
                    }
                    console::log("%s", diff.reasoning_content_delta.c_str());
                }
                
                // 处理正式回复
                if (!diff.content_delta.empty()) {
                    if (is_thinking) {
                        console::log("\n[结束思考]\n\n");
                        console::set_display(DISPLAY_TYPE_RESET);
                        is_thinking = false;
                    }
                    full_response += diff.content_delta;
                    console::log("%s", diff.content_delta.c_str());
                }
                console::flush();
            }
        }
        
        // 处理最终结果
        auto res_final = dynamic_cast<server_task_result_cmpl_final *>(result.get());
        if (res_final) {
            out_timings = std::move(res_final->timings);
            break;
        }
    }
    
    return full_response;
}
```

### 24.1.4 推理预算控制

**源码位置**：`tools/cli/cli.cpp` (第700-900行)

```cpp
// 设置推理预算（控制思考长度）
void setup_reasoning_budget(server_task & task, cli_context & ctx) {
    if (ctx.reasoning_budget <= 0) return;
    
    const llama_vocab * vocab = llama_model_get_vocab(
        llama_get_model(ctx.ctx_server.get_llama_context())
    );
    
    // 配置推理预算采样器
    task.params.sampling.reasoning_budget_tokens = ctx.reasoning_budget;
    task.params.sampling.generation_prompt = chat_params.generation_prompt;
    
    // 标记思考开始和结束
    if (!chat_params.thinking_start_tag.empty()) {
        task.params.sampling.reasoning_budget_start =
            common_tokenize(vocab, chat_params.thinking_start_tag, false, true);
    }
    task.params.sampling.reasoning_budget_end =
        common_tokenize(vocab, chat_params.thinking_end_tag, false, true);
    task.params.sampling.reasoning_budget_forced =
        common_tokenize(vocab, ctx.reasoning_budget_message + chat_params.thinking_end_tag, false, true);
}
```

---

## 24.2 llama-server HTTP服务

### 24.2.1 架构设计

**源码位置**：`tools/server/server.cpp` (第1-300行)

```cpp
// 服务器路由结构
struct server_routes {
    // 健康检查
    std::function<void(const httplib::Request &, httplib::Response &)> get_health;
    
    // 模型信息
    std::function<void(const httplib::Request &, httplib::Response &)> get_models;
    std::function<void(const httplib::Request &, httplib::Response &)> get_props;
    
    // 生成接口
    std::function<void(const httplib::Request &, httplib::Response &)> post_completions;
    std::function<void(const httplib::Request &, httplib::Response &)> post_chat_completions;
    
    // 嵌入接口
    std::function<void(const httplib::Request &, httplib::Response &)> post_embeddings;
    
    // 分词接口
    std::function<void(const httplib::Request &, httplib::Response &)> post_tokenize;
    std::function<void(const httplib::Request &, httplib::Response &)> post_detokenize;
    
    // LoRA适配器管理
    std::function<void(const httplib::Request &, httplib::Response &)> get_lora_adapters;
    std::function<void(const httplib::Request &, httplib::Response &)> post_lora_adapters;
    
    // Slot管理（多会话）
    std::function<void(const httplib::Request &, httplib::Response &)> get_slots;
    std::function<void(const httplib::Request &, httplib::Response &)> post_slots;
};
```

### 24.2.2 OpenAI兼容API

**源码位置**：`tools/server/server.cpp` (第300-600行)

```cpp
// OpenAI兼容的聊天完成接口
void handle_chat_completions(const httplib::Request & req, httplib::Response & res) {
    json body = json::parse(req.body);
    
    // 解析请求
    std::string model = body.value("model", "default");
    json messages = body["messages"];
    float temperature = body.value("temperature", 0.8f);
    int max_tokens = body.value("max_tokens", -1);
    bool stream = body.value("stream", false);
    
    // 转换为llama.cpp内部格式
    server_task task = server_task(SERVER_TASK_TYPE_COMPLETION);
    task.params.sampling.temp = temperature;
    task.params.n_predict = max_tokens;
    task.params.stream = stream;
    
    // 应用聊天模板
    auto chat_params = common_chat_format(
        tmpl,
        messages,
        body.value("tools", ""),
        body.value("tool_choice", "")
    );
    task.cli_prompt = chat_params.prompt;
    
    // 提交任务
    auto result = server_queue.submit(task);
    
    if (stream) {
        // 流式响应
        res.set_chunked_content_provider("text/event-stream",
            [&](size_t offset, httplib::DataSink & sink) {
                // 发送SSE事件
                for (const auto & chunk : result->chunks) {
                    std::string event = format_sse_chunk(chunk);
                    sink.write(event.c_str(), event.size());
                }
                return true;
            }
        );
    } else {
        // 非流式响应
        json response = {
            {"id", generate_id()},
            {"object", "chat.completion"},
            {"created", time(nullptr)},
            {"model", model},
            {"choices", json::array({
                {
                    {"index", 0},
                    {"message", {
                        {"role", "assistant"},
                        {"content", result->text}
                    }},
                    {"finish_reason", "stop"}
                }
            })},
            {"usage", {
                {"prompt_tokens", result->n_prompt_tokens},
                {"completion_tokens", result->n_generated_tokens},
                {"total_tokens", result->n_prompt_tokens + result->n_generated_tokens}
            }}
        };
        res.set_content(response.dump(), "application/json");
    }
}
```

### 24.2.3 路由注册

**源码位置**：`tools/server/server.cpp` (第600-900行)

```cpp
// 注册API路由
void register_routes(server_http_context & ctx_http, server_routes & routes) {
    // 健康检查（公开端点）
    ctx_http.get("/health", ex_wrapper(routes.get_health));
    ctx_http.get("/v1/health", ex_wrapper(routes.get_health));
    
    // 模型信息（公开端点）
    ctx_http.get("/models", ex_wrapper(routes.get_models));
    ctx_http.get("/v1/models", ex_wrapper(routes.get_models));
    ctx_http.get("/api/tags", ex_wrapper(routes.get_models));  // Ollama兼容
    
    // 生成接口
    ctx_http.post("/completion", ex_wrapper(routes.post_completions));  // 传统接口
    ctx_http.post("/completions", ex_wrapper(routes.post_completions));
    ctx_http.post("/v1/completions", ex_wrapper(routes.post_completions_oai));
    ctx_http.post("/chat/completions", ex_wrapper(routes.post_chat_completions));
    ctx_http.post("/v1/chat/completions", ex_wrapper(routes.post_chat_completions));
    ctx_http.post("/api/chat", ex_wrapper(routes.post_chat_completions));  // Ollama兼容
    
    // 嵌入接口
    ctx_http.post("/embeddings", ex_wrapper(routes.post_embeddings));
    ctx_http.post("/v1/embeddings", ex_wrapper(routes.post_embeddings_oai));
    
    // 分词接口
    ctx_http.post("/tokenize", ex_wrapper(routes.post_tokenize));
    ctx_http.post("/detokenize", ex_wrapper(routes.post_detokenize));
    
    // LoRA适配器管理
    ctx_http.get("/lora-adapters", ex_wrapper(routes.get_lora_adapters));
    ctx_http.post("/lora-adapters", ex_wrapper(routes.post_lora_adapters));
    
    // Slot管理
    ctx_http.get("/slots", ex_wrapper(routes.get_slots));
    ctx_http.post("/slots", ex_wrapper(routes.post_slots));
}
```

---

## 24.3 llama-quantize 量化工具

### 24.3.1 量化选项

**源码位置**：`tools/quantize/quantize.cpp` (第1-150行)

```cpp
// 量化选项定义
struct quant_option {
    std::string name;       // 选项名称
    llama_ftype ftype;      // 文件类型
    std::string desc;       // 描述
};

static const std::vector<quant_option> QUANT_OPTIONS = {
    { "Q4_0",     LLAMA_FTYPE_MOSTLY_Q4_0,     " 4.34G, +0.4685 ppl @ Llama-3-8B" },
    { "Q4_1",     LLAMA_FTYPE_MOSTLY_Q4_1,     " 4.78G, +0.4511 ppl @ Llama-3-8B" },
    { "Q5_0",     LLAMA_FTYPE_MOSTLY_Q5_0,     " 5.21G, +0.1316 ppl @ Llama-3-8B" },
    { "Q5_1",     LLAMA_FTYPE_MOSTLY_Q5_1,     " 5.65G, +0.1062 ppl @ Llama-3-8B" },
    { "Q8_0",     LLAMA_FTYPE_MOSTLY_Q8_0,     " 7.96G, +0.0026 ppl @ Llama-3-8B" },
    { "Q4_K_S",   LLAMA_FTYPE_MOSTLY_Q4_K_S,   " 4.37G, +0.2689 ppl @ Llama-3-8B" },
    { "Q4_K_M",   LLAMA_FTYPE_MOSTLY_Q4_K_M,   " 4.58G, +0.1754 ppl @ Llama-3-8B" },
    { "Q5_K_S",   LLAMA_FTYPE_MOSTLY_Q5_K_S,   " 5.21G, +0.1049 ppl @ Llama-3-8B" },
    { "Q5_K_M",   LLAMA_FTYPE_MOSTLY_Q5_K_M,   " 5.33G, +0.0569 ppl @ Llama-3-8B" },
    { "Q6_K",     LLAMA_FTYPE_MOSTLY_Q6_K,     " 6.14G, +0.0217 ppl @ Llama-3-8B" },
    { "IQ2_XXS",  LLAMA_FTYPE_MOSTLY_IQ2_XXS,  " 2.06 bpw quantization" },
    { "IQ2_XS",   LLAMA_FTYPE_MOSTLY_IQ2_XS,   " 2.31 bpw quantization" },
    { "IQ3_XXS",  LLAMA_FTYPE_MOSTLY_IQ3_XXS,  " 3.06 bpw quantization" },
    { "IQ3_S",    LLAMA_FTYPE_MOSTLY_IQ3_S,    " 3.44 bpw quantization" },
    { "IQ4_NL",   LLAMA_FTYPE_MOSTLY_IQ4_NL,   " 4.50 bpw non-linear quantization" },
    { "F16",      LLAMA_FTYPE_MOSTLY_F16,      "14.00G, +0.0020 ppl @ Mistral-7B" },
    { "BF16",     LLAMA_FTYPE_MOSTLY_BF16,     "14.00G, -0.0050 ppl @ Mistral-7B" },
    { "F32",      LLAMA_FTYPE_ALL_F32,         "26.00G              @ 7B" },
    { "COPY",     LLAMA_FTYPE_ALL_F32,         "only copy tensors, no quantizing" },
};
```

### 24.3.2 高级量化选项

**源码位置**：`tools/quantize/quantize.cpp` (第150-400行)

```cpp
// 张量类型覆盖选项
struct tensor_type_option {
    std::string name;           // 张量名称模式
    ggml_type type = GGML_TYPE_COUNT;
};

// 量化参数
struct quantize_params {
    std::string input_model;           // 输入模型路径
    std::string output_model;          // 输出模型路径
    llama_ftype ftype;                 // 目标量化类型
    
    // 高级选项
    bool allow_requantize = false;     // 允许重新量化
    bool leave_output_tensor = false;  // 保持输出层不量化
    bool pure = false;                 // 禁用K-quant混合
    
    // imatrix选项
    std::string imatrix_file;          // 重要性矩阵文件
    
    // 张量类型覆盖
    std::vector<tensor_type_option> tensor_types;  // --tensor-type
    ggml_type output_tensor_type;       // --output-tensor-type
    ggml_type token_embedding_type;     // --token-embedding-type
    
    // 层剪枝
    std::vector<int> prune_layers;      // --prune-layers
    
    // 其他
    bool keep_split = false;           // 保持分片
    bool dry_run = false;              // 仅预览，不执行
};

// 使用示例：
// ./llama-quantize \
//     --imatrix imatrix.dat \
//     --tensor-type attn_q=Q8_0 \
//     --tensor-type attn_k=Q8_0 \
//     --tensor-type attn_v=Q8_0 \
//     model-f32.gguf \
//     model-Q4_K_M.gguf \
//     Q4_K_M
```

### 24.3.3 imatrix（重要性矩阵）

```cpp
// imatrix用于指导量化过程，根据数据重要性分配精度
// 生成imatrix：
// ./llama-imatrix -m model.gguf -f training-data.txt -o imatrix.dat

// 使用imatrix量化：
// ./llama-quantize --imatrix imatrix.dat model-f32.gguf model-Q4_K_M.gguf Q4_K_M

// imatrix效果：
// - 无imatrix: Q4_K_M @ Llama-3-8B = +0.1754 ppl
// - 有imatrix: Q4_K_M @ Llama-3-8B = +0.0892 ppl (显著提升)
```

---

## 24.4 llama-bench 基准测试

### 24.4.1 测试参数

**源码位置**：`tools/llama-bench/llama-bench.cpp` (第1-200行)

```cpp
// 基准测试参数
struct bench_params {
    // 模型参数
    std::vector<std::string> models;           // 测试的模型
    std::vector<int> n_gpu_layers;             // GPU层数配置
    std::vector<ggml_type> types;              // 量化类型
    
    // 批处理参数
    std::vector<int> n_prompt;                 // 提示长度列表
    std::vector<int> n_gen;                    // 生成长度列表
    std::vector<int> n_batch;                  // 批大小列表
    
    // 其他
    int n_threads = -1;                        // 线程数（-1=自动）
    int repetitions = 3;                       // 重复次数
    bool warmup = true;                        // 是否预热
    bool output_csv = false;                   // CSV输出
    bool output_json = false;                  // JSON输出
    bool output_sqlite = false;                // SQLite输出
};
```

### 24.4.2 性能测试流程

**源码位置**：`tools/llama-bench/llama-bench.cpp` (第200-500行)

```cpp
// 运行单次测试
bench_result run_bench(
    const std::string & model_path,
    const bench_params & params
) {
    bench_result result;
    
    // 1. 加载模型
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = params.n_gpu_layers[0];
    
    llama_model * model = llama_load_model_from_file(
        model_path.c_str(),
        model_params
    );
    
    // 2. 创建上下文
    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_batch = params.n_batch[0];
    ctx_params.n_ubatch = params.n_batch[0];
    
    llama_context * ctx = llama_new_context_with_model(model, ctx_params);
    
    // 3. 预热（可选）
    if (params.warmup) {
        run_warmup(ctx);
    }
    
    // 4. 测试提示处理速度（tg）
    for (int r = 0; r < params.repetitions; r++) {
        auto start = std::chrono::high_resolution_clock::now();
        
        // 处理n_prompt个token
        process_prompt(ctx, params.n_prompt[0]);
        
        auto end = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(end - start).count();
        
        result.tg_ms.push_back(ms / params.n_prompt[0]);  // ms per token
    }
    
    // 5. 测试生成速度（pp）
    for (int r = 0; r < params.repetitions; r++) {
        auto start = std::chrono::high_resolution_clock::now();
        
        // 生成n_gen个token
        generate_tokens(ctx, params.n_gen[0]);
        
        auto end = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(end - start).count();
        
        result.pp_ms.push_back(ms / params.n_gen[0]);  // ms per token
    }
    
    // 6. 计算统计
    result.tg_avg = avg(result.tg_ms);
    result.tg_std = stdev(result.tg_ms);
    result.pp_avg = avg(result.pp_ms);
    result.pp_std = stdev(result.pp_ms);
    
    // 清理
    llama_free(ctx);
    llama_free_model(model);
    
    return result;
}
```

### 24.4.3 结果输出

```cpp
// 输出格式示例：
// | model                          | size    | params | backend | ngl | test       |    t/s |
// | ------------------------------ | ------- | ------ | ------- | --- | ---------- | ------ |
// | llama 7B Q4_0                  | 3.56 GiB | 6.74 B | CPU     |   0 | pp 512     |  45.23 |
// | llama 7B Q4_0                  | 3.56 GiB | 6.74 B | CPU     |   0 | tg 128     |   8.12 |
// | llama 7B Q4_0                  | 3.56 GiB | 6.74 B | CUDA    |  33 | pp 512     | 892.45 |
// | llama 7B Q4_0                  | 3.56 GiB | 6.74 B | CUDA    |  33 | tg 128     |  45.67 |

// CSV输出（便于导入Excel分析）
// model,size,params,backend,ngl,test,t/s
// llama 7B Q4_0,3.56 GiB,6.74 B,CPU,0,pp 512,45.23
// ...
```

---

## 设计中的取舍

### 为什么llama-cli复用server代码？

| 方案 | 优点 | 缺点 | 选择 |
|-----|------|------|------|
| 独立实现 | 无依赖 | 代码重复，维护困难 | 否 |
| 复用server | 代码复用，功能一致 | 需要链接server库 | **是** |
| 提取公共库 | 架构清晰 | 重构成本高 | 未来可能 |

**复用server代码的好处**：
1. **功能一致**：CLI和Server使用相同的推理逻辑
2. **维护简单**：修复bug只需改一处
3. **功能丰富**：自动获得server的所有功能（如多模态、工具调用）

### 为什么量化工具是独立的？

```cpp
// 量化是"离线"操作：
// - 不需要加载完整模型到GPU
// - 可以处理比内存大的模型（分块处理）
// - 不需要推理上下文

// 如果集成到主程序：
// - 增加二进制体积
// - 增加复杂性
// - 使用频率低

// 独立工具的优势：
// - 专注单一任务
// - 可以单独优化
// - 便于脚本调用
```

---

## 动手练习

### 练习1：阅读CLI交互循环

阅读 `tools/cli/cli.cpp` 第200-500行，回答：
1. 如何处理Ctrl+C信号？
2. 流式输出的实现机制是什么？
3. 推理预算如何控制思考长度？

### 练习2：测试HTTP API

使用curl测试llama-server的API：
```bash
# 启动服务器
./llama-server -m model.gguf

# 测试聊天完成
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Hello!"}],
    "temperature": 0.7,
    "stream": false
  }'

# 测试嵌入
curl -X POST http://localhost:8080/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Hello world",
    "model": "default"
  }'
```

### 练习3：量化实验

使用llama-quantize进行量化实验：
```bash
# 1. 生成imatrix
./llama-imatrix -m model-f32.gguf -f wiki.txt -o imatrix.dat

# 2. 对比有/无imatrix的量化质量
./llama-quantize model-f32.gguf model-Q4_K_M-no-imatrix.gguf Q4_K_M
./llama-quantize --imatrix imatrix.dat model-f32.gguf model-Q4_K_M-with-imatrix.gguf Q4_K_M

# 3. 使用perplexity评估质量
./llama-perplexity -m model-Q4_K_M-no-imatrix.gguf -f test.txt
./llama-perplexity -m model-Q4_K_M-with-imatrix.gguf -f test.txt
```

---

## 本课小结

| 工具 | 用途 | 核心参数 |
|-----|------|---------|
| llama-cli | 交互式对话 | `-m`, `-p`, `-cnv`, `-ngl` |
| llama-server | HTTP服务 | `-m`, `--port`, `--host` |
| llama-quantize | 模型量化 | `--imatrix`, `--tensor-type` |
| llama-bench | 性能测试 | `-m`, `-p`, `-ngl`, `-r` |

---

*本章对应源码版本：master (2026-04-07)*
