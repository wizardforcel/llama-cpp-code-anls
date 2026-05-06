# 第10章 推理上下文（llama_context） —— 推理过程的"总控制中心"

## 学习目标
1. 理解llama_context的结构组成
2. 掌握llama_batch的数据组织和批处理
3. 理解解码策略的实现原理
4. 掌握状态保存与恢复机制

---

## 生活类比：交响乐团的指挥中心

想象llama_context是一位**交响乐团的总指挥**：

- **llama_context** = 指挥台+总指挥
  - 掌控整个演出的节奏和流程
  - 协调各个乐手（计算资源）的配合
  - 记忆演奏进度（推理状态）
- **llama_batch** = 当前要演奏的乐谱片段
  - 包含多个声部（序列）的音符（tokens）
  - 每个声部有独立的演奏位置（位置编码）
- **KV缓存** = 乐手们的"记忆笔记"
  - 记录已经演奏过的部分，避免重复演奏
  - 每个声部有自己的笔记
- **采样器** = 即兴创作系统
  - 根据乐谱提示（logits）决定下一个音符
  - 可以是严格按谱（贪心）或自由发挥（随机）
- **状态保存** = 演出中场休息
  - 记录当前进度，下半场可以继续

就像指挥需要协调各方，llama_context需要协调模型、缓存、采样器等组件完成推理。

---

## 源码地图

```
src/llama-context.h
├── llama_context_params     # 上下文参数
│   ├── n_ctx/n_batch/n_ubatch     # 批次大小配置
│   ├── n_threads/n_threads_batch  # 线程配置
│   ├── rope_scaling_type          # RoPE缩放类型
│   ├── pooling_type               # 池化类型
│   ├── attention_type             # 注意力类型
│   ├── flash_attn_type            # Flash Attention类型
│   ├── type_k/type_v              # KV缓存数据类型
│   ├── yarn_ext_factor            # YaRN扩展因子
│   ├── samplers/n_samplers        # 采样器配置
│   └── offload_kqv/op_offload     # 卸载配置
├── llama_context            # 上下文结构体
│   ├── llama_cparams        # 计算参数
│   ├── llama_memory         # 内存分配器
│   ├── llama_batch          # 批次数据
│   ├── llama_kv_cache       # KV缓存
│   ├── ggml_cgraph          # 计算图
│   └── t_perf               # 性能统计
├── llama_batch              # 批次结构体
│   ├── token/embd           # 输入数据
│   ├── pos/n_seq_id/seq_id  # 位置/序列信息
│   └── logits               # 输出标记
└── llama_input_buffers      # 输入缓冲区

src/llama-context.cpp
├── llama_new_context()      # 创建上下文（第1-300行）
├── llama_decode()           # 主解码函数（第300-800行）
├── 资源管理（第800-1000行）
└── 状态管理（第1000-1500行）

src/llama-batch.h/cpp
├── llama_batch_init()       # 初始化批次
├── llama_batch_clear()      # 清空批次
└── llama_batch_add()        # 添加token到批次

src/llama-sampler.cpp
├── llama_sampler_init_greedy()      # 贪心采样
├── llama_sampler_init_dist()        # 分布采样
├── llama_sampler_init_top_k()       # Top-K采样
├── llama_sampler_init_top_p()       # Top-P采样
├── llama_sampler_init_min_p()       # Min-P采样
├── llama_sampler_init_typical()     # Typical采样
├── llama_sampler_init_temp()        # 温度缩放
├── llama_sampler_init_penalties()   # 重复惩罚
├── llama_sampler_init_mirostat()    # Mirostat采样
├── llama_sampler_init_xtc()         # XTC采样
├── llama_sampler_chain_*            # 采样器链
└── llama_sampler_apply()            # 应用采样器

include/llama.h
├── llama_token_data         # Token数据（logit/p）
├── llama_token_data_array   # Token数组
├── llama_logit_bias         # Logit偏置
├── llama_sampler_chain_params  # 采样器链参数
├── llama_sampler_seq_config    # 序列采样器配置
└── llama_chat_message       # 聊天消息

examples/save-load-state/
└── save-load-state.cpp      # 状态保存示例
```

---

## 10.1 上下文结构

### 10.1.1 llama_context_params 配置参数

**源码位置**：`include/llama.h` (第330-382行)

```c
struct llama_context_params {
    // 上下文窗口
    uint32_t n_ctx;             // 最大上下文长度（默认512）
    uint32_t n_batch;           // 逻辑最大批处理大小（默认512）
    uint32_t n_ubatch;          // 物理批处理大小（用于分块）
    uint32_t n_seq_max;         // 最大序列数（用于循环模型）

    // 线程配置
    uint32_t n_threads;         // CPU线程数（生成）
    uint32_t n_threads_batch;   // CPU线程数（提示处理）

    // RoPE配置
    enum llama_rope_scaling_type rope_scaling_type; // RoPE扩展类型
    float    rope_freq_base;    // RoPE基数覆盖
    float    rope_freq_scale;   // RoPE缩放覆盖
    float    yarn_ext_factor;   // YaRN扩展因子
    float    yarn_attn_factor;  // YaRN注意力因子
    float    yarn_beta_fast;    // YaRN快速beta
    float    yarn_beta_slow;    // YaRN慢速beta
    uint32_t yarn_orig_ctx;     // YaRN原始上下文大小

    // 注意力配置
    enum llama_pooling_type pooling_type;     // 池化类型（用于嵌入）
    enum llama_attention_type attention_type; // 注意力类型
    enum llama_flash_attn_type flash_attn_type; // Flash Attention类型

    // KV缓存数据类型
    enum ggml_type type_k;      // K缓存数据类型[EXPERIMENTAL]
    enum ggml_type type_v;      // V缓存数据类型[EXPERIMENTAL]

    // 回调函数
    ggml_backend_sched_eval_callback cb_eval;     // 评估回调
    void * cb_eval_user_data;                     // 回调用户数据
    ggml_abort_callback abort_callback;           // 中止回调
    void * abort_callback_data;                   // 中止回调数据

    // 采样器链配置[EXPERIMENTAL]
    struct llama_sampler_seq_config * samplers;
    size_t                            n_samplers;

    // 标志
    bool flash_attn;            // 使用Flash Attention
    bool embeddings;            // 提取嵌入向量
    bool offload_kqv;           // 将KQV卸载到GPU
    bool no_perf;               // 不收集性能数据
    bool op_offload;            // 将主机张量操作卸载到设备
    bool swa_full;              // 使用完整SWA缓存
    bool kv_unified;            // 使用统一KV缓存
};
```

### 10.1.2 llama_context 结构体

**源码位置**：`src/llama-context.h` (第1-150行)

```cpp
struct llama_context {
    // 关联的模型
    llama_model * model;
    llama_cparams cparams;      // 计算参数（运行时）

    // 内存分配器
    std::unique_ptr<llama_memory> memory;

    // 输入输出缓冲区
    struct llama_batch batch;
    struct llama_input_buffers inp;

    // KV缓存
    struct llama_kv_cache kv_cache;

    // 计算图
    struct ggml_cgraph * graph = nullptr;
    struct ggml_context * ctx_compute = nullptr;

    // 状态
    bool has_evaluated_once = false;

    // 输出 logits
    float * logits = nullptr;           // 当前logits
    std::vector<float> logits_buf;      // logits缓冲区

    // 嵌入输出（用于嵌入模型）
    float * embd = nullptr;
    std::vector<float> embd_buf;

    // 位置编码缓存
    std::vector<llama_pos> pos;
    std::vector<llama_pos> pos_view;

    // 性能统计
    struct {
        int64_t t_sample_us = 0;        // 采样时间
        int64_t t_eval_us = 0;          // 评估时间
        int64_t t_p_eval_us = 0;        // 提示评估时间
        int32_t n_sample = 0;           // 采样次数
        int32_t n_eval = 0;             // 评估次数
        int32_t n_p_eval = 0;           // 提示评估token数
    } t_perf;

    // 异步计算
    std::unique_ptr<llama_async_context> async;
};
```

### 10.1.3 上下文创建流程

**源码位置**：`src/llama-context.cpp` (第1-300行)

```cpp
struct llama_context * llama_new_context_with_model(
        struct llama_model * model,
        struct llama_context_params params) {

    // ① 验证参数
    if (params.n_ctx == 0) {
        params.n_ctx = model->hparams.n_ctx_train;
    }
    if (params.n_batch < 1) {
        params.n_batch = 512;
    }

    // ② 创建上下文对象
    llama_context * ctx = new llama_context();
    ctx->model = model;
    ctx->cparams = llama_compute_params(model->hparams, params);

    // ③ 初始化KV缓存
    ctx->kv_cache = llama_kv_cache_init(
        model->hparams,
        model->backend,
        params.n_ctx,
        model->hparams.n_layer,
        model->hparams.n_embd_head_k,
        model->hparams.n_head_kv
    );

    // ④ 分配logits缓冲区
    size_t logits_size = params.n_batch * model->hparams.n_vocab;
    ctx->logits_buf.resize(logits_size);
    ctx->logits = ctx->logits_buf.data();

    // ⑤ 分配计算图上下文
    ctx->ctx_compute = ggml_init({
        .mem_size = compute_graph_size(model->hparams),
        .mem_buffer = nullptr,
        .no_alloc = false,
    });

    // ⑥ 初始化批次
    llama_batch_init(&ctx->batch, params.n_batch, 0, 1);

    return ctx;
}
```

---

## 10.2 推理批次（llama_batch）

### 10.2.1 批次数据结构

**源码位置**：`src/llama-batch.h` (第1-80行)

```cpp
struct llama_batch {
    // token数据
    int32_t * token;            // token ID数组 [n_tokens]
    float   * embd;             // 嵌入输入（替代token）

    // 位置信息
    llama_pos * pos;            // 位置索引 [n_tokens]
    int32_t   * n_seq_id;       // 每个token对应的序列数 [n_tokens]
    llama_seq_id ** seq_id;     // 序列ID数组 [n_tokens][n_seq_id]

    // 注意力掩码
    int8_t * logits;            // 是否输出该token的logits [n_tokens]

    // 规模
    int32_t n_tokens;           // 当前批次中的token数
    int32_t n_seq_id_max;       // seq_id缓冲区大小
    int32_t n_embd;             // 嵌入维度（使用embd时）
    bool    all_pos_0;          // 所有位置是否为0（优化提示）
    bool    all_same_seq;       // 所有token是否同序列（优化）
};
```

**字段说明**：
| 字段 | 说明 | 示例 |
|-----|------|------|
| `token` | token ID数组 | [1, 1500, 2034] |
| `pos` | 每个token的位置 | [0, 1, 2] 或 [10, 11, 12] |
| `seq_id` | 序列标识（多序列时区分） | [[0], [0], [0]] |
| `logits` | 是否计算该位置的logits | [0, 0, 1]（只输出最后一个）|
| `n_tokens` | token总数 | 3 |

### 10.2.2 批次初始化与操作

**源码位置**：`src/llama-batch.cpp` (第1-200行)

```cpp
// 初始化批次
void llama_batch_init(struct llama_batch * batch,
                      int32_t n_tokens,      // 最大token数
                      int32_t n_embd,        // 嵌入维度（可选）
                      int32_t n_seq_max) {   // 最大序列数

    batch->n_tokens = 0;
    batch->n_seq_id_max = n_seq_max;
    batch->n_embd = n_embd;

    // 分配数组
    batch->token = new int32_t[n_tokens];
    batch->embd = n_embd > 0 ? new float[n_tokens * n_embd] : nullptr;
    batch->pos = new llama_pos[n_tokens];
    batch->n_seq_id = new int32_t[n_tokens];
    batch->seq_id = new llama_seq_id*[n_tokens];
    for (int i = 0; i < n_tokens; i++) {
        batch->seq_id[i] = new llama_seq_id[n_seq_max];
    }
    batch->logits = new int8_t[n_tokens];
}

// 添加单个token到批次
void llama_batch_add(struct llama_batch * batch,
                     llama_token token,      // token ID
                     llama_pos pos,          // 位置
                     const std::vector<llama_seq_id> & seq_ids, // 序列ID
                     bool logits) {          // 是否输出logits

    int idx = batch->n_tokens;

    batch->token[idx] = token;
    batch->pos[idx] = pos;
    batch->n_seq_id[idx] = seq_ids.size();
    for (size_t i = 0; i < seq_ids.size(); i++) {
        batch->seq_id[idx][i] = seq_ids[i];
    }
    batch->logits[idx] = logits ? 1 : 0;

    batch->n_tokens++;
}

// 清空批次
void llama_batch_clear(struct llama_batch * batch) {
    batch->n_tokens = 0;
    batch->all_pos_0 = false;
    batch->all_same_seq = true;
}
```

### 10.2.3 批处理推理示例

**场景**：同时生成3个不同序列

```cpp
// 批次设置
llama_batch batch;
llama_batch_init(&batch, 512, 0, 1);

// 序列1: "Hello" -> [15496] at pos 0
llama_batch_add(&batch, 15496, 0, {0}, false);

// 序列2: "Hi there" -> [5026, 612] at pos 0,1
llama_batch_add(&batch, 5026, 0, {1}, false);
llama_batch_add(&batch, 612, 1, {1}, false);

// 序列3: "How are" -> [13225, 389] at pos 0,1
llama_batch_add(&batch, 13225, 0, {2}, false);
llama_batch_add(&batch, 389, 1, {2}, true);  // 输出logits

// 解码
llama_decode(ctx, batch);

// 获取各序列的logits
float * logits = llama_get_logits(ctx);
// logits[2 * n_vocab] 是序列3最后一个token的logits
```

---

## 10.3 解码策略实现

### 10.3.1 llama_decode 核心流程

**源码位置**：`src/llama-context.cpp` (第300-800行)

```cpp
int llama_decode(
        struct llama_context * ctx,
        struct llama_batch   batch) {

    // ① 参数验证
    GGML_ASSERT(batch.n_tokens > 0);
    GGML_ASSERT(batch.n_tokens <= ctx->cparams.n_batch);

    // ② 分块处理（如果batch太大）
    const int n_ubatch = ctx->cparams.n_ubatch;
    for (int i = 0; i < batch.n_tokens; i += n_ubatch) {
        const int n_tokens = std::min(n_ubatch, batch.n_tokens - i);

        // 提取子批次
        llama_batch ubatch = split_batch(batch, i, n_tokens);

        // ③ 构建计算图
        ctx->graph = llama_build_graph(ctx, ubatch, /*worst_case=*/false);

        // ④ 分配张量内存
        ggml_gallocr_alloc_graph(ctx->galloc, ctx->graph);

        // ⑤ 准备输入数据（拷贝到device）
        copy_batch_to_device(ctx, ubatch);

        // ⑥ 执行计算图
        ggml_backend_graph_compute(ctx->model->backend, ctx->graph);

        // ⑦ 提取输出（从device拷贝回来）
        extract_logits(ctx, ubatch);
    }

    return 0;
}
```

### 10.3.2 贪心解码

**源码位置**：`src/llama-sampler.cpp` (第100-200行)

```cpp
// 贪心采样：总是选择概率最高的token
llama_token llama_sample_greedy(
        const float * logits,
        int n_vocab) {

    int max_idx = 0;
    float max_val = logits[0];

    // 遍历所有token找到最大值
    for (int i = 1; i < n_vocab; i++) {
        if (logits[i] > max_val) {
            max_val = logits[i];
            max_idx = i;
        }
    }

    return (llama_token)max_idx;
}
```

### 10.3.3 随机采样与温度

**源码位置**：`src/llama-sampler.cpp` (第200-400行)

```cpp
// 温度缩放 + 随机采样
llama_token llama_sample_temperature(
        const float * logits,
        int n_vocab,
        float temp) {  // 温度参数（通常0.5-1.5）

    // ① 温度缩放：logits /= temp
    // temp > 1: 分布更平缓，随机性增加
    // temp < 1: 分布更尖锐，确定性增加
    std::vector<float> probs(n_vocab);
    float max_logit = *std::max_element(logits, logits + n_vocab);

    float sum_exp = 0.0f;
    for (int i = 0; i < n_vocab; i++) {
        probs[i] = expf((logits[i] - max_logit) / temp);
        sum_exp += probs[i];
    }

    // ② 归一化为概率分布
    for (int i = 0; i < n_vocab; i++) {
        probs[i] /= sum_exp;
    }

    // ③ 采样
    float r = (float)rand() / RAND_MAX;
    float cumsum = 0.0f;

    for (int i = 0; i < n_vocab; i++) {
        cumsum += probs[i];
        if (cumsum >= r) {
            return (llama_token)i;
        }
    }

    return (llama_token)(n_vocab - 1);
}
```

### 10.3.4 Top-K与Top-P采样

**Top-K采样**（限制候选集大小）：
```cpp
// 源码位置：src/llama-sampler.cpp (第400-500行)
void llama_sample_top_k(
        float * logits,
        int n_vocab,
        int k) {  // 通常20-50

    // ① 找到第K大的阈值
    std::vector<float> sorted_logits(logits, logits + n_vocab);
    std::nth_element(sorted_logits.begin(),
                     sorted_logits.begin() + k,
                     sorted_logits.end(),
                     std::greater<float>());
    float threshold = sorted_logits[k];

    // ② 将低于阈值的logits设为-inf
    for (int i = 0; i < n_vocab; i++) {
        if (logits[i] < threshold) {
            logits[i] = -INFINITY;
        }
    }
}
```

**Top-P采样**（核采样，按累积概率截断）：
```cpp
// 源码位置：src/llama-sampler.cpp (第500-700行)
void llama_sample_top_p(
        float * logits,
        int n_vocab,
        float p) {  // 通常0.9-0.95

    // ① softmax得到概率
    softmax(logits, n_vocab);

    // ② 按概率降序排序
    std::vector<std::pair<float, int>> probs_idx;
    for (int i = 0; i < n_vocab; i++) {
        probs_idx.push_back({logits[i], i});
    }
    std::sort(probs_idx.begin(), probs_idx.end(),
              std::greater<std::pair<float, int>>());

    // ③ 累积概率直到超过p
    float cumsum = 0.0f;
    for (int i = 0; i < n_vocab; i++) {
        cumsum += probs_idx[i].first;
        if (cumsum >= p) {
            // 将剩余token的logits设为-inf
            for (int j = i + 1; j < n_vocab; j++) {
                logits[probs_idx[j].second] = -INFINITY;
            }
            break;
        }
    }
}
```

---

## 10.4 状态保存与恢复

### 10.4.1 状态序列化

**源码位置**：`src/llama-context.cpp` (第1000-1300行)

```cpp
// 保存上下文状态
size_t llama_state_get_size(struct llama_context * ctx) {
    // 计算需要的缓冲区大小
    size_t size = 0;

    // KV缓存大小
    size += ctx->kv_cache.total_size();

    // logits输出
    size += ctx->logits_buf.size() * sizeof(float);

    // 性能统计
    size += sizeof(ctx->t_perf);

    return size;
}

// 写入状态
size_t llama_state_get_data(struct llama_context * ctx, uint8_t * dst) {
    uint8_t * pos = dst;

    // ① 写入KV缓存
    pos += ctx->kv_cache.serialize(pos);

    // ② 写入logits
    memcpy(pos, ctx->logits_buf.data(),
           ctx->logits_buf.size() * sizeof(float));
    pos += ctx->logits_buf.size() * sizeof(float);

    // ③ 写入性能统计
    memcpy(pos, &ctx->t_perf, sizeof(ctx->t_perf));
    pos += sizeof(ctx->t_perf);

    return pos - dst;
}
```

### 10.4.2 状态恢复

**源码位置**：`src/llama-context.cpp` (第1300-1500行)

```cpp
// 从缓冲区恢复状态
size_t llama_state_set_data(struct llama_context * ctx, const uint8_t * src) {
    const uint8_t * pos = src;

    // ① 读取KV缓存
    pos += ctx->kv_cache.deserialize(pos);

    // ② 读取logits
    memcpy(ctx->logits_buf.data(), pos,
           ctx->logits_buf.size() * sizeof(float));
    ctx->logits = ctx->logits_buf.data();
    pos += ctx->logits_buf.size() * sizeof(float);

    // ③ 读取性能统计
    memcpy(&ctx->t_perf, pos, sizeof(ctx->t_perf));
    pos += sizeof(ctx->t_perf);

    return pos - src;
}
```

### 10.4.3 使用示例

**源码位置**：`examples/save-load-state/save-load-state.cpp`

```cpp
// 保存会话
std::vector<uint8_t> session_state(llama_state_get_size(ctx));
llama_state_get_data(ctx, session_state.data());

// 保存到文件
FILE * fp = fopen("session.bin", "wb");
fwrite(session_state.data(), 1, session_state.size(), fp);
fclose(fp);

// ... 之后恢复 ...

// 从文件读取
fp = fopen("session.bin", "rb");
fread(session_state.data(), 1, session_state.size(), fp);
fclose(fp);

// 恢复状态
llama_state_set_data(ctx, session_state.data());

// 继续生成
llama_token next_token = llama_sample_token(...);
```

---

## 设计中的取舍

### 为什么batch设计得这么复杂（多个seq_id）？

**简单方案**：每个token只对应一个序列
```cpp
// 简化版
struct simple_batch {
    llama_token * token;
    llama_pos * pos;
    llama_seq_id * seq_id;  // 每个token一个序列ID
};
```

**GGML方案**：每个token可以属于多个序列
```cpp
// GGML版
struct llama_batch {
    llama_seq_id ** seq_id;  // 每个token可以有多个序列ID
    int32_t * n_seq_id;      // 每个token的序列数
};
```

**为什么需要多序列支持？**
- **束搜索（Beam Search）**：同一个token可能属于多个beam
- **投机解码**：草稿token可能需要验证多个候选
- **共享前缀**：多个对话共享相同的系统提示

### 为什么区分n_batch和n_ubatch？

| 参数 | 作用 | 默认值 |
|-----|------|-------|
| `n_batch` | 逻辑批次大小（最大token数） | 2048 |
| `n_ubatch` | 物理批次大小（实际一次处理） | 512 |

**原因**：
1. 大模型可能无法一次性处理2048个token的计算图
2. 分块处理避免OOM，同时保持接口简洁
3. 提示处理（prompt processing）可以分块，生成时通常n_tokens=1

---

## 动手练习

### 练习1：阅读批次处理代码
阅读 `src/llama-context.cpp` 第300-500行，回答：
1. `llama_decode`如何处理超过`n_ubatch`的批次？
2. 什么是`worst_case`参数，何时使用？
3. `copy_batch_to_device`做了什么？

### 练习2：实现自定义采样
实现一个Top-K + Temperature的组合采样器：
```cpp
llama_token sample_top_k_temperature(
    const float* logits,
    int n_vocab,
    int k,
    float temp
);
```

### 练习3：状态保存实验
修改 `examples/save-load-state/save-load-state.cpp`：
1. 生成10个token后保存状态
2. 恢复状态后继续生成
3. 验证两次生成的结果是否一致

---

## 本课小结

| 概念 | 一句话总结 |
|------|-----------|
| llama_context | 推理总控中心，包含KV缓存、计算图、输出缓冲区 |
| llama_batch | 批次数据，支持多token多序列 |
| llama_decode | 核心解码函数，构建图→分配→执行→提取 |
| 贪心采样 | 选概率最高的token，确定性输出 |
| Top-K/Top-P | 限制候选集，平衡多样性和质量 |
| 状态保存 | 序列化KV缓存，实现会话恢复 |

---

*本章对应源码版本：master (2026-04-07)*
