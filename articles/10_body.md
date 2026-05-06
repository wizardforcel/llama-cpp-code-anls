# 第10章 推理上下文（llama_context）—— 推理过程的"总控制中心"

- 难度: ★★★★☆ (4/5)
- 预备知识: C++基础、LLM推理流程基础

---

## 学习目标

1. 理解llama_context的结构组成及其在推理中的核心地位
2. 掌握llama_batch的数据组织和批处理机制
3. 理解解码策略的实现原理（贪心、Top-K、Top-P等）
4. 掌握状态保存与恢复机制，实现会话持久化

---

## 生活类比：交响乐团的指挥中心

想象llama_context是一位**交响乐团的总指挥**：

**llama_context** = 指挥台 + 总指挥
- 掌控整个演出的节奏和流程
- 协调各个乐手（计算资源）的配合
- 记忆演奏进度（推理状态）

**llama_batch** = 当前要演奏的乐谱片段
- 包含多个声部（序列）的音符（tokens）
- 每个声部有独立的演奏位置（位置编码）

**KV缓存** = 乐手们的"记忆笔记"
- 记录已经演奏过的部分，避免重复演奏
- 每个声部有自己的笔记

**采样器** = 即兴创作系统
- 根据乐谱提示（logits）决定下一个音符
- 可以是严格按谱（贪心）或自由发挥（随机）

**状态保存** = 演出中场休息
- 记录当前进度，下半场可以继续

就像指挥需要协调各方，llama_context需要协调模型、缓存、采样器等组件完成推理。

---

## 模块地图

以下是本章涉及的核心文件结构：

```
src/llama-context.h          # llama_context类定义
src/llama-context.cpp        # 上下文实现（创建、解码、状态管理）
src/llama-batch.h            # llama_ubatch和llama_batch_allocr定义
src/llama-sampler.cpp        # 采样器实现
include/llama.h              # C API接口和参数结构体
examples/save-load-state/    # 状态保存示例
```

### 核心类与函数

| 组件 | 文件 | 关键类/函数 |
|------|------|-------------|
| 上下文管理 | `src/llama-context.h/cpp` | `llama_context`类 |
| 批次处理 | `src/llama-batch.h` | `llama_ubatch`, `llama_batch_allocr` |
| 采样策略 | `src/llama-sampler.cpp` | `llama_sampler_*`系列函数 |
| 状态持久化 | `src/llama-context.cpp` | `state_get_data()`, `state_set_data()` |

---

## 10.1 上下文结构

### 10.1.1 上下文参数（llama_context_params）

**源码位置**：`include/llama.h` (第330-382行) - `llama_context_params`

先理解类型，就像先看懂建筑图纸再施工。

```c
// 上下文参数结构体 - 配置推理环境的"蓝图"
struct llama_context_params {
    uint32_t n_ctx;             // ① 最大上下文长度（默认512，0表示从模型获取）
    uint32_t n_batch;           // ② 逻辑最大批处理大小（默认512）
    uint32_t n_ubatch;          // ③ 物理批处理大小（用于分块处理）
    uint32_t n_seq_max;         // ④ 最大序列数（用于循环模型）

    int32_t  n_threads;         // ⑤ CPU线程数（生成阶段）
    int32_t  n_threads_batch;   // ⑥ CPU线程数（批处理/提示处理阶段）

    // RoPE（旋转位置编码）配置
    enum llama_rope_scaling_type rope_scaling_type; // ⑦ RoPE扩展类型
    float    rope_freq_base;    // ⑧ RoPE基数覆盖（0表示从模型获取）
    float    rope_freq_scale;   // ⑨ RoPE缩放覆盖
    float    yarn_ext_factor;   // ⑩ YaRN扩展因子
    float    yarn_attn_factor;  // ⑪ YaRN注意力因子
    float    yarn_beta_fast;    // ⑫ YaRN快速beta
    float    yarn_beta_slow;    // ⑬ YaRN慢速beta
    uint32_t yarn_orig_ctx;     // ⑭ YaRN原始上下文大小

    // 注意力机制配置
    enum llama_pooling_type      pooling_type;      // ⑮ 池化类型（用于嵌入）
    enum llama_attention_type    attention_type;    // ⑯ 注意力类型
    enum llama_flash_attn_type   flash_attn_type;   // ⑰ Flash Attention类型

    // KV缓存数据类型[实验性功能]
    enum ggml_type type_k;      // ⑱ K缓存数据类型
    enum ggml_type type_v;      // ⑲ V缓存数据类型

    // 回调函数
    ggml_backend_sched_eval_callback cb_eval;       // ⑳ 评估回调
    void * cb_eval_user_data;                       // 回调用户数据
    ggml_abort_callback abort_callback;             // ㉑ 中止回调
    void * abort_callback_data;                     // 中止回调数据

    // 采样器链配置[实验性功能]
    struct llama_sampler_seq_config * samplers;     // ㉒ 采样器配置
    size_t                            n_samplers;   // 采样器数量

    // 布尔标志（放在一起避免内存对齐问题）
    bool flash_attn;            // ㉓ 使用Flash Attention
    bool embeddings;            // ㉔ 提取嵌入向量
    bool offload_kqv;         // ㉕ 将KQV卸载到GPU
    bool no_perf;             // ㉖ 不收集性能数据
    bool op_offload;          // ㉗ 将主机张量操作卸载到设备
    bool swa_full;            // ㉘ 使用完整SWA缓存
    bool kv_unified;          // ㉙ 使用统一KV缓存
};
```

这段代码定义了llama_context的配置参数结构体。每个参数都有明确的默认值和用途说明。

**关键参数说明**：

- `n_ctx`：决定模型能"记住"多少token（上下文窗口大小）
- `n_batch` vs `n_ubatch`：逻辑批次 vs 物理批次，后者用于分块处理大批次避免OOM
- `n_threads` vs `n_threads_batch`：生成阶段和提示处理阶段可能使用不同线程数
- `rope_*`：位置编码的扩展配置，用于支持长上下文
- `flash_attn`：是否使用Flash Attention优化内存和计算

---

### 10.1.2 llama_context 类结构

**源码位置**：`src/llama-context.h` (第36-359行) - `llama_context`

```cpp
// llama_context类 - 推理的"总控制中心"
struct llama_context {
    // 关联的模型和计算参数
    const llama_model & model;          // ① 关联的模型（只读引用）
    llama_cparams cparams;              // ② 计算参数（运行时配置）

    // 内存管理
    std::unique_ptr<llama_memory_i> memory;  // ③ 内存分配器（抽象接口）

    // 采样相关
    struct sampling_info {              // ④ 采样信息聚合
        std::map<llama_seq_id, llama_sampler *> samplers;  // 每个序列的采样器
        buffer_view<float>       logits;      // logits输出缓冲区
        buffer_view<llama_token> sampled;     // 已采样token
        buffer_view<float>       probs;       // 概率分布
        buffer_view<llama_token> candidates;  // 候选token
        // ... 计数器省略
    } sampling;

    // 输出缓冲区
    buffer_view<float> logits = {nullptr, 0};   // ⑤ logits输出 [n_outputs][n_vocab]
    buffer_view<float> embd = {nullptr, 0};       // ⑥ 嵌入输出 [n_outputs][n_embd]

    // 序列嵌入（用于池化模式）
    std::map<llama_seq_id, std::vector<float>> embd_seq;  // ⑦ 每个序列的嵌入

    // 批次分配器（避免重复分配内存）
    std::unique_ptr<llama_batch_allocr> balloc;  // ⑧ 批次分配器

    // 后端调度器
    ggml_backend_sched_ptr sched;               // ⑨ 后端调度器
    std::vector<ggml_backend_ptr> backends;     // ⑩ 后端列表
    ggml_backend_t backend_cpu = nullptr;       // ⑪ CPU后端

    // 计算图结果
    llm_graph_result_ptr gf_res_prev;       // ⑫ 上一次计算结果
    llm_graph_result_ptr gf_res_reserve;    // ⑬ 预留的计算结果

    // 输出管理
    uint32_t n_outputs = 0;                   // ⑭ 实际使用的输出数量
    std::vector<int32_t> output_ids;        // ⑮ token位置到输出ID的映射
    std::vector<swap_info> output_swaps;    // ⑯ 输出重排序信息

    // 多线程
    ggml_threadpool_t threadpool = nullptr;        // ⑰ 线程池
    ggml_threadpool_t threadpool_batch = nullptr;  // ⑱ 批处理线程池

    // 中止回调
    ggml_abort_callback abort_callback = nullptr;       // ⑲ 中止回调函数
    void *              abort_callback_data = nullptr;  // 回调数据

    // 性能统计
    mutable int64_t t_start_us  = 0;        // ⑳ 开始时间
    mutable int64_t t_load_us   = 0;        // 加载时间
    mutable int64_t t_p_eval_us = 0;        // 提示评估时间
    mutable int64_t t_eval_us   = 0;        // 评估时间
    mutable int32_t n_p_eval = 0;           // 提示token数
    mutable int32_t n_eval   = 0;           // 评估次数
    mutable int32_t n_reused = 0;           // 图复用次数

    // 适配器（LoRA等）
    llama_adapter_cvec_ptr  cvec;         // ㉑ 控制向量
    llama_adapter_loras_ptr loras;        // ㉒ LoRA适配器

    // 标志
    bool has_evaluated_once = false;      // ㉓ 是否已执行过评估
    bool sched_need_reserve = true;       // ㉔ 是否需要预留调度器
    bool graph_reuse_disable = false;     // ㉕ 是否禁用图复用

    // ... 构造函数、析构函数和成员函数省略
};
```

这段代码定义了llama_context类的核心成员。它聚合了模型引用、内存管理、采样状态、后端调度器、性能统计等所有推理所需的状态。

**关键设计洞察**：

- **引用语义**：`model`是const引用，保证上下文不会修改模型
- **智能指针**：使用`unique_ptr`和`shared_ptr`自动管理内存
- **视图模式**：`buffer_view`提供非拥有式的数组访问，避免拷贝
- **惰性初始化**：许多成员延迟到首次使用时才分配

---

### 10.1.3 上下文创建流程

**源码位置**：`src/llama-context.cpp` (第22-365行) - 构造函数

```cpp
// llama_context构造函数 - 初始化推理环境
llama_context::llama_context(
        const llama_model & model,
              llama_context_params params) :
    model(model),
    cvec(std::make_unique<llama_adapter_cvec>()),      // ① 初始化控制向量
    loras(std::make_unique<llama_adapter_loras>()),    // ② 初始化LoRA适配器
    balloc(std::make_unique<llama_batch_allocr>(
        model.hparams.n_pos_per_embd())) {              // ③ 创建批次分配器

    LLAMA_LOG_INFO("%s: constructing llama_context\n", __func__);

    // ④ 记录模型加载时间
    t_start_us = model.t_start_us;
    t_load_us  = model.t_load_us;

    const auto & hparams = model.hparams;

    // ⑤ 验证并设置序列数限制
    cparams.n_seq_max = std::max(1u, params.n_seq_max);
    if (cparams.n_seq_max > LLAMA_MAX_SEQ) {
        throw std::runtime_error("n_seq_max must be <= " + std::to_string(LLAMA_MAX_SEQ));
    }

    // ⑥ 复制基本参数
    cparams.n_threads        = params.n_threads;
    cparams.n_threads_batch  = params.n_threads_batch;
    cparams.embeddings       = params.embeddings;
    cparams.offload_kqv      = params.offload_kqv;
    cparams.no_perf          = params.no_perf;
    cparams.pooling_type     = params.pooling_type;

    // ⑦ YaRN参数处理（负值表示"未设置"，使用模型默认值）
    cparams.yarn_ext_factor  = params.yarn_ext_factor  >= 0.0f
        ? params.yarn_ext_factor  : hparams.yarn_ext_factor;
    cparams.yarn_attn_factor = params.yarn_attn_factor >= 0.0f
        ? params.yarn_attn_factor : hparams.yarn_attn_factor;
    cparams.yarn_beta_fast   = params.yarn_beta_fast   >= 0.0f
        ? params.yarn_beta_fast   : hparams.yarn_beta_fast;
    cparams.yarn_beta_slow   = params.yarn_beta_slow   >= 0.0f
        ? params.yarn_beta_slow   : hparams.yarn_beta_slow;

    // ⑧ 上下文大小（0表示使用训练时的上下文大小）
    cparams.n_ctx = params.n_ctx == 0
        ? hparams.n_ctx_train
        : params.n_ctx;

    // ⑨ RoPE参数（0.0f表示使用模型默认值）
    cparams.rope_freq_base   = params.rope_freq_base  == 0.0f
        ? hparams.rope_freq_base_train
        : params.rope_freq_base;
    cparams.rope_freq_scale  = params.rope_freq_scale == 0.0f
        ? hparams.rope_freq_scale_train
        : params.rope_freq_scale;

    // ⑩ YaRN原始上下文大小
    cparams.n_ctx_orig_yarn  = params.yarn_orig_ctx    != 0
        ? params.yarn_orig_ctx
        : hparams.n_ctx_orig_yarn != 0
            ? hparams.n_ctx_orig_yarn
            : hparams.n_ctx_train;

    // ⑪ 设置后端采样器（实验性功能）
    if (params.samplers != nullptr && params.n_samplers > 0) {
        for (size_t i = 0; i < params.n_samplers; ++i) {
            const auto & config = params.samplers[i];
            if (set_sampler(config.seq_id, config.sampler)) {
                LLAMA_LOG_INFO("%s: setting backend sampler for seq_id %d\n",
                    __func__, config.seq_id);
            }
        }
    }

    // ⑫ RoPE缩放类型处理
    auto rope_scaling_type = params.rope_scaling_type;
    if (rope_scaling_type == LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED) {
        rope_scaling_type = hparams.rope_scaling_type_train;
    }
    if (rope_scaling_type == LLAMA_ROPE_SCALING_TYPE_NONE) {
        cparams.rope_freq_scale = 1.0f;  // 不缩放
    }

    // ⑬ YaRN扩展因子计算（复杂的缩放公式）
    if (cparams.yarn_ext_factor < 0.0f) {
        cparams.yarn_ext_factor = rope_scaling_type == LLAMA_ROPE_SCALING_TYPE_YARN
            ? 1.0f : 0.0f;
    }

    if (cparams.yarn_ext_factor != 0) {
        // mscale计算：用于调整注意力分布
        auto get_mscale = [](float scale, float mscale) {
            return scale <= 1.0f ? 1.0f : (0.1f * mscale * logf(scale) + 1.0f);
        };
        const float factor = 1.0f / cparams.rope_freq_scale;
        cparams.yarn_attn_factor = get_mscale(factor, 1.0f);
        cparams.yarn_attn_factor *= hparams.rope_attn_factor;
    }

    // ⑭ 池化类型（默认NONE）
    if (cparams.pooling_type == LLAMA_POOLING_TYPE_UNSPECIFIED) {
        cparams.pooling_type = hparams.pooling_type == LLAMA_POOLING_TYPE_UNSPECIFIED
            ? LLAMA_POOLING_TYPE_NONE
            : hparams.pooling_type;
    }

    // ⑮ 因果注意力设置
    if (params.attention_type == LLAMA_ATTENTION_TYPE_UNSPECIFIED) {
        cparams.causal_attn = hparams.causal_attn;
    } else {
        cparams.causal_attn = params.attention_type == LLAMA_ATTENTION_TYPE_CAUSAL;
    }

    // ⑯ Flash Attention设置
    cparams.flash_attn = params.flash_attn_type != LLAMA_FLASH_ATTN_TYPE_DISABLED;
    cparams.auto_fa    = params.flash_attn_type == LLAMA_FLASH_ATTN_TYPE_AUTO;

    // ⑰ 批次大小处理（因果注意力时受限于上下文大小）
    cparams.n_batch = cparams.causal_attn
        ? std::min(cparams.n_ctx, params.n_batch)
        : params.n_batch;
    cparams.n_ubatch = std::min(cparams.n_batch,
        params.n_ubatch == 0 ? params.n_batch : params.n_ubatch);

    // ⑱ 填充上下文大小到256的倍数（内存对齐优化）
    cparams.n_ctx = GGML_PAD(cparams.n_ctx, 256);

    // ⑲ 统一KV缓存 vs 分离KV缓存
    if (cparams.kv_unified) {
        cparams.n_ctx_seq = cparams.n_ctx;
    } else {
        cparams.n_ctx_seq = cparams.n_ctx / cparams.n_seq_max;
        cparams.n_ctx_seq = GGML_PAD(cparams.n_ctx_seq, 256);
        if (cparams.n_ctx_seq == 0) {
            throw std::runtime_error("n_ctx_seq == 0");
        }
    }

    // ⑳ 初始化后端（GPU、CPU等）
    if (!hparams.vocab_only) {
        // 初始化模型指定的设备后端
        for (auto * dev : model.devices) {
            ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
            if (backend == nullptr) {
                throw std::runtime_error("failed to initialize backend");
            }
            backends.emplace_back(backend);
        }

        // 添加加速后端（如BLAS）
        for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
            ggml_backend_dev_t dev = ggml_backend_dev_get(i);
            if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_ACCEL) {
                ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
                if (backend) backends.emplace_back(backend);
            }
        }

        // 添加CPU后端
        backend_cpu = ggml_backend_init_by_type(
            GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
        backends.emplace_back(backend_cpu);

        // 初始化内存模块
        llama_memory_params params_mem = {
            /*.type_k   =*/ params.type_k,
            /*.type_v   =*/ params.type_v,
            /*.swa_full =*/ params.swa_full,
        };
        memory.reset(model.create_memory(params_mem, cparams));
    }

    // ㉑ 初始化完整词表token ID（用于后端采样器）
    const int n_vocab = model.vocab.n_tokens();
    sampling.token_ids_full_vocab.resize(n_vocab);
    for (int i = 0; i < n_vocab; ++i) {
        sampling.token_ids_full_vocab[i] = i;
    }
}
```

这段代码实现了llama_context的构造函数，完成了从参数配置到运行时环境的完整初始化流程。

**初始化流程总结**：

1. **成员初始化**：控制向量、LoRA适配器、批次分配器
2. **参数继承**：从模型超参数继承未指定的值
3. **YaRN计算**：复杂的位置编码扩展公式
4. **后端枚举**：GPU → ACCEL → CPU的优先级顺序
5. **内存分配**：根据参数创建KV缓存

---

## 10.2 推理批次（llama_batch）

### 10.2.1 统一批次数据结构（llama_ubatch）

**源码位置**：`src/llama-batch.h` (第14-69行) - `llama_ubatch`

```cpp
// 统一批次结构 - 实际传递给计算图的数据格式
struct llama_ubatch {
    // 辅助方法
    bool equal_seqs() const {
        return b_equal_seqs != 0;  // 所有token是否属于相同序列集
    }

    bool is_pos_2d() const {
        // 位置编码是否为2D（用于图像模型）
        return n_pos >= 3;
    }

    // 批次规模信息
    uint32_t b_equal_seqs;      // ① 布尔标志：序列是否相同（用int32对齐）
    uint32_t n_tokens;          // ② 总token数 = n_seq_tokens * n_seqs
    uint32_t n_seq_tokens;      // ③ 每个序列集的token数
    uint32_t n_seqs;            // ④ 序列集数量
    uint32_t n_seqs_unq;        // ⑤ 唯一序列ID数量
    uint32_t n_pos;             // ⑥ 每个token的位置输入数（1D/2D/3D/4D）

    // 核心数据指针（指向实际数据存储）
    llama_token  *  token;      // ⑦ token ID数组 [n_tokens]
    float        *  embd;       // ⑧ 嵌入输入（替代token）[n_embd, n_tokens]
    llama_pos    *  pos;        // ⑨ 位置索引 [n_tokens * n_pos]
    int32_t      *  n_seq_id;   // ⑩ 每个token的序列数 [n_tokens]
    llama_seq_id ** seq_id;     // ⑪ 序列ID数组 [n_tokens][n_seq_id]
    llama_seq_id *  seq_id_unq; // ⑫ 唯一序列ID [n_seqs_unq]
    int32_t      *  seq_idx;    // ⑬ 序列索引 [LLAMA_MAX_SEQ]
    int8_t       *  output;     // ⑭ 是否输出该token的logits [n_tokens]

    // 数据存储（可选，指针可能指向这里）
    struct data_t {
        std::vector<llama_token>    token;
        std::vector<float>          embd;
        std::vector<llama_pos>      pos;
        std::vector<int32_t>        n_seq_id;
        std::vector<llama_seq_id *> seq_id;
        std::vector<llama_seq_id>   seq_id_unq;
        std::vector<int32_t>        seq_idx;
        std::vector<int8_t>         output;
        std::vector<llama_seq_id>   seq_id_data;  // 实际seq_id存储
    };
    std::shared_ptr<data_t> data;  // 共享数据存储
};
```

这段代码定义了llama_ubatch结构，它是实际传递给计算图的批次格式。相比C API中的llama_batch，这个结构更紧凑，支持多位置编码（用于多模态模型）。

**为什么需要分离llama_batch和llama_ubatch？**

| 特性 | llama_batch (C API) | llama_ubatch (内部) |
|------|---------------------|---------------------|
| 用户可见 | 是 | 否 |
| 数据所有权 | 外部管理 | 内部管理（data_t） |
| 多位置支持 | 否 | 是（最多4D） |
| 序列组织 | 原始 | 优化后的序列集 |

---

### 10.2.2 批次分配器（llama_batch_allocr）

**源码位置**：`src/llama-batch.h` (第71-173行) - `llama_batch_allocr`

```cpp
// 批次分配器 - 将用户输入转换为内部批次格式
class llama_batch_allocr {
public:
    llama_batch_allocr(uint32_t n_pos_per_embd);

    // 初始化：验证输入批次并生成缺失的数据
    bool init(
        const llama_batch & batch_inp,        // 用户输入批次
        const llama_vocab & vocab,             // 词表（用于验证）
        const llama_memory_i * memory,         // 内存（用于位置检查）
        uint32_t n_embd,                       // 嵌入维度
        uint32_t n_seq_max,                    // 最大序列数
        bool output_all);                      // 是否输出所有位置

    // 获取处理后的批次
    const llama_batch & get_batch() const;

    // 获取输出相关信息
    uint32_t get_n_tokens()  const;
    uint32_t get_n_outputs() const;
    uint32_t get_n_used()    const;

    // 批次分割方法（处理大批次）
    void split_reset();                          // 重置分割状态
    llama_ubatch split_simple(uint32_t n_ubatch);  // 简单分割
    llama_ubatch split_equal(uint32_t n_ubatch, bool sequential);  // 等长分割
    llama_ubatch split_seq(uint32_t n_ubatch);   // 按序列分割

    // 创建预留批次
    llama_ubatch ubatch_reserve(uint32_t n_seq_tokens, uint32_t n_seqs);

private:
    void clear();
    llama_ubatch ubatch_add(const std::vector<int32_t> & idxs,
                           uint32_t n_seqs, bool equal_seqs);

    llama_batch batch;           // 存储的批次数据
    const llama_vocab * vocab;   // 词表指针（调试用）
    const uint32_t n_pos_per_embd; // 每个嵌入的位置数

    // 内部状态
    uint32_t n_embd;
    uint32_t n_seq_max;
    uint32_t n_outputs;
    std::array<llama_seq_id, 1> seq_id_0 = {{ 0 }};  // 默认序列ID

    // 临时缓冲区
    std::vector<llama_pos>      pos;
    std::vector<int32_t>        n_seq_id;
    std::vector<llama_seq_id *> seq_id;
    std::vector<llama_seq_id>   seq_id_unq;
    std::vector<int32_t>        seq_idx;
    std::vector<int8_t>         output;

    // 序列位置追踪
    std::vector<std::set<llama_pos>> seq_pos;  // 每个序列的位置集合
    std::vector<std::vector<bool>>   seq_cpl;  // 序列耦合矩阵

    // 序列集映射
    using seq_set_t = std::bitset<LLAMA_MAX_SEQ>;
    std::vector<seq_set_t> seq_set;
    std::unordered_map<seq_set_t, std::vector<int32_t>> seq_set_map;

    // 输出追踪
    std::vector<int32_t> out_ids;   // 输出token索引
    uint32_t n_used;                // 已使用的token数
    std::vector<bool> used;         // token使用标记
};
```

这段代码定义了llama_batch_allocr类，它是连接用户输入（llama_batch）和内部计算（llama_ubatch）的桥梁。

**核心职责**：

1. **数据验证**：检查token ID有效性、序列ID范围
2. **位置生成**：为缺失位置的token自动生成位置
3. **批次分割**：将大批次分割成适合n_ubatch的小块
4. **序列组织**：优化序列布局以提高缓存命中率

---

## 10.3 解码策略实现

### 10.3.1 llama_decode 主解码流程

**源码位置**：`src/llama-context.cpp` (第1296-1300行, 第3432-3441行)

```cpp
// C API包装函数
int32_t llama_decode(llama_context * ctx, llama_batch batch) {
    const int ret = ctx->decode(batch);  // ① 调用成员函数
    if (ret != 0 && ret != 1) {
        LLAMA_LOG_ERROR("%s: failed to decode, ret = %d\n", __func__, ret);
    }
    return ret;
}

// 成员函数 - 实际解码实现
int llama_context::decode(const llama_batch & batch_inp) {
    // ② 参数验证
    GGML_ASSERT((!batch_inp.token && batch_inp.embd) ||
                (batch_inp.token && !batch_inp.embd));  // token和embd必须二选一

    if (batch_inp.n_tokens == 0) {
        LLAMA_LOG_ERROR("%s: n_tokens == 0\n", __func__);
        return -1;
    }

    const auto & hparams = model.hparams;
    const int64_t n_embd  = hparams.n_embd_inp();
    const int64_t n_vocab = model.vocab.n_tokens();

    // ③ 初始化批次分配器
    if (!balloc->init(batch_inp, model.vocab, memory.get(), n_embd,
                      cparams.kv_unified ? LLAMA_MAX_SEQ : cparams.n_seq_max,
                      true)) {
        LLAMA_LOG_ERROR("%s: failed to initialize batch\n", __func__);
        return -1;
    }

    const uint32_t n_tokens = balloc->get_n_tokens();

    // ④ 检查是否需要重置缓存（用于非因果注意力模式）
    // TODO: 添加新模式以支持padding序列
    const llama_ubatch ubatch = balloc->split_simple(n_tokens);

    // ⑤ 微批处理断言（编码器需要完整批次）
    GGML_ASSERT(cparams.n_ubatch >= n_tokens &&
                "encoder requires n_ubatch >= n_tokens");

    // ⑥ 记录开始时间
    if (t_compute_start_us == 0) {
        t_compute_start_us = ggml_time_us();
    }

    // ⑦ 清空序列嵌入缓存
    embd_seq.clear();

    // ⑧ 确保调度器已预留
    sched_reserve();

    // ⑨ 累加排队token数
    n_queued_tokens += n_tokens;

    // ⑩ 预留输出缓冲区
    if (output_reserve(n_tokens) < n_tokens) {
        LLAMA_LOG_ERROR("%s: could not reserve space\n", __func__);
        return -2;
    }

    // ⑪ 设置输出ID映射
    for (uint32_t i = 0; i < n_tokens; ++i) {
        output_ids[i] = i;
    }
    n_outputs = n_tokens;

    // ⑫ 执行计算
    ggml_status status;
    const auto * res = process_ubatch(ubatch, LLM_GRAPH_TYPE_DECODER,
                                       nullptr, status);

    if (!res) {
        return -3;  // 计算失败
    }

    // ⑬ 提取输出
    // ... logits和embeddings提取逻辑

    return 0;  // 成功
}
```

这段代码实现了llama_decode的核心流程：验证输入、初始化批次、预留资源、执行计算、提取输出。

---

### 10.3.2 process_ubatch - 批次处理核心

**源码位置**：`src/llama-context.cpp` (第1167-1237行)

```cpp
llm_graph_result * llama_context::process_ubatch(
        const llama_ubatch & ubatch,
        llm_graph_type       gtype,        // 图类型（编码器/解码器）
        llama_memory_context_i * mctx,     // 内存上下文
        ggml_status &        ret) {       // 返回状态

    // ① 应用内存上下文（如KV缓存更新）
    if (mctx && !mctx->apply()) {
        LLAMA_LOG_ERROR("%s: failed to apply memory context\n", __func__);
        ret = GGML_STATUS_FAILED;
        return nullptr;
    }

    auto * res = gf_res_prev.get();  // 复用上一次结果对象
    auto * gf  = res->get_gf();      // 获取计算图

    // ② 构建图参数
    const auto gparams = graph_params(res, ubatch, mctx, gtype);

    // ③ 图复用检查（性能优化关键）
    if (!graph_reuse_disable && res->can_reuse(gparams)) {
        // 流水线并行需要同步
        if (cparams.pipeline_parallel) {
            ggml_backend_sched_synchronize(sched.get());
        }
        n_reused++;  // 统计复用次数
    } else {
        // ④ 构建新计算图
        res->reset();
        ggml_backend_sched_reset(sched.get());
        ggml_backend_sched_set_eval_callback(sched.get(),
            cparams.cb_eval, cparams.cb_eval_user_data);

        gf = model.build_graph(gparams);  // 构建图

        if (!gf) {
            LLAMA_LOG_ERROR("%s: failed to initialize graph\n", __func__);
            ret = GGML_STATUS_FAILED;
            return nullptr;
        }

        // ⑤ 分配图内存
        if (!ggml_backend_sched_alloc_graph(sched.get(), gf)) {
            LLAMA_LOG_ERROR("%s: failed to allocate graph\n", __func__);
            ret = GGML_STATUS_ALLOC_FAILED;
            return nullptr;
        }
    }

    // ⑥ 设置输入数据
    res->set_inputs(&ubatch);

    // ⑦ 执行计算
    const auto status = graph_compute(res->get_gf(), ubatch.n_tokens > 1);
    if (status != GGML_STATUS_SUCCESS) {
        LLAMA_LOG_ERROR("%s: failed to compute graph\n", __func__);
        ret = status;
        return nullptr;
    }

    ret = GGML_STATUS_SUCCESS;
    return res;
}
```

这段代码实现了process_ubatch，它是连接批次数据和计算图的核心函数。

**关键优化点**：

- **图复用**：如果参数相同，复用上一次的计算图结构
- **流水线并行**：支持GPU流水线以提高吞吐量
- **延迟分配**：只在需要时构建和分配计算图

---

### 10.3.3 贪心采样实现

**源码位置**：`src/llama-sampler.cpp` (第265-287行) - `llama_sampler_temp_impl`

```cpp
// 温度缩放 + 贪心采样（温度=0时的特例）
static void llama_sampler_temp_impl(llama_token_data_array * cur_p, float temp) {
    if (temp <= 0.0f) {
        // ① 贪心采样：选择logit最大的token
        size_t max_i = 0;
        float  max_l = cur_p->data[0].logit;

        // ② 遍历找到最大值
        for (size_t i = 1; i < cur_p->size; ++i) {
            if (cur_p->data[i].logit > max_l) {
                // ③ 将非最大值设为负无穷
                cur_p->data[max_i].logit = -INFINITY;
                max_i = i;
                max_l = cur_p->data[i].logit;
            } else {
                cur_p->data[i].logit = -INFINITY;
            }
        }
        return;  // 只有max_i位置的logit保持原值
    }

    // ④ 温度缩放（temp > 0）
    for (size_t i = 0; i < cur_p->size; ++i) {
        cur_p->data[i].logit /= temp;  // logits /= temperature
    }
}
```

这段代码实现了温度缩放和贪心采样。当temperature=0时，执行贪心选择；否则对logits进行温度缩放。

**温度参数效果**：

| temp值 | 效果 |
|--------|------|
| temp → 0 | 接近贪心，确定性输出 |
| temp = 1 | 保持原始分布 |
| temp > 1 | 分布更平缓，增加随机性 |
| temp < 1 | 分布更尖锐，减少随机性 |

---

### 10.3.4 Softmax与Top-K采样

**源码位置**：`src/llama-sampler.cpp` (第289-334行)

```cpp
// Softmax实现
static void llama_sampler_softmax_impl(llama_token_data_array * cur_p,
                                        bool do_sort) {
    GGML_ASSERT(cur_p->size > 0);

    // ① 按logits降序排序（如果需要）
    if (do_sort && !cur_p->sorted) {
        llama_token_data_array_partial_sort_inplace(cur_p, cur_p->size);
    }

    // ② 找最大logit（数值稳定性）
    float max_l = cur_p->data[0].logit;
    if (!cur_p->sorted) {
        for (size_t i = 1; i < cur_p->size; ++i) {
            max_l = std::max(max_l, cur_p->data[i].logit);
        }
    }

    // ③ 计算exp并累加
    float cum_sum = 0.0f;
    for (size_t i = 0; i < cur_p->size; ++i) {
        float p = expf(cur_p->data[i].logit - max_l);  // 减去max防止溢出
        cur_p->data[i].p = p;  // 存储概率
        cum_sum += p;
    }

    // ④ 归一化
    for (size_t i = 0; i < cur_p->size; ++i) {
        cur_p->data[i].p /= cum_sum;
    }
}

// Top-K采样
static void llama_sampler_top_k_impl(llama_token_data_array * cur_p, int32_t k) {
    if (k <= 0) return;  // 禁用

    k = std::min(k, (int)cur_p->size);

    // ① 降序排序（只排前k个）
    if (!cur_p->sorted) {
        llama_token_data_array_partial_sort_inplace(cur_p, k);
    }

    // ② 截断到前k个
    cur_p->size = k;
}
```

这段代码实现了softmax概率计算和Top-K截断采样。

**部分排序优化**：

```cpp
// 桶排序 + 部分排序的混合算法（处理大规模词表）
static void llama_token_data_array_partial_sort(
        const llama_token_data_array & cur,
        int npartial,
        std::vector<llama_token_data> & res) {

    constexpr int   nbuckets     = 128;           // ① 128个桶
    constexpr float bucket_low   = -10.0f;        // logit范围下限
    constexpr float bucket_high  =  10.0f;        // logit范围上限
    constexpr float bucket_scale = nbuckets/(bucket_high - bucket_low);

    std::vector<int> histo(nbuckets, 0);           // ② 直方图

    // ③ 统计每个桶的元素数量
    for (int i = 0; i < (int)cur.size; ++i) {
        const float val = cur.data[i].logit;
        int ib = int(bucket_scale * val + bucket_low * bucket_scale);
        ib = std::max(0, std::min(nbuckets - 1, ib));
        ++histo[ib];
    }

    // ④ 找到包含前npartial个元素的桶边界
    int nhave = 0, ib = nbuckets - 1;
    for (; ib >= 0; --ib) {
        nhave += histo[ib];
        if (nhave >= npartial) break;
    }

    // ⑤ 只对该桶内的元素进行精确排序
    // ... 部分排序逻辑
}
```

这种优化对大规模词表（如100k+ tokens）特别重要，能将O(n log n)降到接近O(n)。

---

## 10.4 状态保存与恢复

### 10.4.1 状态序列化

**源码位置**：`src/llama-context.cpp` (第2371-2389行)

```cpp
// 获取状态数据大小
size_t llama_context::state_get_size() {
    llama_io_write_dummy io;  // ① 虚拟写入器（只计算大小）
    try {
        return state_write_data(io);  // ② 模拟写入获取大小
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error getting state size: %s\n", __func__, err.what());
        return 0;
    }
}

// 将状态写入缓冲区
size_t llama_context::state_get_data(uint8_t * dst, size_t size) {
    llama_io_write_buffer io(dst, size);  // ③ 创建缓冲区写入器
    try {
        return state_write_data(io);  // ④ 执行写入
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error saving state: %s\n", __func__, err.what());
        return 0;
    }
}

// 从缓冲区恢复状态
size_t llama_context::state_set_data(const uint8_t * src, size_t size) {
    llama_io_read_buffer io(src, size);  // ⑤ 创建缓冲区读取器
    try {
        return state_read_data(io);  // ⑥ 执行读取
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error loading state: %s\n", __func__, err.what());
        return 0;
    }
}
```

这段代码实现了状态序列化的基本框架，使用IO抽象层支持内存缓冲区和文件。

---

### 10.4.2 状态写入实现

**源码位置**：`src/llama-context.cpp` (第2544-2562行)

```cpp
// 写入状态数据
size_t llama_context::state_write_data(llama_io_write_i & io) {
    LLAMA_LOG_DEBUG("%s: writing state\n", __func__);

    // ① 写入模型信息（用于版本检查）
    {
        LLAMA_LOG_DEBUG("%s: - writing model info\n", __func__);
        const std::string arch_str = llm_arch_name(model.arch);
        io.write_string(arch_str);  // 写入架构名称
        // TODO: 添加更多模型特定信息用于兼容性检查
    }

    // ② 写入内存模块状态（主要是KV缓存）
    if (memory != nullptr) {
        LLAMA_LOG_DEBUG("%s: - writing memory module\n", __func__);
        memory->state_write(io);
    }

    return io.n_bytes();  // 返回写入字节数
}
```

这段代码实现了状态数据的实际写入，主要包括模型架构信息和KV缓存状态。

---

### 10.4.3 状态文件保存与加载

**源码位置**：`src/llama-context.cpp` (第2464-2479行, 第2421-2462行)

```cpp
// 保存状态到文件
bool llama_context::state_save_file(
        const char * filepath,
        const llama_token * tokens,       // 提示token
        size_t       n_token_count) {     // token数量

    llama_file file(filepath, "wb");    // ① 打开文件

    // ② 写入魔数和版本
    file.write_u32(LLAMA_SESSION_MAGIC);     // 魔数：标识文件类型
    file.write_u32(LLAMA_SESSION_VERSION);   // 版本：兼容性检查

    // ③ 写入提示token
    file.write_u32((uint32_t)n_token_count);
    file.write_raw(tokens, sizeof(llama_token) * n_token_count);

    // ④ 写入上下文状态
    llama_io_write_file io(&file);
    state_write_data(io);

    return true;
}

// 从文件加载状态
bool llama_context::state_load_file(
        const char * filepath,
        llama_token * tokens_out,          // 输出：恢复的token
        size_t       n_token_capacity,     // token缓冲区容量
        size_t *     n_token_count_out) { // 输出：实际token数

    llama_file file(filepath, "rb");     // ① 打开文件

    // ② 验证魔数和版本
    {
        const uint32_t magic   = file.read_u32();
        const uint32_t version = file.read_u32();

        if (magic != LLAMA_SESSION_MAGIC || version != LLAMA_SESSION_VERSION) {
            LLAMA_LOG_ERROR("%s: unknown (magic, version): %08x, %08x\n",
                __func__, magic, version);
            return false;
        }
    }

    // ③ 读取提示token
    {
        const uint32_t n_token_count = file.read_u32();
        if (n_token_count > n_token_capacity) {
            LLAMA_LOG_ERROR("%s: token count exceeded capacity!\n", __func__);
            return false;
        }
        file.read_raw(tokens_out, sizeof(llama_token) * n_token_count);
        *n_token_count_out = n_token_count;
    }

    // ④ 恢复上下文状态
    {
        const size_t n_state_size = file.size() - file.tell();
        llama_io_read_file io(&file);
        const size_t n_read = state_read_data(io);

        if (n_read != n_state_size) {
            LLAMA_LOG_ERROR("%s: did not read all data!\n", __func__);
            return false;
        }
    }

    return true;
}
```

这段代码实现了完整的文件级状态保存和加载功能。

**文件格式**：

```
+------------------+
| Magic (4 bytes)  |  <- LLAMA_SESSION_MAGIC
+------------------+
| Version (4 bytes)|  <- LLAMA_SESSION_VERSION
+------------------+
| n_tokens (4B)    |
+------------------+
| tokens[]         |  <- 提示token数组
+------------------+
| KV Cache State   |  <- 内存模块状态
+------------------+
| ...              |
+------------------+
```

---

### 10.4.4 序列状态保存（单序列粒度）

**源码位置**：`src/llama-context.cpp` (第2524-2542行)

```cpp
// 保存单个序列的状态
size_t llama_context::state_seq_save_file(
        llama_seq_id seq_id,               // 目标序列ID
        const char * filepath,
        const llama_token * tokens,
        size_t       n_token_count) {

    llama_file file(filepath, "wb");

    // 使用不同的魔数区分完整状态和序列状态
    file.write_u32(LLAMA_STATE_SEQ_MAGIC);
    file.write_u32(LLAMA_STATE_SEQ_VERSION);

    // 写入提示token
    file.write_u32((uint32_t)n_token_count);
    file.write_raw(tokens, sizeof(llama_token) * n_token_count);

    // 写入该序列的KV缓存
    llama_io_write_file io(&file);
    state_seq_write_data(io, seq_id, 0);  // 0 = 默认flags

    return file.tell();
}
```

这段代码实现了单序列粒度的状态保存，支持多序列场景下的部分状态导出。

---

## 10.5 使用示例

### 状态保存与恢复完整示例

**源码位置**：`examples/save-load-state/save-load-state.cpp`

```cpp
// 保存状态示例
std::vector<uint8_t> session_state(llama_state_get_size(ctx));
llama_state_get_data(ctx, session_state.data());

// 保存到文件
FILE * fp = fopen("session.bin", "wb");
fwrite(session_state.data(), 1, session_state.size(), fp);
fclose(fp);

// ... 之后恢复 ...

// 创建新上下文（相同模型）
llama_context * ctx2 = llama_new_context_with_model(model, params);

// 从文件读取
fp = fopen("session.bin", "rb");
fread(session_state.data(), 1, session_state.size(), fp);
fclose(fp);

// 恢复状态
llama_state_set_data(ctx2, session_state.data());

// 继续生成（与原始上下文状态完全一致）
llama_token next_token = llama_sampler_sample(smpl, ctx2, -1);
```

---

## 设计中的取舍

### 为什么区分n_batch和n_ubatch？

**简单方案**：只有一个batch大小参数
```cpp
// 简化版
struct simple_params {
    uint32_t n_batch;  // 批处理大小
};
```

**llama.cpp方案**：逻辑批次 vs 物理批次分离
```cpp
// 实际实现
struct llama_context_params {
    uint32_t n_batch;   // 逻辑最大批次（API限制）
    uint32_t n_ubatch;  // 物理批次（实际计算单元）
};
```

**为什么这样设计？**

1. **内存限制**：大模型无法一次性处理2048个token的计算图，需要分块
2. **接口简洁**：用户只需设置逻辑批次，内部自动分块
3. **提示处理优化**：提示处理可以分块，生成时通常n_tokens=1

### 为什么每个token可以有多个seq_id？

**场景1：束搜索（Beam Search）**
- 同一个token可能属于多个beam候选
- 需要`seq_id[0] = {0, 1, 2}`表示token同时属于beam 0、1、2

**场景2：投机解码（Speculative Decoding）**
- 草稿token可能需要验证多个候选序列
- 通过多seq_id支持高效的批量验证

**场景3：共享前缀**
- 多个对话共享相同的系统提示
- 前缀token的seq_id包含所有使用该前缀的序列

---

## 动手练习

### 练习1：阅读解码源码

阅读 `src/llama-context.cpp` 第1167-1300行，回答：
1. `process_ubatch`如何处理图复用？什么情况下会触发重建？
2. `pipeline_parallel`为true时需要额外的同步操作，为什么？
3. `set_inputs`方法的作用是什么？

### 练习2：实现自定义采样

实现一个Temperature + Top-P的组合采样器：
```cpp
llama_token sample_temp_topp(
    const float* logits,
    int n_vocab,
    float temp,
    float p
);
```

### 练习3：状态保存实验

修改 `examples/save-load-state/save-load-state.cpp`：
1. 生成10个token后保存状态
2. 恢复状态后继续生成
3. 验证两次生成的结果是否完全一致
4. 尝试修改温度参数后继续生成，观察结果变化

---

## 本章小结

这一章我们深入学习了llama.cpp的推理上下文系统。首先，我们了解了llama_context作为推理"总控制中心"的角色，它协调模型、KV缓存、采样器和后端资源。其次，我们学习了llama_batch的数据组织方式，以及如何通过批次分配器处理大规模输入。接着，我们探讨了解码策略的实现，包括贪心采样、温度缩放、Top-K和Top-P采样。最后，我们掌握了状态保存与恢复机制，实现了会话持久化。

本章涉及的核心概念：

| 概念 | 解释 |
|------|------|
| llama_context | 推理总控中心，包含所有运行时状态 |
| llama_ubatch | 内部批次格式，支持多位置编码 |
| n_batch vs n_ubatch | 逻辑批次 vs 物理批次，用于分块处理 |
| KV缓存 | 存储注意力Key/Value，避免重复计算 |
| 温度采样 | 通过温度参数控制输出随机性 |
| Top-P采样 | 按累积概率截断候选集 |
| 状态保存 | 序列化KV缓存和运行时状态，支持会话恢复 |

下一章中，我们将学习采样器的链式组合机制，探索如何构建复杂的采样策略。

---

*本章对应源码版本：master (2026-05-06)*
