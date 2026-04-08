# 附录A：API参考手册 —— 开发者的"速查宝典"

## 学习目标

1. 掌握GGML张量操作的核心API
2. 理解llama.cpp的模型加载与推理接口
3. 熟悉采样链的构建与配置
4. 学会在实际项目中正确使用API

---

## 生活类比：厨师的工具箱

想象你是一位厨师，API就像你厨房里的各种工具。GGML API是基础的"刀工"——切、剁、片、雕，是每一道菜的基础；llama.cpp API是"烹饪技法"——炒、煮、蒸、烤，决定菜品的最终呈现；采样API则是"调味"——咸淡、酸甜、辣度，赋予菜品独特风味。一个优秀的厨师不仅要会用这些工具，更要懂得何时用、如何用。本章将为你展示llama.cpp这个"厨房"里的全套"工具"及其正确用法。

---

## A.1 GGML C API 详解

### A.1.1 核心数据结构

**ggml_tensor —— 张量的"身份证"**

```c
// ggml/include/ggml.h:357-378
struct ggml_tensor {
    enum ggml_type type;                    // ① 数据类型（F32/F16/Q4_0等）
    struct ggml_backend_buffer * buffer;    // ② 后端缓冲区
    int64_t ne[GGML_MAX_DIMS];              // ③ 各维度元素数量 (number of elements)
    size_t  nb[GGML_MAX_DIMS];              // ④ 各维度字节步长 (stride in bytes)
    void * data;                            // ⑤ 数据指针
    char name[GGML_MAX_NAME];               // ⑥ 张量名称
    struct ggml_tensor * src[GGML_MAX_SRC]; // ⑦ 源张量（计算图连接）
    struct ggml_tensor * view_src;          // ⑧ 视图源张量
    int64_t view_offs;                      // ⑨ 视图偏移
    struct ggml_context * ctx;              // ⑩ 所属上下文
    enum ggml_op op;                        // ⑪ 产生此张量的操作
    int32_t op_params[GGML_MAX_OP_PARAMS];  // ⑫ 操作参数
    int32_t flags;                          // ⑬ 标志位
};
```

**关键字段详解：**

| 字段 | 说明 | 示例 |
|------|------|------|
| `ne[4]` | 各维度大小 | `ne[0]=512, ne[1]=64` 表示 512×64 矩阵 |
| `nb[4]` | 各维度字节步长 | `nb[0]=4` 表示 F32 每元素4字节 |
| `src[]` | 父节点指针 | 构建计算图时自动填充 |
| `data` | 实际数据地址 | 通过 `ggml_get_data()` 访问 |

**为什么使用 `ne` 和 `nb` 分离的设计？**

这种设计支持**非连续内存布局**（如转置张量），`nb` 允许灵活的数据排布，是GGML支持视图操作的基础。

---

**ggml_context —— 内存池管理器**

```c
// ggml/include/ggml.h:347-352
struct ggml_init_params {
    size_t mem_size;    // ① 内存池大小（字节）
    void * mem_buffer;  // ② 外部内存缓冲区（可为NULL）
    bool   no_alloc;    // ③ 是否延迟分配
};
```

**上下文生命周期管理：**

```c
// 1. 初始化参数（申请16MB内存池）
struct ggml_init_params params = {
    .mem_size   = 16*1024*1024,
    .mem_buffer = NULL,          // 由GGML分配
    .no_alloc   = false,
};

// 2. 创建上下文
struct ggml_context * ctx = ggml_init(params);
if (!ctx) {
    fprintf(stderr, "Failed to create context\n");
    return -1;
}

// 3. 使用上下文创建张量、构建计算图...

// 4. 释放上下文（自动释放所有关联张量）
ggml_free(ctx);
```

---

### A.1.2 张量创建函数

**创建张量的四种方式：**

```c
// ① 创建通用N维张量（最灵活）
struct ggml_tensor * ggml_new_tensor(
    struct ggml_context * ctx,
    enum ggml_type type,
    int n_dims,
    const int64_t * ne
);

// ②-⑤ 创建1D-4D张量（便捷函数）
struct ggml_tensor * ggml_new_tensor_1d(
    struct ggml_context * ctx,
    enum ggml_type type,
    int64_t ne0
);

struct ggml_tensor * ggml_new_tensor_2d(
    struct ggml_context * ctx,
    enum ggml_type type,
    int64_t ne0, int64_t ne1
);

// ⑥ 复制张量结构（不复制数据）
struct ggml_tensor * ggml_dup_tensor(
    struct ggml_context * ctx,
    const struct ggml_tensor * src
);

// ⑦ 创建张量视图（共享数据）
struct ggml_tensor * ggml_view_tensor(
    struct ggml_context * ctx,
    struct ggml_tensor * src
);
```

**使用示例：**

```c
// 创建 512×768 的 F32 矩阵
struct ggml_tensor * mat = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 512, 768);

// 创建 4096 维向量
struct ggml_tensor * vec = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4096);

// 创建与现有张量同形状的新张量（用于存储结果）
struct ggml_tensor * mat2 = ggml_dup_tensor(ctx, mat);

// 创建视图（共享内存，不同形状）
int64_t ne[2] = {768, 512};
struct ggml_tensor * mat_view = ggml_view_tensor(ctx, mat);
mat_view->ne[0] = 768;
mat_view->ne[1] = 512;  // 逻辑转置，不移动数据
```

---

### A.1.3 张量运算函数

**基础数学运算：**

```c
// 逐元素加法: result = a + b
struct ggml_tensor * ggml_add(
    struct ggml_context * ctx,
    struct ggml_tensor * a,
    struct ggml_tensor * b
);

// 逐元素减法: result = a - b
struct ggml_tensor * ggml_sub(
    struct ggml_context * ctx,
    struct ggml_tensor * a,
    struct ggml_tensor * b
);

// 逐元素乘法: result = a * b (Hadamard积)
struct ggml_tensor * ggml_mul(
    struct ggml_context * ctx,
    struct ggml_tensor * a,
    struct ggml_tensor * b
);

// 缩放: result = a * s
struct ggml_tensor * ggml_scale(
    struct ggml_context * ctx,
    struct ggml_tensor * a,
    float s
);
```

**矩阵运算（核心）：**

```c
// 矩阵乘法: result = a @ b
// a: [M, K], b: [K, N], result: [M, N]
struct ggml_tensor * ggml_mul_mat(
    struct ggml_context * ctx,
    struct ggml_tensor * a,
    struct ggml_tensor * b
);

// 转置
struct ggml_tensor * ggml_transpose(
    struct ggml_context * ctx,
    struct ggml_tensor * a
);

// 维度重排
struct ggml_tensor * ggml_permute(
    struct ggml_context * ctx,
    struct ggml_tensor * a,
    int axis0, int axis1, int axis2, int axis3
);
```

**神经网络专用运算：**

```c
// SiLU激活: result = x * sigmoid(x)
struct ggml_tensor * ggml_silu(
    struct ggml_context * ctx,
    struct ggml_tensor * a
);

// GELU激活
struct ggml_tensor * ggml_gelu(
    struct ggml_context * ctx,
    struct ggml_tensor * a
);

// RMS归一化（Llama使用）
struct ggml_tensor * ggml_rms_norm(
    struct ggml_context * ctx,
    struct ggml_tensor * a,
    float eps           // 防止除零的小值
);

// Softmax归一化
struct ggml_tensor * ggml_soft_max(
    struct ggml_context * ctx,
    struct ggml_tensor * a
);

// RoPE位置编码
struct ggml_tensor * ggml_rope(
    struct ggml_context * ctx,
    struct ggml_tensor * a,
    struct ggml_tensor * b,  // 位置信息
    int n_dims,              // 编码维度
    int mode,                // 模式
    int n_ctx                // 上下文长度
);
```

**完整计算流程示例：**

```c
// 线性层 + SiLU激活: y = silu(x @ W + b)
struct ggml_tensor * x = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 10);
struct ggml_tensor * W = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 10, 5);
struct ggml_tensor * b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 5);

// 构建计算图
struct ggml_tensor * wx = ggml_mul_mat(ctx, W, x);   // x @ W
struct ggml_tensor * y = ggml_add(ctx, wx, b);        // + b
struct ggml_tensor * out = ggml_silu(ctx, y);         // SiLU激活

// 创建并构建计算图
struct ggml_cgraph * gf = ggml_new_graph(ctx, GGML_DEFAULT_GRAPH_SIZE);
ggml_build_forward(gf, out);

// 设置输入数据
float x_data[10] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f};
memcpy(x->data, x_data, sizeof(x_data));

// 执行计算
ggml_graph_compute(gf, NULL);

// 获取结果
float result = ggml_get_f32_1d(out, 0);
```

---

### A.1.4 后端系统 API

**设备管理：**

```c
// 获取可用设备数量
size_t ggml_backend_dev_count(void);

// 获取指定索引设备
ggml_backend_dev_t ggml_backend_dev_get(size_t index);

// 按名称获取设备
ggml_backend_dev_t ggml_backend_dev_by_name(const char * name);

// 获取设备名称
const char * ggml_backend_dev_name(ggml_backend_dev_t device);

// 获取设备类型 (CPU/GPU/ACCEL)
enum ggml_backend_dev_type ggml_backend_dev_type(ggml_backend_dev_t device);
```

**调度器使用：**

```c
// 创建多后端调度器
ggml_backend_t backends[2] = {cpu_backend, cuda_backend};
ggml_backend_buffer_type_t bufts[2] = {cpu_buft, cuda_buft};

ggml_backend_sched_t sched = ggml_backend_sched_new(
    backends,      // 后端数组
    bufts,         // 缓冲区类型数组
    2,             // 后端数量
    4096,          // 图大小
    true,          // 是否并行
    true           // 是否操作卸载
);

// 为图分配内存
ggml_backend_sched_alloc_graph(sched, gf);

// 执行计算
ggml_backend_sched_graph_compute(sched, gf);
```

---

## A.2 Llama C API 详解

### A.2.1 模型加载参数

**llama_model_params —— 模型配置的结构体：**

```c
// include/llama.h:200-230
typedef struct llama_model_params {
    ggml_backend_dev_t * devices;           // ① 设备列表
    int32_t n_gpu_layers;                   // ② GPU层数 (-1=全部)
    enum llama_split_mode split_mode;       // ③ 多GPU分割模式
    int32_t main_gpu;                       // ④ 主GPU索引
    const float * tensor_split;             // ⑤ 张量分割比例
    llama_progress_callback progress_callback; // ⑥ 加载进度回调
    void * progress_callback_user_data;     // ⑦ 回调用户数据
    bool vocab_only;                        // ⑧ 仅加载词表
    bool use_mmap;                          // ⑨ 使用内存映射
    bool use_mlock;                         // ⑩ 使用mlock锁定内存
    bool check_tensors;                     // ⑪ 验证张量数据
} llama_model_params;
```

**推荐配置组合：**

```c
// 配置A: 本地开发（快速加载）
struct llama_model_params params = llama_model_default_params();
params.n_gpu_layers = 35;       // 部分GPU卸载
params.use_mmap = true;         // 内存映射
params.use_mlock = false;       // 不锁定内存

// 配置B: 生产部署（最大性能）
struct llama_model_params params = llama_model_default_params();
params.n_gpu_layers = -1;       // 全部GPU卸载
params.use_mmap = true;
params.use_mlock = true;        // 锁定内存避免交换

// 配置C: 纯CPU（资源受限）
struct llama_model_params params = llama_model_default_params();
params.n_gpu_layers = 0;        // 不使用GPU
params.use_mmap = true;
```

---

**llama_context_params —— 上下文参数：**

```c
// include/llama.h:232-280
typedef struct llama_context_params {
    uint32_t n_ctx;             // ① 上下文长度（最大token数）
    uint32_t n_batch;           // ② 逻辑批大小
    uint32_t n_ubatch;          // ③ 物理批大小
    uint32_t n_seq_max;         // ④ 最大序列数
    int32_t  n_threads;         // ⑤ 生成线程数
    int32_t  n_threads_batch;   // ⑥ 批处理线程数
    
    // RoPE参数
    enum llama_rope_scaling_type rope_scaling_type;  // 缩放类型
    float    rope_freq_base;                         // 基础频率
    float    rope_freq_scale;                        // 频率缩放
    
    // 缓存类型
    enum ggml_type type_k;      // K缓存数据类型
    enum ggml_type type_v;      // V缓存数据类型
    
    bool embeddings;            // 提取嵌入而非生成
    bool offload_kqv;           // 卸载KQV到GPU
} llama_context_params;
```

---

### A.2.2 核心推理 API

**模型加载与释放：**

```c
// 获取默认参数（推荐始终从此开始）
struct llama_model_params llama_model_default_params(void);
struct llama_context_params llama_context_default_params(void);

// 从文件加载模型
struct llama_model * llama_model_load_from_file(
    const char * path_model,
    struct llama_model_params params
);

// 从分片加载（大模型通常分片存储）
struct llama_model * llama_model_load_from_splits(
    const char ** paths,
    size_t n_paths,
    struct llama_model_params params
);

// 释放模型
void llama_model_free(struct llama_model * model);

// 从模型创建上下文
struct llama_context * llama_init_from_model(
    struct llama_model * model,
    struct llama_context_params params
);

// 释放上下文
void llama_free(struct llama_context * ctx);
```

**完整加载示例：**

```c
// 1. 设置模型参数
struct llama_model_params mparams = llama_model_default_params();
mparams.n_gpu_layers = 35;  // 将35层加载到GPU
mparams.use_mmap = true;    // 使用内存映射
mparams.use_mlock = false;

// 2. 加载模型
struct llama_model * model = llama_model_load_from_file(
    "/path/to/model.gguf", 
    mparams
);
if (model == NULL) {
    fprintf(stderr, "Failed to load model\n");
    return 1;
}

// 3. 设置上下文参数
struct llama_context_params cparams = llama_context_default_params();
cparams.n_ctx = 4096;       // 4K上下文
cparams.n_threads = 4;      // 4线程生成
cparams.n_threads_batch = 8; // 8线程批处理

// 4. 创建上下文
struct llama_context * ctx = llama_init_from_model(model, cparams);
if (ctx == NULL) {
    fprintf(stderr, "Failed to create context\n");
    llama_model_free(model);
    return 1;
}

// 使用完毕后清理
llama_free(ctx);
llama_model_free(model);
```

---

**推理执行：**

```c
// 解码/前向传播（最常用的函数）
int32_t llama_decode(
    struct llama_context * ctx,
    struct llama_batch batch    // 批处理输入
);

// 获取logits（用于采样）
float * llama_get_logits(struct llama_context * ctx);

// 获取特定token的logits
float * llama_get_logits_ith(struct llama_context * ctx, int32_t i);

// 获取嵌入向量（用于RAG等场景）
float * llama_get_embeddings(struct llama_context * ctx);
float * llama_get_embeddings_ith(struct llama_context * ctx, int32_t i);
```

**llama_batch 结构：**

```c
typedef struct llama_batch {
    int32_t n_tokens;           // Token数量
    llama_token  *  token;      // Token ID数组
    float        *  embd;       // 嵌入向量（用于多模态）
    llama_pos    *  pos;        // 位置数组
    int32_t      *  n_seq_id;   // 每个token的序列ID数量
    llama_seq_id ** seq_id;     // 序列ID数组
    int8_t       *  logits;     // 是否输出logits标记
} llama_batch;
```

**推理示例：**

```c
// 准备输入token
llama_token tokens[] = {1, 100, 200, 300};  // 假设的token序列
int n_tokens = 4;

// 创建batch
struct llama_batch batch = llama_batch_init(n_tokens, 0, 1);
for (int i = 0; i < n_tokens; i++) {
    batch.token[i] = tokens[i];
    batch.pos[i] = i;
    batch.n_seq_id[i] = 1;
    batch.seq_id[i][0] = 0;
    batch.logits[i] = 0;  // 不输出logits
}
batch.logits[n_tokens - 1] = 1;  // 只输出最后一个token的logits
batch.n_tokens = n_tokens;

// 执行推理
int ret = llama_decode(ctx, batch);
if (ret != 0) {
    fprintf(stderr, "Decode failed: %d\n", ret);
    // 错误处理
}

// 获取logits进行采样
float * logits = llama_get_logits(ctx);
int n_vocab = llama_vocab_n_tokens(vocab);

// 找到概率最高的token（贪心解码）
llama_token next_token = 0;
float max_logit = -INFINITY;
for (int i = 0; i < n_vocab; i++) {
    if (logits[i] > max_logit) {
        max_logit = logits[i];
        next_token = i;
    }
}

// 清理
llama_batch_free(batch);
```

---

### A.2.3 分词 API

```c
// 文本分词
int32_t llama_tokenize(
    const struct llama_vocab * vocab,
    const char * text,
    int32_t text_len,
    llama_token * tokens,       // 输出token数组
    int32_t n_tokens_max,       // 数组容量
    bool add_special,           // 是否添加特殊token（如BOS）
    bool parse_special          // 是否解析特殊token
);

// Token转文本
int32_t llama_detokenize(
    const struct llama_vocab * vocab,
    const llama_token * tokens,
    int32_t n_tokens,
    char * text,
    int32_t text_len_max,
    bool remove_special,
    bool unparse_special
);

// Token转片段（更安全的接口）
int32_t llama_token_to_piece(
    const struct llama_vocab * vocab,
    llama_token token,
    char * buf,
    int32_t length,
    int32_t lstrip,             // 左侧去除字符数
    bool special                // 是否包含特殊token
);
```

**分词示例：**

```c
const char * text = "Hello, world!";
const struct llama_vocab * vocab = llama_model_get_vocab(model);

// 第一次调用获取token数量
int n_tokens = llama_tokenize(vocab, text, strlen(text), 
                              NULL, 0, true, false);

// 分配内存并分词
llama_token * tokens = malloc(n_tokens * sizeof(llama_token));
llama_tokenize(vocab, text, strlen(text), tokens, n_tokens, true, false);

// 打印token
for (int i = 0; i < n_tokens; i++) {
    char buf[32];
    int len = llama_token_to_piece(vocab, tokens[i], buf, sizeof(buf), 0, false);
    printf("%d: %d -> '%.*s'\n", i, tokens[i], len, buf);
}

free(tokens);
```

---

### A.2.4 采样 API

**采样链构建：**

```c
// 初始化采样链
struct llama_sampler * llama_sampler_chain_init(
    struct llama_sampler_chain_params params
);

// 添加采样器到链
void llama_sampler_chain_add(
    struct llama_sampler * chain,
    struct llama_sampler * smpl
);

// 执行采样
llama_token llama_sampler_sample(
    struct llama_sampler * smpl,
    struct llama_context * ctx,
    int32_t idx             // token索引，-1表示最后一个
);

// 接受token（更新采样器状态）
void llama_sampler_accept(
    struct llama_sampler * smpl,
    llama_token token
);

// 释放采样器
void llama_sampler_free(struct llama_sampler * smpl);
```

**标准采样器：**

```c
// 贪心采样
struct llama_sampler * llama_sampler_init_greedy(void);

// 温度缩放
struct llama_sampler * llama_sampler_init_temp(float t);

// Top-K采样
struct llama_sampler * llama_sampler_init_top_k(int32_t k);

// Top-P (Nucleus)采样
struct llama_sampler * llama_sampler_init_top_p(float p, size_t min_keep);

// Min-P采样
struct llama_sampler * llama_sampler_init_min_p(float p, size_t min_keep);

// 重复惩罚
struct llama_sampler * llama_sampler_init_penalties(
    int32_t penalty_last_n,
    float penalty_repeat,
    float penalty_freq,
    float penalty_present
);

// 语法约束
struct llama_sampler * llama_sampler_init_grammar(
    const struct llama_vocab * vocab,
    const char * grammar_str,
    const char * grammar_root
);

// 随机分布采样
struct llama_sampler * llama_sampler_init_dist(uint32_t seed);
```

**完整采样链示例：**

```c
// 创建采样链
struct llama_sampler_chain_params sparams = llama_sampler_chain_default_params();
struct llama_sampler * smpl = llama_sampler_chain_init(sparams);

// 添加采样器（按顺序）
llama_sampler_chain_add(smpl, llama_sampler_init_penalties(
    64,     // 检查最后64个token
    1.0f,   // 重复惩罚系数
    0.0f,   // 频率惩罚
    0.0f    // 存在惩罚
));
llama_sampler_chain_add(smpl, llama_sampler_init_top_k(40));
llama_sampler_chain_add(smpl, llama_sampler_init_top_p(0.9f, 1));
llama_sampler_chain_add(smpl, llama_sampler_init_min_p(0.05f, 1));
llama_sampler_chain_add(smpl, llama_sampler_init_temp(0.8f));
llama_sampler_chain_add(smpl, llama_sampler_init_dist(12345));  // 随机种子

// 采样循环
for (int i = 0; i < max_tokens; i++) {
    // 解码
    llama_decode(ctx, batch);
    
    // 采样
    llama_token next_token = llama_sampler_sample(smpl, ctx, -1);
    
    // 检查结束
    if (llama_vocab_is_eog(vocab, next_token)) {
        break;
    }
    
    // 接受token
    llama_sampler_accept(smpl, next_token);
    
    // 添加到batch继续生成...
}

// 清理
llama_sampler_free(smpl);
```

---

## A.3 Common 库 API 参考

### A.3.1 参数解析

**common_params —— 一站式参数结构体：**

```c
// common/common.h:100-200
struct common_params {
    // 模型参数
    std::string model;              // 模型路径
    std::string model_url;          // 模型下载URL
    int32_t n_gpu_layers = -1;      // GPU层数
    
    // 上下文参数
    int32_t n_ctx = 4096;           // 上下文大小
    int32_t n_batch = 2048;         // 批大小
    int32_t n_threads = -1;         // 线程数（-1=自动）
    
    // 生成参数
    int32_t n_predict = -1;         // 预测token数（-1=无限制）
    float   temp = 0.8f;            // 温度
    int32_t top_k = 40;             // Top-K
    float   top_p = 0.9f;           // Top-P
    float   min_p = 0.05f;          // Min-P
    
    // 输入输出
    std::string prompt;             // 提示词
    std::vector<std::string> antiprompt;  // 停止词
    bool interactive = false;       // 交互模式
    
    // ... 更多参数
};
```

**参数解析：**

```c
// 解析命令行参数
bool common_params_parse(int argc, char ** argv, common_params & params, 
                         llama_example example, void(*print_usage)(int, char **));

// 转换为llama参数
llama_model_params common_model_params_to_llama(common_params & params);
llama_context_params common_context_params_to_llama(common_params & params);
```

---

### A.3.2 便捷工具函数

```c
// 分词（自动处理内存）
std::vector<llama_token> common_tokenize(
    const struct llama_context * ctx,
    const std::string & text,
    bool add_special,
    bool parse_special
);

// 反分词
std::string common_detokenize(
    const struct llama_context * ctx,
    const std::vector<llama_token> & tokens
);

// Token转片段
std::string common_token_to_piece(
    const struct llama_context * ctx,
    llama_token token
);

// 初始化common库
void common_init();
```

---

## A.4 API 最佳实践

### A.4.1 内存管理原则

```c
// ✅ 好的做法：预先分配足够大的内存池
struct ggml_init_params params = {
    .mem_size = 512*1024*1024,  // 512MB
    .mem_buffer = NULL,
};

// ❌ 避免：频繁创建/销毁上下文
for (int i = 0; i < n; i++) {
    struct ggml_context * ctx = ggml_init(params);  // 不要这样做！
    // ...
    ggml_free(ctx);
}

// ✅ 好的做法：复用上下文
ggml_reset(ctx);  // 重置但不释放
```

### A.4.2 错误处理

```c
// 检查模型加载
struct llama_model * model = llama_model_load_from_file(path, params);
if (model == NULL) {
    // 处理错误
    return -1;
}

// 检查解码返回值
int ret = llama_decode(ctx, batch);
if (ret != 0) {
    switch (ret) {
        case 1:  // KV槽不足
            // 减少批大小或增加上下文
            break;
        case -1: // 无效输入
            // 检查输入batch
            break;
        default:
            // 其他错误
            break;
    }
}
```

### A.4.3 线程安全

- `llama_model` 是线程安全的，可被多个 `llama_context` 共享
- `llama_context` **不是**线程安全的，每个线程应使用独立上下文
- 分词 API 是线程安全的

```c
// ✅ 多线程推理（共享模型）
void * thread_func(void * arg) {
    struct llama_context * ctx = llama_init_from_model(model, cparams);
    // 独立推理
    llama_free(ctx);
    return NULL;
}
```

---

## 本课小结

| API类别 | 核心函数 | 用途 |
|---------|----------|------|
| GGML张量 | `ggml_new_tensor_2d`, `ggml_mul_mat` | 底层张量运算 |
| GGML计算图 | `ggml_build_forward`, `ggml_graph_compute` | 构建执行计算 |
| 模型加载 | `llama_model_load_from_file` | 加载GGUF模型 |
| 推理 | `llama_decode` | 执行前向传播 |
| 分词 | `llama_tokenize`, `llama_token_to_piece` | 文本/token转换 |
| 采样 | `llama_sampler_chain_add`, `llama_sampler_sample` | 生成token |

**快速参考代码：**

```c
// 最小完整推理流程
struct llama_model_params mparams = llama_model_default_params();
mparams.n_gpu_layers = 35;
struct llama_model * model = llama_model_load_from_file("model.gguf", mparams);

struct llama_context_params cparams = llama_context_default_params();
struct llama_context * ctx = llama_init_from_model(model, cparams);

// 分词
llama_token tokens[256];
int n = llama_tokenize(llama_model_get_vocab(model), "Hello", 5, 
                        tokens, 256, true, false);

// 推理
struct llama_batch batch = llama_batch_init(n, 0, 1);
// ... 填充batch ...
llama_decode(ctx, batch);

// 清理
llama_free(ctx);
llama_model_free(model);
```

---

*本附录对应源码版本：master (2026-04-07)*

