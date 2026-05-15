# 附录A：API参考手册 —— 开发者的"速查宝典"

## A.1 GGML C API 参考

### A.1.1 核心数据结构

#### ggml_tensor —— 张量结构体

```c
struct ggml_tensor {
    enum ggml_type type;                    // 数据类型（F32/F16/Q4_0等）
    struct ggml_backend_buffer * buffer;    // 后端缓冲区
    int64_t ne[GGML_MAX_DIMS];              // 各维度元素数量 (number of elements)
    size_t  nb[GGML_MAX_DIMS];              // 各维度步长 (stride in bytes)
    void * data;                            // 数据指针
    char name[GGML_MAX_NAME];               // 张量名称
    struct ggml_tensor * src[GGML_MAX_SRC]; // 源张量（计算图连接）
    struct ggml_tensor * view_src;          // 视图源张量
    int64_t view_offs;                      // 视图偏移
    struct ggml_context * ctx;              // 所属上下文
    enum ggml_op op;                        // 产生此张量的操作
    int32_t op_params[GGML_MAX_OP_PARAMS];  // 操作参数
    int32_t flags;                          // 标志位
};
```

**关键字段说明：**

| 字段 | 说明 | 示例 |
|------|------|------|
| `ne[4]` | 各维度大小 | `ne[0]=512, ne[1]=64` 表示 512x64 矩阵 |
| `nb[4]` | 各维度字节步长 | `nb[0]=4` 表示 F32 每元素4字节 |
| `src[]` | 父节点指针 | 构建计算图时自动填充 |
| `data` | 实际数据地址 | 通过 `ggml_get_data()` 访问 |

#### ggml_context —— 上下文管理器

```c
struct ggml_context;

struct ggml_init_params {
    size_t mem_size;    // 内存池大小（字节）
    void * mem_buffer;  // 外部内存缓冲区（可为NULL）
    bool   no_alloc;    // 是否延迟分配
};
```

**上下文生命周期：**

```c
// 1. 初始化参数
struct ggml_init_params params = {
    .mem_size   = 16*1024*1024,  // 16MB内存池
    .mem_buffer = NULL,          // 由GGML分配
    .no_alloc   = false,
};

// 2. 创建上下文
struct ggml_context * ctx = ggml_init(params);

// 3. 使用上下文创建张量、构建计算图...

// 4. 释放上下文
ggml_free(ctx);
```

#### ggml_cgraph —— 计算图

```c
struct ggml_cgraph {
    int n_nodes;                    // 节点数量
    int n_leafs;                    // 叶子节点数量
    struct ggml_tensor ** nodes;    // 计算节点数组
    struct ggml_tensor ** grads;    // 梯度节点数组（用于反向传播）
    struct ggml_tensor ** leafs;    // 叶子节点数组（输入/参数）
    void * visited_hash_table;      // 访问标记哈希表
    enum ggml_cgraph_eval_order order; // 计算顺序
};
```

### A.1.2 张量创建函数

| 函数 | 签名 | 说明 |
|------|------|------|
| `ggml_new_tensor` | `struct ggml_tensor * ggml_new_tensor(struct ggml_context * ctx, enum ggml_type type, int n_dims, const int64_t * ne)` | 创建通用N维张量 |
| `ggml_new_tensor_1d` | `struct ggml_tensor * ggml_new_tensor_1d(struct ggml_context * ctx, enum ggml_type type, int64_t ne0)` | 创建1D张量（向量） |
| `ggml_new_tensor_2d` | `struct ggml_tensor * ggml_new_tensor_2d(struct ggml_context * ctx, enum ggml_type type, int64_t ne0, int64_t ne1)` | 创建2D张量（矩阵） |
| `ggml_new_tensor_3d` | `struct ggml_tensor * ggml_new_tensor_3d(struct ggml_context * ctx, enum ggml_type type, int64_t ne0, int64_t ne1, int64_t ne2)` | 创建3D张量 |
| `ggml_new_tensor_4d` | `struct ggml_tensor * ggml_new_tensor_4d(struct ggml_context * ctx, enum ggml_type type, int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3)` | 创建4D张量 |
| `ggml_dup_tensor` | `struct ggml_tensor * ggml_dup_tensor(struct ggml_context * ctx, const struct ggml_tensor * src)` | 复制张量结构（不复制数据） |
| `ggml_view_tensor` | `struct ggml_tensor * ggml_view_tensor(struct ggml_context * ctx, struct ggml_tensor * src)` | 创建张量视图 |

**使用示例：**

```c
// 创建 512x768 的 F32 矩阵
struct ggml_tensor * mat = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 512, 768);

// 创建 4096 维向量
struct ggml_tensor * vec = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4096);

// 创建与现有张量同形状的新张量
struct ggml_tensor * mat2 = ggml_dup_tensor(ctx, mat);
```

### A.1.3 张量运算函数

#### 基础数学运算

| 函数 | 签名 | 说明 |
|------|------|------|
| `ggml_add` | `struct ggml_tensor * ggml_add(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b)` | 逐元素加法 `a + b` |
| `ggml_sub` | `struct ggml_tensor * ggml_sub(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b)` | 逐元素减法 `a - b` |
| `ggml_mul` | `struct ggml_tensor * ggml_mul(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b)` | 逐元素乘法 `a * b` |
| `ggml_div` | `struct ggml_tensor * ggml_div(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b)` | 逐元素除法 `a / b` |
| `ggml_sqr` | `struct ggml_tensor * ggml_sqr(struct ggml_context * ctx, struct ggml_tensor * a)` | 平方 `a^2` |
| `ggml_sqrt` | `struct ggml_tensor * ggml_sqrt(struct ggml_context * ctx, struct ggml_tensor * a)` | 平方根 `sqrt(a)` |
| `ggml_scale` | `struct ggml_tensor * ggml_scale(struct ggml_context * ctx, struct ggml_tensor * a, float s)` | 缩放 `a * s` |

#### 矩阵运算

| 函数 | 签名 | 说明 |
|------|------|------|
| `ggml_mul_mat` | `struct ggml_tensor * ggml_mul_mat(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b)` | 矩阵乘法 `a @ b` |
| `ggml_out_prod` | `struct ggml_tensor * ggml_out_prod(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b)` | 外积 |
| `ggml_transpose` | `struct ggml_tensor * ggml_transpose(struct ggml_context * ctx, struct ggml_tensor * a)` | 转置 |
| `ggml_permute` | `struct ggml_tensor * ggml_permute(struct ggml_context * ctx, struct ggml_tensor * a, int axis0, int axis1, int axis2, int axis3)` | 维度重排 |
| `ggml_reshape` | `struct ggml_tensor * ggml_reshape(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b)` | 重塑形状 |

**矩阵乘法示例：**

```c
// C = A @ B, A:[M,K], B:[K,N], C:[M,N]
struct ggml_tensor * A = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 512, 768);  // [768, 512]
struct ggml_tensor * B = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 768, 256);  // [256, 768]
struct ggml_tensor * C = ggml_mul_mat(ctx, A, B);  // [256, 512]
```

#### 神经网络专用运算

| 函数 | 签名 | 说明 |
|------|------|------|
| `ggml_silu` | `struct ggml_tensor * ggml_silu(struct ggml_context * ctx, struct ggml_tensor * a)` | SiLU激活函数 `x * sigmoid(x)` |
| `ggml_gelu` | `struct ggml_tensor * ggml_gelu(struct ggml_context * ctx, struct ggml_tensor * a)` | GELU激活函数 |
| `ggml_relu` | `struct ggml_tensor * ggml_relu(struct ggml_context * ctx, struct ggml_tensor * a)` | ReLU激活函数 `max(0, x)` |
| `ggml_soft_max` | `struct ggml_tensor * ggml_soft_max(struct ggml_context * ctx, struct ggml_tensor * a)` | Softmax归一化 |
| `ggml_rms_norm` | `struct ggml_tensor * ggml_rms_norm(struct ggml_context * ctx, struct ggml_tensor * a, float eps)` | RMS归一化 |
| `ggml_layer_norm` | `struct ggml_tensor * ggml_layer_norm(struct ggml_context * ctx, struct ggml_tensor * a, float eps)` | Layer归一化 |
| `ggml_rope` | `struct ggml_tensor * ggml_rope(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b, int n_dims, int mode, int n_ctx)` | RoPE位置编码 |

### A.1.4 计算图构建与执行

| 函数 | 签名 | 说明 |
|------|------|------|
| `ggml_new_graph` | `struct ggml_cgraph * ggml_new_graph(struct ggml_context * ctx, size_t size)` | 创建计算图 |
| `ggml_build_forward` | `void ggml_build_forward(struct ggml_cgraph * cgraph, struct ggml_tensor * tensor)` | 构建前向图 |
| `ggml_graph_compute` | `void ggml_graph_compute(struct ggml_cgraph * cgraph, struct ggml_cplan * cplan)` | 执行计算图 |
| `ggml_graph_reset` | `void ggml_graph_reset(struct ggml_cgraph * cgraph)` | 重置计算图 |
| `ggml_graph_print` | `void ggml_graph_print(const struct ggml_cgraph * cgraph)` | 打印计算图 |
| `ggml_graph_dump_dot` | `void ggml_graph_dump_dot(const struct ggml_cgraph * gb, const struct ggml_cgraph * gf, const char * filename)` | 导出为DOT格式 |

**完整计算流程示例：**

```c
// 1. 初始化上下文
struct ggml_init_params params = {
    .mem_size = 16*1024*1024,
    .mem_buffer = NULL,
};
struct ggml_context * ctx = ggml_init(params);

// 2. 创建张量
struct ggml_tensor * x = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 10);
struct ggml_tensor * w = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 10, 5);
struct ggml_tensor * b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 5);

// 3. 构建计算图
struct ggml_tensor * wx = ggml_mul_mat(ctx, w, x);
struct ggml_tensor * y = ggml_add(ctx, wx, b);
struct ggml_tensor * out = ggml_silu(ctx, y);

// 4. 创建并构建计算图
struct ggml_cgraph * gf = ggml_new_graph(ctx, GGML_DEFAULT_GRAPH_SIZE);
ggml_build_forward(gf, out);

// 5. 设置输入数据
float x_data[10] = {1.0f, 2.0f, 3.0f, ...};
memcpy(x->data, x_data, sizeof(x_data));

// 6. 执行计算
ggml_graph_compute(gf, NULL);

// 7. 获取结果
float result = ggml_get_f32_1d(out, 0);

// 8. 清理
ggml_free(ctx);
```

### A.1.5 后端系统 API

#### 后端设备管理

| 函数 | 签名 | 说明 |
|------|------|------|
| `ggml_backend_dev_count` | `size_t ggml_backend_dev_count(void)` | 获取可用设备数量 |
| `ggml_backend_dev_get` | `ggml_backend_dev_t ggml_backend_dev_get(size_t index)` | 获取指定索引设备 |
| `ggml_backend_dev_by_name` | `ggml_backend_dev_t ggml_backend_dev_by_name(const char * name)` | 按名称获取设备 |
| `ggml_backend_dev_name` | `const char * ggml_backend_dev_name(ggml_backend_dev_t device)` | 获取设备名称 |
| `ggml_backend_dev_type` | `enum ggml_backend_dev_type ggml_backend_dev_type(ggml_backend_dev_t device)` | 获取设备类型 |

#### 后端缓冲区管理

| 函数 | 签名 | 说明 |
|------|------|------|
| `ggml_backend_alloc_buffer` | `ggml_backend_buffer_t ggml_backend_alloc_buffer(ggml_backend_t backend, size_t size)` | 分配后端缓冲区 |
| `ggml_backend_buffer_get_base` | `void * ggml_backend_buffer_get_base(ggml_backend_buffer_t buffer)` | 获取缓冲区基地址 |
| `ggml_backend_buffer_clear` | `void ggml_backend_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value)` | 清空缓冲区 |

#### 后端调度器

| 函数 | 签名 | 说明 |
|------|------|------|
| `ggml_backend_sched_new` | `ggml_backend_sched_t ggml_backend_sched_new(ggml_backend_t * backends, ggml_backend_buffer_type_t * bufts, int n_backends, size_t graph_size, bool parallel, bool op_offload)` | 创建调度器 |
| `ggml_backend_sched_alloc_graph` | `bool ggml_backend_sched_alloc_graph(ggml_backend_sched_t sched, struct ggml_cgraph * graph)` | 为图分配内存 |
| `ggml_backend_sched_graph_compute` | `enum ggml_status ggml_backend_sched_graph_compute(ggml_backend_sched_t sched, struct ggml_cgraph * graph)` | 执行计算 |

---

## A.2 Llama C API 参考

### A.2.1 核心数据结构

#### llama_model_params —— 模型加载参数

```c
typedef struct llama_model_params {
    ggml_backend_dev_t * devices;                    // 设备列表
    const struct llama_model_tensor_buft_override * tensor_buft_overrides; // 张量缓冲区类型覆盖
    int32_t n_gpu_layers;                            // GPU层数 (-1=全部)
    enum llama_split_mode split_mode;                // 多GPU分割模式
    int32_t main_gpu;                                // 主GPU索引
    const float * tensor_split;                      // 张量分割比例
    llama_progress_callback progress_callback;       // 加载进度回调
    void * progress_callback_user_data;              // 回调用户数据
    const struct llama_model_kv_override * kv_overrides; // KV覆盖
    bool vocab_only;                                 // 仅加载词表
    bool use_mmap;                                   // 使用内存映射
    bool use_direct_io;                              // 使用直接IO
    bool use_mlock;                                  // 使用mlock锁定内存
    bool check_tensors;                              // 验证张量数据
    bool use_extra_bufts;                            // 使用额外缓冲区类型
    bool no_host;                                    // 绕过主机缓冲区
    bool no_alloc;                                   // 仅加载元数据
} llama_model_params;
```

#### llama_context_params —— 上下文参数

```c
typedef struct llama_context_params {
    uint32_t n_ctx;                                  // 上下文长度
    uint32_t n_batch;                                // 逻辑批大小
    uint32_t n_ubatch;                               // 物理批大小
    uint32_t n_seq_max;                              // 最大序列数
    int32_t  n_threads;                              // 生成线程数
    int32_t  n_threads_batch;                        // 批处理线程数
    enum llama_rope_scaling_type rope_scaling_type;  // RoPE缩放类型
    enum llama_pooling_type      pooling_type;       // 池化类型
    enum llama_attention_type    attention_type;     // 注意力类型
    enum llama_flash_attn_type   flash_attn_type;    // Flash Attention类型
    float    rope_freq_base;                         // RoPE基础频率
    float    rope_freq_scale;                        // RoPE频率缩放
    float    yarn_ext_factor;                        // YaRN外推因子
    float    yarn_attn_factor;                       // YaRN注意力因子
    float    yarn_beta_fast;                         // YaRN快速beta
    float    yarn_beta_slow;                         // YaRN慢速beta
    uint32_t yarn_orig_ctx;                          // YaRN原始上下文
    ggml_backend_sched_eval_callback cb_eval;        // 评估回调
    void * cb_eval_user_data;                        // 回调用户数据
    enum ggml_type type_k;                           // K缓存数据类型
    enum ggml_type type_v;                           // V缓存数据类型
    ggml_abort_callback abort_callback;              // 中止回调
    void *              abort_callback_data;         // 中止回调数据
    bool embeddings;                                 // 提取嵌入
    bool offload_kqv;                                // 卸载KQV到GPU
    bool no_perf;                                    // 禁用性能计时
    bool op_offload;                                 // 卸载操作到设备
    bool swa_full;                                   // 使用完整SWA缓存
    bool kv_unified;                                 // 统一KV缓存
    struct llama_sampler_seq_config * samplers;      // 采样器配置
    size_t                            n_samplers;    // 采样器数量
} llama_context_params;
```

#### llama_batch —— 批处理输入

```c
typedef struct llama_batch {
    int32_t n_tokens;           // Token数量
    llama_token  *  token;      // Token ID数组
    float        *  embd;       // 嵌入向量（当token为NULL时使用）
    llama_pos    *  pos;        // 位置数组
    int32_t      *  n_seq_id;   // 每个token的序列ID数量
    llama_seq_id ** seq_id;     // 序列ID数组
    int8_t       *  logits;     // 是否输出logits
} llama_batch;
```

### A.2.2 模型加载与初始化

| 函数 | 签名 | 说明 |
|------|------|------|
| `llama_model_default_params` | `struct llama_model_params llama_model_default_params(void)` | 获取默认模型参数 |
| `llama_model_load_from_file` | `struct llama_model * llama_model_load_from_file(const char * path_model, struct llama_model_params params)` | 从文件加载模型 |
| `llama_model_load_from_splits` | `struct llama_model * llama_model_load_from_splits(const char ** paths, size_t n_paths, struct llama_model_params params)` | 从分片加载模型 |
| `llama_model_free` | `void llama_model_free(struct llama_model * model)` | 释放模型 |
| `llama_context_default_params` | `struct llama_context_params llama_context_default_params(void)` | 获取默认上下文参数 |
| `llama_init_from_model` | `struct llama_context * llama_init_from_model(struct llama_model * model, struct llama_context_params params)` | 从模型创建上下文 |
| `llama_free` | `void llama_free(struct llama_context * ctx)` | 释放上下文 |

**模型加载示例：**

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
```

### A.2.3 推理 API

| 函数 | 签名 | 说明 |
|------|------|------|
| `llama_encode` | `int32_t llama_encode(struct llama_context * ctx, struct llama_batch batch)` | 编码（用于encoder-decoder模型） |
| `llama_decode` | `int32_t llama_decode(struct llama_context * ctx, struct llama_batch batch)` | 解码/推理 |
| `llama_get_logits` | `float * llama_get_logits(struct llama_context * ctx)` | 获取logits |
| `llama_get_logits_ith` | `float * llama_get_logits_ith(struct llama_context * ctx, int32_t i)` | 获取第i个token的logits |
| `llama_get_embeddings` | `float * llama_get_embeddings(struct llama_context * ctx)` | 获取嵌入向量 |
| `llama_get_embeddings_ith` | `float * llama_get_embeddings_ith(struct llama_context * ctx, int32_t i)` | 获取第i个token的嵌入 |

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
}

// 获取logits
float * logits = llama_get_logits(ctx);
int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));

// 找到概率最高的token
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

### A.2.4 分词 API

| 函数 | 签名 | 说明 |
|------|------|------|
| `llama_tokenize` | `int32_t llama_tokenize(const struct llama_vocab * vocab, const char * text, int32_t text_len, llama_token * tokens, int32_t n_tokens_max, bool add_special, bool parse_special)` | 文本分词 |
| `llama_detokenize` | `int32_t llama_detokenize(const struct llama_vocab * vocab, const llama_token * tokens, int32_t n_tokens, char * text, int32_t text_len_max, bool remove_special, bool unparse_special)` | Token转文本 |
| `llama_token_to_piece` | `int32_t llama_token_to_piece(const struct llama_vocab * vocab, llama_token token, char * buf, int32_t length, int32_t lstrip, bool special)` | Token转片段 |

**分词示例：**

```c
const char * text = "Hello, world!";
const struct llama_vocab * vocab = llama_model_get_vocab(model);

// 分词
int n_tokens = llama_tokenize(vocab, text, strlen(text), 
                              NULL, 0, true, false);
llama_token * tokens = malloc(n_tokens * sizeof(llama_token));
llama_tokenize(vocab, text, strlen(text), 
               tokens, n_tokens, true, false);

// 打印token
for (int i = 0; i < n_tokens; i++) {
    char buf[32];
    llama_token_to_piece(vocab, tokens[i], buf, sizeof(buf), 0, false);
    printf("%d: %d -> '%s'\n", i, tokens[i], buf);
}

free(tokens);
```

### A.2.5 采样 API

| 函数 | 签名 | 说明 |
|------|------|------|
| `llama_sampler_chain_init` | `struct llama_sampler * llama_sampler_chain_init(struct llama_sampler_chain_params params)` | 初始化采样链 |
| `llama_sampler_chain_add` | `void llama_sampler_chain_add(struct llama_sampler * chain, struct llama_sampler * smpl)` | 添加采样器到链 |
| `llama_sampler_sample` | `llama_token llama_sampler_sample(struct llama_sampler * smpl, struct llama_context * ctx, int32_t idx)` | 采样 |
| `llama_sampler_accept` | `void llama_sampler_accept(struct llama_sampler * smpl, llama_token token)` | 接受token |
| `llama_sampler_free` | `void llama_sampler_free(struct llama_sampler * smpl)` | 释放采样器 |

**采样器类型：**

| 函数 | 说明 | 参数 |
|------|------|------|
| `llama_sampler_init_greedy` | 贪心采样 | 无 |
| `llama_sampler_init_dist` | 随机分布采样 | seed |
| `llama_sampler_init_top_k` | Top-K采样 | k |
| `llama_sampler_init_top_p` | Top-P (Nucleus)采样 | p, min_keep |
| `llama_sampler_init_min_p` | Min-P采样 | p, min_keep |
| `llama_sampler_init_temp` | 温度缩放 | t |
| `llama_sampler_init_penalties` | 重复惩罚 | penalty_last_n, penalty_repeat, penalty_freq, penalty_present |
| `llama_sampler_init_grammar` | 语法约束 | vocab, grammar_str, grammar_root |

**采样示例：**

```c
// 创建采样链
struct llama_sampler_chain_params sparams = llama_sampler_chain_default_params();
struct llama_sampler * smpl = llama_sampler_chain_init(sparams);

// 添加采样器
llama_sampler_chain_add(smpl, llama_sampler_init_top_k(40));
llama_sampler_chain_add(smpl, llama_sampler_init_top_p(0.9, 1));
llama_sampler_chain_add(smpl, llama_sampler_init_temp(0.8));
llama_sampler_chain_add(smpl, llama_sampler_init_dist(12345));

// 采样
llama_token next_token = llama_sampler_sample(smpl, ctx, -1);

// 接受token
llama_sampler_accept(smpl, next_token);

// 清理
llama_sampler_free(smpl);
```

---

## A.3 Common 库 API 参考

### A.3.1 参数解析

| 结构体/函数 | 说明 |
|-------------|------|
| `struct common_params` | 通用参数结构体，包含所有CLI参数 |
| `common_params_parse` | 解析命令行参数 |
| `common_params_to_llama` | 转换参数为llama格式 |

**common_params 关键字段：**

```c
struct common_params {
    int32_t n_predict;          // 预测token数 (-1=无限制)
    int32_t n_ctx;              // 上下文大小
    int32_t n_batch;            // 批大小
    int32_t n_gpu_layers;       // GPU层数
    int32_t n_threads;          // 线程数
    float   rope_freq_base;     // RoPE基础频率
    float   temp;               // 温度
    int32_t top_k;              // Top-K
    float   top_p;              // Top-P
    std::string model;          // 模型路径
    std::string prompt;         // 提示词
    // ... 更多参数
};
```

### A.3.2 采样封装

| 函数 | 说明 |
|------|------|
| `common_sampler_init` | 初始化通用采样器 |
| `common_sampler_sample` | 执行采样 |
| `common_sampler_accept` | 接受token |
| `common_sampler_free` | 释放采样器 |

### A.3.3 工具函数

| 函数 | 说明 |
|------|------|
| `common_tokenize` | 分词工具函数 |
| `common_detokenize` | 反分词工具函数 |
| `common_token_to_piece` | Token转文本片段 |
| `common_init` | 初始化common库 |

---

## A.4 API 使用最佳实践

### A.4.1 内存管理

1. **上下文内存池**：预先分配足够大的内存池，避免运行时分配
2. **及时释放**：使用完毕后立即释放上下文和模型
3. **避免重复创建**：复用上下文进行多次推理

### A.4.2 性能优化

1. **批处理**：使用 `llama_batch` 进行批量推理
2. **GPU卸载**：合理设置 `n_gpu_layers` 利用GPU加速
3. **线程配置**：根据CPU核心数调整 `n_threads` 和 `n_threads_batch`

### A.4.3 错误处理

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

### A.4.4 线程安全

- `llama_model` 是线程安全的，可被多个 `llama_context` 共享
- `llama_context` **不是**线程安全的，每个线程应使用独立上下文
- 分词 API 是线程安全的

---

## A.5 扩展 API 参考 (llama-ext.h)

`llama-ext.h` 提供了llama.cpp的**扩展API**，包括计算图预留、量化工具函数等。这些API通常用于高级用例和内部工具开发。

### A.5.1 计算图预留

**函数**：`llama_graph_reserve`

**用途**：预留一个新的计算图，有效期直到下次调用

```c
LLAMA_API struct ggml_cgraph * llama_graph_reserve(
    struct llama_context * ctx,
    uint32_t n_tokens,      // token数量
    uint32_t n_seqs,        // 序列数量
    uint32_t n_outputs);    // 输出数量
```

**使用场景**：
- 预分配计算图以评估内存需求
- 在不实际运行推理的情况下测试图构建

**示例**：

```c
// 预留计算图
struct ggml_cgraph * graph = llama_graph_reserve(ctx, 512, 1, 1);
if (graph == NULL) {
    // 预留失败，可能需要减少token数
}

// 图在下次llama_graph_reserve调用前有效
// 或在llama_decode等操作后失效
```

### A.5.2 量化类型工具

**llama_ftype_get_default_type**

获取指定量化格式对应的GGML数据类型：

```c
LLAMA_API ggml_type llama_ftype_get_default_type(llama_ftype ftype);

// 使用示例
ggml_type type = llama_ftype_get_default_type(LLAMA_FTYPE_MOSTLY_Q4_0);
// 返回 GGML_TYPE_Q4_0
```

**量化状态管理**

```c
// 初始化量化状态
LLAMA_API quantize_state_impl * llama_quant_init(
    const llama_model * model,
    const llama_model_quantize_params * params);

// 释放量化状态
LLAMA_API void llama_quant_free(quantize_state_impl * qs);
```

### A.5.3 量化模型描述

**llama_quant_model_desc** - 用于测试的模型描述结构：

```c
struct llama_quant_model_desc {
    const char * architecture;  // 架构名称（如"llama"）
    uint32_t n_embd;            // 嵌入维度
    uint32_t n_ff;              // FFN中间层维度
    uint32_t n_layer;           // 层数
    uint32_t n_head;            // 注意力头数
    uint32_t n_head_kv;         // KV头数
    uint32_t n_expert;          // 专家数量（MoE）
    uint32_t n_embd_head_k;     // K头维度
    uint32_t n_embd_head_v;     // V头维度
};

// 从描述创建模拟模型（用于测试）
LLAMA_API llama_model * llama_quant_model_from_metadata(
    const llama_quant_model_desc * desc);
```

**使用场景**：
- 在没有实际模型文件的情况下测试量化逻辑
- 验证量化参数对不同架构的影响

```c
// 创建测试模型描述
llama_quant_model_desc desc = {
    .architecture = "llama",
    .n_embd = 4096,
    .n_ff = 11008,
    .n_layer = 32,
    .n_head = 32,
    .n_head_kv = 32,
    .n_expert = 0,
    .n_embd_head_k = 128,
    .n_embd_head_v = 128
};

// 创建模拟模型
llama_model * model = llama_quant_model_from_metadata(&desc);

// 测试量化
quantize_state_impl * qs = llama_quant_init(model, &params);
// ...
llama_quant_free(qs);
llama_model_free(model);
```

### A.5.4 张量量化判断

**llama_quant_tensor_allows_quantization**

判断指定张量是否允许量化：

```c
LLAMA_API bool llama_quant_tensor_allows_quantization(
    const quantize_state_impl * qs,
    const ggml_tensor * tensor);
```

**判断依据**：
- 张量名称（某些张量如bias通常不量化）
- 张量维度（小张量可能不值得量化）
- 量化参数配置

### A.5.5 量化类型计算

**llama_quant_compute_types**

为一组张量计算量化类型分配：

```c
LLAMA_API void llama_quant_compute_types(
    const quantize_state_impl * qs,
    llama_ftype ftype,
    ggml_tensor ** tensors,      // 输入张量数组
    ggml_type * result_types,    // 输出类型数组（调用者分配）
    size_t n_tensors);           // 张量数量
```

**工作流程**：

```
┌─────────────────────────────────────────┐
│  llama_quant_compute_types              │
├─────────────────────────────────────────┤
│                                         │
│  输入: tensors[] + ftype                │
│       │                                 │
│       ▼                                 │
│  遍历每个张量                           │
│       │                                 │
│       ├───→ 检查是否允许量化           │
│       │        │                        │
│       │        ├───→ 不允许: 保持F32    │
│       │        │                        │
│       │        └───→ 允许: 选择量化类型 │
│       │                                 │
│       ▼                                 │
│  输出: result_types[]                   │
│       [Q4_0, F32, Q6_K, Q4_0, ...]      │
│                                         │
└─────────────────────────────────────────┘
```

**示例代码**：

```c
// 收集模型中的所有张量
std::vector<ggml_tensor *> tensors;
// ... 填充tensors ...

// 分配输出数组
std::vector<ggml_type> result_types(tensors.size());

// 计算每个张量的量化类型
llama_quant_compute_types(qs, LLAMA_FTYPE_MOSTLY_Q4_K_M, 
                          tensors.data(), result_types.data(), 
                          tensors.size());

// 查看结果
for (size_t i = 0; i < tensors.size(); i++) {
    printf("%s: %s -> %s\n",
           tensors[i]->name,
           ggml_type_name(tensors[i]->type),
           ggml_type_name(result_types[i]));
}
```

### A.5.6 扩展API使用注意事项

1. **稳定性**：扩展API可能在未来版本中变化，生产代码建议使用稳定API
2. **头文件**：使用扩展API需要包含 `llama-ext.h`
3. **链接**：某些扩展功能可能需要链接额外的库
4. **线程安全**：扩展API的线程安全性与对应的标准API一致

### A.5.7 扩展API速查表

| 函数 | 用途 | 稳定版本 |
|------|------|----------|
| `llama_graph_reserve` | 预留计算图 | 是 |
| `llama_ftype_get_default_type` | 获取默认量化类型 | 是 |
| `llama_quant_init` | 初始化量化状态 | 是 |
| `llama_quant_free` | 释放量化状态 | 是 |
| `llama_quant_model_from_metadata` | 创建模拟模型 | 是 |
| `llama_quant_tensor_allows_quantization` | 判断可否量化 | 是 |
| `llama_quant_compute_types` | 计算量化类型 | 是 |
