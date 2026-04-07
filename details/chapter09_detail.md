# 第9章 计算图构建（llama_graph） —— Transformer的"动态组装工厂"

## 学习目标
1. 理解llama_build_graph的架构设计
2. 掌握Transformer层计算图的构建过程
3. 了解图优化技术（算子融合、常量折叠）
4. 能调试和可视化计算图

---

## 生活类比：乐高积木的自动化工厂

想象llama_graph是一家**智能乐高工厂**：

- **计算图构建器** = 自动组装机器人
  - 根据"设计图纸"（模型架构配置）
  - 从"零件仓库"（ggml_context）取零件
  - 按"组装说明书"（Transformer算法）搭建
- **ggml_context** = 零件仓库
  - 预先准备好各种尺寸的积木块（张量）
  - 按需求分配存储空间
- **计算节点** = 加工工位
  - 词嵌入党：把token变成向量
  - Transformer层：特征提取流水线
  - 输出口：生成下一个token的概率分布
- **图优化** = 智能排产系统
  - 把相邻的小工序合并（算子融合）
  - 预计算固定值（常量折叠）
  - 去除无用步骤（死代码消除）

就像工厂需要精确安排每个零件的加工顺序，llama_graph需要精确构建每个张量的计算依赖。

---

## 源码地图

```
src/llama-graph.h
├── llm_build_context          # 图构建上下文
├── llm_build_llama()          # Llama架构构建
├── llm_build_transformer()    # Transformer层构建
└── llm_build_ffn()            # FFN层构建

src/llama-graph.cpp
├── 主构建函数（第1-500行）
│   └── llama_build_graph()
├── 架构特定构建（第500-2000行）
│   ├── llm_build_llama()
│   ├── llm_build_mistral()
│   ├── llm_build_qwen()
│   └── ...
├── Transformer层构建（第2000-3000行）
│   ├── llm_build_attn()       # 注意力
│   └── llm_build_ffn()        # 前馈网络
└── 工具函数（第3000-4000行）
    └── ggml_graph_*()

ggml/src/ggml.c
├── 图构建（第15000-16000行）
│   └── ggml_build_forward()
├── 图执行（第16000-17000行）
│   └── ggml_graph_compute()
└── 图调试（第17000-18000行）
    ├── ggml_graph_print()
    └── ggml_graph_dump_dot()
```

---

## 9.1 推理计算图设计

### 9.1.1 图构建器架构

**源码位置**：`src/llama-graph.h` (第1-100行)

```cpp
// 图构建上下文
struct llm_build_context {
    // 输入
    const llama_hparams & hparams;    // 模型超参数
    const llama_model   & model;      // 模型数据
    const llama_batch   & batch;      // 输入批次
    const llama_kv_cache& kv_cache;   // KV缓存

    // GGML上下文（张量仓库）
    struct ggml_context * ctx0;

    // 构建的计算图
    struct ggml_cgraph * graph;

    // 位置编码参数
    float freq_scale;
    float freq_base;
    float ext_factor;
    float attn_factor;
    float beta_fast;
    float beta_slow;
};

// 主构建函数入口
struct ggml_cgraph * llama_build_graph(
        llama_context * lctx,
        const llama_batch & batch,
        bool worst_case);  // 为最坏情况分配（最大batch）
```

### 9.1.2 图构建流程概览

```
┌─────────────────────────────────────────────────────────────┐
│                    llama_build_graph                        │
├─────────────────────────────────────────────────────────────┤
│  ① 初始化ggml_context                                       │
│     - 估算需要的内存                                         │
│     - 创建临时计算缓冲区                                     │
├─────────────────────────────────────────────────────────────┤
│  ② 选择架构特定的构建函数                                    │
│     - Llama架构 → llm_build_llama()                         │
│     - Mistral架构 → llm_build_mistral()                     │
│     - Qwen架构 → llm_build_qwen()                           │
├─────────────────────────────────────────────────────────────┤
│  ③ 构建输入层                                                │
│     - token嵌入查找                                          │
│     - 位置编码（RoPE）                                       │
├─────────────────────────────────────────────────────────────┤
│  ④ 构建Transformer层（循环n_layer次）                        │
│     - 每层：Attention → FFN → 残差连接                      │
├─────────────────────────────────────────────────────────────┤
│  ⑤ 构建输出层                                                │
│     - 最终归一化                                             │
│     - LM Head（映射到词表）                                  │
│     - Softmax生成概率                                        │
├─────────────────────────────────────────────────────────────┤
│  ⑥ 构建前向图                                                │
│     - ggml_build_forward()                                   │
│     - 拓扑排序所有节点                                       │
├─────────────────────────────────────────────────────────────┤
│  ⑦ 返回计算图                                                │
└─────────────────────────────────────────────────────────────┘
```

---

## 9.2 前向传播实现

### 9.2.1 Llama架构构建入口

**源码位置**：`src/llama-graph.cpp` (第500-800行)

```cpp
// Llama架构计算图构建
static struct ggml_cgraph * llm_build_llama(
        llama_context * lctx,
        const llama_batch & batch) {

    struct llm_build_context * bctx = (struct llm_build_context *)lctx;
    const llama_hparams & hparams = bctx->hparams;
    const int n_layer = hparams.n_layer;

    // ① 创建输入嵌入
    struct ggml_tensor * inp_tokens = ggml_new_tensor_1d(
        bctx->ctx0, GGML_TYPE_I32, batch.n_tokens);

    // 词嵌入查找
    struct ggml_tensor * inpL = ggml_get_rows(bctx->ctx0,
        bctx->model.tok_embd,  // 词嵌入矩阵
        inp_tokens);            // token索引

    // ② 遍历每一层Transformer
    for (int il = 0; il < n_layer; il++) {
        struct ggml_tensor * cur = inpL;

        // --- Attention层 ---
        {
            // 输入归一化
            cur = ggml_rms_norm(bctx->ctx0, cur, hparams.f_norm_rms_eps);
            bctx->inpSA = cur;  // 保存用于后续

            // Q投影
            struct ggml_tensor * Q = ggml_mul_mat(bctx->ctx0,
                bctx->model.layers[il].wq, cur);

            // K投影
            struct ggml_tensor * K = ggml_mul_mat(bctx->ctx0,
                bctx->model.layers[il].wk, cur);

            // V投影
            struct ggml_tensor * V = ggml_mul_mat(bctx->ctx0,
                bctx->model.layers[il].wv, cur);

            // 应用RoPE位置编码
            Q = ggml_rope(bctx->ctx0, Q, ...);
            K = ggml_rope(bctx->ctx0, K, ...);

            // 注意力计算（见下文详细解析）
            struct ggml_tensor * attn_out = llm_build_attn(Q, K, V, ...);

            // 输出投影
            cur = ggml_mul_mat(bctx->ctx0,
                bctx->model.layers[il].wo, attn_out);

            // 残差连接
            cur = ggml_add(bctx->ctx0, cur, inpL);
        }

        // --- FFN层 ---
        {
            struct ggml_tensor * tmp = cur;

            // 归一化
            cur = ggml_rms_norm(bctx->ctx0, cur, hparams.f_norm_rms_eps);

            // SwiGLU: gate = sigmoid(Wg @ x) * (Wu @ x)
            struct ggml_tensor * gate = ggml_mul_mat(bctx->ctx0,
                bctx->model.layers[il].wgate, cur);
            gate = ggml_silu(bctx->ctx0, gate);

            struct ggml_tensor * up = ggml_mul_mat(bctx->ctx0,
                bctx->model.layers[il].wup, cur);

            cur = ggml_mul(bctx->ctx0, gate, up);

            // 下投影
            cur = ggml_mul_mat(bctx->ctx0,
                bctx->model.layers[il].wdown, cur);

            // 残差连接
            cur = ggml_add(bctx->ctx0, cur, tmp);
        }

        inpL = cur;  // 下一层的输入
    }

    // ③ 输出层
    {
        // 最终归一化
        inpL = ggml_rms_norm(bctx->ctx0, inpL, hparams.f_norm_rms_eps);

        // LM Head：映射到词表大小
        struct ggml_tensor * logits = ggml_mul_mat(bctx->ctx0,
            bctx->model.output, inpL);

        // 提取目标token的logits
        bctx->inpl_logits = logits;
    }

    // ④ 构建前向图
    struct ggml_cgraph * graph = ggml_build_forward(bctx->inpl_logits);

    return graph;
}
```

### 9.2.2 多头注意力实现

**源码位置**：`src/llama-graph.cpp` (第2000-2300行)

```cpp
// 注意力层构建
struct ggml_tensor * llm_build_attn(
        struct llm_build_context * bctx,
        struct ggml_tensor * Q,
        struct ggml_tensor * K,
        struct ggml_tensor * V,
        int il) {  // 层索引

    const llama_hparams & hparams = bctx->hparams;
    const int n_head = hparams.n_head;
    const int n_head_kv = hparams.n_head_kv;
    const int n_embd_head = hparams.n_embd_head();

    // ① 维度重塑：分离head维度
    // Q: [n_embd, n_tokens] -> [n_embd_head, n_head, n_tokens]
    Q = ggml_reshape_3d(bctx->ctx0, Q, n_embd_head, n_head, Q->ne[1]);
    K = ggml_reshape_3d(bctx->ctx0, K, n_embd_head, n_head_kv, K->ne[1]);
    V = ggml_reshape_3d(bctx->ctx0, V, n_embd_head, n_head_kv, V->ne[1]);

    // ② KV缓存更新
    // 将当前K,V追加到KV缓存中
    struct ggml_tensor * k_cache = bctx->kv_cache.k_l[il];
    struct ggml_tensor * v_cache = bctx->kv_cache.v_l[il];

    // K = [K_cache, K_new]
    K = ggml_kv_cache_concat(bctx->ctx0, k_cache, K);
    V = ggml_kv_cache_concat(bctx->ctx0, v_cache, V);

    // ③ 计算注意力分数: Q @ K^T / sqrt(d_k)
    struct ggml_tensor * KQ = ggml_mul_mat(bctx->ctx0, K, Q);
    KQ = ggml_scale(bctx->ctx0, KQ, 1.0f / sqrtf(n_embd_head));

    // ④ 应用注意力掩码（causal mask）
    // 防止attend到未来的token
    KQ = ggml_diag_mask_inf(bctx->ctx0, KQ, 0);  // 上三角设为-inf

    // ⑤ Softmax归一化
    KQ = ggml_soft_max(bctx->ctx0, KQ);

    // ⑥ 加权求和: Attention @ V
    struct ggml_tensor * KQV = ggml_mul_mat(bctx->ctx0, V, KQ);

    // ⑦ 重塑回原始维度
    // [n_embd_head, n_head, n_tokens] -> [n_embd, n_tokens]
    KQV = ggml_reshape_2d(bctx->ctx0, KQV, hparams.n_embd, KQV->ne[2]);

    return KQV;
}
```

**注意力计算流程图解**：
```
输入: Q, K, V (每个形状: [n_embd, n_tokens])

Step 1: 分离head
    Q -> reshape -> [head_dim, n_head, n_tokens]
    K -> reshape -> [head_dim, n_kv_head, n_tokens]
    V -> reshape -> [head_dim, n_kv_head, n_tokens]

Step 2: 拼接KV缓存
    K_cache + K -> [head_dim, n_kv_head, n_cache + n_tokens]
    V_cache + V -> [head_dim, n_kv_head, n_cache + n_tokens]

Step 3: 计算注意力分数
    Q @ K^T -> [n_head, n_tokens, n_cache + n_tokens]

Step 4: 掩码 + Softmax
    -> 应用causal mask
    -> softmax -> 概率分布

Step 5: 加权求和
    attention @ V -> [head_dim, n_head, n_tokens]

Step 6: 合并heads
    -> reshape -> [n_embd, n_tokens]
```

### 9.2.3 FFN层实现

**源码位置**：`src/llama-graph.cpp` (第2500-2700行)

```cpp
// SwiGLU FFN构建
struct ggml_tensor * llm_build_ffn(
        struct llm_build_context * bctx,
        struct ggml_tensor * cur,
        int il) {

    const llama_hparams & hparams = bctx->hparams;
    const llama_layer & layer = bctx->model.layers[il];

    // SwiGLU结构: Swish(Wg @ x) * (Wu @ x) @ Wd

    // 门控分支
    struct ggml_tensor * gate = ggml_mul_mat(bctx->ctx0, layer.wgate, cur);
    gate = ggml_silu(bctx->ctx0, gate);  // SiLU激活

    // 上采样分支
    struct ggml_tensor * up = ggml_mul_mat(bctx->ctx0, layer.wup, cur);

    // 逐元素相乘（门控）
    struct ggml_tensor * activated = ggml_mul(bctx->ctx0, gate, up);

    // 下投影
    struct ggml_tensor * output = ggml_mul_mat(bctx->ctx0, layer.wdown, activated);

    return output;
}
```

---

## 9.3 图优化技术

### 9.3.1 算子融合

**原理**：将相邻的轻量算子合并为一个kernel，减少内存访问。

```cpp
// 融合前（两次内存读写）:
// tmp = x + y
// z = tmp * scale

// 融合后（一次内存读写）:
// z = (x + y) * scale

// GGML中的融合算子示例:
ggml_add_mul()      // 加法+乘法融合
ggml_scale_inplace() // 原地缩放
ggml_silu_mul()     // SiLU+乘法融合（SwiGLU优化）
```

**源码位置**：`ggml/src/ggml.c` - 融合算子实现

### 9.3.2 常量折叠

**原理**：预计算不依赖于输入的常量表达式。

```cpp
// 可被折叠的情况:
// - 位置编码表（RoPE sin/cos值）
// - 注意力掩码（causal mask）
// - 归一化的epsilon偏移

// 实现:
// 在图构建时计算，而不是每次推理
struct ggml_tensor * rope_cache = precompute_rope_cache(...);
// 后续重复使用，避免重复计算sin/cos
```

### 9.3.3 死代码消除

**原理**：移除不会被输出的计算节点。

```cpp
// 示例：如果只需要最后一个token的logits
// 中间的token计算可以被优化（但需谨慎）

// GGML的贪心执行策略自动处理:
// - 只计算graph->output需要的节点
// - 不被引用的中间结果不计算
```

---

## 9.4 调试与可视化

### 9.4.1 打印计算图

**源码位置**：`ggml/src/ggml.c` (第17000-17200行)

```cpp
// 打印计算图信息
void ggml_graph_print(const struct ggml_cgraph * graph) {
    printf("===== GGML Computation Graph =====\n");
    printf("Nodes: %d\n", graph->n_nodes);
    printf("Leafs: %d\n", graph->n_leafs);

    printf("\n--- Leaf Nodes (Inputs/Params) ---\n");
    for (int i = 0; i < graph->n_leafs; i++) {
        struct ggml_tensor * node = graph->leafs[i];
        printf("  %s: [%s] shape=[%lld, %lld, %lld]\n",
            node->name,
            ggml_type_name(node->type),
            node->ne[0], node->ne[1], node->ne[2]);
    }

    printf("\n--- Compute Nodes ---\n");
    for (int i = 0; i < graph->n_nodes; i++) {
        struct ggml_tensor * node = graph->nodes[i];
        printf("  [%d] %s: %s -> [%lld, %lld, %lld]\n",
            i,
            node->name,
            ggml_op_name(node->op),
            node->ne[0], node->ne[1], node->ne[2]);
    }
}
```

### 9.4.2 导出DOT图

**源码位置**：`ggml/src/ggml.c` (第17200-17500行)

```cpp
// 导出Graphviz DOT格式
void ggml_graph_dump_dot(const struct ggml_cgraph * graph,
                         const struct ggml_cgraph * gb_grad,
                         const char * filename) {
    FILE * fp = fopen(filename, "w");
    fprintf(fp, "digraph G {\n");

    // 绘制节点
    for (int i = 0; i < graph->n_nodes; i++) {
        struct ggml_tensor * node = graph->nodes[i];
        fprintf(fp, "  \"%s\" [label=\"%s|%s\", shape=record];\n",
            node->name, node->name, ggml_op_name(node->op));
    }

    // 绘制边
    for (int i = 0; i < graph->n_nodes; i++) {
        struct ggml_tensor * node = graph->nodes[i];
        for (int j = 0; j < GGML_MAX_SRC && node->src[j]; j++) {
            fprintf(fp, "  \"%s\" -> \"%s\";\n",
                node->src[j]->name, node->name);
        }
    }

    fprintf(fp, "}\n");
    fclose(fp);
}
```

**使用方式**：
```bash
# 导出DOT文件
dot -Tpng graph.dot -o graph.png  # 转换为PNG图片
```

---

## 动手练习

### 练习1：阅读图构建代码
阅读 `src/llama-graph.cpp` 第500-1000行，回答：
1. 如何根据batch大小调整计算图？
2. 每层的输入输出是如何连接的？
3. RoPE参数是如何传入的？

### 练习2：计算图节点统计
编写程序统计一个Llama2-7B模型的计算图：
1. 总共有多少计算节点？
2. 每种类型的算子（op）各有多少个？
3. 最大的张量形状是什么？

### 练习3：可视化计算图
修改 `examples/simple/simple.cpp`，添加计算图导出功能：
```cpp
// 提示：在ggml_build_forward后调用ggml_graph_dump_dot()
```
然后用Graphviz生成可视化图片。

---

## 本课小结

| 概念 | 一句话总结 |
|------|-----------|
| llama_build_graph | 根据架构配置动态构建推理计算图 |
| llm_build_llama | Llama架构特定的图构建函数 |
| 注意力层 | Q/K/V投影 → RoPE → 缓存 → 注意力分数 → Softmax → 加权求和 |
| FFN层 | SwiGLU结构：SiLU(Wg@x) * (Wu@x) @ Wd |
| 算子融合 | 合并相邻轻量算子，减少内存访问 |
| ggml_graph_dump_dot | 导出Graphviz格式，可视化计算图 |

---

*本章对应源码版本：master (2026-04-07)*
