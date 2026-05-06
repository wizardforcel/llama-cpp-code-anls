# 第4章 张量运算实现 —— 探索"数值计算的瑞士军刀"

## 学习目标
1. 理解GGML基础数学运算的实现原理
2. 掌握神经网络专用算子（SiLU、RMSNorm、RoPE）的源码
3. 了解内存优化技术（in-place、view、复用）的实现机制
4. 能阅读并修改自定义算子

---

## 生活类比：工厂的生产流水线

想象GGML的计算图是一家**智能工厂的生产流水线**：
- **张量** = 传送带上的物料箱
- **运算操作** = 加工机器（加法机、乘法机、混合机）
- **内存池** = 仓库货架（预先规划好的存储空间）
- **in-place优化** = 原地改造（不搬动物料，直接加工）
- **view机制** = 贴标签（多个箱子共享同一批货物）

就像工厂经理需要精心安排每台机器的位置和物料流转，GGML的计算图也需要精确规划每个算子的内存使用和计算顺序。

---

## 源码地图

```
ggml/src/ggml.c
├── 基础运算（15000-16000行）
│   ├── ggml_add()           # 张量加法
│   ├── ggml_add_inplace()   # 原地加法
│   ├── ggml_mul()           # 张量乘法
│   ├── ggml_scale()         # 缩放
│   ├── ggml_sum()           # 求和
│   ├── ggml_mean()          # 平均值
│   ├── ggml_max()           # 最大值
│   ├── ggml_argmax()        # 最大索引
│   └── ggml_repeat()        # 重复
├── 矩阵运算（16000-17000行）
│   ├── ggml_mul_mat()       # 矩阵乘法 GEMM
│   ├── ggml_out_prod()      # 外积
│   ├── ggml_transpose()     # 转置
│   └── ggml_cont()          # 转连续
├── 神经网络运算（17000-18000行）
│   ├── ggml_silu()          # SiLU激活
│   ├── ggml_silu_inplace()  # 原地SiLU
│   ├── ggml_gelu()          # GELU激活
│   ├── ggml_relu()          # ReLU激活
│   ├── ggml_rms_norm()      # RMS归一化
│   ├── ggml_norm()          # LayerNorm
│   ├── ggml_rope()          # 旋转位置编码
│   ├── ggml_soft_max()      # Softmax
│   ├── ggml_diag_mask_inf() # 对角掩码
│   └── ggml_alibi()         # ALiBi位置编码
├── 内存优化（18000-19000行）
│   ├── ggml_view_tensor()   # 视图创建
│   ├── ggml_view_1d/2d/3d() # 多维视图
│   ├── ggml_reshape()       # 重塑形状
│   ├── ggml_reshape_2d/3d() # 多维重塑
│   ├── ggml_permute()       # 维度置换
│   └── ggml_cont()          # 转连续
└── 量化运算（19000-20000行）
    ├── ggml_quantize()      # 量化
    └── ggml_dequantize()    # 反量化

ggml/include/ggml.h
├── enum ggml_op             # 操作类型枚举
│   ├── GGML_OP_ADD/MUL/SUB/DIV
│   ├── GGML_OP_MUL_MAT
│   ├── GGML_OP_SILU/GELU/RELU
│   ├── GGML_OP_NORM/RMS_NORM
│   ├── GGML_OP_ROPE
│   └── GGML_OP_SOFT_MAX
└── 算子API声明
```

---

## 4.1 基础数学运算

### 4.1.1 元素级运算 —— 传送带上的并行加工

**源码位置**：`ggml/src/ggml.c` (第15000-15200行)

#### ggml_add 实现剖析

```c
// 创建加法运算节点
struct ggml_tensor * ggml_add(
        struct ggml_context * ctx,
        struct ggml_tensor * a,
        struct ggml_tensor * b) {
    // ① 形状兼容性检查
    GGML_ASSERT(ggml_are_same_shape(a, b));

    // ② 创建结果张量（形状与输入相同）
    struct ggml_tensor * result = ggml_new_tensor(ctx, a->type, a->n_dims, a->ne);

    // ③ 设置运算类型为 ADD
    result->op = GGML_OP_ADD;

    // ④ 记录输入张量（构建计算图时用到）
    result->src[0] = a;
    result->src[1] = b;

    return result;
}
```

**为什么需要 `src[]` 数组？**
- 计算图需要知道每个节点的"原料"来自哪里
- 执行时根据 `op` 类型和 `src` 输入进行计算
- 反向传播时（训练场景）需要追踪梯度来源

#### 实际执行：ggml_compute_forward_add

```c
// 源码位置：ggml/src/ggml.c (第21000-21200行)
static void ggml_compute_forward_add(
        const struct ggml_compute_params * params,
        struct ggml_tensor * dst) {
    // dst = dst->src[0] + dst->src[1]
    const struct ggml_tensor * src0 = dst->src[0];
    const struct ggml_tensor * src1 = dst->src[1];

    // 根据数据类型分发到不同实现
    switch (src0->type) {
        case GGML_TYPE_F32:
            ggml_compute_forward_add_f32(params, dst);
            break;
        case GGML_TYPE_F16:
            ggml_compute_forward_add_f16(params, dst);
            break;
        // ... 量化类型的特殊处理
    }
}
```

### 4.1.2 矩阵乘法（GEMM）—— 工厂的核心机组

**GEMM = General Matrix Multiply** 是深度学习的核心运算，占推理时间的60%以上。

**源码位置**：`ggml/src/ggml.c` - `ggml_mul_mat()` (第16000-16200行)

```c
struct ggml_tensor * ggml_mul_mat(
        struct ggml_context * ctx,
        struct ggml_tensor * a,   // 权重矩阵 [M, K]
        struct ggml_tensor * b) { // 输入矩阵 [K, N]

    // ① 维度检查：a的列数必须等于b的行数
    GGML_ASSERT(a->ne[0] == b->ne[0]);

    // ② 结果形状：[M, N]
    const int64_t ne[2] = { a->ne[1], b->ne[1] };
    struct ggml_tensor * result = ggml_new_tensor(ctx, GGML_TYPE_F32, 2, ne);

    result->op = GGML_OP_MUL_MAT;
    result->src[0] = a;
    result->src[1] = b;

    return result;
}
```

**形状计算图解**：
```
a: [M, K] @ b: [K, N] = result: [M, N]

示例：
权重 W: [4096, 512]  输入 x: [512, 1]
结果 y: [4096, 1]  (输出 logits)

在LLM中：
- W是模型权重（固定）
- x是输入token的embedding（变化）
- y是下一层输入
```

### 4.1.3 归约运算 —— 物料汇总统计

**源码位置**：`ggml/src/ggml.c` - `ggml_sum()` (第15500-15600行)

```c
// 将所有元素求和，返回标量
struct ggml_tensor * ggml_sum(struct ggml_context * ctx, struct ggml_tensor * a) {
    // 结果是一个0维张量（标量）
    struct ggml_tensor * result = ggml_new_tensor_1d(ctx, a->type, 1);

    result->op = GGML_OP_SUM;
    result->src[0] = a;

    return result;
}
```

**常用归约操作**：
| 函数 | 操作 | 用途 |
|-----|------|------|
| `ggml_sum()` | 求和 | 损失计算 |
| `ggml_mean()` | 求平均 | 归一化 |
| `ggml_max()` | 求最大值 | 注意力掩码 |
| `ggml_argmax()` | 求最大索引 | 采样选择 |

---

## 4.2 神经网络专用运算

### 4.2.1 激活函数 —— 物料的非线性变换

#### SiLU（Swish）激活函数

**数学公式**：`SiLU(x) = x * sigmoid(x) = x / (1 + e^(-x))`

**为什么LLM偏爱SiLU？**
- 平滑非单调，表达能力更强
- 在负值区域有非零梯度（优于ReLU）
- SwiGLU架构中的关键组件

**源码位置**：`ggml/src/ggml.c` - `ggml_silu()` (第17200-17300行)

```c
struct ggml_tensor * ggml_silu(struct ggml_context * ctx, struct ggml_tensor * a) {
    struct ggml_tensor * result = ggml_new_tensor(ctx, a->type, a->n_dims, a->ne);

    result->op = GGML_OP_SILU;
    result->src[0] = a;

    return result;
}

// 实际计算实现
template <typename T>
static void ggml_compute_forward_silu_f32(
        const struct ggml_compute_params * params,
        struct ggml_tensor * dst) {
    const struct ggml_tensor * src0 = dst->src[0];
    const float * src0_data = (const float *)src0->data;
    float * dst_data = (float *)dst->data;

    // 逐元素计算：x * sigmoid(x)
    for (int i = 0; i < ggml_nelements(dst); i++) {
        float x = src0_data[i];
        dst_data[i] = x / (1.0f + expf(-x));  // SiLU公式
    }
}
```

### 4.2.2 RMS归一化 —— 标准化的流水线

**RMSNorm = Root Mean Square Layer Normalization**

**与LayerNorm的区别**：
- LayerNorm: `(x - mean) / sqrt(var + eps)`
- RMSNorm: `x / sqrt(mean(x^2) + eps)`

**优势**：
- 计算更简单（无需计算均值）
- LLaMA2/Mistral等模型的首选

**源码位置**：`ggml/src/ggml.c` - `ggml_rms_norm()` (第17300-17400行)

```c
struct ggml_tensor * ggml_rms_norm(
        struct ggml_context * ctx,
        struct ggml_tensor * a,
        float eps) {              // 防止除零的小数，通常1e-6

    struct ggml_tensor * result = ggml_new_tensor(ctx, a->type, a->n_dims, a->ne);

    result->op = GGML_OP_RMS_NORM;
    result->src[0] = a;

    // eps存储在参数中
    memcpy(result->op_params, &eps, sizeof(eps));

    return result;
}
```

**计算流程图解**：
```
输入 x: [batch, hidden_size]

Step 1: 计算平方均值
    mean_square = sum(x^2) / hidden_size

Step 2: 计算RMS
    rms = sqrt(mean_square + eps)

Step 3: 归一化
    output = x / rms

结果: 每个样本的向量模长被归一化到接近1
```

### 4.2.3 RoPE（旋转位置编码）—— 给物料添加"时间戳"

**RoPE = Rotary Position Embedding**

**为什么需要位置编码？**
- Transformer本身对位置不敏感（自注意力是置换等变的）
- 需要显式注入位置信息

**RoPE的独特之处**：
- 通过旋转矩阵编码相对位置
- 支持外推（处理比训练时更长的序列）

**源码位置**：`ggml/src/ggml.c` - `ggml_rope()` (第17500-17700行)

```c
struct ggml_tensor * ggml_rope(
        struct ggml_context * ctx,
        struct ggml_tensor * a,        // 输入 [head_size/2, n_heads, n_tokens]
        struct ggml_tensor * b,        // 位置索引
        int n_dims,                    // 旋转维度
        int mode,                      // 模式（普通/GQA）
        int n_ctx,                     // 上下文长度
        int n_orig_ctx) {              // 原始上下文（用于外推）

    struct ggml_tensor * result = ggml_new_tensor(ctx, a->type, a->n_dims, a->ne);

    result->op = GGML_OP_ROPE;
    result->src[0] = a;
    result->src[1] = b;

    // 将参数编码到op_params数组
    int32_t params[] = { n_dims, mode, n_ctx, n_orig_ctx };
    memcpy(result->op_params, params, sizeof(params));

    return result;
}
```

**旋转编码数学原理**：
```
对于一对维度 (x_m, x_{m+d/2}) 在位置 m 上：

[x_m']     [cos(m*θ)  -sin(m*θ)]   [x_m]
[x_{m+d/2}'] = [sin(m*θ)   cos(m*θ)] * [x_{m+d/2}]

其中 θ = 10000^(-2i/d)，i是维度索引

这相当于在2D平面上旋转角度 m*θ
```

---

## 4.3 内存优化技术 —— 工厂的空间管理艺术

### 4.3.1 原地操作（In-Place）—— 原地改造

**核心思想**：结果直接写入输入张量的内存，不分配新空间。

**适用条件**：
- 输入张量之后不再被使用
- 形状完全相同

**源码位置**：`ggml/src/ggml.c` - `ggml_silu_inplace()` (第17250行)

```c
struct ggml_tensor * ggml_silu_inplace(struct ggml_context * ctx, struct ggml_tensor * a) {
    // 不创建新张量，直接修改a
    struct ggml_tensor * result = ggml_view_tensor(ctx, a);

    result->op = GGML_OP_SILU;
    result->src[0] = a;
    result->flags |= GGML_TENSOR_FLAG_INPLACE;  // 标记为原地操作

    return result;
}
```

**内存节省示例**：
```
无in-place:
    a(100MB) → silu → result(100MB)  总内存: 200MB

有in-place:
    a(100MB) → silu_inplace → a(100MB)  总内存: 100MB
```

### 4.3.2 视图（View）机制 —— 贴标签的艺术

**核心思想**：多个张量共享同一块内存，只是"看法"不同。

**源码位置**：`ggml/src/ggml.c` - `ggml_view_tensor()` (第18000-18100行)

```c
struct ggml_tensor * ggml_view_tensor(
        struct ggml_context * ctx,
        struct ggml_tensor * a) {

    // 创建新张量，但指向同一块数据
    struct ggml_tensor * result = ggml_new_tensor(ctx, a->type, a->n_dims, a->ne);

    // 关键：共享data指针
    result->data = a->data;
    result->buffer = a->buffer;

    // 标记为视图（不需要单独释放内存）
    result->flags |= GGML_TENSOR_FLAG_VIEW;

    return result;
}
```

**典型应用场景**：
| 操作 | 说明 |
|-----|------|
| `ggml_view_1d()` | 1D切片 |
| `ggml_view_2d()` | 2D切片（取某行/列） |
| `ggml_view_3d()` | 3D切片（取某张量片） |
| `ggml_reshape()` | 改变形状，数据不变 |
| `ggml_permute()` | 维度重排 |
| `ggml_cont()` | 转不连续为连续 |

### 4.3.3 内存复用策略 —— 智能仓库调度

**源码位置**：`ggml/src/ggml-alloc.c`

```c
// 分配器分析计算图，规划内存复用
struct ggml_gallocr * allocr = ggml_gallocr_new(backend);

// 关键优化：生命周期不重叠的张量共享内存
// 例如：
//   op1: a + b = tmp1
//   op2: tmp1 * c = result
//
// tmp1在op2执行后就不再需要
// 因此result可以和tmp1共用内存
```

**内存复用图解**：
```
时间轴 →

张量A  [==========]
张量B  [====      ]
张量C       [==== ]
结果          [==========]

复用后内存布局：
内存块1: [A/B共享][C/结果共享]
内存块2: [      ]

总内存从4块减少到2块！
```

---

## 设计中的取舍

### 为什么GGML要自己实现算子，不调用BLAS/cuBLAS？

| 方案 | 优点 | 缺点 | GGML选择 |
|-----|------|------|---------|
| 纯自研 | 完全控制，量化友好 | 开发量大 | 是 |
| OpenBLAS/MKL | CPU性能最优 | 仅CPU，依赖重 | 可选 |
| cuBLAS | GPU性能最优 | 仅NVIDIA | 部分使用 |

**GGML的混合策略**：
- 简单算子：纯C实现，全平台通用
- GEMM大矩阵：根据后端选择最优实现
- 量化算子：必须自研（标准库不支持）

### 为什么张量操作是"惰性"的（只建图不计算）？

```c
// GGML风格：先建图，后计算
struct ggml_tensor* c = ggml_add(ctx, a, b);  // 不计算，只记录
// ... 构建完整图 ...
ggml_graph_compute(ctx, graph);                // 统一计算

// 对比Eager模式（PyTorch默认）：
tensor_c = tensor_a + tensor_b  // 立即计算
```

**惰性求值的优势**：
1. **优化机会**：全局视角优化内存使用和计算顺序
2. **异步执行**：可以整体丢给GPU，减少CPU-GPU往返
3. **图复用**：同一图结构可重复执行（不同输入）

---

## 动手练习

### 练习1：阅读GGML算子实现
阅读 `ggml/src/ggml.c` 第17200-17800行，理解以下算子的实现：
1. `ggml_gelu()` - GELU激活函数
2. `ggml_norm()` - LayerNorm
3. `ggml_soft_max()` - Softmax

回答问题：
- 这三个算子的 `op_params` 各存储了什么参数？
- 它们的输入输出形状关系是什么？

### 练习2：实现自定义算子
基于以下框架，实现一个LeakyReLU算子：

```c
// LeakyReLU: x < 0 ? alpha * x : x
struct ggml_tensor * ggml_leaky_relu(
        struct ggml_context * ctx,
        struct ggml_tensor * a,
        float alpha) {
    // TODO: 实现算子创建
}
```

### 练习3：分析内存使用
给定计算图：
```
a(100MB) --\
            +--> tmp1(100MB) --\
b(100MB) --/                     +--> result(100MB)
                           c(100MB) --/
```

问题：
1. 不使用内存优化时，总内存需求是多少？
2. 使用in-place和视图优化后，最低内存需求是多少？
3. 画出优化后的内存时间线。

---

## 本课小结

| 概念 | 一句话总结 |
|------|-----------|
| `ggml_add/mul` | 元素级运算，形状必须完全相同 |
| `ggml_mul_mat` | 矩阵乘法，深度学习核心运算 |
| `ggml_silu` | Swish激活，LLM首选激活函数 |
| `ggml_rms_norm` | RMS归一化，LayerNorm的简化版 |
| `ggml_rope` | 旋转位置编码，注入位置信息 |
| `ggml_view_*` | 视图操作，零拷贝形状变换 |
| `ggml_*_inplace` | 原地操作，节省50%内存 |

---

*本章对应源码版本：master (2026-04-07)*
