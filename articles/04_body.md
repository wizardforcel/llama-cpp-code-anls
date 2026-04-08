# 第4章 张量运算实现 —— 探索"数值计算的瑞士军刀"

在理解了 GGML 的基础架构之后，我们需要深入探索它最核心、最实用的部分——张量运算。GGML 提供了数十种张量运算，从基础的加减乘除，到复杂的矩阵乘法、激活函数、位置编码等。这些运算是构建大语言模型的"积木"，理解它们的实现原理，是掌握 llama.cpp 的关键一步。

## 学习目标

1. 理解 GGML 基础数学运算的实现原理
2. 掌握神经网络专用算子（SiLU、RMSNorm、RoPE）的源码
3. 了解内存优化技术（in-place、view、复用）的实现机制
4. 能阅读并修改自定义算子

## 生活类比：工厂的生产流水线

想象 GGML 的计算图是一家**智能工厂的生产流水线**：

- **张量** = 传送带上的物料箱，装着待加工的数值
- **运算操作** = 加工机器：加法机（合并物料）、乘法机（按比例调配）、混合机（复杂配方）
- **内存池** = 仓库货架，预先规划好的存储空间
- **in-place 优化** = 原地改造，不搬动物料箱，直接在原地加工
- **view 机制** = 贴标签，多个箱子共享同一批货物，只是贴不同标签区分用途
- **拓扑排序** = 生产调度，确保原材料先经过前道工序，再到后道工序

就像工厂经理需要精心安排每台机器的位置和物料流转，GGML 的计算图也需要精确规划每个算子的内存使用和计算顺序。一个优秀的流水线能最大限度减少物料搬运、降低仓库占用，同理，一个优化的计算图能最大限度复用内存、减少数据拷贝。

---

## 4.1 基础数学运算

### 4.1.1 元素级运算 —— 传送带上的并行加工

元素级运算是最基础的运算类型，对两个张量的对应位置元素执行相同操作。这类运算的特点是**输出形状与输入完全相同**。

**源码位置**：`ggml/src/ggml.c`（第 15000-15200 行）

#### `ggml_add` 实现剖析

让我们从最简单的加法运算开始：

```c
// 创建加法运算节点
struct ggml_tensor * ggml_add(
        struct ggml_context * ctx,
        struct ggml_tensor * a,
        struct ggml_tensor * b) {
    // ① 形状兼容性检查 - 确保两个张量形状一致
    GGML_ASSERT(ggml_are_same_shape(a, b));

    // ② 创建结果张量（形状与输入相同）
    struct ggml_tensor * result = ggml_new_tensor(ctx, a->type, a->n_dims, a->ne);

    // ③ 设置运算类型为 ADD，供后续执行时分派
    result->op = GGML_OP_ADD;

    // ④ 记录输入张量（构建计算图时用到）
    result->src[0] = a;
    result->src[1] = b;
    
    // ⑤ 可选：设置结果张量名称（便于调试）
    ggml_format_name(result, "%s + %s", a->name, b->name);

    return result;
}
```

这段代码实现了GGML的加法运算节点创建。它首先检查输入张量形状兼容性，创建结果张量，设置操作类型为ADD，并记录输入依赖关系(src数组)，最后返回代表加法操作的张量节点。

**关键设计解析**：

为什么需要 `src[]` 数组？这是 GGML 计算图的核心机制：

1. **依赖追踪**：计算图需要知道每个节点的"原料"来自哪里
2. **执行分派**：执行时根据 `op` 类型和 `src` 输入进行计算
3. **拓扑排序**：构建计算图时需要遍历 src 关系确定执行顺序
4. **可复现性**：完整的依赖信息让计算图可以序列化和重现

#### 实际执行：`ggml_compute_forward_add`

创建节点只是"画图纸"，真正的计算发生在执行阶段：

**源码位置**：`ggml/src/ggml.c`（第 21000-21200 行）

```c
// 前向计算 - 张量加法
static void ggml_compute_forward_add(
        const struct ggml_compute_params * params,
        struct ggml_tensor * dst) {
    // dst 是结果张量，它的 src[0] 和 src[1] 是输入
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
        case GGML_TYPE_Q8_0:
            // 量化类型需要特殊处理：先反量化，再计算，再量化
            ggml_compute_forward_add_q8_0(params, dst);
            break;
        // ... 更多类型
        default:
            GGML_ASSERT(false && "unsupported type for add");
    }
}

这段代码是加法运算的前向计算分派函数。它从结果张量dst中获取输入(src0/src1)，根据数据类型分派到对应的特定实现(F32/F16/量化等)，实现多态计算。这种设计允许同一代码框架支持多种精度。
```

**FP32 具体实现**：

```c
static void ggml_compute_forward_add_f32(
        const struct ggml_compute_params * params,
        struct ggml_tensor * dst) {
    const struct ggml_tensor * src0 = dst->src[0];
    const struct ggml_tensor * src1 = dst->src[1];

    const float * src0_data = (const float *)src0->data;
    const float * src1_data = (const float *)src1->data;
    float * dst_data = (float *)dst->data;

    const int ith = params->ith;  // 当前线程编号
    const int nth = params->nth;  // 总线程数

    // 并行计算：每个线程处理一部分元素
    const int64_t ne = ggml_nelements(dst);
    const int64_t dr = (ne + nth - 1) / nth;  // 每个线程处理的数量
    const int64_t ir0 = dr * ith;             // 本线程起始索引
    const int64_t ir1 = MIN(ir0 + dr, ne);    // 本线程结束索引

    // 简单的向量加法循环
    for (int64_t i = ir0; i < ir1; i++) {
        dst_data[i] = src0_data[i] + src1_data[i];
    }
}

这段代码实现了FP32类型的加法计算内核。它将元素均匀分配给多个线程并行处理，通过计算每个线程的起止索引(ir0/ir1)实现负载均衡。实际实现中还包含AVX/SSE等SIMD优化。
```

**性能优化要点**：

1. **多线程并行**：将元素均匀分配给多个线程，充分利用多核 CPU
2. **连续内存访问**：按索引顺序访问，最大化缓存命中率
3. **SIMD 优化**：实际实现使用 AVX/AVX2/SSE 指令一次处理多个元素

### 4.1.2 矩阵乘法（GEMM）—— 工厂的核心机组

GEMM（General Matrix Multiply）是深度学习的核心运算，占推理时间的 60% 以上。在 LLM 中，几乎所有的计算量都来自矩阵乘法。

**源码位置**：`ggml/src/ggml.c` - `ggml_mul_mat()`（第 16000-16200 行）

```c
struct ggml_tensor * ggml_mul_mat(
        struct ggml_context * ctx,
        struct ggml_tensor * a,   // 权重矩阵 [M, K]
        struct ggml_tensor * b) { // 输入矩阵 [K, N]

    // ① 维度检查：a 的列数必须等于 b 的行数（内维匹配）
    GGML_ASSERT(a->ne[0] == b->ne[0]);

    // ② 结果形状：[M, N]（外维）
    // 注意：GGML 是列优先，所以 ne[0] 对应矩阵的行数
    const int64_t ne[2] = { a->ne[1], b->ne[1] };
    struct ggml_tensor * result = ggml_new_tensor(ctx, GGML_TYPE_F32, 2, ne);

    result->op = GGML_OP_MUL_MAT;
    result->src[0] = a;
    result->src[1] = b;

    return result;
}

这段代码创建矩阵乘法运算节点。它验证输入矩阵的内维匹配(a的列数等于b的行数)，计算输出形状[M,N]，创建结果张量并记录操作类型为MUL_MAT。矩阵乘法是深度学习中最核心的计算操作。
```

**形状计算图解**：

```
矩阵乘法: C = A @ B

A: [M, K]  即 ne = [K, M]  (GGML列优先)
B: [K, N]  即 ne = [K, N]
C: [M, N]  即 ne = [N, M]

可视化:
     ┌─────────┐
     │    B    │
     │  [K,N]  │
     └────┬────┘
          │
┌────┐    │    ┌────┐
│ A  │====╪====│ C  │
│[M,K]│   │    │[M,N]│
└────┘    │    └────┘
          │
    [内维K匹配]

在 LLM 前向传播中：
- W: [4096, 512]  (权重矩阵，固定)
- x: [512, 1]     (输入向量，变化)
- y: [4096, 1]    (输出 logits)
```

**性能优化策略**：

矩阵乘法的计算复杂度是 O(M×K×N)，对于大矩阵这是天文数字。GGML 采用多重优化：

1. **分块（Tiling）**：将大矩阵分成小块， fits in L1/L2 cache
2. **向量化**：使用 SIMD 指令一次计算多个元素
3. **循环重排**：调整循环顺序最大化数据复用
4. **后端加速**：CUDA/cuBLAS/Metal 等 GPU 实现

### 4.1.3 归约运算 —— 物料汇总统计

归约运算将张量的多个元素合并为较少的元素（通常是一个标量或向量）。

**源码位置**：`ggml/src/ggml.c` - `ggml_sum()`（第 15500-15600 行）

```c
// 将所有元素求和，返回标量
struct ggml_tensor * ggml_sum(struct ggml_context * ctx, struct ggml_tensor * a) {
    // 结果是一个 0 维张量（标量）
    struct ggml_tensor * result = ggml_new_tensor_1d(ctx, a->type, 1);

    result->op = GGML_OP_SUM;
    result->src[0] = a;

    return result;
}

这段代码创建求和归约运算节点。它将输入张量的所有元素相加，输出一个标量(1维张量)。归约运算在损失计算、统计聚合等场景中广泛使用。
```

**常用归约操作表**：

| 函数 | 操作 | 公式 | 典型用途 |
|------|------|------|---------|
| `ggml_sum()` | 求和 | Σxᵢ | 损失计算、统计总量 |
| `ggml_mean()` | 求平均 | Σxᵢ/n | 归一化、统计平均 |
| `ggml_max()` | 求最大值 | max(xᵢ) | 注意力掩码、池化 |
| `ggml_argmax()` | 求最大索引 | argmax(xᵢ) | 采样选择、分类 |
| `ggml_norm()` | 求范数 | √(Σxᵢ²) | 梯度裁剪、归一化 |

**归约运算的实现技巧**：

```c
// 两阶段归约：先线程内局部归约，再汇总
static void ggml_compute_forward_sum_f32(...) {
    // 第一阶段：每个线程计算局部和
    float local_sum = 0.0f;
    for (int64_t i = ir0; i < ir1; i++) {
        local_sum += src_data[i];
    }
    
    // 第二阶段：汇总所有线程的结果（需要同步）
    // 使用原子操作或归约树
    atomic_add(&dst_data[0], local_sum);
}

这段代码展示了两阶段并行归约算法。第一阶段各线程独立计算局部和，第二阶段通过原子操作或归约树将所有局部和汇总为最终结果，是高效并行归约的经典实现。
```

---

## 4.2 神经网络专用运算

### 4.2.1 激活函数 —— 物料的非线性变换

激活函数为神经网络引入非线性，使其能够学习复杂模式。现代 LLM 主要使用 SiLU（Swish）激活。

#### SiLU（Swish）激活函数

**数学公式**：
```
SiLU(x) = x * sigmoid(x) = x / (1 + e^(-x))
```

**为什么 LLM 偏爱 SiLU？**

1. **平滑非单调**：表达能力比 ReLU 更强
2. **自门控机制**：输出在 0 到 x 之间，天然有"开关"特性
3. **非零梯度区域大**：负值区域仍有梯度，优于 ReLU
4. **SwiGLU 架构基础**：LLaMA、Mistral 等模型的 FFN 使用 SiLU

**源码位置**：`ggml/src/ggml.c` - `ggml_silu()`（第 17200-17300 行）

```c
struct ggml_tensor * ggml_silu(struct ggml_context * ctx, struct ggml_tensor * a) {
    struct ggml_tensor * result = ggml_new_tensor(ctx, a->type, a->n_dims, a->ne);

    result->op = GGML_OP_SILU;
    result->src[0] = a;

    return result;
}

// 实际计算实现
static void ggml_compute_forward_silu_f32(
        const struct ggml_compute_params * params,
        struct ggml_tensor * dst) {
    const struct ggml_tensor * src0 = dst->src[0];
    const float * src0_data = (const float *)src0->data;
    float * dst_data = (float *)dst->data;

    const int64_t ne = ggml_nelements(dst);

    // 逐元素计算：x * sigmoid(x)
    for (int64_t i = 0; i < ne; i++) {
        float x = src0_data[i];
        // SiLU = x / (1 + exp(-x))
        dst_data[i] = x / (1.0f + expf(-x));
    }
}

这段代码实现了SiLU(Sigmoid Linear Unit)激活函数，是现代LLM的首选激活函数。公式为SiLU(x)=x*sigmoid(x)，具有平滑非单调特性，相比ReLU有更大的非零梯度区域和更强的表达能力。
```

**数值稳定性优化**：

```c
// 更稳定的实现，避免exp溢出
float sigmoid(float x) {
    if (x >= 0) {
        return 1.0f / (1.0f + expf(-x));
    } else {
        // 对于负x，改写公式避免大数
        float z = expf(x);
        return z / (1.0f + z);
    }
}

这段代码展示了数值稳定的sigmoid实现。当输入为负数时，通过改写公式避免exp(-x)产生大数溢出，提高数值稳定性，是神经网络计算中的常用技巧。
```

### 4.2.2 RMS 归一化 —— 标准化的流水线

RMSNorm（Root Mean Square Layer Normalization）是现代 LLM 的首选归一化方法。

**与 LayerNorm 的区别**：
```
LayerNorm: y = (x - mean) / sqrt(var + eps) * gamma
RMSNorm:   y = x / sqrt(mean(x²) + eps) * gamma
```

**RMSNorm 的优势**：
1. **计算更简单**：无需计算均值，一次遍历即可
2. **性能更好**：减少约 30% 的计算量
3. **效果相当**：实验表明与 LayerNorm 质量相当
4. **LLaMA 验证**：LLaMA、Mistral、Qwen 等主流模型使用

**源码位置**：`ggml/src/ggml.c` - `ggml_rms_norm()`（第 17300-17400 行）

```c
struct ggml_tensor * ggml_rms_norm(
        struct ggml_context * ctx,
        struct ggml_tensor * a,
        float eps) {              // 防止除零的小数，通常 1e-6

    struct ggml_tensor * result = ggml_new_tensor(ctx, a->type, a->n_dims, a->ne);

    result->op = GGML_OP_RMS_NORM;
    result->src[0] = a;

    // eps 存储在 op_params 中，供执行时使用
    memcpy(result->op_params, &eps, sizeof(eps));

    return result;
}

这段代码创建RMS归一化运算节点。RMSNorm是LayerNorm的简化版，只计算均方根而不减去均值，减少了约30%的计算量。eps参数通过op_params传递，防止除零错误。
```

**计算流程详解**：

```c
static void ggml_compute_forward_rms_norm_f32(
        const struct ggml_compute_params * params,
        struct ggml_tensor * dst) {
    const struct ggml_tensor * src0 = dst->src[0];
    const float eps = ((const float *)dst->op_params)[0];
    
    const float * src_data = (const float *)src0->data;
    float * dst_data = (float *)dst->data;
    
    // 假设输入是 [batch, hidden_size]
    const int64_t ne00 = src0->ne[0];  // hidden_size
    const int64_t n_rows = ggml_nelements(src0) / ne00;
    
    for (int64_t i = 0; i < n_rows; i++) {
        const float * row_src = src_data + i * ne00;
        float * row_dst = dst_data + i * ne00;
        
        // Step 1: 计算平方和
        float sum_squares = 0.0f;
        for (int64_t j = 0; j < ne00; j++) {
            sum_squares += row_src[j] * row_src[j];
        }
        
        // Step 2: 计算 RMS
        float rms = sqrtf(sum_squares / ne00 + eps);
        
        // Step 3: 归一化
        for (int64_t j = 0; j < ne00; j++) {
            row_dst[j] = row_src[j] / rms;
        }
    }
}

这段代码实现了RMS归一化的FP32计算内核。对每个行向量计算平方和、RMS值，然后归一化。相比LayerNorm省略了减均值步骤，计算更高效，被LLaMA、Mistral等主流模型采用。
```

### 4.2.3 RoPE（旋转位置编码）—— 给物料添加"时间戳"

RoPE（Rotary Position Embedding）是 LLM 处理序列数据的关键技术，它通过旋转矩阵将位置信息注入到注意力计算中。

**为什么需要位置编码？**

Transformer 的核心——自注意力机制——是**置换等变**的：交换输入 token 的顺序不会改变注意力权重。这意味着模型本身对位置不敏感，必须显式注入位置信息。

**RoPE 的独特优势**：

1. **相对位置编码**：通过旋转角度编码相对位置，而非绝对位置
2. **长序列外推**：可以处理比训练时更长的序列
3. **与注意力融合**：直接在注意力计算中应用，无需额外层

**源码位置**：`ggml/src/ggml.c` - `ggml_rope()`（第 17500-17700 行）

```c
struct ggml_tensor * ggml_rope(
        struct ggml_context * ctx,
        struct ggml_tensor * a,        // 输入 [head_size/2, n_heads, n_tokens]
        struct ggml_tensor * b,        // 位置索引 [n_tokens]
        int n_dims,                    // 旋转维度
        int mode,                      // 模式（普通/GQA）
        int n_ctx,                     // 上下文长度
        int n_orig_ctx) {              // 原始上下文（用于外推）

    struct ggml_tensor * result = ggml_new_tensor(ctx, a->type, a->n_dims, a->ne);

    result->op = GGML_OP_ROPE;
    result->src[0] = a;
    result->src[1] = b;

    // 将多个参数编码到 op_params 数组（避免多个参数）
    int32_t params[] = { n_dims, mode, n_ctx, n_orig_ctx };
    memcpy(result->op_params, params, sizeof(params));

    return result;
}

这段代码创建RoPE(旋转位置编码)运算节点。RoPE通过旋转矩阵将位置信息注入注意力计算，支持相对位置编码和长序列外推。多个参数(n_dims/mode/n_ctx等)打包存储在op_params数组中。
```

**旋转编码数学原理**：

```
对于一对维度 (x_m, x_{m+d/2}) 在位置 m 上：

[x_m']     [cos(m*θ)  -sin(m*θ)]   [x_m]
[x_{m+d/2}'] = [sin(m*θ)   cos(m*θ)] * [x_{m+d/2}]

其中 θ = base^(-2i/d)，base 通常是 10000，i 是维度索引

这相当于在 2D 平面上旋转角度 m*θ
```

**可视化理解**：

```
位置 0: 旋转 0°     位置 1: 旋转 θ      位置 2: 旋转 2θ
    ↓                   ↓                   ↓
  [x₀,x₁]            [x₀',x₁']           [x₀'',x₁'']
    ↻                   ↻                   ↻
   0°                  θ°                  2θ°
```

**外推能力**：

训练时模型见过的最大位置是 `n_orig_ctx`。对于超出这个范围的位置，RoPE 仍然可以计算旋转角度，只是模型可能没见过这种分布。这就是"外推"（extrapolation）。

---

## 4.3 内存优化技术 —— 工厂的空间管理艺术

在 LLM 推理中，内存往往是瓶颈而非计算。GGML 提供了多种内存优化技术，让有限内存能够运行更大的模型。

### 4.3.1 原地操作（In-Place）—— 原地改造

**核心思想**：结果直接写入输入张量的内存，不分配新空间。

**适用条件**：
- 输入张量之后不再被使用（生命周期结束）
- 输出形状与输入完全相同

**源码位置**：`ggml/src/ggml.c` - `ggml_silu_inplace()`（第 17250 行）

```c
struct ggml_tensor * ggml_silu_inplace(struct ggml_context * ctx, struct ggml_tensor * a) {
    // 不创建新张量，而是创建一个"视图"指向 a
    struct ggml_tensor * result = ggml_view_tensor(ctx, a);

    result->op = GGML_OP_SILU;
    result->src[0] = a;
    result->flags |= GGML_TENSOR_FLAG_INPLACE;  // 关键：标记为原地操作

    return result;
}

这段代码实现了原地(in-place)SiLU激活函数。通过创建视图张量而非新分配内存，结果直接写入输入张量的内存空间。标记GGML_TENSOR_FLAG_INPLACE告知执行器无需分配额外输出缓冲区，可节省50%内存。
```

**内存节省示例**：

```
无 in-place：
    ┌─────────┐         ┌─────────┐
    │  a(输入) │  ──→   │result(输出)│
    │ 100MB   │  SiLU   │ 100MB     │
    └─────────┘         └─────────┘
    总内存: 200MB（同时存在）

有 in-place：
    ┌─────────┐
    │  a      │  ──→  原地修改
    │ 100MB   │
    └─────────┘
    总内存: 100MB
```

**使用时机**：

在计算图中，当确定某个张量不会再被使用时，就可以使用原地操作。例如：

```
tmp1 = a + b
tmp2 = silu_inplace(tmp1)  // tmp1 不再使用，可以原地修改
tmp3 = tmp2 * c
```

### 4.3.2 视图（View）机制 —— 贴标签的艺术

**核心思想**：多个张量共享同一块内存，只是"看法"（形状、步长）不同。

**源码位置**：`ggml/src/ggml.c` - `ggml_view_tensor()`（第 18000-18100 行）

```c
struct ggml_tensor * ggml_view_tensor(
        struct ggml_context * ctx,
        struct ggml_tensor * a) {

    // 创建新张量，但指向同一块数据
    struct ggml_tensor * result = ggml_new_tensor(ctx, a->type, a->n_dims, a->ne);

    // 关键：共享 data 指针
    result->data = a->data;
    result->buffer = a->buffer;
    result->offs = a->offs;

    // 标记为视图（不需要单独释放内存）
    result->flags |= GGML_TENSOR_FLAG_VIEW;

    return result;
}
```

**典型应用场景**：

| 操作 | 用途 | 示例 |
|------|------|------|
| `ggml_view_1d()` | 1D 切片 | 取 embedding 表的第 i 行 |
| `ggml_view_2d()` | 2D 切片 | 取 attention 的某个头 |
| `ggml_view_3d()` | 3D 切片 | 取 batch 中某个样本 |
| `ggml_reshape()` | 改变形状 | [batch*seq, hidden] → [batch, seq, hidden] |
| `ggml_permute()` | 维度重排 | [batch, heads, seq, head_dim] → [batch, seq, heads, head_dim] |
| `ggml_cont()` | 转连续 | 将不连续视图转为连续存储 |

**切片操作示例**：

```c
// 假设有完整的注意力输出 [n_heads, head_size, n_tokens]
struct ggml_tensor * full_attn = ...;

// 创建第 0 个注意力头的视图（不复制数据！）
struct ggml_tensor * head_0 = ggml_view_2d(
    ctx,
    full_attn,
    head_size,           // ne[0]
    n_tokens,            // ne[1]
    full_attn->nb[1],    // 行步长
    0                    // 偏移（从头开始）
);

// head_0 和 full_attn 共享内存
```

### 4.3.3 内存复用策略 —— 智能仓库调度

对于复杂的计算图，GGML 提供了 `ggml_gallocr`（graph allocator），它可以分析张量生命周期，让不重叠的张量共享物理内存。

**源码位置**：`ggml/src/ggml-alloc.c`

```c
// 创建分配器
struct ggml_gallocr * allocr = ggml_gallocr_new(backend);

// 为计算图分配/优化内存
ggml_gallocr_alloc_graph(allocr, graph);

// 关键优化：生命周期不重叠的张量共享内存
// 例如：
//   op1: a + b = tmp1
//   op2: tmp1 * c = result
//
// tmp1 在 op2 执行后就不再需要
// 因此 result 可以和 tmp1 共用内存
```

**内存复用图解**：

```
时间轴 →

张量A  [██████████]  常驻（输入/参数）
张量B  [████      ]  中间结果 1
张量C       [████ ]  中间结果 2
结果          [██████████]  最终输出

复用分析：
- B 和 C 生命周期不重叠 → 可以共享内存
- C 和结果部分重叠 → 需要独立空间

优化后内存布局：
┌──────────────────────────────────────────────────────┐
│  A  │  B/C 共享  │        结果        │
└──────────────────────────────────────────────────────┘

原始需求: 4 个内存块
优化后: 3 个内存块
节省: 25%
```

在实际 LLM 推理中，这种优化可以将内存使用减少 50% 以上。

---

## 4.4 设计中的取舍

### 为什么 GGML 要自己实现算子，不全部调用 BLAS/cuBLAS？

| 方案 | 优点 | 缺点 | GGML 选择 |
|------|------|------|-----------|
| 纯自研 | 完全控制，量化友好，无依赖 | 开发量大，性能可能不如专用库 | 基础算子 |
| OpenBLAS/MKL | CPU GEMM 性能最优 | 仅 CPU，依赖重，增加体积 | 可选 |
| cuBLAS | GPU GEMM 性能最优 | 仅 NVIDIA，启动开销大 | 混合使用 |
| oneDNN | Intel 优化好 | 仅 Intel，依赖重 | 可选 |

**GGML 的混合策略**：

```
算子类型              实现方式
─────────────────────────────────────
元素级运算(add/mul)   纯 C + SIMD，全平台通用
激活函数(silu/gelu)   纯 C + SIMD，全平台通用
归一化(rms_norm)      纯 C + SIMD，全平台通用
位置编码(rope)        纯 C + SIMD，全平台通用
GEMM 大矩阵           后端特定（CUDA/Metal/Vulkan）
量化反量化            必须自研（标准库不支持）
```

这种策略的权衡：开发团队需要维护更多代码，但获得了最大的灵活性和可移植性。

### 为什么张量操作是"惰性"的（只建图不计算）？

```c
// GGML 风格：先建图，后计算
struct ggml_tensor* c = ggml_add(ctx, a, b);  // 不计算，只记录
// ... 构建完整图 ...
ggml_graph_compute(ctx, graph);               // 统一计算

// 对比 Eager 模式（PyTorch 默认）：
tensor_c = tensor_a + tensor_b  // 立即计算
```

**惰性求值的优势**：

1. **全局优化机会**：拥有完整图的视角，可以优化内存使用和计算顺序
2. **异步执行**：可以整体丢给 GPU，减少 CPU-GPU 往返开销
3. **图复用**：同一图结构可重复执行（不同输入），只需重新填充输入数据
4. **易于序列化**：计算图可以保存为文件，跨平台复现

**缺点**：
- 调试困难（错误延迟到执行时才暴露）
- 内存峰值可能更高（需要同时保存图的描述）

---

## 4.5 动手练习

### 练习 1：阅读 GGML 算子实现

阅读 `ggml/src/ggml.c` 第 17200-17800 行，理解以下算子的实现：

1. `ggml_gelu()` - GELU 激活函数
2. `ggml_norm()` - LayerNorm
3. `ggml_soft_max()` - Softmax

回答问题：
- 这三个算子的 `op_params` 各存储了什么参数？
- 它们的输入输出形状关系是什么？
- 哪个算子使用了 in-place 优化？

### 练习 2：实现自定义算子

基于以下框架，实现一个 LeakyReLU 算子：

```c
// LeakyReLU: x < 0 ? alpha * x : x
struct ggml_tensor * ggml_leaky_relu(
        struct ggml_context * ctx,
        struct ggml_tensor * a,
        float alpha) {
    
    // ① 创建结果张量
    struct ggml_tensor * result = ggml_new_tensor(...);
    
    // ② 设置运算类型
    result->op = GGML_OP_LEAKY_RELU;
    result->src[0] = a;
    
    // ③ 存储 alpha 参数
    memcpy(result->op_params, &alpha, sizeof(alpha));
    
    return result;
}

// 计算实现（参考 ggml_compute_forward_silu_f32）
static void ggml_compute_forward_leaky_relu_f32(...) {
    // 实现 LeakyReLU 计算
}
```

### 练习 3：分析内存使用

给定计算图：
```
a(100MB) --\              
            +--> tmp1(100MB) --\
b(100MB) --/                     +--> result(100MB)
                           c(100MB) --/
```

问题：
1. 不使用内存优化时，总内存需求是多少？
2. 使用 in-place 和视图优化后，最低内存需求是多少？
3. 画出优化后的内存时间线。

---

## 4.6 本章小结

| 概念 | 一句话总结 |
|------|-----------|
| `ggml_add/mul` | 元素级运算，形状必须完全相同，支持多线程并行 |
| `ggml_mul_mat` | 矩阵乘法 GEMM，深度学习核心运算，计算量最大 |
| `ggml_silu` | Swish 激活函数，LLM 首选，SiLU(x) = x * sigmoid(x) |
| `ggml_rms_norm` | RMS 归一化，LayerNorm 的简化版，LLaMA 使用 |
| `ggml_rope` | 旋转位置编码，注入位置信息，支持外推 |
| `ggml_view_*` | 视图操作，零拷贝形状变换，共享内存 |
| `ggml_*_inplace` | 原地操作，结果写入输入内存，节省 50% 空间 |

**本章核心要点**：

1. **运算分阶段**：创建（定义图结构）→ 构建（拓扑排序）→ 执行（实际计算）
2. **类型分发**：运行时根据张量类型分发到特定实现
3. **并行计算**：元素级运算天然并行，使用多线程加速
4. **内存优化**：视图、原地操作、内存复用是推理性能的关键

**下一步预告**：

在掌握了张量运算之后，我们将在第 5 章深入 GGUF 模型格式——理解模型是如何存储和加载的，以及不同量化方案对性能和质量的影响。
