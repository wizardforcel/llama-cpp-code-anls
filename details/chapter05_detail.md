# 第5章 量化技术深度解析 —— 模型压缩的"瘦身魔法"

## 学习目标
1. 理解量化原理（对称/非对称、位数选择）
2. 掌握GGML量化块的内存布局与计算方式
3. 了解Q4_0到Q8_0各类量化格式的区别
4. 能根据场景选择最佳量化策略

---

## 生活类比：图书馆的书籍压缩

想象你经营一座巨大的**数字图书馆**：

- **原始FP32模型** = 每本书都用高清彩色印刷（4字节/字）
- **量化压缩** = 将书籍转为精简版：
  - **Q8_0** = 高保真扫描（1字节/字，质量接近原版）
  - **Q4_0** = 精要摘录（4比特/字，体积25%，质量可接受）
- **量化块（block）** = 每64页为一组，共用一套"缩放说明书"
- **反量化** = 阅读时根据说明书还原内容
- **IQ系列** = 智能压缩（重要章节保真，次要章节精简）

就像图书馆需要在"藏书数量"和"阅读质量"之间权衡，大模型部署也需要在"模型体积"和"推理质量"之间找到平衡点。

---

## 源码地图

```
ggml/include/ggml.h
├── 量化类型枚举（第200-350行）
│   └── enum ggml_type
│       ├── GGML_TYPE_F32/F16        # 浮点类型
│       ├── GGML_TYPE_Q4_0/Q4_1      # 4位量化
│       ├── GGML_TYPE_Q5_0/Q5_1      # 5位量化
│       ├── GGML_TYPE_Q8_0/Q8_1      # 8位量化
│       ├── GGML_TYPE_IQ2_XXS/XS/S   # 2位智能量化
│       ├── GGML_TYPE_IQ3_XXS/XS/M   # 3位智能量化
│       ├── GGML_TYPE_IQ4_NL/XS      # 4位智能量化
│       └── GGML_TYPE_TQ1_0/TQ2_0    # TriLM量化
├── 量化块大小常量（第250-300行）
│   ├── GGML_QK4_0 (32)              # Q4_0块大小
│   ├── GGML_QK5_0 (32)              # Q5_0块大小
│   ├── GGML_QK8_0 (32)              # Q8_0块大小
│   └── GGML_QK_K (256)              # K-quants块大小
├── 量化块结构（第300-500行）
│   ├── block_q4_0 {d, qs[16]}       # Q4_0块结构
│   ├── block_q4_1 {d, m, qs[16]}    # Q4_1块结构
│   ├── block_q8_0 {d, qs[32]}       # Q8_0块结构
│   └── block_qk_k {d, scales, qs[]} # K-quants块
└── 量化类型特性（第300-350行）
    └── ggml_type_traits
        ├── blck_size              # 块大小
        ├── type_size              # 类型大小
        ├── is_quantized           # 是否量化
        └── nrows                  # 行数

ggml/src/ggml-quants.c
├── 量化函数（第1000-2000行）
│   ├── quantize_row_q4_0_reference()
│   ├── quantize_row_q4_1_reference()
│   ├── quantize_row_q5_0_reference()
│   ├── quantize_row_q8_0_reference()
│   └── quantize_row_q8_1_reference()
├── 反量化函数（第2000-3000行）
│   ├── dequantize_row_q4_0()
│   ├── dequantize_row_q4_1()
│   ├── dequantize_row_q5_0()
│   ├── dequantize_row_q8_0()
│   └── dequantize_row_q8_1()
├── 量化矩阵乘（第3000-5000行）
│   ├── ggml_vec_dot_q4_0_q8_0()
│   ├── ggml_vec_dot_q4_1_q8_1()
│   ├── ggml_vec_dot_q5_0_q8_0()
│   └── ggml_vec_dot_q8_0_q8_0()
└── K-quants实现（第5000+行）
    ├── quantize_row_q_k()
    ├── dequantize_row_q_k()
    └── ggml_vec_dot_q_k()

src/llama-quant.cpp
├── llama_model_quantize()   # 模型量化入口
├── llama_quantize()         # 量化接口
└── 量化参数解析
```

---

## 5.1 量化原理概述

### 5.1.1 为什么要量化？

**内存对比**（以7B模型为例）：
| 格式 | 每参数位数 | 模型大小 | 加载所需RAM |
|-----|-----------|---------|------------|
| FP32 | 32 | 28 GB | ~32 GB |
| FP16 | 16 | 14 GB | ~16 GB |
| Q8_0 | 8 | 7 GB | ~8 GB |
| Q4_0 | 4 | 3.5 GB | ~4 GB |
| Q4_K_M | 4 | 3.8 GB | ~4.5 GB |
| IQ2_XXS | 2 | 1.75 GB | ~2.5 GB |

**应用场景**：
- **Q8_0**：追求精度，服务器部署
- **Q4_K_M**：平衡之选，桌面/笔记本
- **IQ2_XXS**：极致压缩，手机/嵌入式

### 5.1.2 对称量化 vs 非对称量化

**对称量化（Symmetric）**：
```
量化公式：
    q = round(x / scale)
反量化：
    x' = q * scale

存储：
    - 每块存储：scale（float32）+ quantized_weights（int4/int8）
    - 无zero point
```

**非对称量化（Asymmetric）**：
```
量化公式：
    q = round((x - zero) / scale)
反量化：
    x' = q * scale + zero

存储：
    - 每块存储：scale + zero_point + quantized_weights
    - 适合数据分布不对称的场景
```

**GGML实现**：
| 类型 | 量化方式 | 源码位置 |
|-----|---------|---------|
| Q4_0 | 对称 | `ggml/include/ggml.h:350` |
| Q4_1 | 非对称 | `ggml/include/ggml.h:360` |
| Q5_0 | 对称 | `ggml/include/ggml.h:370` |
| Q5_1 | 非对称 | `ggml/include/ggml.h:380` |
| Q8_0 | 对称 | `ggml/include/ggml.h:390` |

---

## 5.2 GGML量化实现

### 5.2.1 量化块（Block）结构 —— 压缩的基本单元

**核心思想**：将连续的一组权重（通常32或256个）打包成一个"块"，每块共享一个缩放因子。

#### Q4_0块结构详解

**源码位置**：`ggml/include/ggml.h` (第400-420行)

```c
// Q4_0: 4-bit对称量化，32个元素/块
// 每块32个权重，每个权重4比特
// 压缩率：32个FP32(128字节) -> 18字节，压缩比 ~7x

struct block_q4_0 {
    float d;              // delta/scale，2字节
    uint8_t qs[16];       // 32个4位权重，打包成16字节
                          // qs[i]的高4位 = 权重2*i
                          // qs[i]的低4位 = 权重2*i+1
};

static_assert(sizeof(block_q4_0) == sizeof(float) + QK4_0/2, "wrong block_q4_0 size");
// sizeof(block_q4_0) = 4 + 16 = 20字节？不对，实际是18字节（有对齐）
```

**内存布局图解**：
```
原始FP32数据（32个元素）: 128字节
[0.5] [-0.3] [1.2] ... [0.8]
  ↓ 量化（找到最大绝对值，计算scale）

Q4_0块：18字节
┌─────────────────────────────────────┐
│  d (float)  │  qs[0] │ qs[1] │ ... │ qs[15] │
│  4字节      │  1字节 │ 1字节 │     │ 1字节  │
├─────────────────────────────────────┤
│  scale=0.1  │w0|w1  │w2|w3  │ ... │w30|w31 │
│             │4b+4b  │4b+4b  │     │4b+4b   │
└─────────────────────────────────────┘

反量化：
  weight[i] = ((qs[i/2] >> (4*(i%2))) & 0xF - 8) * d
  // 解释：unpack 4-bit，映射到[-8, 7]，乘以scale
```

#### Q4_1块结构（非对称）

**源码位置**：`ggml/include/ggml.h` (第420-440行)

```c
// Q4_1: 4-bit非对称量化，32个元素/块
struct block_q4_1 {
    float d;              // delta/scale
    float m;              // min/zero point
    uint8_t qs[16];       // 32个4位权重
};
// sizeof(block_q4_1) = 4 + 4 + 16 = 24字节

// 反量化公式：
// weight[i] = ((qs[i/2] >> (4*(i%2))) & 0xF) * d + m
// 范围映射到 [m, m+15*d]
```

### 5.2.2 量化过程详解

**源码位置**：`ggml/src/ggml-quants.c` - `quantize_row_q4_0_reference()` (第1000-1100行)

```c
// 量化一行FP32数据为Q4_0
void quantize_row_q4_0_reference(
    const float * x,           // 输入：原始FP32数据
    block_q4_0 * y,            // 输出：量化后的块数组
    int64_t k) {               // 元素数量（必须是32的倍数）

    const int64_t nb = k / QK4_0;  // 块数

    for (int64_t i = 0; i < nb; i++) {
        // ① 找到当前块的最大绝对值
        float amax = 0.0f;
        for (int l = 0; l < QK4_0; l++) {
            const float v = x[i*QK4_0 + l];
            amax = fmaxf(amax, fabsf(v));
        }

        // ② 计算scale（d = max / 7，因为4位范围是[-8, 7]）
        const float d = amax / 7.0f;
        const float id = d ? 1.0f/d : 0.0f;
        y[i].d = d;

        // ③ 量化每个元素
        for (int l = 0; l < QK4_0; l += 2) {
            // 映射到[0, 15]范围，然后打包成4位
            const float v0 = x[i*QK4_0 + l + 0]*id;
            const float v1 = x[i*QK4_0 + l + 1]*id;

            const uint8_t vi0 = (int8_t)roundf(v0) + 8;  // [-7, 7] -> [1, 15]
            const uint8_t vi1 = (int8_t)roundf(v1) + 8;

            y[i].qs[l/2] = (vi0 & 0xF) | ((vi1 & 0xF) << 4);  // 打包
        }
    }
}
```

### 5.2.3 量化矩阵乘法 —— 在压缩空间计算

**核心优化**：两个量化矩阵相乘时，不需要完全反量化，可以部分展开计算。

**源码位置**：`ggml/src/ggml-quants.c` - `ggml_vec_dot_q4_0_q8_0()` (第3000-3200行)

```c
// 计算两个量化向量的点积
// x: Q4_0量化向量（来自权重矩阵）
// y: Q8_0量化向量（来自激活，通常Q8精度更高）
float ggml_vec_dot_q4_0_q8_0(
    int n,                      // 向量长度
    const block_q4_0 * x,       // 量化权重
    const block_q8_0 * y) {     // 量化激活

    float sumf = 0.0f;
    const int nb = n / QK8_0;   // 块数

    for (int i = 0; i < nb; i++) {
        // ① 获取每块的scale
        const float dx = x[i].d;  // Q4_0 scale
        const float dy = y[i].d;  // Q8_0 scale

        // ② 计算块内点积（在整数空间）
        int sumi = 0;
        for (int j = 0; j < QK4_0/2; j++) {
            // 解包Q4_0的两个4位权重
            const uint8_t v = x[i].qs[j];
            const int vi0 = (v & 0xF) - 8;    // 高4位
            const int vi1 = (v >> 4) - 8;     // 低4位

            // 对应的Q8_0值（8位，已存储）
            const int yi0 = y[i].qs[2*j + 0];
            const int yi1 = y[i].qs[2*j + 1];

            // 整数点积累加
            sumi += vi0 * yi0 + vi1 * yi1;
        }

        // ③ 应用scale（在浮点空间）
        sumf += sumi * dx * dy;
    }

    return sumf;
}
```

**优化要点**：
1. **块级计算**：按块计算，复用scale
2. **整数点积**：大部分计算在整数域，更快
3. **延迟反量化**：只在最后一步乘以scale

---

## 5.3 自定义量化类型

### 5.3.1 量化类型对比表

**源码位置**：`ggml/include/ggml.h` (第200-250行)

```c
enum ggml_type {
    GGML_TYPE_F32  = 0,   // 32位浮点
    GGML_TYPE_F16  = 1,   // 16位浮点
    GGML_TYPE_Q4_0 = 2,   // 4位对称，32元素/块，18字节/块
    GGML_TYPE_Q4_1 = 3,   // 4位非对称，32元素/块，24字节/块
    GGML_TYPE_Q5_0 = 6,   // 5位对称，32元素/块，22字节/块
    GGML_TYPE_Q5_1 = 7,   // 5位非对称，32元素/块，28字节/块
    GGML_TYPE_Q8_0 = 8,   // 8位对称，32元素/块，36字节/块
    GGML_TYPE_Q8_1 = 9,   // 8位非对称，32元素/块，40字节/块
    // ... 更多IQ系列
};
```

| 类型 | 位宽 | 块大小 | 块字节数 | 压缩比 | 适用场景 |
|-----|------|-------|---------|-------|---------|
| FP32 | 32 | - | 4 | 1x | 训练 |
| FP16 | 16 | - | 2 | 2x | 混合精度 |
| Q8_0 | 8 | 32 | 36 | 3.5x | 高精度推理 |
| Q5_0 | 5 | 32 | 22 | 5.8x | 中等精度 |
| Q4_0 | 4 | 32 | 18 | 7.1x | 平衡之选 |
| Q4_1 | 4 | 32 | 24 | 5.3x | 非对称分布 |

### 5.3.2 IQ（改进量化）系列

**IQ2_XXS**: 2-bit极端压缩
- 目标：手机/嵌入式部署
- 技巧：重要权重用更高精度，次要权重降低精度
- 感知损失：通过重要性矩阵（imatrix）指导量化

**源码位置**：`ggml/src/ggml-quants.c` (第5000+行)

```c
// IQ系列使用混合精度策略
// 例如IQ2_XXS：
// - 大部分权重2-bit
// - 异常值（outliers）用更高精度存储
// - 每块有额外的scale/指数信息
```

---

## 设计中的取舍

### 为什么Q4_0块大小是32而不是64或256？

| 块大小 | 优点 | 缺点 | GGML选择 |
|-------|------|------|---------|
| 16 | scale精度高 | 开销大（25%） | 否 |
| 32 | 平衡 | 平衡 | **是** |
| 64 | 开销小（6%） | 局部精度损失 | 部分使用 |
| 256 | 开销极小 | 长尾效应明显 | K-quants使用 |

**Q4_0 (32元素/块)**:
- 开销：18字节存储32个数，vs 128字节原始 = 14% overhead
- 局部性：32个连续的权重通常分布相似

**K-quants (256元素/块)**:
- 开销更低，但用更复杂的scale分配策略补偿

### 为什么量化激活常用Q8_0而不是Q4_0？

```
权重矩阵 W: Q4_0量化（静态，只读）
激活向量 x: Q8_0量化（动态，每次推理不同）

原因：
1. 激活比权重对精度更敏感
2. Q8_0 * Q8_0的矩阵乘法更快（SIMD友好）
3. 激活占用内存少，量化到8位收益足够
```

---

## 动手练习

### 练习1：计算量化后大小
给定一个形状为 `[4096, 4096]` 的FP32矩阵，计算：
1. 原始大小（字节）
2. Q4_0量化后大小
3. Q8_0量化后大小
4. 验证你的计算与 `ggml/ggml.c` 中的 `ggml_type_size` 一致

### 练习2：理解量化误差
阅读 `tests/test-quantize-perf.cpp`，运行量化精度测试：
```bash
./test-quantize-perf
```
观察不同量化类型的误差指标（RMSE、cosine similarity）。

### 练习3：自定义量化类型设计
假设你要设计一种新的量化格式 `Q3_0`（3-bit对称量化），回答：
1. 每块多少元素最合适？
2. 块结构如何定义？
3. 压缩比是多少？
4. 反量化公式是什么？

参考 `ggml/include/ggml.h` 中的已有定义，写出你的结构体。

---

## 本课小结

| 概念 | 一句话总结 |
|------|-----------|
| Q4_0 | 4位对称量化，32元素/块，压缩比7x |
| Q4_1 | 4位非对称量化，多存储min值 |
| block_q4_0 | 含scale+16字节权重数据 |
| quantize_row | 找max算scale，映射到[0,15] |
| vec_dot_q4_0 | 整数点积+延迟scale，效率最高 |
| IQ系列 | 智能混合精度，极致压缩 |

---

*本章对应源码版本：master (2026-04-07)*
