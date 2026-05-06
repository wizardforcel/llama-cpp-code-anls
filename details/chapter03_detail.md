# 第3章 GGML 核心架构 —— 理解"张量世界的乐高积木"

## 学习目标
1. 掌握GGML的设计哲学与核心概念
2. 深入理解ggml_tensor数据结构
3. 理解计算图（ggml_cgraph）的构建与执行机制
4. 掌握上下文（ggml_context）的内存管理

---

## 生活类比：乐高积木工厂

想象GGML是一座**智能乐高工厂**：

- **ggml_tensor（张量）** = 标准化的乐高积木块
  - 有不同形状（维度）和颜色（数据类型）
  - 每块有编号和标签（元数据）
- **ggml_context（上下文）** = 积木仓库
  - 预先分配一大块空间（内存池）
  - 按需取出积木，用完统一归还
- **ggml_cgraph（计算图）** = 组装说明书
  - 告诉工厂按什么顺序组装积木
  - 哪些积木需要先准备（依赖关系）
- **计算执行** = 自动化组装流水线
  - 按说明书顺序加工每块积木
  - 多工位并行作业（多线程/SIMD）

就像乐高工厂需要精确管理积木和组装流程，GGML需要精确管理张量和计算流程。

---

## 源码地图

```
ggml/include/ggml.h (第1-900行)
├── struct ggml_tensor       # 张量结构体
│   ├── type                 # 数据类型(enum ggml_type)
│   ├── backend              # 所属后端
│   ├── ne[GGML_MAX_DIMS]    # 各维度元素数
│   ├── nb[GGML_MAX_DIMS]    # 各维度字节步长
│   ├── op                   # 产生此张量的操作
│   ├── src[]                # 输入张量
│   ├── data                 # 数据指针
│   └── name                 # 名称
├── struct ggml_cgraph       # 计算图结构体
│   ├── n_nodes/n_leafs      # 节点/叶子数
│   ├── nodes[]/leafs[]      # 节点/叶子数组
│   └── order                # 评估顺序
├── struct ggml_context      # 上下文结构体
│   ├── mem_data/mem_size    # 内存池
│   ├── n_objects            # 对象数量
│   └── objects_begin/end    # 对象链表
├── enum ggml_type           # 数据类型枚举
│   ├── GGML_TYPE_F32/F16    # 浮点类型
│   ├── GGML_TYPE_Q4_0/Q4_1  # 4位量化
│   ├── GGML_TYPE_Q5_0/Q5_1  # 5位量化
│   ├── GGML_TYPE_Q8_0       # 8位量化
│   └── GGML_TYPE_IQ*        # 改进量化
├── enum ggml_op             # 操作类型枚举
│   ├── GGML_OP_ADD/MUL      # 基础运算
│   ├── GGML_OP_MUL_MAT      # 矩阵乘法
│   ├── GGML_OP_SILU/RELU    # 激活函数
│   ├── GGML_OP_NORM/RMS_NORM# 归一化
│   └── GGML_OP_ROPE         # 旋转位置编码
└── 核心API声明
    ├── ggml_init()          # 初始化上下文
    ├── ggml_new_tensor*()   # 创建张量
    ├── ggml_build_forward() # 构建前向图
    ├── ggml_graph_compute() # 执行计算图
    └── ggml_*()             # 各种算子

ggml/src/ggml.c
├── ggml_new_tensor()        # 创建张量（第5000-5500行）
├── ggml_build_forward()     # 构建计算图（第15000-16000行）
├── ggml_graph_compute()     # 执行计算图（第16000-17000行）
└── 各种算子实现

ggml/src/ggml-alloc.c
├── ggml_gallocr_new()       # 创建分配器
├── ggml_gallocr_alloc_graph() # 为计算图分配内存
└── ggml_gallocr_free()      # 释放分配器
```

---

## 3.1 GGML 设计概述

### 3.1.1 从PyTorch/TensorFlow到GGML

**设计对比**：
| 特性 | PyTorch | TensorFlow | GGML |
|-----|---------|-----------|------|
| 主要用途 | 训练+推理 | 训练+推理 | 推理优化 |
| 依赖 | Python+大量库 | Python+大量库 | 零依赖C/C++ |
| 运行时 | 动态图 | 静态图 | 静态计算图 |
| 内存管理 | 自动GC | 手动+自动 | 手动内存池 |
| 部署 | 重量级 | 重量级 | 轻量级 |

### 3.1.2 GGML核心设计理念

**源码位置**：`ggml/include/ggml.h` (第1-100行)

1. **推理优先**：专注推理场景优化，舍弃训练相关复杂度
2. **零依赖**：纯C实现，不依赖Python/NumPy等
3. **静态图**：预构建计算图，运行时零开销
4. **内存池**：预分配内存，避免动态分配碎片
5. **多后端**：统一抽象，支持CPU/GPU异构计算

---

## 3.2 张量（ggml_tensor）数据结构

### 3.2.1 结构体定义

**源码位置**：`ggml/include/ggml.h` (第500-600行)

```c
struct ggml_tensor {
    // 类型系统
    enum ggml_type type;           // 数据类型（FP32/FP16/Q4_0等）
    enum ggml_backend_type backend; // 所属后端

    // 维度信息
    int64_t ne[GGML_MAX_DIMS];     // 各维度元素数（number of elements）
    size_t  nb[GGML_MAX_DIMS];     // 各维度步长（number of bytes）

    // 运算图相关
    enum ggml_op op;               // 产生此张量的操作类型
    struct ggml_tensor * src[GGML_MAX_SRC];  // 输入张量
    struct ggml_tensor * grad;               // 梯度张量（训练用）

    // 内存管理
    struct ggml_backend_buffer * buffer;
    void * data;                   // 数据指针

    // 名称（调试用途）
    char name[GGML_MAX_NAME];
};
```

### 3.2.2 维度与步长

**内存布局示例**：2D矩阵 [3, 4]
```
形状: 3行4列，FP32类型（4字节/元素）

ne[0] = 4  (列数)
ne[1] = 3  (行数)

nb[0] = 4  (一个元素的字节数)
nb[1] = 16 (一行的字节数 = 4列 * 4字节)

地址计算: &tensor[i][j] = data + i * nb[1] + j * nb[0]
```

**源码示例**：`ggml/src/ggml.c` - `ggml_new_tensor_2d()`

```c
struct ggml_tensor * ggml_new_tensor_2d(
        struct ggml_context * ctx,
        enum ggml_type type,
        int64_t ne0,   // 第一维度大小
        int64_t ne1) { // 第二维度大小
    const int64_t ne[2] = { ne0, ne1 };
    return ggml_new_tensor(ctx, type, 2, ne);
}
```

### 3.2.3 数据类型系统

**源码位置**：`ggml/include/ggml.h` (第200-300行)

```c
enum ggml_type {
    // 浮点类型
    GGML_TYPE_F32  = 0,   // 32位浮点
    GGML_TYPE_F16  = 1,   // 16位浮点

    // 量化类型
    GGML_TYPE_Q4_0 = 2,   // 4位量化，每块32个元素
    GGML_TYPE_Q4_1 = 3,   // 4位量化，带偏移
    GGML_TYPE_Q8_0 = 8,   // 8位量化
    // ... 更多类型
};
```

---

## 3.3 计算图（ggml_cgraph）机制

### 3.3.1 计算图结构

**源码位置**：`ggml/include/ggml.h` (第700-800行)

```c
struct ggml_cgraph {
    int n_nodes;          // 节点数量
    int n_leafs;          // 叶子节点数量

    struct ggml_tensor ** nodes;   // 计算节点数组
    struct ggml_tensor ** leafs;   // 叶子节点数组

    // 执行顺序
    enum ggml_cgraph_eval_order order;

    // 性能统计
    int64_t perf_runs;
    int64_t perf_cycles;
};
```

### 3.3.2 计算图构建示例

```c
// 构建 y = x * W + b 的计算图
struct ggml_context * ctx = ggml_init(params);

// 定义输入和参数
struct ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 768, 1);
struct ggml_tensor * W = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 768, 3072);
struct ggml_tensor * b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 3072);

// 构建计算图
struct ggml_tensor * tmp = ggml_mul_mat(ctx, W, x);  // tmp = W @ x
struct ggml_tensor * y   = ggml_add(ctx, tmp, b);    // y = tmp + b

// 创建前向图
struct ggml_cgraph * graph = ggml_build_forward(y);

// 执行计算
ggml_graph_compute_with_ctx(ctx, graph, n_threads);
```

### 3.3.3 计算图执行

**源码位置**：`ggml/src/ggml.c` (第16000-17000行)

```c
void ggml_graph_compute(struct ggml_cgraph * graph,
                        struct ggml_cplan * cplan) {
    // 按拓扑顺序遍历节点
    for (int i = 0; i < graph->n_nodes; i++) {
        struct ggml_tensor * node = graph->nodes[i];

        // 根据操作类型执行
        switch (node->op) {
            case GGML_OP_ADD:
                ggml_compute_forward_add(node);
                break;
            case GGML_OP_MUL_MAT:
                ggml_compute_forward_mul_mat(node);
                break;
            // ... 更多操作
        }
    }
}
```

---

## 3.4 上下文（ggml_context）管理

### 3.4.1 内存池设计

**源码位置**：`ggml/include/ggml.h` (第350-400行)

```c
struct ggml_context {
    // 内存池
    void * mem_data;         // 内存池起始地址
    size_t mem_size;         // 总大小
    size_t mem_offset;       // 当前分配偏移

    // 对象管理
    int n_objects;
    struct ggml_object * objects_begin;
    struct ggml_object * objects_end;

    // 计算图缓存
    struct ggml_cgraph * graph;
};
```

### 3.4.2 上下文生命周期

```c
// 1. 初始化参数
struct ggml_init_params params = {
    .mem_size   = 512*1024*1024,  // 512MB内存池
    .mem_buffer = NULL,            // 由GGML分配
    .no_alloc   = false,
};

// 2. 创建上下文
struct ggml_context * ctx = ggml_init(params);

// 3. 创建张量（从内存池分配）
struct ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 100, 100);

// 4. 构建计算图并执行
struct ggml_tensor * c = ggml_add(ctx, a, b);
struct ggml_cgraph * graph = ggml_build_forward(c);
ggml_graph_compute_with_ctx(ctx, graph, 4);

// 5. 释放上下文（统一释放所有内存）
ggml_free(ctx);
```

### 3.4.3 内存分配器

**源码位置**：`ggml/src/ggml-alloc.c`

```c
// 分配器可以复用张量内存
struct ggml_gallocr * allocr = ggml_gallocr_new(
    ggml_backend_get_default()
);

// 为计算图分配/优化内存
ggml_gallocr_alloc_graph(allocr, graph);

// 关键优化：
// - 生命周期不重叠的张量共享内存
// - 计算完成后立即释放中间结果
```

---

## 动手练习

1. **张量创建**：编写程序创建各种形状的张量，打印ne和nb数组
2. **内存计算**：计算Q4_0量化下，形状为[4096, 4096]的矩阵占用多少内存
3. **计算图构建**：手动构建一个y = (a + b) * c的计算图并执行

---

## 本课小结

| 概念 | 一句话总结 |
|------|-----------|
| ggml_tensor | 张量结构体，包含维度、步长、数据指针 |
| ggml_cgraph | 计算图，描述操作顺序和依赖关系 |
| ggml_context | 上下文，内存池管理器 |
| 内存池 | 预分配大块内存，避免频繁malloc/free |
| 静态图 | 预构建计算图，运行时零开销 |

---

*本章对应源码版本：master (2026-04-07)*
