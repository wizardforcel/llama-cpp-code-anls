# 第12章 高级缓存策略 —— 长文本与多会话的"智慧管家"

## 学习目标
1. 理解滑动窗口注意力（SWA）的实现原理
2. 掌握多序列缓存管理（Multi-Sequence）
3. 了解ISWA（Interleaved SWA）缓存架构
4. 掌握长文本处理的缓存优化技巧

---

## 生活类比：图书馆的"智能书架系统"

想象llama.cpp的KV缓存是一位**管理巨型图书馆的智慧管家**：

- **普通缓存** = 传统书架
  - 每本书（token）按顺序摆放
  - 空间满了就从头开始覆盖
  - 适合短文本，长文本会丢失前文

- **SWA（滑动窗口注意力）** = 热门书籍专区
  - 只保留最近N本书（如4096本）
  - 超出窗口的旧书自动移入"冷存储"
  - 适合无限长文本生成

- **多序列缓存** = 多用户借阅系统
  - 每个读者（序列）有自己的借阅区域
  - 可以独立添加、删除、复制书籍
  - 支持多对话并行处理

- **ISWA** = 混合书架布局
  - 普通书籍区 + 热门书籍区
  - 不同层使用不同策略
  - 平衡长文本能力和上下文理解

就像图书馆需要针对不同读者和书籍类型优化布局，KV缓存需要根据模型架构和使用场景选择最佳策略。

---

## 源码地图

```
src/llama-kv-cache.h
├── llama_kv_cache           # 标准KV缓存
│   ├── n_swa                # 滑动窗口大小
│   ├── swa_type             # SWA类型
│   └── is_masked_swa()      # SWA掩码判断
└── llama_kv_cache_context   # 缓存操作上下文

src/llama-kv-cache-iswa.h
├── llama_kv_cache_iswa      # ISWA缓存（组合标准+SWA）
│   ├── kv_base              # 非SWA层缓存
│   └── kv_swa               # SWA层缓存
└── llama_kv_cache_iswa_context

src/llama-hparams.h
├── is_swa()                 # 判断某层是否使用SWA
├── n_swa()                  # 获取SWA窗口大小
└── is_masked_swa()          # SWA掩码计算
```

---

## 12.1 滑动窗口注意力（SWA）

### 12.1.1 SWA设计动机

**问题背景**：
- 标准Attention的复杂度是O(n^2)，长文本时KV缓存爆炸
- 但远距离token的注意力权重通常很小
- 能否只保留最近的K个token的KV？

**SWA解决方案**：
- 只缓存最近`n_swa`个token的KV
- 注意力计算时只attend到窗口内的token
- 内存复杂度从O(n)降为O(n_swa)

### 12.1.2 SWA类型定义

**源码位置**：`src/llama-hparams.h`

```cpp
enum llama_swa_type {
    LLAMA_SWA_TYPE_NONE = 0,      // 不使用SWA
    LLAMA_SWA_TYPE_NORMAL = 1,    // 标准SWA（Mistral风格）
};

// SWA掩码判断
static bool is_masked_swa(
        uint32_t       n_swa,      // 窗口大小
        llama_swa_type swa_type,   // SWA类型
        llama_pos      p0,         // KV位置
        llama_pos      p1) {       // 当前查询位置
    if (swa_type == LLAMA_SWA_TYPE_NONE || n_swa == 0) {
        return false;  // 不掩码
    }
    // 如果KV位置距离当前位置超过窗口大小，掩码掉
    return p1 - p0 > (llama_pos) n_swa;
}
```

### 12.1.3 SWA在KV缓存中的应用

**源码位置**：`src/llama-kv-cache.cpp` (第960-968行)

```cpp
// 在find_slot中判断cell是否可用
if (!can_use && cells.seq_count(idx) == 1) {
    const llama_pos pos_cell = cells.pos_get(idx);
    const llama_seq_id seq_id_cell = cells.seq_get(idx);

    // SWA掩码：如果cell位置超出窗口范围，可以复用
    if (llama_hparams::is_masked_swa(n_swa, swa_type, 
                                     pos_cell, 
                                     cells.seq_pos_max(seq_id_cell) + 1)) {
        can_use = true;  // 这个cell虽然被占用，但SWA会掩码它，可以覆盖
    }
}
```

**SWA缓存复用图解**：
```
时间轴 →
位置:   0    1    2    ...  4092 4093 4094 4095 4096 4097
        │    │    │         │    │    │    │    │    │
SWA窗口(4096):                    └────────────────────┘
                                    当前窗口范围

Cell状态:
        [已占][已占][已占] ... [已占][已占][已占][已占][可复用][可复用]
                              ↑                        ↑
                           窗口起点                  当前位置

当生成第4097个token时:
- 位置0的KV已经被掩码（超出窗口）
- 可以安全地覆盖位置0的cell
- 实现循环复用，支持无限长文本
```

### 12.1.4 SWA的注意力掩码

**源码位置**：`src/llama-kv-cache.cpp` (第1546-1550行)

```cpp
// 在KQ掩码中应用SWA
if (swa) {
    if (llama_hparams::is_masked_swa(n_swa, swa_type, p0, p1)) {
        goto skip;  // 掩码掉超出窗口的KV
    }
}

// skip标签处将mask设为-INFINITY
data[idst + j] = -INFINITY;
```

**SWA掩码效果**：
```
KQ Mask矩阵（n_swa=4）:
       KV位置
       0    1    2    3    4    5    6    7
      ┌────┬────┬────┬────┬────┬────┬────┬────┐
Q  0  │ 0  │-inf│-inf│-inf│-inf│-inf│-inf│-inf│  只能attend到自己
u  1  │ 0  │ 0  │-inf│-inf│-inf│-inf│-inf│-inf│  只能attend到前2个
e  2  │ 0  │ 0  │ 0  │-inf│-inf│-inf│-inf│-inf│  只能attend到前3个
r  3  │ 0  │ 0  │ 0  │ 0  │-inf│-inf│-inf│-inf│  窗口满，attend到4个
y  4  │-inf│ 0  │ 0  │ 0  │ 0  │-inf│-inf│-inf│  位置0被掩码，滑动窗口
位 5  │-inf│-inf│ 0  │ 0  │ 0  │ 0  │-inf│-inf│
置 6  │-inf│-inf│-inf│ 0  │ 0  │ 0  │ 0  │-inf│
   7  │-inf│-inf│-inf│-inf│ 0  │ 0  │ 0  │ 0  │
      └────┴────┴────┴────┴────┴────┴────┴────┘
```

---

## 12.2 ISWA（交错SWA）缓存架构

### 12.2.1 为什么需要ISWA

**Mistral等模型的设计**：
- 部分层使用SWA（如偶数层）
- 部分层不使用SWA（如奇数层）
- 兼顾长文本能力和上下文理解

**实现挑战**：
- 需要维护两个独立的KV缓存
- 需要协调两个缓存的更新
- 需要统一对外接口

### 12.2.2 ISWA类设计

**源码位置**：`src/llama-kv-cache-iswa.h`

```cpp
class llama_kv_cache_iswa : public llama_memory_i {
public:
    llama_kv_cache_iswa(
        const llama_model & model,
        ggml_type   type_k,
        ggml_type   type_v,
        bool   v_trans,
        bool   offload,
        bool   swa_full,      // SWA层是否使用全量缓存
        bool   unified,       // 统一流模式
        uint32_t   kv_size,
        uint32_t   n_seq_max,
        uint32_t   n_ubatch,
        uint32_t   n_pad,
        const layer_filter_cb & filter,
        const  layer_reuse_cb & reuse);

    // 获取底层缓存实例
    llama_kv_cache * get_base() const;  // 非SWA层缓存
    llama_kv_cache * get_swa () const;  // SWA层缓存

private:
    const bool unified;
    std::unique_ptr<llama_kv_cache> kv_base;  // 标准缓存
    std::unique_ptr<llama_kv_cache> kv_swa;   // SWA缓存
};
```

### 12.2.3 ISWA初始化流程

```cpp
llama_kv_cache_iswa::llama_kv_cache_iswa(...)
    : hparams(model.hparams), unified(unified) {

    // ① 创建标准缓存（用于非SWA层）
    auto filter_base = [&](uint32_t il) -> bool {
        return !hparams.is_swa(il);  // 只保留非SWA层
    };

    kv_base = std::make_unique<llama_kv_cache>(
        model, type_k, type_v, v_trans, offload, unified,
        kv_size, n_seq_max, n_pad, 0,  // n_swa=0
        LLAMA_SWA_TYPE_NONE,           // 不使用SWA
        filter_base, reuse);

    // ② 创建SWA缓存（用于SWA层）
    auto filter_swa = [&](uint32_t il) -> bool {
        return hparams.is_swa(il);  // 只保留SWA层
    };

    // SWA缓存大小：如果swa_full则使用kv_size，否则使用n_swa
    const uint32_t kv_size_swa = swa_full ? kv_size : n_swa;

    kv_swa = std::make_unique<llama_kv_cache>(
        model, type_k, type_v, v_trans, offload, unified,
        kv_size_swa, n_seq_max, n_pad, n_swa,
        hparams.swa_type,
        filter_swa, reuse);
}
```

### 12.2.4 ISWA架构图解

```
┌─────────────────────────────────────────────────────────────────┐
│                      ISWA缓存架构                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────────┐      ┌─────────────────────┐          │
│  │    kv_base (标准)    │      │    kv_swa (滑动窗口) │          │
│  │   用于非SWA层        │      │   用于SWA层          │          │
│  ├─────────────────────┤      ├─────────────────────┤          │
│  │  Layer 1 (非SWA)    │      │  Layer 0 (SWA)      │          │
│  │  Layer 3 (非SWA)    │      │  Layer 2 (SWA)      │          │
│  │  Layer 5 (非SWA)    │      │  Layer 4 (SWA)      │          │
│  │  ...                │      │  ...                │          │
│  ├─────────────────────┤      ├─────────────────────┤          │
│  │  kv_size = 8192     │      │  kv_size = 4096     │          │
│  │  (全量缓存)          │      │  (窗口大小)          │          │
│  └─────────────────────┘      └─────────────────────┘          │
│                                                                 │
│  模型层分配示例（Mistral 8x7B）:                                 │
│  ┌────┬────┬────┬────┬────┬────┬────┬────┐                     │
│  │ L0 │ L1 │ L2 │ L3 │ L4 │ L5 │ ...     │                     │
│  │SWA │base│SWA │base│SWA │base│ ...     │                     │
│  └────┴────┴────┴────┴────┴────┴────┴────┘                     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 12.3 多序列缓存管理

### 12.3.1 多序列场景

**使用场景**：
1. **批处理推理**：同时处理多个独立对话
2. **束搜索（Beam Search）**：维护多个候选序列
3. **对话分叉**：从某点创建多个分支继续生成

### 12.3.2 流（Stream）概念

**源码位置**：`src/llama-kv-cache.h` (第229-261行)

```cpp
class llama_kv_cache {
    // 序列到流的映射
    std::vector<uint32_t> seq_to_stream;

    // 每流的cell数组
    std::vector<llama_kv_cells> v_cells;

    // 每流的头指针（循环缓冲区用）
    std::vector<uint32_t> v_heads;

    // 流数量
    const uint32_t n_stream;
};
```

**流模式对比**：

| 模式 | n_stream | 特点 | 适用场景 |
|-----|---------|------|---------|
| 统一流 | 1 | 所有序列共享cell数组 | 序列数少，需要共享前缀 |
| 多流 | n_seq_max | 每序列独立cell数组 | 序列数多，需要完全隔离 |

### 12.3.3 多序列批次处理

**源码位置**：`src/llama-kv-cache.cpp` (第614-649行)

```cpp
llama_memory_context_ptr llama_kv_cache::init_batch(
        llama_batch_allocr & balloc,
        uint32_t n_ubatch,
        bool embd_all) {

    // ① 将批次分割为ubatches
    std::vector<llama_ubatch> ubatches;
    while (true) {
        // 单流模式：简单分割
        // 多流模式：按序列均匀分割
        auto ubatch = n_stream == 1 ? 
            balloc.split_simple(n_ubatch) : 
            balloc.split_equal(n_ubatch, true);

        if (ubatch.n_tokens == 0) break;
        ubatches.push_back(std::move(ubatch));
    }

    // ② 为每个ubatch查找槽位
    auto sinfos = prepare(ubatches);
    if (sinfos.empty()) {
        return error_context;  // 找不到足够空间
    }

    // ③ 创建处理上下文
    return std::make_unique<llama_kv_cache_context>(
        this, std::move(sinfos), std::move(ubatches));
}
```

**多序列处理图解**：
```
批次输入（6个token，3个序列）:
┌─────────────────────────────────────────────────────┐
│ token:  [T0] [T1] [T2] [T3] [T4] [T5]              │
│ seq_id: [ 0 ] [ 0 ] [ 1 ] [ 1 ] [ 2 ] [ 2 ]        │
│ pos:    [ 10] [ 11] [ 5 ] [ 6 ] [ 20] [ 21]        │
└─────────────────────────────────────────────────────┘

多流模式处理:
┌─────────────────────────────────────────────────────┐
│ Stream 0: [T0] [T1]  → 写入cells[0]的位置10,11      │
│ Stream 1: [T2] [T3]  → 写入cells[1]的位置5,6        │
│ Stream 2: [T4] [T5]  → 写入cells[2]的位置20,21      │
└─────────────────────────────────────────────────────┘

统一流模式处理:
┌─────────────────────────────────────────────────────┐
│ 所有token写入同一个cell数组，通过seq_id区分归属       │
│ Cell 10: [T0], seq={0}                              │
│ Cell 11: [T1], seq={0}                              │
│ Cell 5:  [T2], seq={1}                              │
│ ...                                                 │
└─────────────────────────────────────────────────────┘
```

---

## 12.4 长文本优化技巧

### 12.4.1 K-Shift（位置偏移）

**问题**：当KV缓存满时，新token无处存放。

**解决方案**：
- 删除最旧的token（seq_rm）
- 将所有剩余token的位置减去偏移量
- 应用K-Shift更新RoPE编码

**源码位置**：`src/llama-kv-cache.cpp` (第1787-1830行)

```cpp
ggml_cgraph * llama_kv_cache::build_graph_shift(
        llm_graph_result * res,
        llama_context * lctx) const {

    // 为每层构建K-Shift计算图
    for (const auto & layer : layers) {
        // 获取该层的K缓存
        ggml_tensor * k = ggml_view_3d(ctx, layer.k, ...);

        // 应用RoPE偏移
        ggml_tensor * cur = build_rope_shift(
            cparams, ctx, k, 
            inp->k_shift,  // 偏移量
            inp->k_rot,    // 旋转矩阵（量化时使用）
            rope_factors,
            freq_base_l, freq_scale_l,
            il);

        ggml_build_forward_expand(gf, cur);
    }

    return gf;
}
```

### 12.4.2 缓存压缩策略

**策略1：动态SWA**
- 根据序列长度自动启用SWA
- 短文本用全量缓存，长文本用窗口

**策略2：层共享**
- 相邻层共享KV缓存
- 减少内存占用，轻微质量损失

**策略3：量化缓存**
```cpp
// 使用Q8_0或Q4_0量化KV缓存
llama_kv_cache kv_cache(
    model,
    GGML_TYPE_Q8_0,  // K缓存8位量化
    GGML_TYPE_Q4_0,  // V缓存4位量化
    ...
);
```

### 12.4.3 内存与质量的权衡

| 配置 | 内存占用 | 长文本能力 | 质量 |
|-----|---------|-----------|------|
| F16全量 | 100% | 受限于显存 | 最佳 |
| Q8_0全量 | 50% | 受限于显存 | 优秀 |
| Q4_0全量 | 25% | 受限于显存 | 良好 |
| SWA (4096) | 与长度无关 | 无限 | 良好 |
| ISWA | 中等 | 很长 | 优秀 |

---

## 设计中的取舍

### 为什么ISWA比纯SWA更好？

**纯SWA**：
- 优点：内存固定，支持无限长文本
- 缺点：所有层都受限，可能丢失重要上下文

**ISWA**：
- 优点：部分层保留全量上下文，部分层使用窗口
- 缺点：实现复杂，需要维护两个缓存

**llama.cpp的选择**：
- 支持纯SWA、纯标准、ISWA三种模式
- 通过模型配置自动选择最优策略

### 统一流 vs 多流

**统一流**：
- 适合：序列数少，需要共享系统提示
- 缺点：序列数多时cell碎片化严重

**多流**：
- 适合：序列数多，需要完全隔离
- 缺点：内存预分配，可能浪费

---

## 动手练习

### 练习1：计算ISWA内存占用
给定配置：
- n_layer = 32（其中16层SWA，16层非SWA）
- kv_size = 8192
- n_swa = 4096
- type_k = Q8_0, type_v = Q8_0
- n_embd_k_gqa = n_embd_v_gqa = 1024

计算：
1. 纯标准缓存的总内存
2. 纯SWA缓存的总内存
3. ISWA缓存的总内存

**答案**：
```
1. 纯标准: 32 * 8192 * 1024 * 2 * 1B = 512 MB
2. 纯SWA: 32 * 4096 * 1024 * 2 * 1B = 256 MB
3. ISWA: 16 * 8192 * 1024 * 2 * 1B + 16 * 4096 * 1024 * 2 * 1B = 384 MB
```

### 练习2：分析SWA掩码
阅读 `src/llama-kv-cache.cpp` 第1546-1550行，回答：
1. SWA掩码在KQ Mask的哪个阶段应用？
2. 被掩码的KV会如何处理？
3. 为什么SWA cell可以被复用？

### 练习3：设计缓存策略
假设你需要部署一个支持100K上下文的应用，但GPU只有24GB显存，设计一个合理的缓存策略。

**提示**：
- 考虑使用SWA或ISWA
- 考虑量化KV缓存
- 考虑层共享

---

## 本课小结

| 概念 | 一句话总结 |
|------|-----------|
| SWA | 滑动窗口注意力，固定内存支持无限长文本 |
| ISWA | 交错SWA，部分层用窗口，部分层用全量 |
| 多流 | 每序列独立的cell数组，完全隔离 |
| 统一流 | 所有序列共享cell数组，支持共享前缀 |
| K-Shift | 位置偏移，支持缓存内token移动 |
| seq_rm/cp | 序列操作，支持删除、复制、分叉 |

---

*本章对应源码版本：master (2026-04-07)*
