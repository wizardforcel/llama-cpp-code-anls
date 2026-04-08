# 第11章 KV缓存架构 —— 自注意力的"记忆外挂"

在Transformer模型的推理过程中，KV缓存（Key-Value Cache）是提升效率的关键技术。它避免了自注意力机制中的重复计算，将二次方复杂度降为线性。本章将深入解析llama.cpp的KV缓存架构。

## 学习目标

1. 理解KV缓存的核心作用和内存布局
2. 掌握`llama_kv_cache`的类设计与关键数据结构
3. 理解KV缓存的更新、查找和回收机制
4. 了解注意力旋转优化技术

## 生活类比：超级大脑的"记忆宫殿"

想象Transformer模型是一位拥有**超级记忆能力的大脑**：

- **KV缓存** = 记忆宫殿中的储物柜
  - K（Key）= 知识的索引卡片（用来快速查找相关内容）
  - V（Value）= 知识的详细内容（实际存储的信息）
  - 每个Transformer层都有自己的储物柜组
  
- **Cell（单元格）** = 单个储物格
  - 每个token对应一个储物格
  - 存储该token的K和V向量
  - 记录token的位置信息和所属序列
  
- **Slot查找** = 寻找空闲储物格
  - 循环缓冲区策略：从头开始找空位
  - 支持多序列：不同对话使用不同的储物区域
  
- **序列操作** = 记忆管理
  - `seq_rm` = 删除某段记忆
  - `seq_cp` = 复制记忆到新的序列
  - `seq_keep` = 只保留指定序列的记忆

就像记忆宫殿需要精心管理空间，KV缓存需要高效地分配、回收和复用内存。

---

## 11.1 KV缓存核心设计

### 11.1.1 为什么需要KV缓存

**问题背景**：
在Transformer的自注意力机制中，每个token需要与所有之前的token计算注意力分数。如果不使用缓存，每次生成都需要重新计算所有历史token的K和V。

**计算复杂度对比**：

| 方式 | 时间复杂度 | 空间复杂度 | 适用场景 |
|-----|-----------|-----------|---------|
| 无缓存 | O(n²) 每步 | O(1) | 仅训练 |
| 有缓存 | O(n) 每步 | O(n) | 推理生成 |

**KV缓存的收益**：
- 将二次方复杂度降为线性
- 避免重复计算历史token的K/V
- 支持增量式token生成

### 11.1.2 内存布局设计

**源码位置**：`src/llama-kv-cache.h`（第215-226行）

```cpp
struct kv_layer {
    uint32_t il;                          // 模型层索引
    ggml_tensor * k;                      // K缓存张量
    ggml_tensor * v;                      // V缓存张量
    std::vector<ggml_tensor *> k_stream;  // 每流的K视图
    std::vector<ggml_tensor *> v_stream;  // 每流的V视图
};

这段代码定义了KV缓存层的结构，每个Transformer层对应一个kv_layer，包含该层的K和V缓存张量，以及为每个流（stream）创建的K/V视图，用于多序列场景下的数据隔离。

```

**内存布局图解**：
```
┌─────────────────────────────────────────────────────────────┐
│                    KV缓存内存布局                            │
├─────────────────────────────────────────────────────────────┤
│  Layer 0 K缓存: [n_embd_k_gqa, kv_size, n_stream]            │
│  Layer 0 V缓存: [n_embd_v_gqa, kv_size, n_stream]            │
├─────────────────────────────────────────────────────────────┤
│  Layer 1 K缓存: [n_embd_k_gqa, kv_size, n_stream]            │
│  Layer 1 V缓存: [n_embd_v_gqa, kv_size, n_stream]            │
├─────────────────────────────────────────────────────────────┤
│  ... 更多层 ...                                              │
├─────────────────────────────────────────────────────────────┤
│  其中:                                                       │
│  - n_embd_k_gqa = n_head_kv * n_embd_head_k                  │
│  - n_embd_v_gqa = n_head_kv * n_embd_head_v                  │
│  - kv_size = 最大缓存token数（由n_ctx决定）                   │
│  - n_stream = 流数量（多序列时为n_seq_max，否则为1）          │
└─────────────────────────────────────────────────────────────┘
```

### 11.1.3 Cell单元格管理

**源码位置**：`src/llama-kv-cells.h`

```cpp
// KV单元格管理（每个token对应一个cell）
class llama_kv_cells {
public:
    // 单元格状态查询
    bool is_empty(uint32_t i) const;           // 是否为空
    bool seq_has(uint32_t i, llama_seq_id seq_id) const;  // 是否属于某序列
    llama_pos pos_get(uint32_t i) const;       // 获取位置
    
    // 单元格操作
    void pos_set(uint32_t i, llama_pos pos);   // 设置位置
    void seq_add(uint32_t i, llama_seq_id seq_id);  // 添加序列归属
    bool seq_rm(uint32_t i, llama_seq_id seq_id);   // 移除序列归属
    void rm(uint32_t i);                       // 清空单元格
    
    // 位置偏移（用于K-shift）
    bool pos_add(uint32_t i, llama_pos delta); // 增加位置偏移
    void pos_div(uint32_t i, int d);           // 位置除法（用于YaRN）
};

这段代码定义了KV缓存单元格管理类，每个token在缓存中对应一个cell，该类提供了查询cell状态（是否为空、所属序列、位置信息）和操作cell（设置位置、添加/移除序列归属、清空、位置偏移等）的接口。

```

**Cell状态图解**：
```
┌────────────────────────────────────────────────────────────┐
│  Cell索引:    0    1    2    3    4    5    6    7          │
├────────────────────────────────────────────────────────────┤
│  序列归属:   [0]  [0]  [0]  [1]  [1]  [0]  [ ]  [ ]        │
│  位置pos:     0    1    2    0    1    3   -1   -1          │
│  偏移shift:   0    0    0    0    0    0    0    0          │
├────────────────────────────────────────────────────────────┤
│  图示:                                                        │
│  序列0: ████░░██  (位置0,1,2,3)                              │
│  序列1: ░░██░░░░  (位置0,1)                                  │
│  空cell: ░░ 表示空                                           │
└────────────────────────────────────────────────────────────┘
```

---

## 11.2 缓存初始化与内存分配

### 11.2.1 构造函数实现

**源码位置**：`src/llama-kv-cache.cpp`（第79-315行）

```cpp
llama_kv_cache::llama_kv_cache(
        const llama_model & model,
                ggml_type   type_k,       // K缓存数据类型
                ggml_type   type_v,       // V缓存数据类型
                     bool   v_trans,      // V是否转置存储
                     bool   offload,      // 是否卸载到GPU
                     bool   unified,      // 统一流模式
                 uint32_t   kv_size,      // 缓存大小（cell数）
                 uint32_t   n_seq_max,    // 最大序列数
                 uint32_t   n_pad,        // 填充对齐
                 uint32_t   n_swa,        // 滑动窗口大小
           llama_swa_type   swa_type,    // SWA类型
    const layer_filter_cb & filter,      // 层过滤回调
    const  layer_reuse_cb & reuse)       // 层复用回调
    : model(model), hparams(model.hparams), v_trans(v_trans),
      n_seq_max(n_seq_max), n_stream(unified ? 1 : n_seq_max), 
      n_pad(n_pad), n_swa(n_swa), swa_type(swa_type) {

    // ① 初始化Cell数组（每流一个）
    v_cells.resize(n_stream);
    for (uint32_t s = 0; s < n_stream; ++s) {
        v_cells[s].resize(kv_size);  // 每个流kv_size个cell
    }

    // ② 初始化头指针（循环缓冲区用）
    v_heads.resize(n_stream);
    for (uint32_t s = 0; s < n_stream; ++s) {
        v_heads[s] = 0;  // 从0开始搜索空闲cell
    }

    // ③ 序列到流的映射
    seq_to_stream.resize(LLAMA_MAX_SEQ, 0);
    if (n_stream > 1) {
        for (uint32_t s = 0; s < n_stream; ++s) {
            seq_to_stream[s] = s;  // 每个序列对应一个流
        }
    }

    // ④ 为每层创建KV张量
    for (uint32_t il = 0; il < hparams.n_layer; il++) {
        if (!hparams.has_kv(il)) continue;  // 跳过无KV的层
        if (filter && !filter(il)) continue;  // 应用过滤器

        // 计算维度
        const uint32_t n_embd_k_gqa = hparams.n_embd_k_gqa(il);
        const uint32_t n_embd_v_gqa = !v_trans ? 
            hparams.n_embd_v_gqa(il) : hparams.n_embd_v_gqa_max();

        // 选择设备
        ggml_backend_buffer_type_t buft = ggml_backend_cpu_buffer_type();
        if (offload) {
            auto * dev = model.dev_layer(il);
            buft = ggml_backend_dev_buffer_type(dev);
        }

        // 创建张量
        ggml_tensor * k = ggml_new_tensor_3d(ctx, type_k, n_embd_k_gqa, kv_size, n_stream);
        ggml_tensor * v = ggml_new_tensor_3d(ctx, type_v, n_embd_v_gqa, kv_size, n_stream);

        // 创建每流视图
        for (uint32_t s = 0; s < n_stream; ++s) {
            k_stream.push_back(ggml_view_2d(ctx, k, n_embd_k_gqa, kv_size, k->nb[1], s*k->nb[2]));
            v_stream.push_back(ggml_view_2d(ctx, v, n_embd_v_gqa, kv_size, v->nb[1], s*v->nb[2]));
        }

        layers.push_back({ il, k, v, k_stream, v_stream });
    }

    // ⑤ 分配后端缓冲区
    for (auto & [buft, ctx] : ctx_map) {
        auto buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx.get(), buft);
        ggml_backend_buffer_clear(buf, 0);  // 初始化为0
        ctxs_bufs.emplace_back(std::move(ctx), buf);
    }
}

这段代码实现了KV缓存的构造函数。流程包括：1)初始化每个流的cell数组和头指针；2)建立序列到流的映射关系；3)为每层Transformer创建K/V张量（考虑GPU卸载和流视图）；4)为所有上下文分配后端缓冲区并初始化为零。
```

### 11.2.2 内存占用计算

**K缓存大小**：
```
size_k = n_layer * kv_size * n_embd_k_gqa * type_size(type_k)
```

**V缓存大小**：
```
size_v = n_layer * kv_size * n_embd_v_gqa * type_size(type_v)
```

**示例：Llama2-7B with Q8_0量化**

| 参数 | 值 |
|-----|---|
| n_layer | 32 |
| n_head_kv | 32 |
| n_embd_head_k | 128 |
| kv_size (n_ctx=4096) | 4096 |
| type_k/v | Q8_0 (1字节) |
| K缓存 | 32 * 4096 * 4096 * 1 = 512 MB |
| V缓存 | 32 * 4096 * 4096 * 1 = 512 MB |
| **总计** | **1024 MB** |

---

## 11.3 槽位查找与分配

### 11.3.1 Slot查找算法

**源码位置**：`src/llama-kv-cache.cpp`（第805-1002行）

```cpp
llama_kv_cache::slot_info llama_kv_cache::find_slot(
        const llama_ubatch & ubatch, 
        bool cont) const {  // cont: 是否要求连续

    uint32_t n_tokens = ubatch.n_tokens;
    uint32_t n_seqs = 1;

    // 多序列模式：token平均分配给各序列
    if (n_stream > 1) {
        n_seqs = ubatch.n_seqs_unq;
        n_tokens = n_tokens / n_seqs;
    }

    slot_info res;
    res.resize(n_seqs);

    // 为每个序列查找槽位
    for (uint32_t s = 0; s < n_seqs; ++s) {
        const auto seq_id = ubatch.seq_id_unq[s];
        res.strm[s] = seq_to_stream[seq_id];
        
        const auto & cells = v_cells[res.strm[s]];
        uint32_t head_cur = v_heads[res.strm[s]];

        // 优化：如果head前有大量未使用cell，从头开始搜索
        if (head_cur > cells.get_used() + 2*n_tokens) {
            head_cur = 0;
        }

        // 循环查找空闲cell
        uint32_t n_tested = 0;
        const uint32_t n_test = cont ? n_tokens : 1;

        while (true) {
            if (head_cur + n_test > cells.size()) {
                n_tested += cells.size() - head_cur;
                head_cur = 0;  // 循环到开头
                continue;
            }

            for (uint32_t i = 0; i < n_test; i++) {
                const auto idx = head_cur++;
                n_tested++;

                // 判断cell是否可用
                bool can_use = cells.is_empty(idx);

                if (!can_use && cells.seq_count(idx) == 1) {
                    const llama_pos pos_cell = cells.pos_get(idx);
                    const llama_seq_id seq_id_cell = cells.seq_get(idx);

                    // SWA掩码：如果cell位置超出窗口范围，可以复用
                    if (is_masked_swa(n_swa, swa_type, pos_cell, 
                                      cells.seq_pos_max(seq_id_cell) + 1)) {
                        can_use = true;
                    }
                }

                if (can_use) {
                    res.idxs[s].push_back(idx);
                } else if (cont) {
                    break;  // 连续模式：遇到占用就重新开始
                }
            }

            if (res.idxs[s].size() == n_tokens) {
                break;  // 找到足够cell
            }

            if (cont) {
                res.idxs[s].clear();  // 连续模式：清空重新开始
            }

            if (n_tested >= cells.size()) {
                return {};  // 找不到足够空间
            }
        }
    }

    return res;
}

这段代码实现了KV缓存的槽位查找算法，用于为新的token找到可用的缓存位置。流程包括：1)计算每个序列需要的token数；2)循环遍历缓存查找空闲cell；3)支持SWA窗口外的cell复用；4)可选连续模式要求cell连续排列。

---

## 11.4 序列操作

### 11.4.1 序列删除 (seq_rm)

**源码位置**：`src/llama-kv-cache.cpp`（第330-391行）

```cpp
bool llama_kv_cache::seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1) {
    // seq_id = -1: 删除所有序列在[p0, p1)范围内的cell
    // seq_id >= 0: 只删除指定序列的cell

    if (seq_id >= 0) {
        auto & cells = v_cells[seq_to_stream[seq_id]];
        auto & head = v_heads[seq_to_stream[seq_id]];

        uint32_t new_head = cells.size();

        for (uint32_t i = 0; i < cells.size(); ++i) {
            if (!cells.pos_in(i, p0, p1)) continue;

            // 如果cell属于该序列，移除归属
            if (cells.seq_has(i, seq_id) && cells.seq_rm(i, seq_id)) {
                if (new_head == cells.size()) {
                    new_head = i;  // 记录第一个释放的cell
                }
            }
        }

        // 更新head到释放的位置，加速后续查找
        if (new_head != cells.size() && new_head < head) {
            head = new_head;
        }
    }

    return true;
}

这段代码实现了序列删除操作seq_rm，用于从KV缓存中移除指定序列的KV数据。流程包括：1)遍历指定范围内的所有cell；2)如果cell属于目标序列，移除其序列归属；3)记录第一个释放的cell位置；4)更新head指针以加速后续槽位查找。

### 11.4.2 序列操作对比

| 操作 | 功能 | 使用场景 |
|-----|------|---------|
| `seq_rm` | 删除序列的某段KV | 截断对话历史 |
| `seq_cp` | 复制序列KV到新序列 | 束搜索、对话分叉 |
| `seq_keep` | 只保留指定序列 | 清理其他序列 |
| `seq_add` | 平移序列位置 | 位置编码调整 |
| `seq_div` | 位置除法 | YaRN缩放 |

---

## 11.5 设计中的取舍

### 为什么使用循环缓冲区而不是LRU？

**LRU方案**：
- 优点：可以精确控制哪些token被保留
- 缺点：需要维护复杂的优先级结构，查找空闲位置慢

**循环缓冲区方案（llama.cpp采用）**：
- 优点：实现简单，O(1)查找，缓存友好
- 缺点：可能提前覆盖仍有用的token

**llama.cpp的优化**：
- SWA（滑动窗口注意力）：自动掩码远距离token
- 智能head重置：当head前空闲cell过多时，从头开始搜索

### 为什么V缓存要转置存储？

**llama.cpp支持两种布局**：
- `v_trans = false`: 非转置，配合Flash Attention
- `v_trans = true`: 转置，传统Attention优化

---

## 11.6 本章小结

| 概念 | 一句话总结 |
|------|-----------|
| **KV缓存** | 存储历史token的K/V向量，避免重复计算 |
| **Cell** | 单个token的KV存储单元，带位置和序列信息 |
| **Slot查找** | 循环缓冲区算法，O(1)查找空闲cell |
| **seq_cp** | 支持同流（元数据）和跨流（数据拷贝）两种复制 |
| **v_trans** | V缓存布局选择，配合不同Attention实现 |

**下一步预告**：

在第12章，我们将探索高级缓存策略——理解SWA、ISWA和长文本处理优化。
