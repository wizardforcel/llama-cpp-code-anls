# 第11章 KV缓存架构 —— 自注意力的"记忆外挂"

在Transformer模型的推理过程中，KV缓存（Key-Value Cache）是提升效率的关键技术。它避免了自注意力机制中的重复计算，将二次方复杂度降为线性。本章将深入解析llama.cpp的KV缓存架构。

## 学习目标

1. 理解KV缓存的核心作用和内存布局
2. 掌握`llama_kv_cache`的类设计与关键数据结构
3. 理解KV缓存的更新、查找和回收机制
4. 了解注意力旋转优化技术

## 生活类比：超级大脑的"记忆宫殿"

想象 Transformer 模型是一位拥有超级记忆能力的大脑，而 KV 缓存就是它精心搭建的记忆宫殿。K（Key）就像知识的索引卡片，用来快速查找相关内容；V（Value）则像知识的详细内容，存储着实际的信息。每个 Transformer 层都有自己的储物柜组，里面排列着整齐的储物格——这就是 Cell（单元格）的概念，每个 token 对应一个储物格，里面存放着该 token 的 K 和 V 向量，还记录着 token 的位置信息和所属序列。

当你需要存放新的内容时，Slot 查找机制就会出动——它采用循环缓冲区策略，从头开始寻找空闲的储物格，找到后就把新 token 的信息存放进去。多序列场景下，不同对话使用不同的储物区域，互不干扰。

记忆不是一成不变的，KV 缓存提供了丰富的序列操作：`seq_rm` 可以删除某段记忆，`seq_cp` 可以把记忆复制到新的序列中，`seq_keep` 则只保留指定序列的记忆。就像记忆宫殿需要精心管理空间以免溢出，KV 缓存需要高效地分配、回收和复用内存，确保在有限的存储空间中服务尽可能长的对话。

---

## 11.0 内存管理抽象层

在深入了解KV缓存之前，我们需要先理解`llama-memory.h`定义的**内存管理抽象接口**。这是llama.cpp中所有内存类型（KV缓存、循环状态缓存等）的统一基类，它提供了一套通用的内存操作接口，使得上层代码无需关心具体的内存实现细节。

### 11.0.1 llama_memory_i - 内存管理接口

**源码位置**：`src/llama-memory.h`（第68-120行）

```cpp
struct llama_memory_i {
    // 析构函数
    virtual ~llama_memory_i() = default;

    // 初始化批次处理，将输入批次分割为ubatches
    virtual llama_memory_context_ptr init_batch(
            llama_batch_allocr & balloc,
            uint32_t n_ubatch,
            bool embd_all) = 0;

    // 模拟满缓存，用于分配最坏情况下的计算缓冲区
    virtual llama_memory_context_ptr init_full() = 0;

    // 准备待处理的内存更新（如偏移、复制等）
    virtual llama_memory_context_ptr init_update(
            llama_context * lctx, 
            bool optimize) = 0;

    // 序列操作接口
    virtual void clear(bool data) = 0;
    virtual bool seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1) = 0;
    virtual void seq_cp(llama_seq_id seq_id_src, llama_seq_id seq_id_dst, 
                        llama_pos p0, llama_pos p1) = 0;
    virtual void seq_keep(llama_seq_id seq_id) = 0;
    virtual void seq_add(llama_seq_id seq_id, llama_pos p0, llama_pos p1, 
                         llama_pos shift) = 0;
    virtual void seq_div(llama_seq_id seq_id, llama_pos p0, llama_pos p1, int d) = 0;

    // 状态读写
    virtual void state_write(llama_io_write_i & io, 
                             llama_seq_id seq_id = -1, 
                             llama_state_seq_flags flags = 0) const = 0;
    virtual void state_read(llama_io_read_i & io, 
                            llama_seq_id seq_id = -1, 
                            llama_state_seq_flags flags = 0) = 0;
};
```

**关键设计思想**：

1. **统一接口**：所有内存类型（KV缓存、循环状态、混合内存）都实现`llama_memory_i`，使得`llama_context`可以用相同的方式处理不同类型的模型。这种设计极大地提高了代码的复用性和可维护性。

2. **批次处理抽象**：`init_batch()`返回`llama_memory_context_i`，用于迭代处理分割后的ubatches。这种设计支持大模型的分批次推理，避免一次性加载过多数据导致内存不足。

3. **状态持久化**：统一的`state_write/read`接口支持会话保存/恢复，使得推理状态可以在不同运行之间持久化。

### 11.0.2 llama_memory_context_i - 批次处理上下文

**源码位置**：`src/llama-memory.h`（第46-62行）

```cpp
struct llama_memory_context_i {
    virtual ~llama_memory_context_i() = default;

    // 消费当前ubatch并切换到下一个
    // 返回false表示处理完成
    virtual bool next() = 0;

    // 将当前ubatch的内存状态应用到内存对象
    // 返回false表示失败
    virtual bool apply() = 0;

    // 获取当前ubatch
    virtual const llama_ubatch & get_ubatch() const = 0;

    // 获取内存上下文状态
    virtual llama_memory_status get_status() const = 0;
};
```

**工作流程**：

```
llama_memory_i
    ↓ init_batch()
llama_memory_context_i
    ↓ next() → apply() → next() → apply() → ...
（直到next()返回false）
```

这种迭代器模式允许分批次处理大量token，每次只处理一个ubatch，有效控制内存使用。

### 11.0.3 llama_memory_status - 操作状态

**源码位置**：`src/llama-memory.h`（第25-30行）

```cpp
enum llama_memory_status {
    LLAMA_MEMORY_STATUS_SUCCESS = 0,       // 操作成功
    LLAMA_MEMORY_STATUS_NO_UPDATE,       // 无需更新
    LLAMA_MEMORY_STATUS_FAILED_PREPARE,  // 准备失败
    LLAMA_MEMORY_STATUS_FAILED_COMPUTE,  // 计算失败
};
```

**辅助函数**：
- `llama_memory_status_combine()` - 合并两个状态（用于混合内存类型）
- `llama_memory_status_is_fail()` - 检查是否为失败状态

### 11.0.4 与KV缓存的关系

`llama_kv_cache`继承并实现`llama_memory_i`：

```cpp
class llama_kv_cache : public llama_memory_i {
    // 实现所有虚函数...
};
```

这种设计使得：
1. **llama_context不需要知道具体内存类型**，只通过`llama_memory_i`指针操作
2. **支持混合模型**（如Jamba、RWKV）：可以同时拥有KV缓存和循环状态
3. **易于扩展**：添加新的内存类型只需实现`llama_memory_i`

### 11.0.5 设计中的取舍

**为什么使用抽象接口而非模板？**

| 方案 | 优点 | 缺点 | llama.cpp选择 |
|-----|------|------|---------------|
| 模板 | 编译时确定，零运行时开销 | 代码膨胀，编译慢 | ❌ |
| **虚函数抽象** | 运行时多态，代码复用 | 轻微虚函数调用开销 | ✅ |

llama.cpp选择虚函数抽象的原因是：
1. **内存类型在运行时才确定**（根据模型架构自动选择）
2. **虚函数开销在推理中可忽略**（相比矩阵计算）
3. **代码复用性高**，易于维护和扩展

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

## 11.6 动手练习

### 练习 1：计算 KV 缓存内存占用

给定模型配置：n_layer=32，kv_size=4096，n_embd_k_gqa=n_embd_v_gqa=4096，type_k=Q8_0（1字节/元素），type_v=Q8_0（1字节/元素）。计算单个序列的 K 缓存和 V 缓存分别占用多少内存，以及总内存占用。

### 练习 2：分析 Slot 查找算法

阅读 `src/llama-kv-cache.cpp` 中 `find_slot` 的实现（约第 805-1002 行），回答以下问题：当缓存已满时，如何为新 token 找到可用位置？SWA 模式下如何复用超出窗口范围的 cell？连续模式和非连续模式在查找策略上有什么不同？

### 练习 3：理解序列操作

使用 llama.cpp 的 API，编写一个简单的测试程序：创建两个序列，分别进行 token 添加，然后对其中一个序列执行 `seq_rm` 删除部分 token，最后检查删除操作后缓存的状态是否正确。

---

## 11.7 本章小结

本章深入解析了KV缓存架构的设计与实现。KV缓存用于存储历史token的K/V向量，避免在自回归生成过程中重复计算。Cell是单个token的KV存储单元，带有位置和序列信息用于管理。Slot查找采用循环缓冲区算法，实现O(1)时间复杂度的空闲cell查找。seq_cp操作支持同流（仅元数据复制）和跨流（数据拷贝）两种复制模式。v_trans参数用于选择V缓存的布局方式，以配合不同的Attention实现需求。

本章我们一起学习了以下概念：

| 概念 | 解释 |
|------|------|
| KV缓存 | 存储历史 token 的 Key 和 Value 向量，避免自回归生成中的重复计算 |
| Cell单元格 | 每个 token 的 KV 存储单元，记录位置和序列归属信息 |
| Slot查找 | 循环缓冲区算法，在 KV 缓存中寻找可用的空闲 cell |
| 多流模式 | 每个序列独立的 cell 数组，实现序列间的完全隔离 |
| 序列操作 | seq_rm/cp/keep 等操作，支持删除、复制、保留序列的 KV 数据 |
| v_trans | 控制 V 缓存布局的参数，转置存储配合 Flash Attention 实现 |

**下一章预告**：

下一章中，我们将学习高级缓存策略，理解 SWA、ISWA 和长文本处理优化。
