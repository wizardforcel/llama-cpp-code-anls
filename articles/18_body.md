# 第18章 高级生成技术 —— 加速推理的"秘密武器"

大语言模型的推理速度一直是实际应用中的关键瓶颈。虽然模型规模决定了能力上限，但推理效率决定了能否在实际场景中落地。传统自回归生成是串行的——每生成一个token都要调用一次大模型，这就像一位作家逐字逐句地斟酌，虽然质量高但速度极慢。而高级生成技术则像一位善用草稿和并行工作的作家，通过"快速猜测+批量验证"或"多路并行尝试"来显著提升推理效率。

## 学习目标

1. 理解投机解码（Speculative Decoding）的原理与实现
2. 掌握前瞻解码（Lookahead Decoding）的并行机制
3. 了解查找解码（Lookup Decoding）的缓存策略
4. 学会配置和调优各类加速技术
5. 能够根据硬件条件选择合适的加速方案

## 生活类比：写作与校对

想象你在写一篇长文章：

- **普通生成**：逐字逐句地写，每写一个字都要深思熟虑，质量高但速度慢

- **投机解码**：先快速草拟一段（草稿模型），然后一次性校对（目标模型）。如果草稿质量不错，就节省了大量时间；如果草稿质量差，就重新写。关键是草稿要快，校对要批量化。

- **前瞻解码**：同时写多个版本，看哪个版本能延续更长的正确内容。就像同时尝试多条故事线，选择发展最顺畅的那条。

- **查找解码**：从之前写过的文章中找相似的段落直接复用。这是写手的常见技巧——如果之前写过类似内容，直接复制修改。

**核心思想**：利用"小模型快速猜测 + 大模型验证"或"并行尝试 + 选择最优"来减少大模型的调用次数。

## 源码地图

```
common/speculative.h/cpp       # 投机解码核心实现
  ├── common_speculative_init()        # 初始化投机解码
  ├── common_speculative_draft()       # 生成草稿token
  ├── common_speculative_accept()      # 接受验证通过的token
  └── common_speculative_state_*       # 各类实现状态

examples/speculative/          # 投机解码示例
  └── speculative.cpp

examples/lookahead/            # 前瞻解码示例
  └── lookahead.cpp            # 前瞻解码实现（约480行）

examples/lookup/               # 查找解码示例
  └── lookup.cpp               # 查找解码实现（约240行）

common/ngram-cache.h/cpp       # N-gram缓存支持
  ├── common_ngram_cache_update()      # 更新缓存
  ├── common_ngram_cache_draft()       # 从缓存生成草稿
  └── common_ngram_cache_*             # 缓存操作

examples/parallel/             # 并行解码示例
  └── parallel.cpp
```

## 18.1 投机解码（Speculative Decoding）

### 18.1.1 核心原理

投机解码的核心思想基于一个观察：小模型（草稿模型）虽然能力弱于大模型（目标模型），但在许多简单token的预测上与小模型一致。因此可以用小模型快速生成候选序列，然后用大模型并行验证，只接受验证通过的部分。

**数学基础**：

对于草稿token x，其接受概率为：
```
接受概率 = min(1, P_target(x) / P_draft(x))

如果 P_target(x) >= P_draft(x)：总是接受
如果 P_target(x) < P_draft(x)：以 P_target(x)/P_draft(x) 概率接受
```

这保证了：
1. 如果大模型认为草稿token概率更高，直接接受
2. 如果大模型认为草稿token概率较低，按一定比例接受（保持分布一致）
3. 拒绝的token从大模型重新采样，保证生成质量

**源码位置**：`common/speculative.h`（第17-42行）

```cpp
struct common_speculative_state {
    const enum common_speculative_type type;
    
    size_t n_call_begin  = 0;  // 刷新调用次数
    size_t n_call_draft  = 0;  // 生成草稿调用次数
    size_t n_call_accept = 0;  // 接受调用次数
    
    size_t n_gen_drafts = 0;   // 生成的草稿数
    size_t n_acc_drafts = 0;   // 接受的草稿数
    size_t n_gen_tokens = 0;   // 生成的token数
    size_t n_acc_tokens = 0;   // 接受的token数
    
    int64_t t_begin_us  = 0;   // 刷新耗时
    int64_t t_draft_us  = 0;   // 生成草稿耗时
    int64_t t_accept_us = 0;   // 接受耗时
};
```

### 18.1.2 草稿模型实现

**源码位置**：`common/speculative.cpp`（第147-438行）

```cpp
struct common_speculative_state_draft : public common_speculative_state {
    llama_context * ctx_tgt;  // 目标模型上下文
    llama_context * ctx_dft;  // 草稿模型上下文
    common_sampler * smpl;    // 草稿模型采样器
    llama_batch  batch;       // 批处理
    llama_tokens prompt_dft;  // 草稿模型prompt缓存
    
    void draft(
            const common_params_speculative & params,
            const llama_tokens & prompt_tgt,
            llama_token id_last,
            llama_tokens & result) override {
        
        // 1. 复用KV缓存：找到草稿模型prompt与当前输入的最长公共前缀
        int reuse_i = -1;
        int reuse_n = 0;
        llama_tokens prompt_cur(prompt_tgt);
        prompt_cur.push_back(id_last);
        
        for (int i = 0; i < (int) prompt_dft.size(); ++i) {
            int cur = 0;
            while (i + cur < (int) prompt_dft.size() && 
                   i_start + cur < (int) prompt_cur.size() &&
                   prompt_cur[i_start + cur] == prompt_dft[i + cur]) {
                cur++;
            }
            if (cur > reuse_n) {
                reuse_i = i;
                reuse_n = cur;
            }
        }
        
        // 2. 评估新token（未缓存的部分）
        if (batch.n_tokens > 0) {
            llama_decode(ctx_dft, batch);
        }
        
        // 3. 采样n_draft个token
        for (int i = 0; i < params.n_max; ++i) {
            common_sampler_sample(smpl, ctx_dft, 0, true);
            const auto * cur_p = common_sampler_get_candidates(smpl, true);
            
            // 只保留高置信度的草稿token
            if (cur_p->data[0].p < params.p_min) {
                break;
            }
            
            llama_token id = cur_p->data[0].id;
            result.push_back(id);
            
            // 在草稿模型上评估（为下一个token做准备）
            common_batch_clear(batch);
            common_batch_add(batch, id, n_past + i + 1, { 0 }, true);
            llama_decode(ctx_dft, batch);
        }
        
        prompt_dft = prompt_cur;
    }
};
```

**投机解码流程图解**：
```
输入prompt: "The quick brown fox"

步骤1: 草稿模型快速生成
┌─────────────────────────────────────────┐
│ 草稿模型 M_q (小模型，快)                 │
│                                         │
│ 输入: "The quick brown fox"              │
│ 输出: ["jumps", "over", "the", "lazy"]   │
│ 耗时: 4 × 10ms = 40ms                   │
└─────────────────────────────────────────┘
                    ↓
步骤2: 目标模型并行验证
┌─────────────────────────────────────────┐
│ 目标模型 M_p (大模型，慢)                 │
│                                         │
│ 输入: prompt + ["jumps", "over", "the", "lazy"] │
│ 输出: 每个位置的logits（并行计算）        │
│ 耗时: 1 × 50ms = 50ms                   │
└─────────────────────────────────────────┘
                    ↓
步骤3: 接受/拒绝决策
┌─────────────────────────────────────────┐
│ 位置0: P_p("jumps") >= P_q("jumps")? ✓  │
│ 位置1: P_p("over") >= P_q("over")?   ✓  │
│ 位置2: P_p("the") < P_q("the")?      ✗  │
│                                         │
│ 结果: 接受2个token，从位置2重新采样       │
└─────────────────────────────────────────┘

总耗时: 40ms + 50ms = 90ms
传统串行: 4 × 50ms = 200ms
加速比: 200/90 ≈ 2.2x
```

### 18.1.3 N-gram自投机

**源码位置**：`common/speculative.cpp`（第465-645行）

如果不想维护两个模型，可以使用N-gram自投机——利用模型自己生成的历史序列进行投机。

```cpp
// 基于N-gram的投机解码（无需草稿模型）
struct common_speculative_state_ngram_mod : public common_speculative_state {
    common_ngram_mod & mod;   // N-gram模型
    size_t i_last = 0;        // 上次处理位置
    size_t n_draft_last = 0;  // 上次草稿长度
    int n_low = 0;            // 低接受率计数
    
    void draft(
            const common_params_speculative & params,
            const llama_tokens & prompt_tgt,
            llama_token id_last,
            llama_tokens & result) override {
        
        const size_t cur_len = prompt_tgt.size();
        const size_t n = mod.get_n();
        
        // 添加新的N-gram到模型
        if (i_last + 32 < cur_len) {
            for (size_t i = i_last; i < cur_len - n; ++i) {
                mod.add(prompt_tgt.data() + i);
            }
            i_last = cur_len - n;
        }
        
        // 基于最近N-1个token预测下一个
        result.resize(n + params.n_max);
        for (size_t i = 0; i < n - 1; ++i) {
            result[i] = prompt_tgt[cur_len - n + 1 + i];
        }
        result[n - 1] = id_last;
        
        // 从N-gram模型获取预测
        for (int i = 0; i < params.n_max; ++i) {
            const llama_token token = mod.get(result.data() + i);
            if (token == common_ngram_mod::EMPTY) {
                if (i < params.n_min) {
                    result.clear();
                    return;
                }
                result.resize(n + i);
                break;
            }
            result[n + i] = token;
        }
        
        // 只返回预测的token（去掉前缀）
        result.resize(result.size() - n);
    }
};
```

**N-gram投机图解**：
```
历史序列: [..., "The", "quick", "brown", "fox"]

N-gram模型 (N=4):
┌─────────────────────────────────────┐
│ Key: ["The", "quick", "brown"]       │
│ Value: {"fox": 5, "cat": 2, "dog": 1} │
│                                     │
│ Key: ["quick", "brown", "fox"]       │
│ Value: {"jumps": 6, "runs": 2}       │
└─────────────────────────────────────┘

预测过程:
当前: ["quick", "brown", "fox"] + "jumps"
查表: ["brown", "fox", "jumps"] → {"over": 5, "high": 1}
预测: "over"

继续:
当前: ["fox", "jumps", "over"] → {"the": 4, "a": 2}
预测: "the"

草稿序列: ["jumps", "over", "the"]
```

N-gram投机的优势：
1. **无需额外模型**：仅使用目标模型的历史输出
2. **内存友好**：N-gram表比草稿模型小得多
3. **适合重复内容**：代码、模板化文本效果好

## 18.2 前瞻解码（Lookahead Decoding）

### 18.2.1 Jacobi迭代原理

前瞻解码基于Jacobi迭代方法，打破传统自回归的串行限制，并行验证多个未来位置的token。

**对比**：
```
传统自回归:  x_1 → x_2 → x_3 → x_4 → ... (串行，每步依赖前一步)
Jacobi迭代:  同时猜测 x_1, x_2, x_3, x_4
             并行验证并修正
             直到收敛
```

**关键参数**：
- `W`（窗口大小）：并行的Jacobi迭代数
- `N`（N-gram大小）：用于验证的N-gram长度  
- `G`（验证组数）：同时验证的N-gram数量

**源码位置**：`examples/lookahead/lookahead.cpp`（第1-484行）

### 18.2.2 批处理构建

前瞻解码的核心是构建一个复杂的批处理，同时包含Jacobi迭代位置和验证N-gram。

**源码位置**：`examples/lookahead/lookahead.cpp`（第180-256行）

```cpp
// 构建前瞻解码的批处理
while (true) {
    common_batch_clear(batch);
    
    // 1. 添加当前token（所有序列共享）
    common_batch_add(batch, id, n_past, seq_id_all, true);
    
    // 2. 添加验证N-gram
    const int g_cur = ngrams_observed.cnt[id];
    for (int g = 0; g < g_cur; g++) {
        const int idx = ngrams_observed.idx[id] + g * (N - 1);
        for (int j = 0; j < N - 1; j++) {
            const llama_token t = ngrams_observed.tokens[idx + j];
            ngrams_cur[g].tokens[j + 1] = t;
            common_batch_add(batch, t, n_past + j + 1, {W + 1 + g}, true);
        }
    }
    
    // 3. 添加前瞻token（Jacobi迭代位置）
    for (int i = 1; i < W; i++) {
        seq_id_look.resize(W - i);
        for (int j = 0; j < W - i; j++) {
            seq_id_look[j] = i + j + 1;
        }
        common_batch_add(batch, tokens_j[0][i], n_past + i, seq_id_look, false);
    }
    
    // 4. 添加多级前瞻token
    for (int j = 1; j < N - 1; j++) {
        for (int i = 0; i < W; i++) {
            common_batch_add(batch, tokens_j[j][i], n_past + j + i, {i + 1}, j == N - 2);
        }
    }
    
    llama_decode(ctx, batch);
    
    // 处理输出、验证、接受...
}
```

**前瞻解码批处理结构图解**（W=5, N=4, G=2）：
```
批处理中的序列布局:

位置:    0    1    2    3    4    5    6    7    8    9
Token:   I    L    L    L    L    L    V    V    V    V
         │    │    │    │    │    │    │    │    │    │
序列:    0    1    2    3    4    5    6    7    8    9
         │    └────┴────┴────┴────┘    └────┴────┘
         │         Jacobi前瞻            验证N-gram
         └──────────────────────────────────────────
                    所有序列共享

序列说明:
- 序列0: 主序列（当前输入位置）
- 序列1-5: Jacobi前瞻序列（每级一个，共W个）
- 序列6-7: 验证N-gram序列（共G组，每组N-1个token）

I = 输入token
L = Jacobi前瞻token  
V = 验证N-gram的token
```

### 18.2.3 验证与接受

前瞻解码的关键是从多个并行的N-gram验证中找出可以接受的token序列。

**源码位置**：`examples/lookahead/lookahead.cpp`（第263-350行）

```cpp
for (int v = 0; v < N; ++v) {
    // 找到最佳匹配的验证N-gram
    bool found = false;
    for (int g = 0; g < (int) ngrams_cur.size(); g++) {
        if (ngrams_cur[g].active) {
            i_batch = ngrams_cur[g].i_batch[v];
            seq_id_best = ngrams_cur[g].seq_id;
            ++n_accept;
            found = true;
            break;
        }
    }
    
    if (!found) break;
    
    // 采样下一个token
    id = common_sampler_sample(smpl, ctx, i_batch);
    
    // 验证其他N-gram是否匹配
    for (int g = 0; g < (int) ngrams_cur.size(); g++) {
        if (ngrams_cur[g].active) {
            llama_token expected = ngrams_cur[g].tokens[v + 1];
            if (id != expected) {
                ngrams_cur[g].active = false;  // 不匹配，停用该N-gram
            }
        }
    }
    
    // 输出接受的token
    // ...
}
```

**前瞻解码优势**：
1. **无需草稿模型**：完全基于目标模型自身
2. **并行验证**：一次decode验证多个可能路径
3. **自适应**：根据验证结果动态调整

## 18.3 查找解码（Lookup Decoding）

### 18.3.1 N-gram缓存机制

查找解码的核心是预建N-gram缓存，在生成时快速查找历史模式。

**源码位置**：`common/ngram-cache.h`（第14-102行）

```cpp
// N-gram定义
struct common_ngram {
    llama_token tokens[LLAMA_NGRAM_MAX];  // 最多4个token
};

// N-gram缓存：从N-gram到下一个token的概率分布
using common_ngram_cache_part = std::unordered_map<llama_token, int32_t>;
using common_ngram_cache = std::unordered_map<common_ngram, common_ngram_cache_part, common_ngram_hash_function>;

// 更新缓存
void common_ngram_cache_update(
    common_ngram_cache & ngram_cache,
    int ngram_min, int ngram_max,
    std::vector<llama_token> & inp_data,
    int nnew, bool print_progress);

// 从缓存生成草稿
void common_ngram_cache_draft(
    std::vector<llama_token> & inp,
    std::vector<llama_token> & draft, int n_draft,
    int ngram_min, int ngram_max,
    common_ngram_cache & nc_context,
    common_ngram_cache & nc_dynamic,
    common_ngram_cache & nc_static);
```

### 18.3.2 三层缓存策略

**源码位置**：`examples/lookup/lookup.cpp`（第46-72行）

```cpp
// 三层N-gram缓存
common_ngram_cache ngram_cache_context;   // 上下文缓存（当前对话）
common_ngram_cache ngram_cache_dynamic;   // 动态缓存（历史对话）
common_ngram_cache ngram_cache_static;    // 静态缓存（预训练数据）

// 初始化时加载缓存
if (!params.speculative.lookup_cache_static.empty()) {
    ngram_cache_static = common_ngram_cache_load(
        params.speculative.lookup_cache_static
    );
}

if (!params.speculative.lookup_cache_dynamic.empty()) {
    ngram_cache_dynamic = common_ngram_cache_load(
        params.speculative.lookup_cache_dynamic
    );
}
```

**三层缓存结构图解**：
```
用户输入: "The quick brown fox"

查找顺序（从高到低优先级）：

1. 上下文缓存 (内存，最快，容量小)
   ├── 最近生成的token序列
   ├── 实时更新
   └── 命中率最高，但数据量少

2. 动态缓存 (磁盘，中等速度，中等容量)
   ├── 之前对话的历史
   ├── 需要显式保存/加载
   └── 对话间共享

3. 静态缓存 (预生成，加载慢，容量大)
   ├── 从大规模语料预建
   ├── 包含通用语言模式
   └── 适用于特定领域

查找示例:
输入: ["The", "quick", "brown"]
上下文: 未命中
动态: 未命中  
静态: 命中 → {"fox": 10, "cat": 3, "dog": 2}

预测: "fox" (选择频率最高的)
```

### 18.3.3 查找解码流程

**源码位置**：`examples/lookup/lookup.cpp`（第116-210行）

```cpp
while (true) {
    // 1. 从目标模型采样
    llama_token id = common_sampler_sample(smpl, ctx, i_dft);
    
    // 2. 检查是否匹配草稿
    if (i_dft < (int) draft.size() && id == draft[i_dft]) {
        // 匹配成功，接受token
        ++n_accept;
        ++i_dft;
        
        // 更新上下文缓存
        common_ngram_cache_update(
            ngram_cache_context, 
            LLAMA_NGRAM_MIN, LLAMA_NGRAM_MAX, 
            inp, 1, false
        );
        continue;
    }
    
    // 3. 不匹配，从缓存重新生成草稿
    draft.clear();
    draft.push_back(id);
    
    common_ngram_cache_draft(
        inp, draft, n_draft,
        LLAMA_NGRAM_MIN, LLAMA_NGRAM_MAX,
        ngram_cache_context,
        ngram_cache_dynamic,
        ngram_cache_static
    );
    
    i_dft = 1;
}
```

**查找解码工作流程**：
1. 从N-gram缓存生成草稿序列
2. 目标模型验证草稿token
3. 匹配则继续，不匹配则重新生成草稿
4. 同时更新上下文缓存

## 18.4 技术对比与选型

### 18.4.1 技术特性对比

| 技术 | 草稿来源 | 硬件要求 | 加速比 | 适用场景 |
|------|----------|----------|--------|----------|
| 草稿模型投机 | 小模型 | 2x显存 | 1.5-2.5x | 有合适小模型时 |
| N-gram自投机 | 自身历史 | 无额外 | 1.2-1.8x | 重复内容多 |
| 前瞻解码 | 并行猜测 | 大KV缓存 | 1.3-2.0x | 批量推理 |
| 查找解码 | 预建缓存 | 磁盘空间 | 1.1-1.5x | 固定领域 |

### 18.4.2 配置建议

**投机解码配置**：
```cpp
struct common_params_speculative {
    int n_max = 16;           // 最大草稿token数
    int n_min = 5;            // 最小草稿token数
    float p_min = 0.9f;       // 草稿置信度阈值
    
    // 草稿模型路径
    std::string model_dft;
    
    // N-gram参数
    int ngram_size_n = 4;     // N-gram大小
    int ngram_min_hits = 2;   // 最小命中次数
    
    // 缓存路径
    std::string lookup_cache_static;
    std::string lookup_cache_dynamic;
};
```

**命令行使用示例**：
```bash
# 草稿模型投机（推荐加速比最高）
./llama-cli -m llama-70b.gguf --model-draft llama-7b.gguf \
    -p "Hello" --draft 16

# N-gram投机（无需额外模型）
./llama-cli -m model.gguf --speculative ngram_mod \
    -p "Hello" -n 16

# 查找解码（适合固定领域）
./llama-cli -m model.gguf \
    --lookup-cache-static medical_cache.gguf \
    -p "Patient symptoms:"

# 前瞻解码（需要专门程序）
./lookahead -m model.gguf -p "Hello" \
    --n-parallel 20 --n-gram 4
```

## 18.5 设计中的取舍

### 18.5.1 内存与速度的权衡

| 技术 | 额外内存需求 | 加速效果 | 适用条件 |
|------|-------------|----------|----------|
| 草稿模型 | 完整小模型 | 最高 | 显存充足 |
| 前瞻解码 | W+G+1倍KV缓存 | 中等 | KV缓存大 |
| 查找解码 | 缓存文件大小 | 较低 | 磁盘空间 |
| N-gram投机 | N-gram表 | 较低 | 内存充足 |

### 18.5.2 质量与效率的权衡

- **高接受率策略**：保守生成草稿，质量高但生成慢
- **高投机率策略**：激进生成草稿，生成快但验证失败多
- **最佳平衡点**：通常接受率在60-80%时整体效率最优

### 18.5.3 实现复杂度

复杂度排序（从低到高）：
1. **查找解码**：简单缓存查询，最易实现
2. **N-gram投机**：状态机匹配，需要维护历史
3. **草稿模型投机**：双模型管理，KV缓存同步
4. **前瞻解码**：复杂批处理调度，最难实现

## 18.6 动手练习

### 练习1：测试投机解码加速比

```bash
# 基准测试（无投机）
./llama-cli -m model.gguf -p "Write a story" \
    --no-display-prompt -n 100 2>&1 | grep "eval time"

# 草稿模型投机
./llama-cli -m model.gguf --model-draft draft.gguf \
    -p "Write a story" --no-display-prompt -n 100 2>&1 | grep "eval time"

# 计算加速比
# 注意：不同硬件和模型效果不同
```

### 练习2：分析接受率

修改`common/speculative.cpp`添加统计输出：

```cpp
void common_speculative_print_stats(const common_speculative * spec) {
    for (const auto & impl : spec->impls) {
        float accept_rate = impl->n_gen_tokens > 0 ? 
            (float)impl->n_acc_tokens / impl->n_gen_tokens : 0;
        float draft_efficiency = impl->n_gen_drafts > 0 ?
            (float)impl->n_acc_drafts / impl->n_gen_drafts : 0;
        
        printf("Speculative Stats:\n");
        printf("  Type: %s\n", 
               common_speculative_type_to_str(impl->type).c_str());
        printf("  Tokens: generated=%zu, accepted=%zu\n",
               impl->n_gen_tokens, impl->n_acc_tokens);
        printf("  Accept rate: %.2f%%\n", accept_rate * 100);
        printf("  Draft efficiency: %.2f%%\n", draft_efficiency * 100);
    }
}
```

### 练习3：构建领域专用缓存

从领域语料构建N-gram缓存：

```python
# build_cache.py
from collections import defaultdict
import json

def build_ngram_cache(texts, n_max=4):
    """从文本列表构建N-gram缓存"""
    cache = defaultdict(lambda: defaultdict(int))
    
    for text in texts:
        tokens = tokenize(text)  # 需要使用llama.cpp的tokenizer
        for n in range(1, n_max + 1):
            for i in range(len(tokens) - n):
                key = tuple(tokens[i:i+n])
                next_token = tokens[i+n] if i+n < len(tokens) else None
                if next_token is not None:
                    cache[key][next_token] += 1
    
    return cache

# 示例：构建医疗领域缓存
with open("medical_texts.txt") as f:
    texts = f.readlines()

cache = build_ngram_cache(texts, n_max=4)

# 保存为llama.cpp格式
# 需要使用gguf-py库
```

## 18.7 本章小结

| 技术 | 核心思想 | 加速原理 | 适用条件 |
|------|----------|----------|----------|
| **草稿模型投机** | 小模型猜测+大模型验证 | 减少大模型调用次数 | 有小模型可用 |
| **N-gram自投机** | 自身历史模式投机 | 利用重复模式 | 文本重复性高 |
| **前瞻解码** | 并行验证多路径 | Jacobi迭代并行化 | KV缓存充足 |
| **查找解码** | 预建缓存查询 | 跳过重复计算 | 领域固定 |

**选型决策树**：
```
是否有合适的小模型？
├── 是 → 草稿模型投机（效果最好）
└── 否 → 文本重复性如何？
    ├── 高 → N-gram投机或查找解码
    └── 低 → 前瞻解码
```

**关键经验**：

1. **草稿质量是关键**：草稿模型不需要很强，但速度要快
2. **接受率60-80%最优**：太低则草稿无效，太高则草稿生成太慢
3. **硬件决定上限**：显存大小决定了可以同时维护多少序列
4. **领域适配很重要**：通用场景用通用草稿模型，专业场景用专业缓存

**下一步预告**：

在第19章，我们将探索LoRA（Low-Rank Adaptation）适配器支持——如何以极小的成本微调模型，实现个性化定制和特定领域优化。
