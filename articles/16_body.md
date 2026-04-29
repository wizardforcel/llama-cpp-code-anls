# 第16章 采样算法详解 —— 让模型"开口说话"的艺术

当大语言模型生成文本时，它会为词汇表中的每个token计算一个分数（logit）。但这只是第一步——如何从这些分数中选出下一个token，才是真正决定生成质量的魔法时刻。这个过程叫做"采样"（Sampling）。想象你面对一个拥有10万道菜的巨型菜单，每道菜都有一个推荐分数，你会如何选择？每次都选最高分？随机选择？还是在高分菜品中随机挑选？这些策略就是采样算法要解决的问题。

## 学习目标

1. 理解大模型生成文本的核心机制——采样算法
2. 掌握贪心采样、随机采样、温度缩放等基础采样方法
3. 深入理解Top-K、Top-P（Nucleus）、Min-P、Typical Sampling等高级采样技术
4. 学会配置重复惩罚机制，避免生成重复内容
5. 能够根据不同任务场景选择合适的采样参数

## 生活类比：餐厅点菜的艺术

想象你来到一家有 10 万道菜品的餐厅，每道菜都有一个"推荐分数"（logit）。采样算法就像是你选择菜品的策略。贪心采样就是每次都点分数最高的那道菜——稳妥但缺乏惊喜，适合需要确定性的场合，比如商务宴请。随机采样则是完全随机选择——可能点到黑暗料理，但偶尔会有意外惊喜，就像闭着眼睛在菜单上乱指。温度缩放是调节"冒险程度"的旋钮：温度高时更愿意尝试低分菜品，温度低时则保守地选择高分菜。

Top-K 策略让你只从评分前 K 名的菜品中选择，在质量下限和多样性之间取得平衡。Top-P 更为灵活，它从累计评分占比达到 P% 的菜品池中动态选择，当评分分布很集中时候选池小，分布很平坦时候选池大。重复惩罚则是为了避免连续点同一道菜，让餐桌更丰富多样——就像你不会每顿都吃同样的菜。

就像点菜需要根据不同场合（商务宴请 vs 朋友聚会）选择不同策略，采样参数也需要根据任务类型（代码生成 vs 创意写作）进行调整。没有一种策略适合所有场景，这正是采样算法的艺术所在。

## 源码地图

```
src/llama-sampler.cpp          # 采样器核心实现（约3800行）
  ├── llama_sampler_init_greedy()      # 贪心采样
  ├── llama_sampler_init_dist()        # 随机分布采样
  ├── llama_sampler_init_top_k()       # Top-K采样
  ├── llama_sampler_init_top_p()       # Top-P采样
  ├── llama_sampler_init_min_p()       # Min-P采样
  ├── llama_sampler_init_typical()     # Typical采样
  ├── llama_sampler_init_temp()        # 温度缩放
  ├── llama_sampler_init_penalties()   # 重复惩罚
  └── llama_sampler_chain_*            # 采样器链

include/llama.h                # C API 声明
  ├── struct llama_sampler_i           # 采样器接口
  ├── struct llama_sampler             # 采样器基类
  └── llama_sampler_init_*             # 各类采样器初始化

common/sampling.h/cpp          # 高级采样封装
  ├── common_sampler_init()            # 采样器初始化
  ├── common_sampler_sample()          # 采样执行
  └── llama_sampling_params            # 采样参数结构
```

## 16.1 基础采样方法

### 16.1.1 贪心采样（Greedy Sampling）

**源码位置**：`src/llama-sampler.cpp`（第931-1019行）

```cpp
struct llama_sampler_greedy : public llama_sampler_backend {
    // 贪心采样：总是选择概率最高的token
};

static void llama_sampler_greedy_apply(struct llama_sampler * smpl, 
                                        llama_token_data_array * cur_p) {
    // 找到logit最大的token，设为1.0，其他设为0
    llama_sampler_softmax_impl(cur_p);
    cur_p->data[0].p = 1.0f;
    for (size_t i = 1; i < cur_p->size; ++i) {
        cur_p->data[i].p = 0.0f;
    }
}
```

**贪心采样流程图解**：
```
输入 logits:    [2.0, 1.5, 0.5, -1.0, -2.0, ...]
                 ↓ Softmax转换
概率分布:       [0.45, 0.27, 0.18, 0.06, 0.04, ...]
                 ↓ 贪心选择
处理后:         [1.0, 0, 0, 0, 0, ...]  ← 只保留最大值
                 ↓
输出 token:     索引0的token（概率最高的那个）
```

**适用场景**：
- **确定性任务**：代码生成、数学计算、事实问答
- **可复现性要求**：需要相同输入产生相同输出的场景
- **测试调试**：便于定位模型问题

**局限性**：
- 容易产生重复、机械的文本
- 缺乏创造性和多样性
- 可能陷入局部最优

贪心采样就像一位极其保守的点菜者——每次都选择评分最高的那道菜。虽然不会出错，但也永远不会尝试可能带来惊喜的新菜品。

### 16.1.2 随机采样（Dist Sampling）

**源码位置**：`src/llama-sampler.cpp`（第1022-1230行）

```cpp
struct llama_sampler_dist : public llama_sampler_backend {
    uint32_t seed;
    std::mt19937 rng;  // Mersenne Twister随机数生成器
};

static void llama_sampler_dist_apply(struct llama_sampler * smpl, 
                                      llama_token_data_array * cur_p) {
    // 根据概率分布随机采样
    llama_sampler_softmax_impl(cur_p);
    std::discrete_distribution<> dist(
        cur_p->size, 0.0, 1.0, 
        [&](double) { return cur_p->data[&_ - cur_p->data].p; }
    );
    // 使用随机数生成器进行采样
    auto * ctx = (llama_sampler_dist *) smpl->ctx;
    size_t selected = dist(ctx->rng);
    // 将选中的token概率设为1，其余设为0
    for (size_t i = 0; i < cur_p->size; ++i) {
        cur_p->data[i].p = (i == selected) ? 1.0f : 0.0f;
    }
}
```

**随机采样流程图解**：
```
概率分布:       [0.40, 0.30, 0.20, 0.10]
累积分布:       [0.40, 0.70, 0.90, 1.00]
                 ↓ 生成随机数
随机数:         0.55 (落在0.40-0.70区间)
                 ↓
选中:           索引1 (概率30%的token)
```

`std::discrete_distribution`是一个离散分布随机数生成器，它根据给定的概率权重进行采样。概率越高的token被选中机会越大，但低概率token也有机会被选中。

**为什么需要随机采样？**

1. **多样性**：避免重复生成相同的文本
2. **创造性**：给低概率但可能有趣的token一个机会
3. **自然性**：人类语言本身就具有随机性和不可预测性

### 16.1.3 温度缩放（Temperature Scaling）

**源码位置**：`src/llama-sampler.cpp`（第1798-1901行）

```cpp
struct llama_sampler_temp : public llama_sampler_backend {
    float temp;  // 温度参数
};

static void llama_sampler_temp_apply(struct llama_sampler * smpl, 
                                      llama_token_data_array * cur_p) {
    const float temp = ctx->temp;
    if (temp != 1.0f) {
        for (size_t i = 0; i < cur_p->size; ++i) {
            cur_p->data[i].logit /= temp;  // 温度缩放
        }
    }
}
```

**数学原理**：
```
P(token_i) = exp(logit_i / T) / Σ exp(logit_j / T)

T → 0:  趋近贪心采样（最确定）
T = 1:  保持原始分布
T → ∞:  趋近均匀分布（最随机）
```

**温度对分布的影响图解**：
```
原始 logits:    [2.0, 1.0, 0.0]
                 ↓ T=0.5 (低温，更确定)
调整后 logits:  [4.0, 2.0, 0.0]
Softmax后:      [0.88, 0.12, 0.00]
                 ↓ 效果：概率集中在最高分

原始 logits:    [2.0, 1.0, 0.0]
                 ↓ T=1.0 (正常温度)
调整后 logits:  [2.0, 1.0, 0.0]
Softmax后:      [0.67, 0.24, 0.09]
                 ↓ 效果：保持原始分布

原始 logits:    [2.0, 1.0, 0.0]
                 ↓ T=2.0 (高温，更随机)
调整后 logits:  [1.0, 0.5, 0.0]
Softmax后:      [0.50, 0.30, 0.20]
                 ↓ 效果：分布更平缓，多样性增加
```

**温度参数调优建议**：

| 温度值 | 特点 | 适用场景 |
|--------|------|----------|
| 0.0-0.3 | 几乎确定性 | 代码生成、数学计算 |
| 0.4-0.7 | 保守但有变化 | 技术文档、事实问答 |
| 0.8-1.0 | 平衡 | 通用对话、文章写作 |
| 1.1-1.5 | 创造性高 | 创意写作、头脑风暴 |
| >1.5 | 非常随机 | 实验性文本、艺术性创作 |

温度就像调节"冒险程度"的旋钮。低温时，模型像一位严谨的科学家，总是选择最可能的答案；高温时，模型像一位艺术家，愿意尝试意想不到的表达。

## 16.2 高级采样技术

### 16.2.1 Top-K 采样

**源码位置**：`src/llama-sampler.cpp`（第1246-1338行）

```cpp
struct llama_sampler_top_k : public llama_sampler_backend {
    int32_t k;  // 只保留前k个候选
};

static void llama_sampler_top_k_apply(struct llama_sampler * smpl, 
                                       llama_token_data_array * cur_p) {
    const int32_t k = ctx->k;
    
    // 按logit排序，只保留前k个
    llama_sampler_top_k_impl(cur_p, k);
    
    // 其余设为负无穷（概率为0）
    for (size_t i = k; i < cur_p->size; ++i) {
        cur_p->data[i].logit = -FLT_MAX;
    }
}
```

**Top-K采样流程图解**：
```
原始概率排序:   [0.40, 0.25, 0.15, 0.10, 0.05, 0.03, 0.02]
                 ↓ Top-K=3
保留前3个:      [0.40, 0.25, 0.15, 0, 0, 0, 0]
                 ↓ Softmax重新归一化
最终概率:       [0.50, 0.31, 0.19, 0, 0, 0, 0]
                 ↓ 从这三个中随机采样
```

**Top-K参数选择建议**：

| K值 | 特点 | 适用场景 |
|-----|------|----------|
| K=1 | 等价于贪心采样 | 确定性任务 |
| K=10 | 高度聚焦 | 代码生成 |
| K=40 | 通用配置 | 日常对话 |
| K=100 | 更多样性 | 创意写作 |

**Top-K的局限性**：

固定K值在不同分布下表现不一致：
- 当概率分布很平坦时，Top-K可能包含很多低质量候选
- 当概率分布很尖锐时，Top-K可能过滤掉合理的低概率候选

### 16.2.2 Top-P（Nucleus）采样

**源码位置**：`src/llama-sampler.cpp`（第1339-1532行）

```cpp
struct llama_sampler_top_p : public llama_sampler_backend {
    float p;         // 累积概率阈值（通常是0.9或0.95）
    size_t min_keep; // 最少保留数量
};

static void llama_sampler_top_p_apply(struct llama_sampler * smpl, 
                                       llama_token_data_array * cur_p) {
    // 先排序
    llama_sampler_softmax_impl(cur_p);
    llama_sampler_top_k_impl(cur_p, cur_p->size);  // 按概率排序
    
    // 计算累积概率，找到 cutoff
    float cum_sum = 0.0f;
    size_t last_idx = cur_p->size;
    for (size_t i = 0; i < cur_p->size; ++i) {
        cum_sum += cur_p->data[i].p;
        if (cum_sum >= p) {
            last_idx = i + 1;
            break;
        }
    }
    
    // 只保留到 cutoff
    for (size_t i = last_idx; i < cur_p->size; ++i) {
        cur_p->data[i].logit = -FLT_MAX;
    }
}
```

**Top-P采样流程图解**：
```
原始概率排序:   [0.40, 0.25, 0.15, 0.10, 0.05, 0.03, 0.02]
累积概率:       [0.40, 0.65, 0.80, 0.90, 0.95, 0.98, 1.00]
                 ↓ Top-P=0.9
保留到:         [0.40, 0.25, 0.15, 0.10] (累积达到0.9)
                 ↓ Softmax重新归一化
最终概率:       [0.44, 0.28, 0.17, 0.11]
```

**Top-P的优势**：

1. **自适应**：根据概率分布动态调整候选池大小
2. **质量保障**：确保候选池累积概率达到阈值
3. **灵活性**：在分布尖锐时自动减少候选，在分布平坦时自动增加候选

**Top-K vs Top-P对比**：

| 特性 | Top-K | Top-P |
|------|-------|-------|
| 候选数量 | 固定 | 动态 |
| 平坦分布 | 可能包含低质量候选 | 自适应过滤 |
| 尖锐分布 | 可能过滤合理候选 | 保留足够候选 |
| 推荐值 | K=40 | P=0.9 |

Top-P是目前最广泛推荐的采样策略，它在质量和多样性之间取得了很好的平衡。

### 16.2.3 Min-P 采样

**源码位置**：`src/llama-sampler.cpp`（第1533-1688行）

```cpp
struct llama_sampler_min_p : public llama_sampler_backend {
    float p;         // 相对于最高概率的最小比例
    size_t min_keep; // 最少保留数量
};

static void llama_sampler_min_p_apply(struct llama_sampler * smpl, 
                                       llama_token_data_array * cur_p) {
    const float p = ctx->p;
    
    llama_sampler_softmax_impl(cur_p);
    llama_sampler_top_k_impl(cur_p, cur_p->size);
    
    // 计算阈值：最高概率 * p
    const float min_p = cur_p->data[0].p * p;
    
    // 只保留概率 >= min_p 的token
    for (size_t i = 0; i < cur_p->size; ++i) {
        if (cur_p->data[i].p < min_p) {
            cur_p->data[i].logit = -FLT_MAX;
        }
    }
}
```

**Min-P采样原理**：

Min-P解决了Top-P的一个关键问题：当概率分布非常平坦时，Top-P可能包含过多低质量候选；当分布非常尖锐时，Top-P可能过滤掉合理候选。

Min-P的策略是：设置一个相对于最高概率的阈值。例如，p=0.1表示只保留概率不低于最高概率10%的候选。

**Min-P优势图解**：
```
场景1：尖锐分布
原始:           [0.80, 0.10, 0.05, 0.03, 0.02]
Top-P=0.9:      保留[0.80, 0.10] (2个)
Min-P=0.1:      保留[0.80, 0.10] (2个，0.10>=0.80*0.1)
效果:           两者相当

场景2：平坦分布
原始:           [0.15, 0.14, 0.13, 0.12, 0.11, ...]
Top-P=0.9:      保留前7-8个
Min-P=0.1:      只保留概率>=0.015的（可能只有3-4个）
效果:           Min-P过滤更多低质量候选
```

### 16.2.4 Typical Sampling

**源码位置**：`src/llama-sampler.cpp`（第1689-1797行）

```cpp
struct llama_sampler_typical : public llama_sampler_backend {
    float p;
    size_t min_keep;
};

static void llama_sampler_typical_apply(struct llama_sampler * smpl, 
                                         llama_token_data_array * cur_p) {
    const float p = ctx->p;
    
    llama_sampler_softmax_impl(cur_p);
    
    // 计算信息熵
    float entropy = 0.0f;
    for (size_t i = 0; i < cur_p->size; ++i) {
        if (cur_p->data[i].p > 0.0f) {
            entropy -= cur_p->data[i].p * logf(cur_p->data[i].p);
        }
    }
    
    // 计算每个token的"典型性"
    // 典型性 = 负对数概率与熵的接近程度
    std::vector<std::pair<float, size_t>> neg_log_probs;
    for (size_t i = 0; i < cur_p->size; ++i) {
        float neg_log_p = -logf(cur_p->data[i].p);
        float typicality = fabs(neg_log_p - entropy);
        neg_log_probs.push_back({typicality, i});
    }
    
    // 按典型性排序，保留累积概率达到p的token
    std::sort(neg_log_probs.begin(), neg_log_probs.end());
    
    float cum_sum = 0.0f;
    size_t cutoff = cur_p->size;
    for (size_t i = 0; i < neg_log_probs.size(); ++i) {
        cum_sum += cur_p->data[neg_log_probs[i].second].p;
        if (cum_sum >= p) {
            cutoff = i + 1;
            break;
        }
    }
    
    // 标记非典型token
    std::vector<bool> is_typical(cur_p->size, false);
    for (size_t i = 0; i < cutoff; ++i) {
        is_typical[neg_log_probs[i].second] = true;
    }
    
    for (size_t i = 0; i < cur_p->size; ++i) {
        if (!is_typical[i]) {
            cur_p->data[i].logit = -FLT_MAX;
        }
    }
}
```

**Typical Sampling原理**：

Typical Sampling基于信息论中的"信息量"概念：
- 高概率token（如"the"、"a"）信息量低，太常见
- 低概率token（如生僻词）信息量高，太罕见
- "典型"token的信息量接近平均熵

Typical Sampling选择信息量接近平均水平的token，产生更自然、更多样化的文本。

## 16.3 重复惩罚机制

### 16.3.1 重复惩罚实现

**源码位置**：`src/llama-sampler.cpp`（第2622-2770行）

```cpp
struct llama_sampler_penalties : public llama_sampler_backend {
    int32_t penalty_last_n;   // 检查最近n个token
    float penalty_repeat;     // 重复惩罚系数
    float penalty_freq;       // 频率惩罚
    float penalty_present;    // 存在惩罚
    bool penalize_nl;         // 是否惩罚换行
};

static void llama_sampler_penalties_apply(struct llama_sampler * smpl, 
                                           llama_token_data_array * cur_p) {
    auto * ctx = (llama_sampler_penalties *) smpl->ctx;
    
    // 统计最近n个token的出现次数
    std::unordered_map<llama_token, int> token_count;
    for (int i = 0; i < ctx->penalty_last_n && i < (int)ctx->prev.size(); ++i) {
        token_count[ctx->prev[ctx->prev.size() - 1 - i]]++;
    }
    
    // 应用惩罚
    for (size_t i = 0; i < cur_p->size; ++i) {
        const auto token = cur_p->data[i].id;
        const auto it = token_count.find(token);
        if (it != token_count.end()) {
            // 重复惩罚：降低已出现token的概率
            if (ctx->penalty_repeat != 1.0f) {
                cur_p->data[i].logit /= ctx->penalty_repeat;
            }
            
            // 频率惩罚：根据出现次数惩罚
            if (ctx->penalty_freq != 0.0f) {
                cur_p->data[i].logit -= it->second * ctx->penalty_freq;
            }
            
            // 存在惩罚：只要出现过就惩罚
            if (ctx->penalty_present != 0.0f) {
                cur_p->data[i].logit -= ctx->penalty_present;
            }
        }
    }
}
```

**三种惩罚方式对比**：

| 惩罚类型 | 公式 | 特点 |
|----------|------|------|
| 重复惩罚 | `logit /= repeat_penalty` | 简单乘法，适合避免单个token重复 |
| 频率惩罚 | `logit -= count * freq_penalty` | 线性惩罚，出现越多惩罚越大 |
| 存在惩罚 | `logit -= present_penalty` | 只要出现过就惩罚，鼓励新内容 |

**重复惩罚效果图解**：
```
历史token:      ["the", "cat", "sat", "on", "the", "mat", "and", "the", ...]
最近10个统计:   "the":3, "cat":1, "sat":1, "on":1, "mat":1, "and":1

应用惩罚前:     token "the" logit = 2.5
penalty_repeat=1.2
应用惩罚后:     token "the" logit = 2.5 / 1.2 = 2.08

结果:           "the"的选中概率降低，鼓励使用其他词汇
```

## 16.4 采样器链（Sampler Chain）

### 16.4.1 采样器链实现

**源码位置**：`src/llama-sampler.cpp`（第526-798行）

```cpp
struct llama_sampler_chain {
    std::vector<llama_sampler *> samplers;
    bool accept_logits;  // 是否接受logits输入
};

void llama_sampler_chain_add(struct llama_sampler * chain, 
                              struct llama_sampler * smpl) {
    auto * ctx = (llama_sampler_chain *) chain->ctx;
    ctx->samplers.push_back(smpl);
}

static void llama_sampler_chain_apply(struct llama_sampler * smpl, 
                                       llama_token_data_array * cur_p) {
    auto * ctx = (llama_sampler_chain *) smpl->ctx;
    // 依次调用链中的每个采样器
    for (auto * s : ctx->samplers) {
        llama_sampler_apply(s, cur_p);
    }
}
```

**采样器链就像一条生产线**：每个采样器对候选token进行一次加工，最终输出采样结果。

### 16.4.2 典型采样链配置

**标准采样链**：
```cpp
// 1. 创建采样器链
struct llama_sampler * chain = llama_sampler_chain_init({});

// 2. 按顺序添加采样器
llama_sampler_chain_add(chain, llama_sampler_init_penalties(
    penalty_last_n, penalty_repeat, penalty_freq, penalty_present, penalize_nl
));  // 1. 应用重复惩罚
llama_sampler_chain_add(chain, llama_sampler_init_temp(0.8));        // 2. 温度缩放
llama_sampler_chain_add(chain, llama_sampler_init_top_k(40));        // 3. Top-K过滤
llama_sampler_chain_add(chain, llama_sampler_init_top_p(0.95, 1));   // 4. Top-P过滤
llama_sampler_chain_add(chain, llama_sampler_init_dist(seed));       // 5. 随机采样

// 3. 使用采样器链
llama_token token = llama_sampler_sample(chain, ctx, idx);
```

**执行流程图解**：
```
原始logits      [2.0, 1.5, 0.5, -1.0, ...] (10万个token)
    ↓
[重复惩罚]      [2.0, 1.3, 0.5, -1.0, ...]  (降低"the"等高频词)
    ↓
[温度缩放 T=0.8] [2.5, 1.6, 0.6, -1.25, ...]  (除以0.8，拉大数据)
    ↓
[Top-K=40]      [2.5, 1.6, 0.6, ...] (只保留40个)
    ↓
[Top-P=0.95]    [2.5, 1.6, 0.6, ...] (保留累积95%概率的)
    ↓
[Softmax+随机]   选中token id = 1234
```

**采样器顺序的重要性**：

采样器的执行顺序非常关键，通常遵循：
1. **惩罚类**：先应用重复惩罚（修改原始logits）
2. **缩放类**：然后温度缩放（调整分布形状）
3. **过滤类**：再Top-K/Top-P过滤（减少候选集）
4. **采样类**：最后随机采样（做出选择）

改变顺序可能导致完全不同的结果！

### 16.4.3 场景化参数配置

**创意写作配置**：
```cpp
// 高温度 + 中等Top-P，鼓励创造性
params.temp = 0.9;
params.top_p = 0.95;
params.top_k = 100;
params.repeat_penalty = 1.1;
// 效果：多样化表达，偶尔有惊喜
```

**代码生成配置**：
```cpp
// 低温度 + 高重复惩罚，确保准确性
params.temp = 0.2;
params.top_p = 0.95;
params.repeat_penalty = 1.2;
params.frequency_penalty = 0.1;
// 效果：确定性输出，避免语法错误
```

**对话系统配置**：
```cpp
// 平衡配置
params.temp = 0.7;
params.top_p = 0.9;
params.top_k = 40;
params.repeat_penalty = 1.15;
// 效果：自然流畅，不过于机械也不过于随意
```

## 16.5 设计中的取舍

### 如何在多样性与质量之间权衡？

不同采样策略在多样性和质量上有各自的侧重。贪心采样总是选最高概率的 token，质量高（局部最优）但多样性极低，适合代码生成等确定性任务。高温搭配 Top-P 的组合产生高多样性，但质量会下降，适合创意写作。低温搭配 Top-K 在两者之间取得平衡，适合技术文档等需要一定准确性又不失变化的任务。典型采样（Typical Sampling）则利用信息论原理选择信息量接近平均水平的 token，在自然语言生成中表现最好。没有一种策略在所有场景下都最优，关键在于根据任务选择恰当的配置。

### 采样器链的性能开销主要在哪？

每次采样最重的计算是 Softmax，它需要遍历整个词表（通常 128K 个 token）做指数运算。优化方式是将 Softmax 推到采样链的末尾统一计算，避免中间步骤重复计算。Top-K/Top-P 需要排序，但可以使用 `std::nth_element` 做部分排序，时间复杂度从 O(n log n) 降到 O(n)。惩罚采样需要访问历史 token 统计，可使用哈希表缓存最近 N 个 token 的出现次数，避免每次全量扫描。采样器链还将多个采样器模块化组合，每个模块只修改需要修改的 logit 值，减少了不必要的数据拷贝。

### 采样算法有哪些固有局限？

贪心采样的最大问题是局部最优不等于全局最优，可能导致生成陷入循环——模型反复输出相同的短语。温度缩放虽然能增加多样性，但不同模型对温度的敏感度差异很大，同一套温度参数在不同模型上的效果可能天差地别，需要针对每个模型实验调优。此外，固定参数难以适应同一段文本中的不同上下文：开头可能需要高确定性来建立主题，中间需要适度的变化来展开论述，结尾则需要精确收尾。动态调整采样参数是解决这个问题的方向，但实现复杂度较高。

## 16.6 动手练习

### 练习1：采样参数实验

使用llama-cli测试不同参数组合，观察输出差异：

```bash
# 测试温度影响
./llama-cli -m model.gguf -p "Once upon a time" --temp 0.2 -n 50
./llama-cli -m model.gguf -p "Once upon a time" --temp 1.0 -n 50
./llama-cli -m model.gguf -p "Once upon a time" --temp 1.5 -n 50

# 测试Top-P影响
./llama-cli -m model.gguf -p "The solution is" --top-p 0.5 -n 50
./llama-cli -m model.gguf -p "The solution is" --top-p 0.95 -n 50

# 测试重复惩罚
./llama-cli -m model.gguf -p "The cat sat on the" --repeat-penalty 1.0 -n 50
./llama-cli -m model.gguf -p "The cat sat on the" --repeat-penalty 1.5 -n 50
```

### 练习2：分析采样概率分布

修改代码打印采样前后的概率分布变化：

```cpp
void print_top_tokens(llama_token_data_array * cur_p, int n = 10) {
    llama_sampler_softmax_impl(cur_p);
    // 复制并排序
    std::vector<llama_token_data> sorted(cur_p->data, cur_p->data + cur_p->size);
    std::sort(sorted.begin(), sorted.end(), 
              [](const auto& a, const auto& b) { return a.p > b.p; });
    
    printf("Top %d tokens:\n", n);
    for (int i = 0; i < n && i < sorted.size(); i++) {
        printf("  %d: token=%d, p=%.4f, logit=%.4f\n", 
               i, sorted[i].id, sorted[i].p, sorted[i].logit);
    }
}

// 在采样链的每个步骤后调用
print_top_tokens(cur_p, 10);
llama_sampler_apply(penalties_sampler, cur_p);
print_top_tokens(cur_p, 10);
llama_sampler_apply(temp_sampler, cur_p);
print_top_tokens(cur_p, 10);
```

### 练习3：自定义采样器

设计一个"多样性增强"采样器，降低语义相似token的概率：

```cpp
// 概念实现
struct diversity_sampler {
    float diversity_factor;
    std::set<llama_token> recent_tokens;
    
    void apply(llama_token_data_array * cur_p) {
        // 对于与最近token语义相近的候选，降低其概率
        // 这可以通过embedding相似度或共享n-gram来实现
        for (size_t i = 0; i < cur_p->size; ++i) {
            if (is_semantically_similar(cur_p->data[i].id, recent_tokens)) {
                cur_p->data[i].logit *= (1.0f - diversity_factor);
            }
        }
    }
};
```

## 16.7 本章小结

本章深入解析了采样算法的原理和应用。贪心采样总是选择概率最高的token，确定性强但缺乏多样性。温度缩放通过调整概率分布控制"冒险程度"，低温时保守、高温时随机。Top-K采样固定保留前K个候选token，简单高效。Top-P采样动态保留累积概率达到P%的候选，具有自适应特性，是推荐的做法。Min-P采样相对于最高概率设置阈值，避免极端情况下的不当选择。Typical采样基于信息熵选择"典型"的token，生成结果更自然。重复惩罚机制降低已出现token的概率，避免输出单调重复。采样器链采用模块化设计，可以组合多种采样策略。

采样算法是在"确定性"和"创造性"之间的权衡艺术。没有一种采样策略适合所有场景——代码生成需要确定性，创意写作需要多样性。理解每种采样算法的原理和适用场景，是调优模型生成质量的关键。

本章我们一起学习了以下概念：

| 概念 | 解释 |
|------|------|
| 贪心采样 | 总是选择概率最高的 token，确定性强但缺乏多样性 |
| 温度缩放 | 通过 logit/T 调整概率分布的尖锐程度，控制"冒险程度" |
| Top-K | 固定保留概率前 K 名的候选 token |
| Top-P | 动态保留累积概率达到 P% 的候选，自适应调整候选池大小 |
| 典型采样 | 基于信息熵选择信息量接近平均水平的 token，生成更自然 |
| 采样器链 | 将多个采样器（惩罚、温度、过滤、采样）模块化串联，依次处理 |

**下一章预告**：

下一章中，我们将学习语法约束，理解如何使用 GBNF 语法强制模型生成符合特定格式的输出，如 JSON、SQL 等结构化数据。
