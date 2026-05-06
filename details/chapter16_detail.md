# 第16章 采样算法详解（llama_sampling）—— 让模型"开口说话"的艺术

## 1. 学习目标

通过本章学习，你将能够：
- 理解大模型生成文本的核心机制——采样算法
- 掌握贪心采样、随机采样、温度缩放等基础采样方法
- 深入理解 Top-K、Top-P（Nucleus）、Min-P、Typical Sampling 等高级采样技术
- 学会配置重复惩罚机制，避免生成重复内容
- 能够根据不同任务场景选择合适的采样参数

## 2. 生活类比：餐厅点菜的艺术

想象你来到一家有1000道菜品的餐厅，每道菜都有一个"推荐分数"（logit）。采样算法就像是你选择菜品的策略：

- **贪心采样**：每次都点分数最高的那道菜——稳妥但缺乏惊喜
- **随机采样**：完全随机选择——可能点到黑暗料理
- **温度缩放**：调整"冒险程度"，温度高更愿意尝试低分菜品，温度低则保守
- **Top-K**：只从评分前K名的菜品中选择
- **Top-P**：从累计评分占比达到P%的菜品池中选择
- **重复惩罚**：避免连续点同一道菜，让餐桌更丰富

## 3. 源码地图

```
src/llama-sampler.cpp          # 采样器核心实现（约3800行）
  ├── struct llama_sampler_base            # 采样器基类
  ├── struct llama_sampler_greedy          # 贪心采样器
  ├── struct llama_sampler_dist            # 分布采样器
  ├── struct llama_sampler_top_k           # Top-K采样器
  ├── struct llama_sampler_top_p           # Top-P采样器
  ├── struct llama_sampler_min_p           # Min-P采样器
  ├── struct llama_sampler_typical         # Typical采样器
  ├── struct llama_sampler_temp            # 温度缩放采样器
  ├── struct llama_sampler_penalties       # 重复惩罚采样器
  ├── struct llama_sampler_mirostat        # Mirostat采样器
  ├── struct llama_sampler_xtc             # XTC采样器
  ├── struct llama_sampler_infill          # Infill采样器
  ├── struct llama_sampler_grammar         # 语法约束采样器
  ├── struct llama_sampler_logit_bias      # Logit偏置采样器
  ├── struct llama_sampler_chain           # 采样器链
  │   └── llama_sampler_chain_add()        # 添加采样器到链
  └── llama_sampler_apply()                # 应用采样器

include/llama.h                # C API 声明
  ├── struct llama_token_data            # Token数据(id/logit/p)
  ├── struct llama_token_data_array      # Token数据数组
  ├── struct llama_sampler_i             # 采样器接口
  ├── struct llama_sampler               # 采样器基类
  ├── struct llama_sampler_chain_params  # 采样器链参数
  ├── struct llama_logit_bias            # Logit偏置
  └── llama_sampler_init_*               # 各类采样器初始化

common/sampling.h/cpp          # 高级采样封装
  ├── common_sampler_init()            # 采样器初始化
  ├── common_sampler_sample()          # 采样执行
  ├── llama_sampling_params            # 采样参数结构
  ├── common_sampler_get_seed()        # 获取采样器种子
  ├── common_sampler_set_seed()        # 设置采样器种子
  └── common_sampler_accept()          # 接受token更新状态
```

## 4. 详细章节内容

### 4.1 基础采样方法

#### 4.1.1 贪心采样（Greedy Sampling）

**源码位置**：`src/llama-sampler.cpp:931-1019`

```cpp
struct llama_sampler_greedy : public llama_sampler_backend {
    // 贪心采样：总是选择概率最高的token
};

static void llama_sampler_greedy_apply(struct llama_sampler * smpl, llama_token_data_array * cur_p) {
    // 找到logit最大的token，设为1.0，其他设为0
    llama_sampler_softmax_impl(cur_p);
    cur_p->data[0].p = 1.0f;
    for (size_t i = 1; i < cur_p->size; ++i) {
        cur_p->data[i].p = 0.0f;
    }
}
```

**图解**：
```
输入 logits:    [2.0, 1.5, 0.5, -1.0, ...]
                 ↓
Softmax:        [0.45, 0.27, 0.18, 0.10, ...]
                 ↓
贪心选择:       [1.0, 0, 0, 0, ...]  ← 只保留最大值
                 ↓
输出 token:     索引0的token
```

**适用场景**：
- 确定性任务（代码生成、数学计算）
- 需要可复现结果的场景

**代码示例**：
```cpp
// 创建贪心采样器
struct llama_sampler * smpl = llama_sampler_init_greedy();

// 应用采样
llama_sampler_apply(smpl, &candidates);
llama_token token = llama_sampler_sample(smpl, ctx, 0);
```

#### 4.1.2 随机采样（Dist Sampling）

**源码位置**：`src/llama-sampler.cpp:1022-1230`

```cpp
struct llama_sampler_dist : public llama_sampler_backend {
    uint32_t seed;
    std::mt19937 rng;
};

static void llama_sampler_dist_apply(struct llama_sampler * smpl, llama_token_data_array * cur_p) {
    // 根据概率分布随机采样
    llama_sampler_softmax_impl(cur_p);
    std::discrete_distribution<> dist(cur_p->size, 0.0, 1.0, 
        [&](double) { return cur_p->data[&_ - cur_p->data].p; });
    // ... 采样逻辑
}
```

**图解**：
```
概率分布:       [0.4, 0.3, 0.2, 0.1]
随机数:         0.35 (落在0.4区间内)
                 ↓
选中:           索引0 (概率40%)
```

#### 4.1.3 温度缩放（Temperature Scaling）

**源码位置**：`src/llama-sampler.cpp:1798-1901`

```cpp
struct llama_sampler_temp : public llama_sampler_backend {
    float temp;  // 温度参数
};

static void llama_sampler_temp_apply(struct llama_sampler * smpl, llama_token_data_array * cur_p) {
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

**图解**：
```
原始 logits:    [2.0, 1.0, 0.0]
                 ↓ T=0.5 (低温，更确定)
调整后:         [4.0, 2.0, 0.0]
Softmax:        [0.88, 0.12, 0.00]
                 ↓
高概率集中在最高分

原始 logits:    [2.0, 1.0, 0.0]
                 ↓ T=2.0 (高温，更随机)
调整后:         [1.0, 0.5, 0.0]
Softmax:        [0.50, 0.30, 0.20]
                 ↓
概率分布更平缓
```

### 4.2 高级采样技术

#### 4.2.1 Top-K 采样

**源码位置**：`src/llama-sampler.cpp:1246-1338`

```cpp
struct llama_sampler_top_k : public llama_sampler_backend {
    int32_t k;  // 只保留前k个
};

static void llama_sampler_top_k_apply(struct llama_sampler * smpl, llama_token_data_array * cur_p) {
    const int32_t k = ctx->k;
    
    // 按logit排序，只保留前k个
    llama_sampler_top_k_impl(cur_p, k);
    
    // 其余设为负无穷（概率为0）
    for (size_t i = k; i < cur_p->size; ++i) {
        cur_p->data[i].logit = -FLT_MAX;
    }
}
```

**图解**：
```
原始排序:       [0.40, 0.25, 0.15, 0.10, 0.05, 0.03, 0.02]
                 ↓ Top-K=3
保留:           [0.40, 0.25, 0.15, 0, 0, 0, 0]
                 ↓
重新归一化:     [0.50, 0.31, 0.19, 0, 0, 0, 0]
```

**参数选择建议**：
- K=1：等价于贪心采样
- K=40：通用对话场景
- K=10：需要更聚焦的回答

#### 4.2.2 Top-P（Nucleus）采样

**源码位置**：`src/llama-sampler.cpp:1339-1532`

```cpp
struct llama_sampler_top_p : public llama_sampler_backend {
    float p;         // 累积概率阈值
    size_t min_keep; // 最少保留数量
};

static void llama_sampler_top_p_apply(struct llama_sampler * smpl, llama_token_data_array * cur_p) {
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

**图解**：
```
原始概率:       [0.40, 0.25, 0.15, 0.10, 0.05, 0.03, 0.02]
累积概率:       [0.40, 0.65, 0.80, 0.90, 0.95, 0.98, 1.00]
                 ↓ Top-P=0.9
保留到:         [0.40, 0.25, 0.15, 0.10] (累积0.9)
                 ↓
重新归一化:     [0.44, 0.28, 0.17, 0.11]
```

**Top-K vs Top-P**：
- Top-K：固定数量，可能包含低质量候选
- Top-P：动态数量，根据分布自适应调整

#### 4.2.3 Min-P 采样

**源码位置**：`src/llama-sampler.cpp:1533-1688`

```cpp
struct llama_sampler_min_p : public llama_sampler_backend {
    float p;         // 相对于最高概率的最小比例
    size_t min_keep;
};

static void llama_sampler_min_p_apply(struct llama_sampler * smpl, llama_token_data_array * cur_p) {
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

**优势**：
- 避免Top-P在分布平坦时包含过多低质量候选
- 避免Top-K在分布尖锐时过滤掉合理候选

#### 4.2.4 Typical Sampling

**源码位置**：`src/llama-sampler.cpp:1689-1797`

```cpp
struct llama_sampler_typical {
    float p;
    size_t min_keep;
};

static void llama_sampler_typical_apply(struct llama_sampler * smpl, llama_token_data_array * cur_p) {
    // 基于信息熵的采样
    // 计算每个token的负对数概率（信息量）
    // 保留"典型"的信息量范围内的token
    
    const float entropy = -std::accumulate(...);  // 计算熵
    
    // 按与熵的距离排序
    // 保留累积概率达到p的token
}
```

**原理**：
- 选择"信息量"接近平均水平的token
- 过滤掉概率过高（太常见）和过低（太罕见）的token
- 产生更自然、更多样化的文本

### 4.3 重复惩罚机制

**源码位置**：`src/llama-sampler.cpp:2622-2770`

```cpp
struct llama_sampler_penalties {
    int32_t penalty_last_n;   // 检查最近n个token
    float penalty_repeat;     // 重复惩罚系数
    float penalty_freq;       // 频率惩罚
    float penalty_present;    // 存在惩罚
};

static void llama_sampler_penalties_apply(struct llama_sampler * smpl, llama_token_data_array * cur_p) {
    // 统计最近n个token的出现次数
    std::unordered_map<llama_token, int> token_count;
    for (int i = 0; i < penalty_last_n && i < (int)prev.size(); ++i) {
        token_count[prev[prev.size() - 1 - i]]++;
    }
    
    // 应用惩罚
    for (size_t i = 0; i < cur_p->size; ++i) {
        const auto token = cur_p->data[i].id;
        const auto it = token_count.find(token);
        if (it != token_count.end()) {
            // 重复惩罚：降低已出现token的概率
            const float penalty = penalty_repeat;
            cur_p->data[i].logit /= penalty;
        }
    }
}
```

**三种惩罚方式**：

1. **重复惩罚（Repeat Penalty）**：
   - 简单除法惩罚
   - 适用于避免单个token重复

2. **频率惩罚（Frequency Penalty）**：
   - 根据出现次数线性惩罚
   - 公式：`logit -= freq * penalty_freq`

3. **存在惩罚（Presence Penalty）**：
   - 只要出现过就惩罚（与次数无关）
   - 鼓励引入新话题

**图解**：
```
历史token:      ["the", "cat", "sat", "on", "the", "mat"]
                 ↓
统计:           "the"出现2次，其他各1次
                 ↓
惩罚应用:       "the"的logit /= 1.2 (假设penalty=1.2)
                 ↓
结果:           降低"the"被选中的概率，鼓励多样性
```

### 4.4 采样参数调优

#### 4.4.1 采样器链（Sampler Chain）

**源码位置**：`src/llama-sampler.cpp:526-798`

```cpp
struct llama_sampler_chain {
    std::vector<llama_sampler *> samplers;
    // 按顺序应用多个采样器
};

void llama_sampler_chain_add(struct llama_sampler * chain, struct llama_sampler * smpl) {
    auto * ctx = (llama_sampler_chain *) chain->ctx;
    ctx->samplers.push_back(smpl);
}

static void llama_sampler_chain_apply(struct llama_sampler * smpl, llama_token_data_array * cur_p) {
    auto * ctx = (llama_sampler_chain *) smpl->ctx;
    // 依次调用链中的每个采样器
    for (auto * s : ctx->samplers) {
        llama_sampler_apply(s, cur_p);
    }
}
```

**典型配置**：
```cpp
// 标准采样链
llama_sampler_chain_add(chain, llama_sampler_init_penalties(...));  // 1. 惩罚
llama_sampler_chain_add(chain, llama_sampler_init_temp(0.8));        // 2. 温度
llama_sampler_chain_add(chain, llama_sampler_init_top_k(40));        // 3. Top-K
llama_sampler_chain_add(chain, llama_sampler_init_top_p(0.95, 1));   // 4. Top-P
llama_sampler_chain_add(chain, llama_sampler_init_dist(seed));       // 5. 采样
```

#### 4.4.2 场景化参数配置

**创意写作**：
```cpp
// 高温度 + 中等Top-P，鼓励创造性
params.temp = 0.9;
params.top_p = 0.95;
params.top_k = 100;
params.repeat_penalty = 1.1;
```

**代码生成**：
```cpp
// 低温度 + 高重复惩罚，确保准确性
params.temp = 0.2;
params.top_p = 0.95;
params.repeat_penalty = 1.2;
params.frequency_penalty = 0.1;
```

**对话系统**：
```cpp
// 平衡配置
params.temp = 0.7;
params.top_p = 0.9;
params.top_k = 40;
params.repeat_penalty = 1.15;
```

## 5. 设计中的取舍

### 5.1 多样性与质量的权衡

| 策略 | 多样性 | 质量 | 适用场景 |
|------|--------|------|----------|
| 贪心 | 低 | 高（局部） | 确定性任务 |
| 高温+Top-P | 高 | 中 | 创意写作 |
| 低温+Top-K | 中 | 高 | 技术文档 |

### 5.2 性能考量

- **Softmax计算**：每次采样都需要，是主要开销
- **排序开销**：Top-K/Top-P需要排序，可用部分排序优化
- **内存访问**：惩罚采样需要访问历史token

### 5.3 采样算法的局限性

1. **贪心陷阱**：局部最优不等于全局最优
2. **温度敏感**：不同模型对温度敏感度不同
3. **上下文依赖**：固定参数难以适应所有上下文

## 6. 动手练习

### 练习1：实现自定义采样器

```cpp
// 实现一个"多样性增强"采样器
struct diversity_sampler {
    float diversity_factor;
    
    void apply(llama_token_data_array * cur_p) {
        // 降低已选token附近token的概率
        // 鼓励选择语义上不同的token
    }
};
```

### 练习2：采样参数实验

使用llama-cli测试不同参数组合：
```bash
# 测试温度影响
./llama-cli -m model.gguf -p "Once upon a time" --temp 0.2
./llama-cli -m model.gguf -p "Once upon a time" --temp 1.0
./llama-cli -m model.gguf -p "Once upon a time" --temp 1.5

# 测试Top-P影响
./llama-cli -m model.gguf -p "The solution is" --top-p 0.5
./llama-cli -m model.gguf -p "The solution is" --top-p 0.95
```

### 练习3：分析采样概率分布

修改代码打印采样前后的概率分布变化：
```cpp
// 在llama_sampler_apply前后打印前10个token的概率
void print_top_tokens(llama_token_data_array * cur_p, int n = 10) {
    llama_sampler_softmax_impl(cur_p);
    llama_sampler_top_k_impl(cur_p, n);
    for (int i = 0; i < n; i++) {
        printf("%d: token=%d, p=%.4f\n", i, cur_p->data[i].id, cur_p->data[i].p);
    }
}
```

## 7. 本课小结

本章我们深入学习了llama.cpp的采样算法实现：

1. **基础采样**：贪心采样确定性强，随机采样引入多样性
2. **温度缩放**：控制"冒险程度"，平衡确定性与创造性
3. **高级技术**：
   - Top-K：固定候选数量
   - Top-P：动态候选数量（推荐）
   - Min-P：基于相对概率过滤
   - Typical：基于信息熵选择
4. **重复惩罚**：频率/存在/重复三种机制避免单调
5. **采样链**：模块化组合多种采样策略

**关键源码文件**：`src/llama-sampler.cpp`、`include/llama.h`、`common/sampling.cpp`

**下一步**：学习如何使用语法约束进一步控制生成内容，让模型输出符合特定格式。
