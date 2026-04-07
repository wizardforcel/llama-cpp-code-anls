# 第23章 高级采样工具（common/sampling.cpp） —— 采样策略的"指挥官"

## 学习目标
1. 理解common_sampler如何封装llama_sampler
2. 掌握采样器链的构建和配置方法
3. 了解语法约束与采样的集成机制
4. 学会使用性能监控和回调功能
5. 能够自定义采样策略组合

---

## 生活类比：餐厅的点餐推荐系统

想象你是一个餐厅的智能点餐助手：

- **候选菜品** = 模型输出的logits（所有可能的Token）
- **温度参数** = 推荐 adventurous 程度（低=保守推荐招牌菜，高=大胆推荐创新菜）
- **Top-K** = 只考虑评分最高的K道菜
- **Top-P** = 考虑评分累积达到P%的所有菜
- **重复惩罚** = 避免重复推荐同一道菜
- **语法约束** =  dietary restrictions（素食、过敏等限制）
- **采样器链** = 多级筛选流程（先按类型筛选，再按评分排序，最后随机推荐）

就像一个好的点餐系统需要平衡多样性和用户偏好，好的采样策略也需要在创造性和连贯性之间找到平衡。

---

## 源码地图

```
common/sampling.h          # 采样器封装接口
├── common_sampler                 # 采样器封装结构
├── common_params_sampling         # 采样参数
├── common_sampler_init()          # 初始化
├── common_sampler_sample()        # 采样主函数
└── common_sampler_accept()        # 接受Token

common/sampling.cpp        # 采样器实现（832行）
├── struct common_sampler          # 内部实现
├── common_sampler_init()          # 构建采样器链
├── common_sampler_sample()        # 采样逻辑
├── grammar support                # 语法约束集成
└── performance metrics            # 性能监控
```

---

## 23.1 采样参数封装

### 23.1.1 参数结构体

**源码位置**：`common/sampling.h` (第1-100行)

```cpp
// 采样参数结构体
struct common_params_sampling {
    // 重复惩罚参数
    int32_t penalty_last_n = 64;      // 检查重复的范围
    float penalty_repeat = 1.0f;      // 重复惩罚系数
    float penalty_freq = 0.0f;        // 频率惩罚
    float penalty_present = 0.0f;     // 存在惩罚

    // DRY（Don't Repeat Yourself）惩罚
    float dry_multiplier = 0.0f;      // DRY惩罚乘数
    float dry_base = 1.75f;           // DRY基数
    int32_t dry_allowed_length = 2;   // 允许重复的最小长度
    int32_t dry_penalty_last_n = -1;  // DRY检查范围

    // 核心采样参数
    int32_t top_k = 40;               // Top-K采样
    float top_p = 0.95f;              // Top-P（Nucleus）采样
    float min_p = 0.05f;              // Min-P采样
    float xtc_probability = 0.0f;     // XTC概率
    float xtc_threshold = 0.1f;       // XTC阈值
    float typ_p = 1.0f;               // Typical采样
    float top_n_sigma = -1.0f;        // Top-N-Sigma采样
    float temp = 0.8f;                // 温度

    // Mirostat自适应采样
    int32_t mirostat = 0;             // Mirostat版本（0=禁用）
    float mirostat_eta = 0.1f;        // Mirostat学习率
    float mirostat_tau = 5.0f;        // Mirostat目标困惑度

    // 其他参数
    int32_t n_prev = 64;              // 保留的历史Token数
    int32_t n_probs = 0;              // 输出的概率数量
    std::vector<llama_token> penalty_prompt_tokens;  // 惩罚提示词
    bool ignore_eos = false;          // 是否忽略EOS

    // 语法约束
    std::string grammar;              // GBNF语法字符串

    // 性能选项
    bool no_perf = false;             // 禁用性能统计
};
```

### 23.1.2 参数打印

**源码位置**：`common/sampling.cpp` (第100-150行)

```cpp
// 将参数格式化为可读字符串
std::string common_params_sampling::print() const {
    char result[1024];

    snprintf(result, sizeof(result),
            "\trepeat_last_n = %d, repeat_penalty = %.3f, frequency_penalty = %.3f, presence_penalty = %.3f\n"
            "\tdry_multiplier = %.3f, dry_base = %.3f, dry_allowed_length = %d, dry_penalty_last_n = %d\n"
            "\ttop_k = %d, top_p = %.3f, min_p = %.3f, xtc_probability = %.3f, xtc_threshold = %.3f, "
            "typical_p = %.3f, top_n_sigma = %.3f, temp = %.3f\n"
            "\tmirostat = %d, mirostat_lr = %.3f, mirostat_ent = %.3f",
            penalty_last_n, penalty_repeat, penalty_freq, penalty_present,
            dry_multiplier, dry_base, dry_allowed_length, dry_penalty_last_n,
            top_k, top_p, min_p, xtc_probability, xtc_threshold, typ_p, top_n_sigma, temp,
            mirostat, mirostat_eta, mirostat_tau);

    return std::string(result);
}
```

---

## 23.2 采样器链构建

### 23.2.1 采样器结构

**源码位置**：`common/sampling.cpp` (第150-250行)

```cpp
// 环形缓冲区（用于存储历史Token）
template<typename T>
struct ring_buffer {
    size_t capacity = 0;
    size_t sz = 0;
    size_t first = 0;
    size_t pos = 0;
    std::vector<T> data;

    void push_back(T value) {
        if (sz < capacity) {
            data.push_back(value);
            sz++;
        } else {
            data[pos] = value;
        }
        pos = (pos + 1) % capacity;
    }

    T & operator[](size_t i) {
        return data[(first + i) % capacity];
    }

    void clear() {
        sz = 0;
        pos = 0;
        first = 0;
    }
};

// 采样器封装结构
struct common_sampler {
    common_params_sampling params;

    // 子采样器
    struct llama_sampler * grmr;      // 语法约束采样器
    struct llama_sampler * rbudget;   // 推理预算采样器
    struct llama_sampler * chain;     // 主采样器链

    // 历史Token缓冲区
    ring_buffer<llama_token> prev;

    // 当前候选Token
    std::vector<llama_token_data> cur;
    llama_token_data_array cur_p;

    // 性能统计
    mutable int64_t t_total_us = 0;
};
```

### 23.2.2 初始化采样器链

**源码位置**：`common/sampling.cpp` (第250-450行)

```cpp
struct common_sampler * common_sampler_init(
    const struct llama_model * model,
    struct common_params_sampling & params
) {
    const llama_vocab * vocab = llama_model_get_vocab(model);

    // 创建采样器链参数
    llama_sampler_chain_params lparams = llama_sampler_chain_default_params();
    lparams.no_perf = params.no_perf;

    // 初始化各子采样器
    llama_sampler * grmr = nullptr;
    llama_sampler * rbudget = nullptr;
    llama_sampler * chain = llama_sampler_chain_init(lparamms);

    std::vector<llama_sampler *> samplers;

    // 1. 添加语法约束采样器（如果指定了语法）
    const std::string & grammar_str = common_grammar_value(params.grammar);
    if (!grammar_str.empty()) {
        if (grammar_str.compare(0, 11, "%llguidance") == 0) {
#ifdef LLAMA_USE_LLGUIDANCE
            // 使用LLGuidance高性能语法引擎
            samplers.push_back(llama_sampler_init_llguidance(vocab, grammar_str.c_str()));
#endif
        } else {
            // 使用标准GBNF语法解析器
            llama_grammar_params gparams;
            gparams.vocab = vocab;
            gparams.grammar_str = grammar_str.c_str();
            grmr = llama_sampler_init_grammar(gparams);
        }
    }

    // 2. 添加推理预算采样器（如果启用）
    if (params.reasoning_budget_tokens > 0) {
        rbudget = llama_sampler_init_reasoning_budget(
            params.reasoning_budget_tokens,
            params.reasoning_budget_start,
            params.reasoning_budget_end,
            params.reasoning_budget_forced
        );
    }

    // 3. 构建主采样器链（按顺序添加）

    // 3.1 添加重复惩罚采样器
    if (params.penalty_last_n > 0) {
        samplers.push_back(llama_sampler_init_penalties(
            llama_vocab_n_tokens(vocab),
            params.penalty_last_n,
            params.penalty_repeat,
            params.penalty_freq,
            params.penalty_present,
            params.penalize_nl,
            params.ignore_eos
        ));
    }

    // 3.2 添加DRY惩罚采样器
    if (params.dry_multiplier > 0) {
        samplers.push_back(llama_sampler_init_dry(
            vocab,
            params.dry_multiplier,
            params.dry_base,
            params.dry_allowed_length,
            params.dry_penalty_last_n < 0 ? params.penalty_last_n : params.dry_penalty_last_n,
            params.dry_sequence_breakers
        ));
    }

    // 3.3 根据温度选择采样策略
    if (params.temp < 0.0f) {
        // 贪婪采样（温度<0）
        samplers.push_back(llama_sampler_init_greedy());
    } else if (params.temp == 0.0f) {
        // 分布采样（温度=0）
        samplers.push_back(llama_sampler_init_dist(params.seed));
    } else {
        // 温度缩放 + 其他采样器
        samplers.push_back(llama_sampler_init_temp(params.temp));

        // 3.4 添加Top-K采样器
        if (params.top_k > 0) {
            samplers.push_back(llama_sampler_init_top_k(params.top_k));
        }

        // 3.5 添加Top-P（Nucleus）采样器
        if (params.top_p < 1.0f) {
            samplers.push_back(llama_sampler_init_top_p(params.top_p, params.min_keep));
        }

        // 3.6 添加Min-P采样器
        if (params.min_p > 0.0f) {
            samplers.push_back(llama_sampler_init_min_p(params.min_p, params.min_keep));
        }

        // 3.7 添加XTC采样器
        if (params.xtc_probability > 0.0f) {
            samplers.push_back(llama_sampler_init_xtc(
                params.xtc_probability,
                params.xtc_threshold,
                params.min_keep,
                params.seed
            ));
        }

        // 3.8 添加Typical采样器
        if (params.typ_p < 1.0f) {
            samplers.push_back(llama_sampler_init_typical(params.typ_p, params.min_keep));
        }

        // 3.9 添加Top-N-Sigma采样器
        if (params.top_n_sigma > 0.0f) {
            samplers.push_back(llama_sampler_init_top_n_sigma(params.top_n_sigma));
        }

        // 3.10 添加Mirostat自适应采样器
        if (params.mirostat > 0) {
            samplers.push_back(llama_sampler_init_mirostat(
                vocab,
                params.seed,
                params.mirostat_tau,
                params.mirostat_eta,
                params.mirostat
            ));
        } else {
            // 3.11 普通分布采样
            samplers.push_back(llama_sampler_init_dist(params.seed));
        }
    }

    // 4. 将所有采样器添加到链中
    for (auto * smpl : samplers) {
        llama_sampler_chain_add(chain, smpl);
    }

    // 5. 创建common_sampler实例
    common_sampler * result = new common_sampler;
    result->params = params;
    result->grmr = grmr;
    result->rbudget = rbudget;
    result->chain = chain;
    result->prev.capacity = params.n_prev;
    result->cur.resize(llama_vocab_n_tokens(vocab));

    return result;
}
```

### 23.2.3 采样器链执行流程

```
输入: logits（模型原始输出）
    │
    ▼
┌─────────────────┐
│ 1. 重复惩罚     │  ← 降低最近出现过的Token概率
│    (penalties)  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 2. DRY惩罚      │  ← 降低序列重复概率
│    (dry)        │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 3. 温度缩放     │  ← 调整分布的"尖锐"程度
│    (temp)       │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 4. Top-K过滤    │  ← 只保留概率最高的K个
│    (top_k)      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 5. Top-P过滤    │  ← 保留累积概率达到P的Token
│    (top_p)      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 6. Min-P过滤    │  ← 过滤掉相对概率太低的Token
│    (min_p)      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 7. 分布采样     │  ← 从剩余候选中随机选择
│    (dist)       │
└────────┬────────┘
         │
         ▼
输出: 选中的Token
```

---

## 23.3 语法约束集成

### 23.3.1 两阶段语法检查

**源码位置**：`common/sampling.cpp` (第450-600行)

```cpp
// 采样主函数（集成语法约束）
llama_token common_sampler_sample(
    struct common_sampler * gsmpl,
    struct llama_context * ctx,
    int idx,
    bool grammar_first
) {
    // 1. 设置logits
    gsmpl->set_logits(ctx, idx);

    // 2. 应用主采样器链
    llama_sampler_apply(gsmpl->chain, &gsmpl->cur_p);

    // 3. 获取采样的Token
    llama_token selected = gsmpl->cur_p.data[gsmpl->cur_p.selected].id;

    // 4. 语法约束检查（快速路径）
    if (gsmpl->grmr && !grammar_first) {
        // 先检查选中的Token是否符合语法
        llama_token_data_array cand = gsmpl->cur_p;
        llama_sampler_apply(gsmpl->grmr, &cand);

        if (cand.data[cand.selected].id == selected) {
            // Token符合语法，直接接受
            return selected;
        }

        // Token不符合语法，需要重新采样（慢速路径）
        // 恢复原始候选
        gsmpl->set_logits(ctx, idx);

        // 先应用语法约束
        llama_sampler_apply(gsmpl->grmr, &gsmpl->cur_p);

        // 再应用其他采样器
        llama_sampler_apply(gsmpl->chain, &gsmpl->cur_p);

        selected = gsmpl->cur_p.data[gsmpl->cur_p.selected].id;
    }

    return selected;
}
```

### 23.3.2 性能优化策略

```cpp
// 语法约束性能优化原理：
//
// 场景1：大部分Token符合语法（常见情况）
//   - 快速路径：采样 → 检查通过 → 完成
//   - 只执行一次语法检查，开销极小
//
// 场景2：Token不符合语法（较少见）
//   - 慢速路径：采样 → 检查失败 → 重新采样
//   - 需要两次应用采样器链，开销较大
//   - 但由于符合语法的Token通常很多，失败率不高
//
// 场景3：grammar_first=true（强制语法优先）
//   - 始终先应用语法约束
//   - 确保所有候选都符合语法
//   - 开销最大，但保证结果正确性
```

---

## 23.4 性能监控

### 23.4.1 时间测量

**源码位置**：`common/sampling.cpp` (第50-100行)

```cpp
// 时间测量辅助类
struct common_time_meas {
    int64_t & t_total_us;
    bool no_perf;
    int64_t t_start_us;

    common_time_meas(int64_t & t_total, bool no_perf)
        : t_total_us(t_total), no_perf(no_perf) {
        if (!no_perf) {
            t_start_us = ggml_time_us();
        }
    }

    ~common_time_meas() {
        if (!no_perf) {
            t_total_us += ggml_time_us() - t_start_us;
        }
    }
};

// 使用示例
void common_sampler::set_logits(struct llama_context * ctx, int idx) {
    common_time_meas tm(t_total_us, params.no_perf);  // 自动计时

    // ... 设置logits的逻辑
}
```

### 23.4.2 性能统计输出

**源码位置**：`common/sampling.cpp` (第700-800行)

```cpp
void common_perf_print(
    const struct llama_context * ctx,
    const struct common_sampler * gsmpl
) {
    if (gsmpl->params.no_perf) {
        return;
    }

    const int64_t t_total_us = gsmpl->t_total_us;

    // 采样时间统计
    LOG_INF("sampling time = %10.2f ms\n", t_total_us / 1000.0);
    LOG_INF("sampling time per token = %10.2f ms\n", 
            t_total_us / 1000.0 / n_tokens);

    // 各子采样器时间（通过llama_sampler_chain获取）
    llama_sampler_chain_perf_print(gsmpl->chain);
}
```

---

## 23.5 高级功能

### 23.5.1 批量采样与投机解码

**源码位置**：`common/sampling.cpp` (第600-700行)

```cpp
// 批量采样（用于投机解码）
std::vector<llama_token> common_sampler_sample_and_accept_n(
    struct common_sampler * gsmpl,
    struct llama_context * ctx,
    const std::vector<int> & idxs,
    const llama_tokens & draft,
    bool grammar_first
) {
    std::vector<llama_token> result;
    result.reserve(idxs.size());

    // 逐个采样，与draft对比
    for (size_t i = 0; i < idxs.size(); i++) {
        llama_token token = common_sampler_sample(
            gsmpl, ctx, idxs[i], grammar_first
        );

        result.push_back(token);

        // 接受Token
        common_sampler_accept(gsmpl, token, true);

        // 如果与draft不一致，停止接受
        if (i < draft.size() && token != draft[i]) {
            break;
        }
    }

    return result;
}
```

### 23.5.2 候选Token访问

```cpp
// 获取当前候选Token（用于分析）
llama_token_data_array * common_sampler_get_candidates(
    struct common_sampler * gsmpl,
    bool do_sort
) {
    if (do_sort && !gsmpl->cur_p.sorted) {
        // 按概率排序
        std::sort(
            gsmpl->cur_p.data,
            gsmpl->cur_p.data + gsmpl->cur_p.size,
            [](const llama_token_data & a, const llama_token_data & b) {
                return a.p > b.p;
            }
        );
        gsmpl->cur_p.sorted = true;
    }

    return &gsmpl->cur_p;
}
```

---

## 设计中的取舍

### 为什么使用"快速路径+慢速路径"的语法检查策略？

| 策略 | 平均延迟 | 最坏延迟 | 正确性 | common选择 |
|-----|---------|---------|-------|-----------|
| 总是先应用语法 | 高 | 高 | 保证 | 否（grammar_first=false） |
| 快速路径优先 | 低 | 高 | 保证 | **是** |
| 无语法检查 | 最低 | 最低 | 不保证 | 否 |

**快速路径的优势**：
1. **常见情况优化**：大部分Token符合语法，无需重新采样
2. **延迟可预测**：平均性能接近无语法检查
3. **正确性保证**：必要时回退到慢速路径

### 为什么采样器链使用链表而不是数组？

```cpp
// 链表结构
struct llama_sampler_chain {
    std::vector<llama_sampler *> samplers;
    // ...
};

// vs 数组结构
llama_sampler * chain[MAX_SAMPLERS];
```

**链表/动态数组的优势**：
1. **灵活性**：运行时动态构建
2. **可扩展性**：支持任意数量的采样器
3. **内存效率**：只分配需要的空间

---

## 动手练习

### 练习1：阅读采样器链构建

阅读 `common/sampling.cpp` 第250-450行，回答：
1. 温度参数如何影响采样器链的组成？
2. Mirostat采样器与其他采样器如何共存？
3. 为什么DRY采样器需要vocab参数？

### 练习2：自定义采样策略

实现一个"Top-A"采样器（只保留概率大于阈值a * max_prob的Token）：

```cpp
// 添加到采样器链
llama_sampler * llama_sampler_init_top_a(float a) {
    return new llama_sampler_top_a{a};
}

// 应用函数
void llama_sampler_top_a_apply(llama_token_data_array * cur_p, float a) {
    // TODO: 实现Top-A过滤逻辑
    // 1. 找到最大概率
    // 2. 计算阈值 = a * max_prob
    // 3. 过滤掉概率 < 阈值的Token
}
```

### 练习3：采样性能分析

编写程序比较不同采样策略的性能：
```cpp
// 测试场景：1000次采样
// 1. 贪婪采样（temp=-1）
// 2. 温度采样（temp=0.8）
// 3. 带语法约束的采样
// 比较平均延迟和吞吐量
```

---

## 本课小结

| 概念 | 一句话总结 |
|------|-----------|
| common_sampler | 采样器封装，管理采样器链和历史 |
| 采样器链 | 按顺序应用多个采样策略 |
| 快速路径 | 先采样后检查，优化常见情况 |
| 慢速路径 | 先应用语法约束，保证正确性 |
| ring_buffer | 环形缓冲区，存储历史Token |
| cur_p | 当前候选Token数组 |
| common_time_meas | RAII时间测量工具 |

---

*本章对应源码版本：master (2026-04-07)*
