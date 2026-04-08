# 第23章 高级采样工具（common/sampling.cpp） —— 采样策略的"指挥官"

## 学习目标

1. 理解`common_sampler`如何封装`llama_sampler`
2. 掌握采样器链的构建和配置方法
3. 了解语法约束与采样的集成机制
4. 学会使用性能监控和回调功能
5. 能够自定义采样策略组合

---

## 生活类比：餐厅的智能点餐推荐系统

想象你走进一家高档餐厅，面对一本厚厚的菜单却不知从何下手。这时，一个智能点餐助手来到你身边，它要根据你的喜好、预算、饮食限制，从上百道菜中为你推荐最合适的选择。这个推荐过程，与大语言模型的采样机制惊人地相似。

**候选菜品与Token候选**

菜单上的每一道菜都对应模型词汇表中的一个Token。当模型生成文本时，它会为每个Token计算一个"分数"（logit），就像每道菜都有一个综合评分。这些评分考虑了食材新鲜度、厨师擅长程度、顾客评价等因素。对模型而言，logit反映了在当前上下文中每个Token的合适程度。

**温度参数与冒险精神**

温度参数就像你告诉推荐系统的"冒险程度"。如果温度很低（如0.2），你会说："请推荐最稳妥的招牌菜"——系统总是选择评分最高的选项，结果可预测但缺乏惊喜。如果温度很高（如1.5），你会说："我想尝试些新鲜刺激的"——系统会更愿意推荐那些评分略低但独特的创意菜品，结果更多样化但也可能踩雷。

**Top-K与精选菜单**

Top-K采样就像是说："只给我看评分最高的前K道菜"。如果K=5，你只在最顶尖的5道菜中选择，忽略其他所有选项。这确保了推荐质量不会太差，但也限制了选择范围。如果K设置得太小，可能会错过一些隐藏 gems；如果K设置得太大，低质量的选项又会混入。

**Top-P与累积满意度**

Top-P（Nucleus采样）更像是一个聪明的选择策略："列出能让我90%满意的菜品集合"。系统会按评分从高到低排列，累加概率直到达到P值（如0.9），然后只在这个集合中随机选择。这样，如果前几道菜已经占据了90%的"推荐价值"，后面的菜自然被排除；如果评分分布分散，入选的菜品就会更多。

**重复惩罚与避免单调**

想象你已经连续三天吃了红烧肉，今天再看到它时，即使它评分很高，你也想换换口味。重复惩罚机制就是这样工作的：最近出现过的Token会被降低概率，就像那些"最近已经吃过"的菜被暂时从推荐列表中降级。这避免了模型像复读机一样重复同样的词语。

**语法约束与饮食限制**

有些人因为健康原因有特殊的饮食限制：素食者、对某些食物过敏的人、或者有宗教信仰的 dietary restrictions。语法约束就像这些限制——它确保生成的Token必须符合特定的结构规则（如必须是有效的JSON格式、必须遵循特定的语法规则）。即使有某个Token概率很高，如果它违反了语法规则，也会被排除。

**采样器链与多级筛选**

一个优秀的点餐系统不会只做一次筛选。它可能先根据你的饮食限制过滤（素食、无坚果），然后按评分排序（Top-K），再考虑价格区间（Top-P），最后加入一些随机性增加惊喜（温度采样）。采样器链就是这样工作的：多个采样策略按顺序应用，每个策略都在前一个的基础上进一步精炼候选集。

通过这个类比，我们可以看到采样不仅仅是"随机选一个Token"那么简单，而是一个复杂的多阶段决策过程，需要在创造性、连贯性和约束条件之间找到平衡。就像一个好的点餐系统能让顾客满意而归，一个好的采样策略能让模型生成高质量、多样化的文本。

---

## 23.1 采样参数封装 —— 配置化的艺术

### 23.1.1 参数结构体详解

在深入采样器实现之前，我们需要先理解配置系统。`common_params_sampling`结构体就像是餐厅推荐系统的"顾客偏好档案"，它包含了所有影响采样行为的参数。这个设计体现了软件工程中"配置与逻辑分离"的原则。

**源码位置**：`common/sampling.h` (第1-100行)

```cpp
// 采样参数结构体 - 完整的采样配置
struct common_params_sampling {
    // ==================== 重复惩罚参数 ====================
    // 这些参数控制模型"避免重复"的行为
    int32_t penalty_last_n = 64;      // 检查重复的范围（最近N个token）
    float penalty_repeat = 1.0f;      // 重复惩罚系数（1.0=无惩罚，>1.0=惩罚）
    float penalty_freq = 0.0f;        // 频率惩罚（出现越多，惩罚越大）
    float penalty_present = 0.0f;     // 存在惩罚（只要出现过就惩罚）

    // ==================== DRY惩罚（高级重复避免）====================
    // DRY = Don't Repeat Yourself，更智能的重复检测
    float dry_multiplier = 0.0f;      // DRY惩罚乘数（0=禁用）
    float dry_base = 1.75f;           // DRY基数（惩罚强度基础值）
    int32_t dry_allowed_length = 2;   // 允许重复的最小长度（短词不禁用）
    int32_t dry_penalty_last_n = -1;  // DRY检查范围（-1表示使用penalty_last_n）

    // ==================== 核心采样参数 ====================
    int32_t top_k = 40;               // Top-K采样：只保留概率最高的K个
    float top_p = 0.95f;              // Top-P（Nucleus）采样：保留累积概率达到P的
    float min_p = 0.05f;              // Min-P采样：过滤相对概率太低的
    float xtc_probability = 0.0f;     // XTC（Exclude Top Choices）概率
    float xtc_threshold = 0.1f;       // XTC阈值
    float typ_p = 1.0f;               // Typical采样参数（1.0=禁用）
    float top_n_sigma = -1.0f;        // Top-N-Sigma采样（-1=禁用）
    float temp = 0.8f;                // 温度参数（控制随机性）

    // ==================== Mirostat自适应采样 ====================
    // Mirostat自动调整温度以维持目标困惑度
    int32_t mirostat = 0;             // Mirostat版本（0=禁用，1或2）
    float mirostat_eta = 0.1f;        // Mirostat学习率（调整速度）
    float mirostat_tau = 5.0f;        // Mirostat目标困惑度

    // ==================== 其他参数 ====================
    int32_t n_prev = 64;              // 保留的历史Token数（用于上下文）
    int32_t n_probs = 0;              // 输出的概率数量（用于调试分析）
    std::vector<llama_token> penalty_prompt_tokens;  // 特定提示词的惩罚
    bool ignore_eos = false;          // 是否忽略EOS（End of Sequence）token

    // ==================== 语法约束 ====================
    std::string grammar;              // GBNF语法字符串（用于结构化输出）

    // ==================== 性能选项 ====================
    bool no_perf = false;             // 禁用性能统计（减少开销）
    
    // 打印参数（用于调试）
    std::string print() const;
};
```

这个结构体的设计体现了几个重要的工程决策：

1. **合理的默认值**：每个参数都有默认值，用户只需要覆盖关心的参数
2. **参数分组**：通过注释将相关参数分组，提高可读性
3. **禁用机制**：使用特殊值（如-1、0、1.0）表示"禁用"某个功能
4. **一致性命名**：使用`penalty_`前缀表示惩罚参数，`mirostat_`前缀表示Mirostat参数

### 23.1.2 参数的字符串表示

为了方便调试和日志记录，`common_params_sampling`提供了`print()`方法，将所有参数格式化为可读的字符串。这在诊断采样问题时非常有用——你可以直接看到当前的采样配置。

**源码位置**：`common/sampling.cpp` (第100-150行)

```cpp
// 将参数格式化为可读字符串
std::string common_params_sampling::print() const {
    // 使用固定大小的缓冲区（足够容纳所有参数）
    char result[1024];

    // snprintf确保不会缓冲区溢出
    snprintf(result, sizeof(result),
            // 第一行：重复惩罚参数
            "\trepeat_last_n = %d, repeat_penalty = %.3f, "
            "frequency_penalty = %.3f, presence_penalty = %.3f\n"
            // 第二行：DRY惩罚参数
            "\tdry_multiplier = %.3f, dry_base = %.3f, "
            "dry_allowed_length = %d, dry_penalty_last_n = %d\n"
            // 第三行：核心采样参数
            "\ttop_k = %d, top_p = %.3f, min_p = %.3f, "
            "xtc_probability = %.3f, xtc_threshold = %.3f, "
            "typical_p = %.3f, top_n_sigma = %.3f, temp = %.3f\n"
            // 第四行：Mirostat参数
            "\tmirostat = %d, mirostat_lr = %.3f, mirostat_ent = %.3f",
            // 参数值
            penalty_last_n, penalty_repeat, penalty_freq, penalty_present,
            dry_multiplier, dry_base, dry_allowed_length, dry_penalty_last_n,
            top_k, top_p, min_p, xtc_probability, xtc_threshold, 
            typ_p, top_n_sigma, temp,
            mirostat, mirostat_eta, mirostat_tau);

    return std::string(result);
}
```

这个方法的设计体现了几个C++最佳实践：

1. **缓冲区安全**：使用`snprintf`而不是`sprintf`，指定缓冲区大小防止溢出
2. **格式化控制**：使用`%.3f`限制浮点数精度，避免输出过长
3. **分行组织**：按逻辑分组将参数分成多行，提高可读性
4. **常量正确性**：方法声明为`const`，表示不会修改对象状态

**为什么使用固定大小的缓冲区而不是std::string拼接？**

在这个场景下，使用固定大小的`char`数组有几个优势：
- **性能**：避免了多次内存分配（`std::string`的`operator+=`可能触发重新分配）
- **简单性**：代码更简洁，不需要处理复杂的字符串操作
- **确定性**：输出大小有上限，不会因为意外情况消耗过多内存

当然，1024字节的缓冲区是保守估计——实际输出通常在500字节左右。如果参数列表大幅扩展，可能需要调整这个大小。

---

## 23.2 采样器链构建 —— 组装你的采样流水线

### 23.2.1 环形缓冲区设计

在构建采样器链之前，我们需要先了解一个基础数据结构：环形缓冲区（ring buffer）。它被用来存储最近生成的Token历史，这是实现重复惩罚的关键。

**源码位置**：`common/sampling.cpp` (第150-200行)

```cpp
// 环形缓冲区模板 - 固定大小的循环队列
// 相比std::deque，它内存连续、缓存友好、无需动态分配
template<typename T>
struct ring_buffer {
    size_t capacity = 0;   // 最大容量
    size_t sz = 0;         // 当前大小
    size_t first = 0;      // 队列头部索引
    size_t pos = 0;        // 下一个写入位置
    std::vector<T> data;   // 底层存储

    // 添加元素到缓冲区
    void push_back(T value) {
        if (sz < capacity) {
            // 还有空间，直接追加
            data.push_back(value);
            sz++;
        } else {
            // 缓冲区已满，覆盖最旧的元素
            data[pos] = value;
        }
        // 移动写入位置（循环）
        pos = (pos + 1) % capacity;
    }

    // 随机访问（按逻辑顺序）
    T & operator[](size_t i) {
        // 从first开始计算实际索引
        return data[(first + i) % capacity];
    }

    // 清空缓冲区
    void clear() {
        sz = 0;
        pos = 0;
        first = 0;
        // 注意：不释放内存，只是重置状态
    }
    
    // 获取当前大小
    size_t size() const { return sz; }
    
    // 检查是否为空
    bool empty() const { return sz == 0; }
};
```

**为什么是环形缓冲区？**

环形缓冲区是处理"滑动窗口"类型数据的经典数据结构。在采样场景中：

1. **固定大小**：我们只需要最近N个Token（如64个），不需要无限历史
2. **高效更新**：新Token到来时，O(1)时间复杂度即可更新
3. **内存友好**：预分配固定大小，无运行时分配开销
4. **缓存友好**：数据连续存储，遍历时CPU缓存命中率高

相比`std::deque`（双端队列），环形缓冲区在只需要"尾部插入、头部弹出"的场景下更高效。`std::deque`为了支持两端操作，通常采用分块存储，这会带来额外的指针追踪开销。

### 23.2.2 采样器结构详解

`common_sampler`是整个采样系统的核心，它将底层`llama_sampler`接口封装成更易于使用的形式。

**源码位置**：`common/sampling.cpp` (第200-280行)

```cpp
// 采样器封装结构 - 用户直接交互的对象
struct common_sampler {
    // ==================== 配置 ====================
    common_params_sampling params;  // 采样参数（值拷贝，线程安全）

    // ==================== 子采样器 ====================
    // 这三个采样器有特殊地位，不在主链中
    struct llama_sampler * grmr = nullptr;      // 语法约束采样器
    struct llama_sampler * rbudget = nullptr;   // 推理预算采样器
    
    // 主采样器链 - 按顺序应用多个采样策略
    struct llama_sampler * chain = nullptr;

    // ==================== 状态 ====================
    // 历史Token缓冲区（用于重复惩罚）
    ring_buffer<llama_token> prev;
    
    // 当前候选Token数组（缓存避免重复分配）
    std::vector<llama_token_data> cur;
    
    // 当前候选Token的数组表示（兼容C接口）
    llama_token_data_array cur_p;

    // ==================== 性能统计 ====================
    mutable int64_t t_total_us = 0;  // 总采样时间（微秒）
    
    // ==================== 方法 ====================
    // 设置当前候选（从模型上下文获取logits）
    void set_logits(struct llama_context * ctx, int idx);
    
    // 接受一个Token（更新历史）
    void accept(llama_token token);
    
    // 获取采样器链（用于高级操作）
    struct llama_sampler * get_chain() const { return chain; }
    
    // 获取语法采样器
    struct llama_sampler * get_grammar() const { return grmr; }
};
```

这个结构体的设计体现了几个重要原则：

1. **所有权清晰**：使用原始指针管理`llama_sampler`，因为底层是C接口
2. **状态封装**：`prev`、`cur`、`cur_p`等状态都在结构内部管理
3. **性能考虑**：`cur`向量预分配，避免每次采样都重新分配内存
4. **常量正确性**：性能统计标记为`mutable`，允许在const方法中修改

### 23.2.3 初始化采样器链

`common_sampler_init`函数是整个采样系统的"装配工厂"，它根据参数配置构建完整的采样器链。这是理解采样流程的关键代码。

**源码位置**：`common/sampling.cpp` (第280-500行)

```cpp
/**
 * 初始化采样器
 * 
 * 这是整个采样系统的核心构建函数。根据传入的参数，
 * 它会创建一个或多个采样器，并将它们组织成采样器链。
 * 
 * @param model 模型对象（用于获取词汇表信息）
 * @param params 采样参数配置
 * @return 配置好的采样器实例
 */
struct common_sampler * common_sampler_init(
    const struct llama_model * model,
    struct common_params_sampling & params
) {
    // 获取模型的词汇表
    const llama_vocab * vocab = llama_model_get_vocab(model);
    
    // 获取词汇表大小（用于预分配数组）
    const int n_vocab = llama_vocab_n_tokens(vocab);

    // ========== 步骤1：创建采样器链参数 ==========
    llama_sampler_chain_params lparams = llama_sampler_chain_default_params();
    lparams.no_perf = params.no_perf;  // 传递性能统计开关

    // ========== 步骤2：初始化特殊采样器 ==========
    // 语法采样器和推理预算采样器不在主链中，需要单独处理
    llama_sampler * grmr = nullptr;
    llama_sampler * rbudget = nullptr;
    
    // 创建主采样器链
    llama_sampler * chain = llama_sampler_chain_init(lparams);
    
    // 临时存储所有要添加到链中的采样器
    std::vector<llama_sampler *> samplers;

    // ========== 步骤3：配置语法约束采样器 ==========
    const std::string & grammar_str = common_grammar_value(params.grammar);
    if (!grammar_str.empty()) {
        // 检查是否使用LLGuidance（高性能语法引擎）
        if (grammar_str.compare(0, 11, "%llguidance") == 0) {
#ifdef LLAMA_USE_LLGUIDANCE
            // LLGuidance是可选依赖，需要编译时启用
            samplers.push_back(llama_sampler_init_llguidance(
                vocab, grammar_str.c_str()
            ));
#endif
        } else {
            // 使用标准GBNF语法解析器
            llama_grammar_params gparams;
            gparams.vocab = vocab;
            gparams.grammar_str = grammar_str.c_str();
            // 注意：这里赋值给grmr而不是加入samplers
            // 因为语法采样器有特殊处理逻辑
            grmr = llama_sampler_init_grammar(gparams);
        }
    }

    // ========== 步骤4：配置推理预算采样器 ==========
    if (params.reasoning_budget_tokens > 0) {
        rbudget = llama_sampler_init_reasoning_budget(
            params.reasoning_budget_tokens,
            params.reasoning_budget_start,
            params.reasoning_budget_end,
            params.reasoning_budget_forced
        );
    }

    // ========== 步骤5：构建主采样器链 ==========
    // 采样器按以下顺序添加，每个都在前一个的基础上处理
    
    // 5.1 重复惩罚采样器（最先应用，基于原始概率）
    if (params.penalty_last_n > 0) {
        samplers.push_back(llama_sampler_init_penalties(
            n_vocab,                          // 词汇表大小
            params.penalty_last_n,            // 检查范围
            params.penalty_repeat,            // 重复惩罚系数
            params.penalty_freq,              // 频率惩罚
            params.penalty_present,           // 存在惩罚
            params.penalize_nl,               // 是否惩罚换行符
            params.ignore_eos                 // 是否忽略EOS
        ));
    }

    // 5.2 DRY惩罚采样器（更智能的重复避免）
    if (params.dry_multiplier > 0) {
        samplers.push_back(llama_sampler_init_dry(
            vocab,
            params.dry_multiplier,
            params.dry_base,
            params.dry_allowed_length,
            // 如果dry_penalty_last_n为-1，使用penalty_last_n
            params.dry_penalty_last_n < 0 
                ? params.penalty_last_n 
                : params.dry_penalty_last_n,
            params.dry_sequence_breakers      // 序列打断符（如换行）
        ));
    }

    // 5.3 温度处理 - 根据温度值选择不同策略
    if (params.temp < 0.0f) {
        // 温度 < 0：贪婪采样（总是选择概率最高的）
        // 这实际上忽略了概率分布，直接取argmax
        samplers.push_back(llama_sampler_init_greedy());
    } else if (params.temp == 0.0f) {
        // 温度 = 0：分布采样（从概率分布中采样，无温度缩放）
        samplers.push_back(llama_sampler_init_dist(params.seed));
    } else {
        // 温度 > 0：先应用温度缩放，再应用其他采样器
        samplers.push_back(llama_sampler_init_temp(params.temp));

        // 5.4 Top-K采样器
        if (params.top_k > 0) {
            samplers.push_back(llama_sampler_init_top_k(params.top_k));
        }

        // 5.5 Top-P（Nucleus）采样器
        if (params.top_p < 1.0f) {
            samplers.push_back(llama_sampler_init_top_p(
                params.top_p, 
                params.min_keep  // 最少保留的token数
            ));
        }

        // 5.6 Min-P采样器
        if (params.min_p > 0.0f) {
            samplers.push_back(llama_sampler_init_min_p(
                params.min_p, 
                params.min_keep
            ));
        }

        // 5.7 XTC（Exclude Top Choices）采样器
        // XTC会随机排除概率最高的部分候选，增加多样性
        if (params.xtc_probability > 0.0f) {
            samplers.push_back(llama_sampler_init_xtc(
                params.xtc_probability,   // 应用XTC的概率
                params.xtc_threshold,     // 排除阈值
                params.min_keep,
                params.seed
            ));
        }

        // 5.8 Typical采样器
        // 基于信息论概念，保留"典型"的token
        if (params.typ_p < 1.0f) {
            samplers.push_back(llama_sampler_init_typical(
                params.typ_p, 
                params.min_keep
            ));
        }

        // 5.9 Top-N-Sigma采样器
        // 基于统计标准差过滤异常值
        if (params.top_n_sigma > 0.0f) {
            samplers.push_back(llama_sampler_init_top_n_sigma(
                params.top_n_sigma
            ));
        }

        // 5.10 Mirostat自适应采样
        // Mirostat会动态调整采样策略以维持目标困惑度
        if (params.mirostat > 0) {
            samplers.push_back(llama_sampler_init_mirostat(
                vocab,
                params.seed,
                params.mirostat_tau,    // 目标困惑度
                params.mirostat_eta,    // 学习率
                params.mirostat         // 版本（1或2）
            ));
        } else {
            // 5.11 普通分布采样（如果没有使用Mirostat）
            samplers.push_back(llama_sampler_init_dist(params.seed));
        }
    }

    // ========== 步骤6：组装采样器链 ==========
    for (auto * smpl : samplers) {
        if (smpl) {  // 安全检查
            llama_sampler_chain_add(chain, smpl);
        }
    }

    // ========== 步骤7：创建并初始化common_sampler ==========
    common_sampler * result = new common_sampler();
    result->params = params;
    result->grmr = grmr;           // 语法采样器单独存储
    result->rbudget = rbudget;     // 推理预算采样器单独存储
    result->chain = chain;         // 主采样器链
    result->prev.capacity = params.n_prev;  // 设置历史缓冲区大小
    
    // 预分配候选token数组，避免运行时分配
    result->cur.resize(n_vocab);
    
    // 初始化cur_p数组结构
    result->cur_p.data = result->cur.data();
    result->cur_p.size = 0;  // 初始时没有候选
    result->cur_p.sorted = false;

    return result;
}
```

这段代码虽然长，但逻辑非常清晰。让我们分解它的关键设计决策：

**采样器顺序的重要性**

采样器链中的顺序不是随意的。每个采样器都在前一个的基础上工作：

1. **惩罚采样器最先**：因为它们需要基于原始概率工作
2. **温度缩放其次**：调整整体分布形状
3. **过滤采样器随后**：Top-K、Top-P、Min-P等逐步缩小候选集
4. **采样采样器最后**：从最终候选集中实际选择一个Token

**为什么Mirostat与普通dist采样互斥？**

```cpp
if (params.mirostat > 0) {
    samplers.push_back(llama_sampler_init_mirostat(...));
} else {
    samplers.push_back(llama_sampler_init_dist(params.seed));
}
```

Mirostat是一个"自包含"的采样策略。它不仅选择Token，还会根据结果动态调整内部状态（如温度）。如果再加上普通的`dist`采样器，两者的随机性会相互干扰。Mirostat内部已经包含了采样逻辑，所以不需要额外的dist采样器。

**特殊采样器的独立管理**

语法采样器(`grmr`)和推理预算采样器(`rbudget`)不在主链中，因为它们需要特殊的处理逻辑：
- 语法采样器可能需要两阶段检查（快速路径vs慢速路径）
- 推理预算需要在特定时机触发

这种设计保持了主采样器链的简单性，同时为特殊需求提供了灵活性。

### 23.2.4 采样器链执行流程

为了更直观地理解采样器链的工作方式，让我们看看数据是如何在其中流动的：

```
输入: logits（模型原始输出，大小为词汇表大小）
    │
    ▼
┌─────────────────────────────────────────────────────┐
│ 1. 重复惩罚 (penalties)                              │
│    • 检查最近N个token                                │
│    • 降低重复出现的token概率                         │
│    • 输出：调整后的logits                            │
└────────────────────────┬────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────┐
│ 2. DRY惩罚 (dry)                                     │
│    • 检测序列重复模式                                │
│    • 对可能导致重复的token施加惩罚                   │
│    • 输出：进一步调整的logits                        │
└────────────────────────┬────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────┐
│ 3. 温度缩放 (temp)                                   │
│    • logits /= temp                                  │
│    • 应用softmax转换为概率                           │
│    • 输出：概率分布                                  │
└────────────────────────┬────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────┐
│ 4. Top-K过滤 (top_k)                                 │
│    • 按概率排序                                      │
│    • 只保留概率最高的K个                             │
│    • 其余概率设为0                                   │
│    • 输出：截断的概率分布                            │
└────────────────────────┬────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────┐
│ 5. Top-P过滤 (top_p)                                 │
│    • 按概率降序排列                                  │
│    • 累加概率直到达到P                               │
│    • 只保留这个集合                                  │
│    • 输出：进一步截断的分布                          │
└────────────────────────┬────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────┐
│ 6. Min-P过滤 (min_p)                                 │
│    • 计算相对概率 = token概率 / 最大概率             │
│    • 过滤掉相对概率 < min_p的token                   │
│    • 输出：过滤后的分布                              │
└────────────────────────┬────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────┐
│ 7. 分布采样 (dist / mirostat)                        │
│    • 对剩余概率重新归一化                            │
│    • 使用随机数生成器采样                            │
│    • 输出：选中的token ID                            │
└─────────────────────────────────────────────────────┘
```

这个流程展示了采样器链的"流水线"特性：每个阶段接收前一阶段的输出，处理后传递给下一阶段。数据在流动过程中逐渐被"精炼"，最终产生一个Token选择。

---

## 23.3 语法约束集成 —— 在创造性与规范性之间平衡

### 23.3.1 两阶段语法检查策略

语法约束是大语言模型应用中非常重要的一项功能，特别是在需要生成结构化输出时（如JSON、特定格式的文本）。llama.cpp采用了一种聪明的"两阶段"策略来平衡性能和正确性。

**源码位置**：`common/sampling.cpp` (第500-650行)

```cpp
/**
 * 采样主函数 - 集成语法约束
 * 
 * 这是采样的核心函数，它实现了"快速路径+慢速路径"的两阶段策略：
 * 
 * 快速路径（常见情况）：
 *   1. 正常采样
 *   2. 检查是否符合语法
 *   3. 符合 → 直接返回（开销小）
 * 
 * 慢速路径（异常情况）：
 *   1. 正常采样
 *   2. 检查不符合语法
 *   3. 重新采样（先应用语法约束）
 *   4. 返回符合语法的token（开销大，但保证正确性）
 * 
 * @param gsmpl 采样器实例
 * @param ctx 模型上下文
 * @param idx 批次中的索引（用于批量解码）
 * @param grammar_first 是否强制先应用语法约束
 * @return 采样的token ID
 */
llama_token common_sampler_sample(
    struct common_sampler * gsmpl,
    struct llama_context * ctx,
    int idx,
    bool grammar_first
) {
    // ========== 阶段1：准备候选 ==========
    // 从模型上下文中获取当前位置的logits
    // 这会填充gsmpl->cur_p数组
    gsmpl->set_logits(ctx, idx);

    // ========== 阶段2A：快速路径 ==========
    // 如果grammar_first=false（默认），尝试快速路径
    if (gsmpl->grmr && !grammar_first) {
        // 2A.1 先应用主采样器链（不含语法约束）
        llama_sampler_apply(gsmpl->chain, &gsmpl->cur_p);
        
        // 2A.2 获取选中的token
        const llama_token selected = 
            gsmpl->cur_p.data[gsmpl->cur_p.selected].id;
        
        // 2A.3 快速检查：这个token是否符合语法？
        // 创建一个临时副本进行检查，不修改原始候选
        llama_token_data_array cand = gsmpl->cur_p;
        llama_sampler_apply(gsmpl->grmr, &cand);
        
        // 2A.4 检查语法约束后的选择是否一致
        if (cand.data[cand.selected].id == selected) {
            // 太棒了！选中的token本身就符合语法
            // 不需要重新采样，直接返回
            return selected;
        }
        
        // 2A.5 不符合语法，准备走慢速路径
        // 注意：此时gsmpl->cur_p已经被修改，需要恢复
    }

    // ========== 阶段2B：慢速路径（或grammar_first=true）==========
    if (gsmpl->grmr) {
        // 2B.1 恢复原始候选（如果需要）
        // 如果之前走了快速路径且失败，需要重新获取logits
        if (!grammar_first) {
            gsmpl->set_logits(ctx, idx);
        }
        
        // 2B.2 先应用语法约束
        // 这会将不符合语法的token概率设为0
        llama_sampler_apply(gsmpl->grmr, &gsmpl->cur_p);
        
        // 2B.3 再应用主采样器链
        // 注意：此时cur_p中不符合语法的token概率已经为0
        // 采样器链会在剩余候选中工作
        llama_sampler_apply(gsmpl->chain, &gsmpl->cur_p);
        
        // 2B.4 返回最终选中的token
        return gsmpl->cur_p.data[gsmpl->cur_p.selected].id;
    }

    // ========== 阶段3：无语法约束 ==========
    // 如果没有启用语法约束，直接应用采样器链
    llama_sampler_apply(gsmpl->chain, &gsmpl->cur_p);
    return gsmpl->cur_p.data[gsmpl->cur_p.selected].id;
}
```

**为什么这个两阶段策略有效？**

这个策略基于一个重要的观察：**在大多数时间，采样的token本来就符合语法**。

想象你在写JSON，当前正在写 `"name": "` 之后的内容。语法要求接下来必须是字符串内容。但模型生成的token大概率就是字符串相关的token（字符、数字、空格等），违反语法的情况其实很少见。

让我们来量化一下：
- 假设词汇表大小为50,000
- 符合当前语法状态的token可能有1,000个（2%）
- 但采样器链选中的token在这1,000个之中的概率可能高达90%以上
- 因为语法约束通常与语义约束一致（例如，JSON字符串位置本就不该出现`}`）

所以：
- **90%以上的情况**：走快速路径，只需额外一次语法检查
- **不到10%的情况**：走慢速路径，需要重新采样
- **平均开销**：接近一次检查，远小于总是先应用语法的开销

### 23.3.2 性能优化策略详解

让我们更详细地分析性能优化策略：

```cpp
/**
 * 语法约束性能优化原理分析
 * 
 * 场景1：大部分Token符合语法（~90-95%的情况）
 * ┌─────────────────────────────────────────────────────┐
 * │ 快速路径执行流程：                                    │
 * │ 1. 应用采样器链        ~O(K)  K=平均候选数            │
 * │ 2. 获取选中token       O(1)                          │
 * │ 3. 语法检查（快速）    O(1)  只需验证单个token        │
 * │                                                     │
 * │ 总开销：~O(K) + 极小常数                             │
 * │ 相比无语法检查的开销增加：<5%                         │
 * └─────────────────────────────────────────────────────┘
 * 
 * 场景2：Token不符合语法（~5-10%的情况）
 * ┌─────────────────────────────────────────────────────┐
 * │ 慢速路径执行流程：                                    │
 * │ 1. 应用采样器链        ~O(K)  （快速路径已做）        │
 * │ 2. 语法检查失败        O(1)                          │
 * │ 3. 恢复候选            O(V)  V=词汇表大小             │
 * │ 4. 应用语法约束        O(V)  需检查所有token          │
 * │ 5. 重新应用采样器链    ~O(K') K'<=K（候选已减少）     │
 * │                                                     │
 * │ 总开销：~O(V) + 2*O(K)                              │
 * │ 词汇表通常50K，比快速路径慢~100倍，但发生率低         │
 * └─────────────────────────────────────────────────────┘
 * 
 * 场景3：grammar_first=true（强制语法优先）
 * ┌─────────────────────────────────────────────────────┐
 * │ 始终先应用语法约束：                                  │
 * │ 1. 应用语法约束        O(V)  每次都要检查所有         │
 * │ 2. 应用采样器链        ~O(K)                          │
 * │                                                     │
 * │ 总开销：~O(V) + O(K)                                │
 * │ 保证正确性，但性能开销最大                            │
 * └─────────────────────────────────────────────────────┘
 */
```

**代码中的关键细节**

```cpp
// 快速路径中的关键一行：
llama_token_data_array cand = gsmpl->cur_p;
```

这行代码创建了一个**副本**，而不是引用。为什么这么重要？

因为`llama_sampler_apply`会修改传入的数组——它会排序、过滤、甚至改变选中索引。如果我们在快速路径中直接修改`gsmpl->cur_p`，那么当需要走慢速路径时，原始信息就已经丢失了。

创建副本的开销是O(K)，其中K是当前候选数（经过Top-K后通常只有几十到几百）。这个开销远小于恢复原始logits的成本（需要从模型重新获取或保存完整副本）。

### 23.3.3 接受Token与状态更新

采样之后，我们需要"接受"这个Token，这会更新采样器的内部状态。

**源码位置**：`common/sampling.cpp` (第650-720行)

```cpp
/**
 * 接受一个token - 更新采样器状态
 * 
 * 接受token意味着：
 * 1. 将token加入历史缓冲区（用于重复惩罚）
 * 2. 通知所有子采样器更新状态
 * 3. 更新语法采样器的状态（如果是语法约束token）
 * 
 * @param gsmpl 采样器实例
 * @param token 要接受的token ID
 * @param apply_grammar 是否应用语法约束
 */
void common_sampler_accept(
    struct common_sampler * gsmpl,
    llama_token token,
    bool apply_grammar
) {
    // ========== 步骤1：更新历史缓冲区 ==========
    // 将token加入环形缓冲区，用于未来的重复惩罚计算
    gsmpl->prev.push_back(token);

    // ========== 步骤2：通知主采样器链 ==========
    // 采样器链中的每个采样器都有机会更新状态
    llama_sampler_accept(gsmpl->chain, token);

    // ========== 步骤3：通知语法采样器 ==========
    // 语法采样器需要知道接受了什么token来更新其内部状态机
    // 例如，如果接受了`{"`，语法状态会进入"期望key"状态
    if (gsmpl->grmr && apply_grammar) {
        llama_sampler_accept(gsmpl->grmr, token);
    }
    
    // ========== 步骤4：通知推理预算采样器 ==========
    if (gsmpl->rbudget) {
        llama_sampler_accept(gsmpl->rbudget, token);
    }
}
```

---

## 23.4 性能监控 —— 测量才能优化

### 23.4.1 RAII时间测量

性能监控是高质量软件的标志。`common/sampling.cpp`使用了一种优雅的RAII（Resource Acquisition Is Initialization）模式来自动测量代码块的执行时间。

**源码位置**：`common/sampling.cpp` (第50-100行)

```cpp
/**
 * RAII时间测量辅助类
 * 
 * 这个类利用C++的析构函数自动调用机制，实现了"进入作用域开始计时，
 * 离开作用域停止计时"的功能，无需手动调用开始/结束函数。
 * 
 * 使用示例：
 *   {
 *       common_time_meas tm(t_total_us, no_perf);
 *       // 被测量的代码
 *   }  // 这里自动停止计时并累加
 */
struct common_time_meas {
    int64_t & t_total_us;    // 引用外部计时变量
    bool no_perf;            // 是否禁用性能统计
    int64_t t_start_us;      // 开始时间戳

    // 构造函数 - 进入作用域时调用
    common_time_meas(int64_t & t_total, bool no_perf)
        : t_total_us(t_total), no_perf(no_perf) {
        if (!no_perf) {
            // 使用ggml的时间函数获取微秒级时间戳
            t_start_us = ggml_time_us();
        }
    }

    // 析构函数 - 离开作用域时自动调用
    ~common_time_meas() {
        if (!no_perf) {
            // 计算经过的时间并累加到总计
            int64_t t_end_us = ggml_time_us();
            t_total_us += t_end_us - t_start_us;
        }
    }
    
    // 禁用拷贝（避免双重计时）
    common_time_meas(const common_time_meas&) = delete;
    common_time_meas& operator=(const common_time_meas&) = delete;
    
    // 允许移动（转移计时责任）
    common_time_meas(common_time_meas&&) = default;
};
```

**使用场景示例**

```cpp
void common_sampler::set_logits(struct llama_context * ctx, int idx) {
    // 创建计时器 - 构造时自动开始计时
    common_time_meas tm(t_total_us, params.no_perf);

    // 获取logits的逻辑...
    const float * logits = llama_get_logits_ith(ctx, idx);
    
    // 填充cur数组...
    for (int i = 0; i < n_vocab; i++) {
        cur[i].id = i;
        cur[i].logit = logits[i];
        cur[i].p = 0.0f;  // 概率稍后计算
    }
    
    // 函数返回时，tm自动销毁，析构函数停止计时
}
```

**为什么使用RAII而不是显式调用？**

```cpp
// 显式调用方式（容易出错）
void some_function() {
    int64_t start = ggml_time_us();
    // ... 代码 ...
    
    // 如果有多个return路径，需要在每个地方都添加结束代码
    if (some_condition) {
        t_total += ggml_time_us() - start;  // 别忘了这个！
        return;
    }
    
    // ... 更多代码 ...
    t_total += ggml_time_us() - start;  // 主路径的结束
}

// RAII方式（自动、安全）
void some_function() {
    common_time_meas tm(t_total, no_perf);  // 开始
    // ... 代码 ...
    
    if (some_condition) {
        return;  // 自动停止计时，无需额外代码
    }
    
    // ... 更多代码 ...
    // 自动停止计时
}
```

RAII模式消除了手动管理的错误风险，特别是在有多个return路径或可能抛出异常的情况下。这是C++资源管理的最佳实践。

### 23.4.2 性能统计输出

收集了性能数据后，我们需要一种方式展示它们。`common_perf_print`函数提供了详细的采样性能统计。

**源码位置**：`common/sampling.cpp` (第720-800行)

```cpp
/**
 * 打印采样性能统计
 * 
 * 输出格式：
 *   sampling time = XXXX.XX ms / YYYY runs (ZZZ.ZZ ms per token)
 * 
 * @param ctx 模型上下文（用于获取token数量）
 * @param gsmpl 采样器实例
 */
void common_perf_print(
    const struct llama_context * ctx,
    const struct common_sampler * gsmpl
) {
    // 如果禁用了性能统计，直接返回
    if (gsmpl->params.no_perf) {
        return;
    }

    // 获取总采样时间（微秒）
    const int64_t t_total_us = gsmpl->t_total_us;
    
    // 获取生成的token数量
    const int n_tokens = ...;  // 从上下文获取

    // ========== 打印总体统计 ==========
    LOG_INF("sampling time = %10.2f ms / %5d runs (%8.3f ms per token)\n",
            t_total_us / 1000.0,           // 转换为毫秒
            n_tokens,                       // 总token数
            t_total_us / 1000.0 / n_tokens // 每个token的平均时间
    );

    // ========== 打印各子采样器的时间 ==========
    // 采样器链内部也维护了各自的计时
    llama_sampler_chain_perf_print(gsmpl->chain);
}
```

**典型输出示例**

```
sampling time =     15.34 ms /   128 runs (  0.120 ms per token)
    [top_k]   :     2.34 ms
    [top_p]   :     5.67 ms
    [min_p]   :     1.23 ms
    [dist]    :     0.50 ms
    [grammar] :     5.60 ms
```

这个输出告诉我们：
- 总共采样了128个token，耗时15.34ms
- 平均每个token采样耗时0.12ms（约8,300 token/秒）
- 最耗时的部分是Top-P（5.67ms）和语法约束（5.60ms）

**基于统计的优化建议**

有了这些数据，我们可以做出优化决策：
- 如果Top-P耗时过高，可以考虑降低`min_keep`参数
- 如果语法约束耗时过高，可以检查语法规则是否过于复杂
- 如果总体采样时间相对推理时间可以忽略，可以关闭统计以减少开销

---

## 23.5 高级功能 —— 采样器的进阶用法

### 23.5.1 批量采样与投机解码

投机解码（Speculative Decoding）是一种加速生成的方法，它使用一个较小的"草稿模型"快速生成候选token，然后由大模型并行验证。这要求采样器支持批量采样功能。

**源码位置**：`common/sampling.cpp` (第800-900行)

```cpp
/**
 * 批量采样并验证 - 用于投机解码
 * 
 * 投机解码流程：
 * 1. 草稿模型快速生成N个候选token
 * 2. 大模型并行计算这N个位置的logits
 * 3. 逐个采样并与草稿对比
 * 4. 如果一致，继续；如果不一致，停止
 * 
 * @param gsmpl 采样器实例
 * @param ctx 模型上下文
 * @param idxs 批次中各位置的索引
 * @param draft 草稿模型生成的候选token
 * @param grammar_first 是否优先语法约束
 * @return 接受的token序列
 */
std::vector<llama_token> common_sampler_sample_and_accept_n(
    struct common_sampler * gsmpl,
    struct llama_context * ctx,
    const std::vector<int> & idxs,
    const llama_tokens & draft,
    bool grammar_first
) {
    // 预分配结果数组
    std::vector<llama_token> result;
    result.reserve(idxs.size());

    // 逐个位置采样
    for (size_t i = 0; i < idxs.size(); i++) {
        // 在当前位置采样
        llama_token token = common_sampler_sample(
            gsmpl, ctx, idxs[i], grammar_first
        );

        // 记录结果
        result.push_back(token);

        // 接受这个token（更新采样器状态）
        common_sampler_accept(gsmpl, token, true);

        // 投机解码的关键：与草稿对比
        // 如果与草稿不一致，停止接受后续token
        // 因为后面的草稿token是基于前面的预测，前面错了后面也无效
        if (i < draft.size() && token != draft[i]) {
            break;
        }
    }

    return result;
}
```

**投机解码的关键洞察**

投机解码能加速的原因是：
1. **草稿模型速度快**：小模型生成N个token的时间 < 大模型生成1个token的时间
2. **并行验证**：大模型可以一次性计算N个位置的logits（利用批处理）
3. **接受率**：如果草稿质量高，大部分token会被接受

假设：
- 大模型速度：10 token/秒
- 草稿模型速度：100 token/秒
- 接受率：80%
- 草稿长度：4

则有效速度 ≈ 10 + (100 * 4 * 0.8) / (1 + 4 * 0.2) ≈ 24 token/秒

这是一个2.4倍的加速！

### 23.5.2 候选Token访问与分析

有时候我们需要检查采样器的内部状态，比如在调试时查看每个token的概率分布。`common_sampler_get_candidates`提供了这个功能。

**源码位置**：`common/sampling.cpp` (第900-950行)

```cpp
/**
 * 获取当前候选token数组
 * 
 * 这个函数允许外部代码访问采样器的内部候选状态，
 * 用于调试、分析或实现自定义采样逻辑。
 * 
 * @param gsmpl 采样器实例
 * @param do_sort 是否按概率排序（如果尚未排序）
 * @return 候选token数组（指针，不要释放）
 */
llama_token_data_array * common_sampler_get_candidates(
    struct common_sampler * gsmpl,
    bool do_sort
) {
    // 如果需要排序且当前未排序
    if (do_sort && !gsmpl->cur_p.sorted) {
        // 使用标准库的sort进行降序排序
        std::sort(
            gsmpl->cur_p.data,           // 开始指针
            gsmpl->cur_p.data + gsmpl->cur_p.size,  // 结束指针
            // 比较函数：按概率降序
            [](const llama_token_data & a, const llama_token_data & b) {
                return a.p > b.p;  // 降序：大的在前
            }
        );
        gsmpl->cur_p.sorted = true;
    }

    // 返回指向内部数组的指针
    // 注意：调用者不应修改此数组，也不应保存指针长期引用
    return &gsmpl->cur_p;
}
```

**使用场景示例**

```cpp
// 查看Top-10候选token
void print_top_candidates(common_sampler * sampler, int top_n = 10) {
    // 获取排序后的候选
    auto * candidates = common_sampler_get_candidates(sampler, true);
    
    printf("Top %d candidates:\n", top_n);
    for (int i = 0; i < std::min(top_n, (int)candidates->size); i++) {
        const auto & cand = candidates->data[i];
        printf("  %2d: token=%5d, prob=%.4f, logit=%.4f\n",
               i + 1, cand.id, cand.p, cand.logit);
    }
}
```

这个功能在调试采样问题时非常有用。例如，如果你发现模型总是输出相同的token，可以检查候选分布是否过于"尖锐"（某个token概率接近1.0）。

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

**代价**：
- 代码复杂度增加（需要维护两条路径）
- 最坏情况下延迟加倍（需要两次采样）
- 需要额外的临时存储（候选数组副本）

这个权衡是值得的，因为：
- 语法约束在结构化输出场景中是必需的（正确性优先）
- 但大部分用户输入不会频繁触发语法违规（性能重要）
- 快速路径的成功率通常>90%

### 为什么采样器链使用动态数组而不是链表？

```cpp
// 实际实现（动态数组）
struct llama_sampler_chain {
    std::vector<llama_sampler *> samplers;
    // ...
};

// 对比：链表实现
struct llama_sampler_chain {
    llama_sampler_node * head;  // 链表头
    // ...
};
struct llama_sampler_node {
    llama_sampler * sampler;
    llama_sampler_node * next;
};
```

**动态数组的优势**：
1. **缓存友好**：采样器指针连续存储，遍历时代码缓存命中率高
2. **内存开销小**：无需为每个节点存储next指针
3. **随机访问**：虽然主要顺序遍历，但偶尔需要索引访问
4. **简单性**：代码更易读、易维护

**为什么不使用链表？**
- 链表的主要优势（O(1)插入/删除）在这个场景下不需要
- 采样器链在初始化后很少修改
- 链表的指针追踪会导致更多的缓存未命中

---

## 动手练习

### 练习1：阅读采样器链构建

阅读 `common/sampling.cpp` 第280-500行，回答：

1. **温度参数如何影响采样器链的组成？**
   - 提示：查找`params.temp < 0`、`params.temp == 0`和`params.temp > 0`的处理逻辑

2. **Mirostat采样器与其他采样器如何共存？**
   - 提示：查找`params.mirostat > 0`的条件分支
   - 为什么Mirostat和`dist`采样器是互斥的？

3. **为什么DRY采样器需要vocab参数，而重复惩罚采样器不需要？**
   - 提示：查看两者的初始化参数
   - DRY需要访问词汇表的什么信息？

### 练习2：自定义采样策略

实现一个"Top-A"采样器（只保留概率大于`a * max_prob`的Token）：

```cpp
// Top-A采样器结构
struct llama_sampler_top_a {
    float a;  // 阈值系数
};

// 初始化函数
llama_sampler * llama_sampler_init_top_a(float a) {
    auto * params = new llama_sampler_top_a{a};
    
    static llama_sampler_i iface = {
        /* .name     = */ [](const llama_sampler *) { return "top-a"; },
        /* .accept   = */ [](llama_sampler *, llama_token) {},
        /* .apply    = */ [](llama_sampler * smpl, llama_token_data_array * cur_p) {
            auto * params = (llama_sampler_top_a *)smpl->params;
            
            // TODO: 实现Top-A过滤逻辑
            // 1. 找到最大概率max_p
            // 2. 计算阈值 = params->a * max_p
            // 3. 过滤掉概率 < 阈值的Token
            // 4. 重新归一化概率
        },
        /* .reset    = */ [](llama_sampler *) {},
        /* .clone    = */ [](const llama_sampler * smpl) {
            auto * params = (llama_sampler_top_a *)smpl->params;
            return llama_sampler_init_top_a(params->a);
        },
        /* .free     = */ [](llama_sampler * smpl) {
            delete (llama_sampler_top_a *)smpl->params;
        },
    };
    
    return new llama_sampler{iface, params};
}
```

**进阶挑战**：
- 添加`min_keep`参数（至少保留的token数）
- 在`common_sampler_init`中集成你的采样器
- 测试不同`a`值的效果

### 练习3：采样性能分析

编写程序比较不同采样策略的性能：

```cpp
#include "common/sampling.h"
#include <chrono>

void benchmark_sampling(const char * name, 
                        common_params_sampling & params,
                        int n_iterations = 1000) {
    // 初始化采样器...
    auto * sampler = common_sampler_init(model, params);
    
    // 计时
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < n_iterations; i++) {
        // 模拟采样流程...
        common_sampler_sample(sampler, ctx, 0, false);
        common_sampler_accept(sampler, token, true);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    
    printf("%s: %ld ms (%.2f ms/token)\n", name, ms, (float)ms / n_iterations);
    
    // 清理...
}

int main() {
    // 测试场景：1000次采样
    
    // 1. 贪婪采样（temp=-1）
    common_params_sampling greedy;
    greedy.temp = -1.0f;
    benchmark_sampling("greedy", greedy);
    
    // 2. 温度采样（temp=0.8）
    common_params_sampling temp;
    temp.temp = 0.8f;
    benchmark_sampling("temperature", temp);
    
    // 3. 复杂采样（top_k + top_p + min_p）
    common_params_sampling complex;
    complex.temp = 0.8f;
    complex.top_k = 40;
    complex.top_p = 0.95f;
    complex.min_p = 0.05f;
    benchmark_sampling("complex", complex);
    
    // 4. 带语法约束的采样
    common_params_sampling grammar;
    grammar.temp = 0.8f;
    grammar.grammar = R"({"type": "object"})";
    benchmark_sampling("grammar", grammar);
    
    return 0;
}
```

**预期发现**：
- 贪婪采样最快（无随机数生成）
- 温度采样比复杂采样快（少过滤步骤）
- 语法约束增加显著开销（取决于语法复杂度）

---

## 本课小结

本课深入解析了common/sampling.cpp的实现。`common_sampler` 是采样器封装类，负责管理采样器链、历史状态和性能统计。`common_params_sampling` 是采样参数结构体，包含所有可配置的采样选项。采样器链按顺序应用多个采样策略，每个采样器都在前一个的基础上工作。ring_buffer实现环形缓冲区，高效存储固定大小的Token历史。快速路径采用"先采样后检查"策略，优化大部分Token本来就符合语法的常见情况。慢速路径先应用语法约束再采样，保证正确性但开销更大。`common_time_meas` 是RAII时间测量工具，自动开始和停止计时。`common_sampler_sample_and_accept_n` 提供批量采样接口，支持投机解码加速。温度策略定义：temp<0使用贪婪采样，temp=0使用分布采样，temp>0应用温度缩放配合其他采样器。Mirostat是一种自适应采样技术，动态调整以维持目标困惑度。

---

## 关联阅读

- **第16章**：深入理解各种采样算法的数学原理
- **第17章**：GBNF语法约束的详细解析
- **第18章**：投机解码和Lookahead解码的完整实现
- **LLaMA.cpp Wiki**：`docs/sampling.md` 官方采样文档

---

*本章对应源码版本：master (2026-04-07)*
*本章作者：llama.cpp社区*
*最后更新：2026-04-08*
