# 第7章 模型架构支持 —— 适配"百花齐放的模型生态"

## 学习目标
1. 理解llama.cpp支持的50+模型架构
2. 掌握llm_arch配置系统的设计
3. 理解模型张量映射机制
4. 能添加对新架构的支持

---

## 生活类比：万能翻译官的语言系统

想象llama.cpp是一位**精通百种语言的联合国翻译官**：

- **不同模型架构** = 不同国家的语言方言（美式英语、英式英语、澳洲英语...）
- **llm_arch配置** = 每种语言的语法规则手册
- **张量映射** = 词汇对照表（把"elevator"映射到"lift"）
- **统一计算图** = 通用的语义理解层（无论哪种语言都转成统一的概念）
- **添加新架构** = 学习新语言（掌握规则后就能翻译）

就像优秀的翻译官需要先掌握语言规则再开口翻译，llama.cpp需要先理解模型架构配置才能正确加载和运行模型。

---

## 源码地图

```
src/llama-arch.h
├── enum llm_arch            # 架构枚举（50+种模型）
├── struct llm_arch_info     # 架构信息
└── llm_arch_from_string()   # 字符串转枚举

src/llama-arch.cpp
├── 架构配置数据（第100-1000行）
│   └── LLM_ARCH_INFO        # 各架构的层数、维度等配置
├── 张量名称映射（第1000-2000行）
│   └── LLM_TENSOR_NAMES     # 各架构的张量命名规则
└── 架构检测逻辑（第2000-3000行）

src/llama-hparams.h/cpp
├── llama_hparams            # 超参数结构体
└── 加载和验证逻辑

src/llama-model-loader.cpp
├── 张量映射逻辑（第1-1000行）
└── 权重加载实现（第1000-2000行）
```

---

## 7.1 支持的模型架构概览

### 7.1.1 架构枚举全景

**源码位置**：`src/llama-arch.h` (第50-150行)

```cpp
enum llm_arch {
    LLM_ARCH_UNKNOWN = 0,

    // Llama家族
    LLM_ARCH_LLAMA = 1,           // Meta Llama 1/2/3
    LLM_ARCH_LLAMA4 = 2,          // Llama 4 (未来)

    // Mistral家族
    LLM_ARCH_MISTRAL = 10,        // Mistral 7B
    LLM_ARCH_MIXTRAL = 11,        // Mixtral MoE

    // 国产模型
    LLM_ARCH_QWEN = 20,           // 通义千问
    LLM_ARCH_QWEN2 = 21,          // Qwen2
    LLM_ARCH_BAICHUAN = 22,       // 百川
    LLM_ARCH_YI = 23,             // 零一万物

    // 其他知名模型
    LLM_ARCH_GPTNEOX = 30,        // GPT-NeoX (Pythia)
    LLM_ARCH_FALCON = 31,         // Falcon
    LLM_ARCH_STARCODER = 32,      // StarCoder
    LLM_ARCH_PERSIMMON = 33,      // Persimmon
    LLM_ARCH_REFACT = 34,         // Refact
    LLM_ARCH_BERT = 35,           // BERT (嵌入模型)
    LLM_ARCH_NOMIC_BERT = 36,     // Nomic Embed

    // 更多架构...
    LLM_ARCH_COUNT
};
```

### 7.1.2 主流架构特性对比

| 架构 | 注意力变体 | 位置编码 | 归一化 | 特色 |
|-----|-----------|---------|-------|------|
| Llama2 | MHA/GQA | RoPE | RMSNorm | 经典开源 |
| Llama3 | GQA | RoPE (theta=500k) | RMSNorm | 超长上下文 |
| Mistral | SWA/GQA | RoPE | RMSNorm | 滑动窗口 |
| Mixtral | GQA + MoE | RoPE | RMSNorm | 专家混合 |
| Qwen2 | GQA | RoPE (NTK) | RMSNorm | 长文本优化 |
| Yi | MHA | RoPE | RMSNorm/LayerNorm | 双语优化 |

---

## 7.2 架构配置系统（llama_arch）

### 7.2.1 超参数结构体

**源码位置**：`src/llama-hparams.h` (第1-100行)

```cpp
struct llama_hparams {
    // 模型维度
    uint32_t n_vocab;          // 词表大小
    uint32_t n_ctx_train;      // 训练时的上下文长度
    uint32_t n_embd;           // 嵌入维度（隐藏层大小）
    uint32_t n_head;           // 注意力头数
    uint32_t n_head_kv;        // KV头数（GQA时n_head_kv < n_head）
    uint32_t n_layer;          // Transformer层数
    uint32_t n_ff;             // 前馈网络维度

    // 注意力配置
    uint32_t n_expert;         // MoE专家数量
    uint32_t n_expert_used;    // 每token激活的专家数

    // 位置编码
    float    rope_theta;       // RoPE基数（默认10000）
    float    rope_scaling_type;// RoPE扩展类型（NTK/线性）
    float    rope_scaling_factor; // 扩展因子

    // 归一化
    float    f_norm_eps;       // RMSNorm epsilon
    float    f_norm_rms_eps;   // RMSNorm专用epsilon

    // 类型标识
    llm_arch arch;             // 模型架构
    enum llama_rope_type rope_type; // RoPE类型
    enum llama_ftype ftype;    // 文件类型（量化方式）
};
```

### 7.2.2 架构配置数据

**源码位置**：`src/llama-arch.cpp` (第100-500行)

```cpp
// 各架构的默认配置
static const std::map<llm_arch, llm_arch_info> LLM_ARCH_INFO = {
    {
        LLM_ARCH_LLAMA,
        {
            .name = "llama",
            .hparams = {
                .n_ctx_train = 4096,
                .n_head_kv = 0,      // 0表示使用MHA（n_head_kv = n_head）
                .rope_theta = 10000.0f,
                .rope_scaling_type = LLAMA_ROPE_SCALING_NONE,
                .f_norm_rms_eps = 1e-6f,
            },
            .layers = {
                .attention = {
                    .q = "blk.%d.attn_q",
                    .k = "blk.%d.attn_k",
                    .v = "blk.%d.attn_v",
                    .o = "blk.%d.attn_output",
                },
                .ffn = {
                    .gate = "blk.%d.ffn_gate",
                    .up = "blk.%d.ffn_up",
                    .down = "blk.%d.ffn_down",
                },
            },
        }
    },
    {
        LLM_ARCH_MISTRAL,
        {
            .name = "mistral",
            .hparams = {
                .n_ctx_train = 8192,
                .n_head_kv = 0,      // Mistral用MHA但支持SWA
                .rope_theta = 10000.0f,
                .f_norm_rms_eps = 1e-5f,
            },
            .features = {
                .sliding_window = 4096,  // 滑动窗口注意力
            },
        }
    },
    // ... 更多架构
};
```

### 7.2.3 GQA（分组查询注意力）配置

**源码位置**：`src/llama-hparams.cpp` (第100-200行)

```cpp
// 判断使用GQA还是MHA
bool llama_hparams::use_gqa() const {
    return n_head_kv != 0 && n_head_kv < n_head;
}

// GQA计算示例：
// n_head = 32, n_head_kv = 8
// 表示每4个Query头共享1组K,V头
// 内存节省：(32-8)/32 = 75%的KV缓存减少

// 头维度计算
uint32_t llama_hparams::n_embd_head() const {
    return n_embd / n_head;  // 每个头的维度
}

uint32_t llama_hparams::n_embd_head_k() const {
    return use_gqa() ? n_embd / n_head_kv : n_embd_head();
}
```

---

## 7.3 模型张量映射

### 7.3.1 张量命名规范

**源码位置**：`src/llama-arch.cpp` (第1000-1500行)

```cpp
// 张量名称模板
struct llm_tensor_names {
    // 嵌入层
    std::string token_embd = "token_embd";  // 词嵌入

    // 输出层
    std::string output_norm = "output_norm";
    std::string output = "output";          // LM head

    // 每层Transformer
    struct {
        std::string attn_norm;   // 注意力前归一化
        std::string attn_q;      // Q投影
        std::string attn_k;      // K投影
        std::string attn_v;      // V投影
        std::string attn_o;      // 输出投影
        std::string ffn_norm;    // FFN前归一化
        std::string ffn_gate;    // SwiGLU gate
        std::string ffn_up;      // SwiGLU up
        std::string ffn_down;    // FFN down
    } layer;
};

// Llama架构的张量名
static const llm_tensor_names LLM_TENSOR_NAMES_LLAMA = {
    .token_embd = "token_embd",
    .output_norm = "output_norm",
    .output = "output",
    .layer = {
        .attn_norm = "blk.%d.attn_norm",
        .attn_q = "blk.%d.attn_q",
        .attn_k = "blk.%d.attn_k",
        .attn_v = "blk.%d.attn_v",
        .attn_o = "blk.%d.attn_o",
        .ffn_norm = "blk.%d.ffn_norm",
        .ffn_gate = "blk.%d.ffn_gate",
        .ffn_up = "blk.%d.ffn_up",
        .ffn_down = "blk.%d.ffn_down",
    }
};

// Qwen架构的张量名（注意命名差异）
static const llm_tensor_names LLM_TENSOR_NAMES_QWEN = {
    .token_embd = "token_embd",
    .output_norm = "output_norm",
    .output = "output",
    .layer = {
        .attn_norm = "blk.%d.attn_norm",
        .attn_q = "blk.%d.attn_q",
        .attn_k = "blk.%d.attn_k",
        .attn_v = "blk.%d.attn_v",
        .attn_o = "blk.%d.attn_output",  // 注意：这里叫attn_output
        .ffn_norm = "blk.%d.ffn_norm",
        .ffn_gate = "blk.%d.ffn_gate",
        .ffn_up = "blk.%d.ffn_up",
        .ffn_down = "blk.%d.ffn_down",
    }
};
```

### 7.3.2 HuggingFace到GGUF的映射

**源码位置**：`convert_hf_to_gguf.py` (第500-1000行)

```python
# HuggingFace格式和GGUF格式的命名映射
TENSOR_MAPPING = {
    # 通用映射
    "model.embed_tokens.weight": "token_embd.weight",
    "model.norm.weight": "output_norm.weight",
    "lm_head.weight": "output.weight",

    # 每层映射
    "model.layers.{i}.self_attn.q_proj.weight": "blk.{i}.attn_q.weight",
    "model.layers.{i}.self_attn.k_proj.weight": "blk.{i}.attn_k.weight",
    "model.layers.{i}.self_attn.v_proj.weight": "blk.{i}.attn_v.weight",
    "model.layers.{i}.self_attn.o_proj.weight": "blk.{i}.attn_o.weight",

    "model.layers.{i}.mlp.gate_proj.weight": "blk.{i}.ffn_gate.weight",
    "model.layers.{i}.mlp.up_proj.weight": "blk.{i}.ffn_up.weight",
    "model.layers.{i}.mlp.down_proj.weight": "blk.{i}.ffn_down.weight",

    "model.layers.{i}.input_layernorm.weight": "blk.{i}.attn_norm.weight",
    "model.layers.{i}.post_attention_layernorm.weight": "blk.{i}.ffn_norm.weight",
}

# 注意不同架构的差异
# Mistral和Llama使用相同的命名规范
# Qwen的o_proj在GGUF中叫attn_output
```

### 7.3.3 张量加载流程

**源码位置**：`src/llama-model-loader.cpp` (第500-1000行)

```cpp
// 加载权重的核心流程
void llama_model_loader::load_tensors(...) {
    // ① 遍历GGUF中的所有张量
    for (const auto & info : gguf_tensors) {
        // ② 解析张量名称，确定所属层和类型
        std::string name = info.name;
        int layer_idx = -1;
        enum llm_tensor_type type;

        // 解析如 "blk.5.attn_q.weight" -> layer=5, type=ATTN_Q
        if (sscanf(name.c_str(), "blk.%d.", &layer_idx) == 1) {
            // 提取层内张量类型
            std::string suffix = name.substr(name.find(".", 4) + 1);
            type = llama_tensor_type_from_name(suffix);
        }

        // ③ 根据架构配置验证形状
        const llama_hparams & hparams = model->hparams;
        std::vector<int64_t> expected_shape = calc_tensor_shape(
            type, layer_idx, hparams);

        GGML_ASSERT(ggml_tensor_shape_matches(info, expected_shape));

        // ④ 创建ggml_tensor并加载数据
        struct ggml_tensor * tensor = ggml_new_tensor(...);
        load_tensor_data(info, tensor->data);

        // ⑤ 添加到模型
        model->tensors[type][layer_idx] = tensor;
    }
}
```

---

## 设计中的取舍

### 为什么用字符串命名而不是固定索引？

| 方案 | 优点 | 缺点 | GGML选择 |
|-----|------|------|---------|
| 字符串命名 | 可读性好，易调试，兼容变化 | 解析开销 | **是** |
| 固定索引 | 访问快，无解析 | 难维护，扩展性差 | 否 |

**GGML的权衡**：
- 加载时一次性解析名称（运行时不重复解析）
- 使用字符串便于人类阅读和调试
- GGUF格式标准化后解析开销可接受

### 如何处理架构的细微差异？

**问题**：Llama2和Llama3非常相似，但rope_theta不同

**解决方案**：
```cpp
// 基础架构 + 参数覆盖
struct llama_hparams {
    llm_arch arch;  // 区分Llama2 vs Llama3

    // 从GGUF文件读取实际参数
    float rope_theta;  // Llama2=10000, Llama3=500000
};

// 而非创建 LLM_ARCH_LLAMA2 和 LLM_ARCH_LLAMA3
// 因为它们的计算图结构完全相同
```

---

## 动手练习

### 练习1：阅读架构配置
阅读 `src/llama-arch.cpp` 中3个不同架构的配置，比较：
1. 哪些超参数不同？
2. 张量命名有什么差异？
3. 各自有什么特色功能？

### 练习2：分析GQA配置
给定以下配置，计算内存节省：
```cpp
n_embd = 4096
n_head = 32
n_head_kv = 8
n_layer = 32
n_ctx = 4096
```

问题：
1. MHA的KV缓存总大小是多少？
2. GQA的KV缓存总大小是多少？
3. 节省了多少百分比？

### 练习3：添加新架构支持
假设要添加一个名为"MyLLM"的新架构，需要修改哪些文件？

提示：
1. `src/llama-arch.h` - 添加枚举
2. `src/llama-arch.cpp` - 添加配置
3. `convert_hf_to_gguf.py` - 添加转换映射
4. `src/llama-graph.cpp` - 可能需要调整图构建

---

## 本课小结

| 概念 | 一句话总结 |
|------|-----------|
| llm_arch | 模型架构枚举，50+种模型 |
| llama_hparams | 超参数配置，决定模型行为 |
| GQA | 分组查询注意力，节省KV缓存内存 |
| 张量映射 | HF格式到GGUF格式的命名转换 |
| 架构检测 | 从GGUF元数据自动识别架构 |

---

*本章对应源码版本：master (2026-04-07)*
