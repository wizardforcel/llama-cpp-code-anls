# 第7章 模型架构支持 —— 适配"百花齐放的模型生态"

大语言模型领域百花齐放：LLaMA、Mistral、Qwen、Yi、Falcon 等架构层出不穷。llama.cpp 需要支持这些不同的架构，同时保持代码的整洁和可维护性。本章将深入解析其模型架构支持系统。

## 学习目标

1. 理解 llama.cpp 支持的 50+ 模型架构
2. 掌握 `llm_arch` 配置系统的设计
3. 理解模型张量映射机制
4. 能添加对新架构的支持

## 生活类比

想象 llama.cpp 是一位精通百种语言的联合国翻译官。在这个比喻中，不同的模型架构就像是来自不同国家的语言方言——它们有着相似的语法结构，但存在细微的差异（比如 RoPE theta 值、归一化的 epsilon 等参数），而且词汇命名习惯也各不相同（比如有的用 `attn_o`，有的用 `attn_output`）。这些看似微小的差异如果不加处理，就可能导致整段翻译出现理解偏差。

翻译官的工作需要依赖几本关键的手册。`llm_arch` 配置就像是每种语言的语法规则手册，它记录了该架构特有的超参数、定义了层结构类型和注意力机制等核心特征。张量映射则是一本词汇对照表，将 HuggingFace 生态中的"model.layers.0.self_attn.q_proj"这样的命名，精确地翻译为 GGUF 格式中的"blk.0.attn_q"。

翻译官还维护着一个统一的语义理解层——这就是统一计算图的角色。无论输入的来源是哪种语言方言，经过处理之后都会转化为统一的张量运算概念，就像无论说哪种语言，"苹果"这个概念在所有语言中指向的都是同一种水果。而当社区推出一个新的模型架构时，就相当于翻译官在学习一门新语言——一旦掌握了这门语言的语法规则（配置），就能顺畅地进行翻译（加载并运行模型）。

就像优秀的翻译官需要先掌握语言规则再开口翻译一样，llama.cpp 也需要先理解模型的架构配置，才能正确地加载权重并构建推理计算图。

---

## 7.1 支持的模型架构概览

### 7.1.1 架构枚举全景

**源码位置**：`src/llama-arch.h`（第 50-150 行）

```cpp
// 模型架构枚举
enum llm_arch {
    LLM_ARCH_UNKNOWN = 0,

    // ===== Llama 家族 =====
    LLM_ARCH_LLAMA = 1,           // Meta Llama 1/2/3
    LLM_ARCH_LLAMA4 = 2,          // Llama 4 (未来)

    // ===== Mistral 家族 =====
    LLM_ARCH_MISTRAL = 10,        // Mistral 7B
    LLM_ARCH_MIXTRAL = 11,        // Mixtral MoE
    LLM_ARCH_CODESTRAL = 12,      // Codestral

    // ===== 国产模型 =====
    LLM_ARCH_QWEN = 20,           // 通义千问 1.0
    LLM_ARCH_QWEN2 = 21,          // Qwen2
    LLM_ARCH_BAICHUAN = 22,       // 百川
    LLM_ARCH_YI = 23,             // 零一万物 Yi
    LLM_ARCH_DEEPSEEK = 24,       // DeepSeek

    // ===== 其他知名模型 =====
    LLM_ARCH_GPTNEOX = 30,        // GPT-NeoX (Pythia)
    LLM_ARCH_FALCON = 31,         // Falcon
    LLM_ARCH_STARCODER = 32,      // StarCoder
    LLM_ARCH_PERSIMMON = 33,      // Persimmon
    LLM_ARCH_REFACT = 34,         // Refact
    LLM_ARCH_BERT = 35,           // BERT (嵌入模型)
    LLM_ARCH_NOMIC_BERT = 36,     // Nomic Embed

    // ... 更多架构持续添加中
    LLM_ARCH_COUNT
};

这段代码定义了llama.cpp支持的模型架构枚举，包含50+种不同的大语言模型架构。从Llama、Mistral到Qwen、DeepSeek等国产模型，每种架构都有独特的注意力变体、位置编码和归一化方式。
```

### 7.1.2 主流架构特性对比

| 架构 | 注意力变体 | 位置编码 | 归一化 | 特色 |
|-----|-----------|---------|-------|------|
| Llama2 | MHA/GQA | RoPE | RMSNorm | 经典开源 |
| Llama3 | GQA | RoPE (theta=500k) | RMSNorm | 超长上下文支持 |
| Mistral | SWA/GQA | RoPE | RMSNorm | 滑动窗口注意力 |
| Mixtral | GQA + MoE | RoPE | RMSNorm | 专家混合模型 |
| Qwen2 | GQA | RoPE (NTK) | RMSNorm | 长文本优化 |
| Yi | MHA | RoPE | RMSNorm/LayerNorm | 双语优化 |

**关键概念解释**：

- **MHA** = Multi-Head Attention（多头注意力）
- **GQA** = Grouped Query Attention（分组查询注意力）
- **SWA** = Sliding Window Attention（滑动窗口注意力）
- **MoE** = Mixture of Experts（专家混合）
- **RoPE** = Rotary Position Embedding（旋转位置编码）
- **RMSNorm** = Root Mean Square Layer Normalization

---

## 7.2 架构配置系统（llm_arch）

### 7.2.1 超参数结构体

**源码位置**：`src/llama-hparams.h`（第 1-100 行）

```cpp
struct llama_hparams {
    // ===== 模型维度 =====
    uint32_t n_vocab;          // 词表大小（如 32000、50000、128256）
    uint32_t n_ctx_train;      // 训练时的上下文长度
    uint32_t n_embd;           // 嵌入维度（隐藏层大小，如 4096）
    uint32_t n_head;           // 注意力头数（如 32）
    uint32_t n_head_kv;        // KV 头数（GQA 时 n_head_kv < n_head）
    uint32_t n_layer;          // Transformer 层数（如 32）
    uint32_t n_ff;             // 前馈网络维度（通常是 2.67 * n_embd）

    // ===== MoE 配置 =====
    uint32_t n_expert;         // 专家数量（MoE 模型）
    uint32_t n_expert_used;    // 每 token 激活的专家数

    // ===== 位置编码 =====
    float    rope_theta;       // RoPE 基数（Llama2=10000，Llama3=500000）
    float    rope_scaling_type;// RoPE 扩展类型（NTK/线性/动态）
    float    rope_scaling_factor; // 扩展因子（如 2.0 表示 2 倍扩展）
    float    rope_freq_base;   // RoPE 频率基
    float    rope_freq_scale;  // RoPE 频率缩放

    // ===== 归一化 =====
    float    f_norm_eps;       // LayerNorm epsilon
    float    f_norm_rms_eps;   // RMSNorm 专用 epsilon

    // ===== 类型标识 =====
    llm_arch arch;             // 模型架构枚举
    enum llama_rope_type rope_type; // RoPE 类型
    enum llama_ftype ftype;    // 文件类型（量化方式）
    
    // ===== 注意力配置 =====
    uint32_t n_embd_head_k;    // 每个 K 头的维度
    uint32_t n_embd_head_v;    // 每个 V 头的维度
    uint32_t n_rep;            // GQA 重复次数
};

这段代码定义了llama模型的超参数结构，包含模型维度(n_vocab/n_embd/n_layer等)、MoE配置、位置编码参数(rope_theta等)、归一化参数及注意力配置。这些参数从GGUF文件读取，决定模型的结构和行为。
```

### 7.2.2 架构配置数据

**源码位置**：`src/llama-arch.cpp`（第 100-500 行）

```cpp
// 各架构的默认配置和元数据
static const std::map<llm_arch, llm_arch_info> LLM_ARCH_INFO = {
    {
        LLM_ARCH_LLAMA,
        {
            .name = "llama",
            .hparams_defaults = {
                .n_ctx_train = 4096,
                .n_head_kv = 0,           // 0 表示使用 MHA
                .rope_theta = 10000.0f,
                .rope_scaling_type = LLAMA_ROPE_SCALING_NONE,
                .f_norm_rms_eps = 1e-6f,
            },
            // 张量名称模板
            .tensor_names = {
                .token_embd = "token_embd",
                .output_norm = "output_norm",
                .output = "output",
                .layers = {
                    .attention = {
                        .q = "blk.%d.attn_q",
                        .k = "blk.%d.attn_k",
                        .v = "blk.%d.attn_v",
                        .o = "blk.%d.attn_o",
                    },
                    .ffn = {
                        .gate = "blk.%d.ffn_gate",
                        .up = "blk.%d.ffn_up",
                        .down = "blk.%d.ffn_down",
                    },
                },
            },
        }
    },
    {
        LLM_ARCH_MISTRAL,
        {
            .name = "mistral",
            .hparams_defaults = {
                .n_ctx_train = 8192,
                .n_head_kv = 0,           // Mistral 用 MHA 但支持 SWA
                .rope_theta = 10000.0f,
                .f_norm_rms_eps = 1e-5f,
            },
            .features = {
                .sliding_window = 4096,   // 滑动窗口注意力大小
                .gqa = false,
            },
        }
    },
    {
        LLM_ARCH_QWEN2,
        {
            .name = "qwen2",
            .hparams_defaults = {
                .n_ctx_train = 32768,     // Qwen2 支持长上下文
                .n_head_kv = 0,
                .rope_theta = 1000000.0f, // 更大的 RoPE theta
                .f_norm_rms_eps = 1e-6f,
            },
            .features = {
                .use_alibi = false,
                .use_ntk_rope = true,     // 使用 NTK 扩展
            },
        }
    },
    // ... 更多架构
};

这段代码定义了不同模型架构的配置信息映射表。每种架构包含默认超参数(如上下文长度、RoPE theta值)、张量命名模板(如blk.0.attn_q)及特性标志(如SWA、GQA支持)，使llama.cpp能适配各种模型变体。
```

### 7.2.3 GQA（分组查询注意力）配置

**源码位置**：`src/llama-hparams.cpp`（第 100-200 行）

```cpp
// 判断是否使用 GQA
bool llama_hparams::use_gqa() const {
    // n_head_kv = 0 表示使用 MHA（n_head_kv = n_head）
    // n_head_kv < n_head 表示使用 GQA
    return n_head_kv != 0 && n_head_kv < n_head;
}

// GQA 计算示例：
// n_head = 32, n_head_kv = 8
// 表示每 4 个 Query 头共享 1 组 K,V 头
// 内存节省：(32-8)/32 = 75% 的 KV 缓存减少

// 头维度计算
uint32_t llama_hparams::n_embd_head() const {
    return n_embd / n_head;  // 每个头的维度
}

uint32_t llama_hparams::n_embd_head_k() const {
    return use_gqa() ? n_embd / n_head_kv : n_embd_head();
}

uint32_t llama_hparams::n_rep() const {
    // GQA 重复次数
    return use_gqa() ? n_head / n_head_kv : 1;
}

这段代码实现了分组查询注意力(GQA)的判断和计算。当n_head_kv < n_head时启用GQA，多个Query头共享一组K/V头，可显著减少KV缓存内存占用。n_rep计算每个KV头需要服务的Query头数量。
```

**GQA 内存节省计算**：

```
MHA（多头注意力）：
  KV 缓存 = 2 * batch * n_ctx * n_head * head_dim * sizeof(dtype)
  
GQA（n_head_kv = n_head / 4）：
  KV 缓存 = 2 * batch * n_ctx * n_head_kv * head_dim * sizeof(dtype)
         = MHA 的 1/4
         
Llama2 70B 示例：
  n_head = 64, n_head_kv = 8（8 倍压缩）
  对于 4096 上下文，batch=1，FP16：
  MHA KV 缓存 = 2 * 1 * 4096 * 64 * 128 * 2 = 134 MB
  GQA KV 缓存 = 2 * 1 * 4096 * 8 * 128 * 2 = 16.8 MB
```

---

## 7.3 模型张量映射

### 7.3.1 张量命名规范

**源码位置**：`src/llama-arch.cpp`（第 1000-1500 行）

```cpp
// 张量名称模板结构
struct llm_tensor_names {
    // 嵌入层
    std::string token_embd = "token_embd";  // 词嵌入
    std::string pos_embd = "";              // 位置嵌入（如有）

    // 输出层
    std::string output_norm = "output_norm";
    std::string output = "output";          // LM head（语言模型头）

    // 每层 Transformer 的张量
    struct {
        std::string attn_norm;       // 注意力前归一化
        std::string attn_q;          // Q 投影
        std::string attn_k;          // K 投影
        std::string attn_v;          // V 投影
        std::string attn_o;          // 输出投影
        std::string attn_q_norm;     // Q 归一化（如使用）
        std::string attn_k_norm;     // K 归一化（如使用）
        std::string ffn_norm;        // FFN 前归一化
        std::string ffn_gate;        // SwiGLU gate 投影
        std::string ffn_up;          // SwiGLU up 投影
        std::string ffn_down;        // FFN down 投影
    } layer;
};

这段代码定义了模型张量命名模板结构，统一不同架构的张量命名方式。包括嵌入层、输出层及每层Transformer的注意力(Q/K/V/O投影)和FFN(gate/up/down)张量名称，支持不同模型的权重映射转换。

```cpp
// Llama 架构的张量名
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

// Qwen 架构的张量名（注意命名差异）
static const llm_tensor_names LLM_TENSOR_NAMES_QWEN = {
    .token_embd = "token_embd",
    .output_norm = "output_norm",
    .output = "output",
    .layer = {
        .attn_norm = "blk.%d.attn_norm",
        .attn_q = "blk.%d.attn_q",
        .attn_k = "blk.%d.attn_k",
        .attn_v = "blk.%d.attn_v",
        .attn_o = "blk.%d.attn_output",  // 注意：这里叫 attn_output
        .ffn_norm = "blk.%d.ffn_norm",
        .ffn_gate = "blk.%d.ffn_gate",
        .ffn_up = "blk.%d.ffn_up",
        .ffn_down = "blk.%d.ffn_down",
    }
};
```

这段代码展示了Llama和Qwen架构的具体张量命名配置。两者主要差异在注意力输出投影的命名(Llama用attn_o，Qwen用attn_output)。%d占位符表示层索引，用于生成blk.0.attn_q、blk.1.attn_q等各层张量名称。

### 7.3.2 HuggingFace 到 GGUF 的映射

**源码位置**：`convert_hf_to_gguf.py`（第 500-1000 行）

```python
# HuggingFace 格式和 GGUF 格式的命名映射
TENSOR_MAPPING = {
    # ===== 通用映射 =====
    "model.embed_tokens.weight": "token_embd.weight",
    "model.norm.weight": "output_norm.weight",
    "lm_head.weight": "output.weight",

    # ===== 每层映射 =====
    # 注意力权重
    "model.layers.{i}.self_attn.q_proj.weight": "blk.{i}.attn_q.weight",
    "model.layers.{i}.self_attn.k_proj.weight": "blk.{i}.attn_k.weight",
    "model.layers.{i}.self_attn.v_proj.weight": "blk.{i}.attn_v.weight",
    "model.layers.{i}.self_attn.o_proj.weight": "blk.{i}.attn_o.weight",

    # FFN 权重（SwiGLU 结构）
    "model.layers.{i}.mlp.gate_proj.weight": "blk.{i}.ffn_gate.weight",
    "model.layers.{i}.mlp.up_proj.weight": "blk.{i}.ffn_up.weight",
    "model.layers.{i}.mlp.down_proj.weight": "blk.{i}.ffn_down.weight",

    # 归一化权重
    "model.layers.{i}.input_layernorm.weight": "blk.{i}.attn_norm.weight",
    "model.layers.{i}.post_attention_layernorm.weight": "blk.{i}.ffn_norm.weight",
}

# 注意不同架构的差异
# - Mistral 和 Llama 使用相同的命名规范
# - Qwen 的 o_proj 在 GGUF 中叫 attn_output
# - 某些模型可能有额外的偏置项
```

### 7.3.3 张量加载流程

**源码位置**：`src/llama-model-loader.cpp`（第 500-1000 行）

```cpp
// 加载权重的核心流程
void llama_model_loader::load_tensors(llama_model * model) {
    // ① 遍历 GGUF 中的所有张量
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
        } else {
            // 全局张量（如 token_embd）
            type = llama_tensor_type_from_name(name);
        }

        // ③ 根据架构配置验证形状
        const llama_hparams & hparams = model->hparams;
        std::vector<int64_t> expected_shape = calc_tensor_shape(
            type, layer_idx, hparams);

        GGML_ASSERT(ggml_tensor_shape_matches(info, expected_shape));

        // ④ 创建 ggml_tensor 并加载数据
        struct ggml_tensor * tensor = ggml_new_tensor(...);
        load_tensor_data(info, tensor->data);

        // ⑤ 添加到模型张量映射表
        model->tensors[type][layer_idx] = tensor;
    }
}

// 根据张量类型和超参数计算期望形状
std::vector<int64_t> calc_tensor_shape(
    llm_tensor_type type,
    int layer_idx,
    const llama_hparams & hparams) {
    
    switch (type) {
        case LLM_TENSOR_TOKEN_EMBD:
            return {hparams.n_embd, hparams.n_vocab};
        case LLM_TENSOR_ATTN_Q:
            return {hparams.n_embd, hparams.n_embd};
        case LLM_TENSOR_ATTN_K:
            return {hparams.n_embd_head_k * hparams.n_head_kv, hparams.n_embd};
        case LLM_TENSOR_ATTN_V:
            return {hparams.n_embd_head_v * hparams.n_head_kv, hparams.n_embd};
        // ... 更多类型
    }
}
```

---

## 7.4 设计中的取舍

### 为什么用字符串命名而不是固定索引？

| 方案 | 优点 | 缺点 | GGML 选择 |
|-----|------|------|-----------|
| 字符串命名 | 可读性好，易调试，兼容变化 | 解析开销 | ✅ 是 |
| 固定索引 | 访问快，无解析 | 难维护，扩展性差 | ❌ |

**GGML 的权衡**：

1. **加载时一次性解析**：运行时只解析一次，之后使用指针访问
2. **便于调试**：错误信息可以显示具体张量名称
3. **兼容性好**：GGUF 格式标准化后，解析开销可接受
4. **易于扩展**：新架构只需添加配置，无需修改核心代码

### 如何处理架构的细微差异？

**问题**：Llama2 和 Llama3 非常相似，但 `rope_theta` 不同（10000 vs 500000）

**解决方案**：

```cpp
// 基础架构 + 参数覆盖
struct llama_hparams {
    llm_arch arch;       // 区分 Llama2 vs Llama3
    float rope_theta;    // 从 GGUF 文件读取实际参数
    // Llama2=10000, Llama3=500000
};

// 而非创建 LLM_ARCH_LLAMA2 和 LLM_ARCH_LLAMA3
// 因为它们的计算图结构完全相同！
```

**架构分类策略**：

1. **同一架构，不同版本**：使用 `arch` 枚举 + 超参数（如 Llama2/Llama3）
2. **不同架构，相似结构**：同一枚举，不同配置（如不同大小的 Mistral）
3. **完全不同的架构**：新建 `arch` 枚举值（如 BERT vs LLaMA）

---

## 7.5 动手练习

### 练习 1：阅读架构配置

阅读 `src/llama-arch.cpp` 中 3 个不同架构的配置，比较：

1. 哪些超参数不同？（n_ctx_train、rope_theta、f_norm_rms_eps 等）
2. 张量命名有什么差异？（如 attn_o vs attn_output）
3. 各自有什么特色功能？（SWA、GQA、MoE 等）

### 练习 2：分析 GQA 配置

给定以下配置，计算内存节省：

```cpp
n_embd = 4096
n_head = 32
n_head_kv = 8
n_layer = 32
n_ctx = 4096
```

问题：

1. **MHA 的 KV 缓存总大小是多少？**
   - 提示：2 * n_layer * n_ctx * n_head * (n_embd/n_head) * sizeof(FP16)

2. **GQA 的 KV 缓存总大小是多少？**
   - 提示：2 * n_layer * n_ctx * n_head_kv * (n_embd/n_head) * sizeof(FP16)

3. **节省了多少百分比？**
   - 答案：75%

### 练习 3：添加新架构支持

假设要添加一个名为 "MyLLM" 的新架构，需要修改哪些文件？

**步骤**：

1. **`src/llama-arch.h`** - 添加枚举值
   ```cpp
   LLM_ARCH_MYLLM = 100,
   ```

2. **`src/llama-arch.cpp`** - 添加配置
   ```cpp
   {
       LLM_ARCH_MYLLM,
       {
           .name = "myllm",
           .hparams_defaults = {...},
           .tensor_names = {...},
       }
   }
   ```

3. **`convert_hf_to_gguf.py`** - 添加转换映射
   ```python
   # 定义 HF 到 GGUF 的张量名映射
   ```

4. **`src/llama-graph.cpp`** - 如有特殊算子，添加图构建逻辑

---

## 7.6 本章小结

本章介绍了 llama.cpp 如何支持多种模型架构。`llm_arch` 枚举定义了支持的架构类型，目前支持 50 多种不同的模型架构。`llama_hparams` 结构存储超参数配置，这些参数决定了模型的结构和行为。GQA（分组查询注意力）是一种优化技术，通过设置 n_head_kv 小于 n_head 来节省 KV 缓存空间。张量映射机制负责将 HuggingFace 格式的张量名称转换为 GGUF 格式的命名。架构检测功能能够从 GGUF 元数据中自动识别模型架构。`n_ctx_train` 表示训练时的上下文长度，这个参数决定了 RoPE 的配置方式。`rope_theta` 是 RoPE 的基数参数，影响模型的长上下文外推能力。

本章我们一起学习了以下概念：

| 概念 | 解释 |
|------|------|
| llm_arch 枚举 | 定义了50+种模型架构类型，从 Llama 到 Qwen、DeepSeek 等 |
| llama_hparams 超参数 | 结构体存储模型维度、注意力头数、RoPE theta 等配置参数 |
| GQA 分组查询注意力 | 多个 Query 头共享一组 K/V 头，可节省 75% 的 KV 缓存 |
| 张量命名映射 | 将 HuggingFace 格式的张量名转换为 GGUF 统一的命名规范 |
| 架构配置信息表 | LLM_ARCH_INFO 映射表，包含每种架构的默认参数和命名模板 |
| RoPE 参数体系 | rope_theta、scaling_type、freq_base 等参数控制位置编码行为 |

下一章中，我们将学习 GGUF 文件格式——理解模型文件的内部结构和元数据系统。
