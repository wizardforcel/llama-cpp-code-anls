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
├── enum llm_arch            # 架构枚举（100+种模型）
├── enum llm_kv              # GGUF元数据键枚举
├── enum llm_tensor          # 张量类型枚举
├── enum llm_tensor_layer    # 层张量类型枚举
├── struct LLM_KV            # 元数据键构建器
├── struct LLM_TN            # 张量名称构建器
├── struct llm_tensor_info   # 张量信息结构
├── struct llm_arch_info     # 架构信息
└── llm_arch_from_string()   # 字符串转枚举

src/llama-arch.cpp
├── 架构配置数据（第100-1000行）
│   └── LLM_ARCH_INFO        # 各架构的层数、维度等配置
├── 张量名称映射（第1000-2000行）
│   └── LLM_TENSOR_NAMES     # 各架构的张量命名规则
└── 架构检测逻辑（第2000-3000行）

src/llama-hparams.h/cpp
├── enum llama_expert_gating_func_type  # 专家门控函数类型
├── enum llama_swa_type                 # 滑动窗口注意力类型
├── struct llama_hparams_posnet         # PosNet参数（WavTokenizer）
├── struct llama_hparams_convnext       # ConvNeXt参数
├── struct llama_hparams                # 主超参数结构体
│   ├── n_embd_head_k_swa/n_embd_head_v_swa  # SWA头维度
│   ├── n_rot_full/n_rot_swa               # RoPE维度
│   ├── n_embd_head_k_mla_impl            # MLA实现维度
│   ├── expert_gating_func                # MoE门控函数
│   ├── swa_type/swa_layers               # SWA配置
│   └── ssm_d_conv/ssm_d_state            # 状态空间模型参数
└── 加载和验证逻辑

src/llama-model-loader.cpp
├── 张量映射逻辑（第1-1000行）
└── 权重加载实现（第1000-2000行）
```

---

## 7.1 支持的模型架构概览

### 7.1.1 架构枚举全景

**源码位置**：`src/llama-arch.h` (第13-140行)

```cpp
enum llm_arch {
    LLM_ARCH_LLAMA,           // Meta Llama 1/2/3
    LLM_ARCH_LLAMA4,          // Llama 4
    LLM_ARCH_DECI,            // Deci
    LLM_ARCH_FALCON,          // Falcon
    LLM_ARCH_BAICHUAN,        // 百川
    LLM_ARCH_GROK,            // Grok
    LLM_ARCH_GPT2,            // GPT-2
    LLM_ARCH_GPTJ,            // GPT-J
    LLM_ARCH_GPTNEOX,         // GPT-NeoX (Pythia)
    LLM_ARCH_MPT,             // MPT
    LLM_ARCH_STARCODER,       // StarCoder
    LLM_ARCH_REFACT,          // Refact
    LLM_ARCH_BERT,            // BERT
    LLM_ARCH_MODERN_BERT,     // ModernBERT
    LLM_ARCH_NOMIC_BERT,      // Nomic Embed
    LLM_ARCH_NOMIC_BERT_MOE,  // Nomic BERT MoE
    LLM_ARCH_NEO_BERT,        // NeoBERT
    LLM_ARCH_JINA_BERT_V2,    // Jina Embeddings v2
    LLM_ARCH_JINA_BERT_V3,    // Jina Embeddings v3
    LLM_ARCH_EUROBERT,        // EuroBERT
    LLM_ARCH_BLOOM,           // BLOOM
    LLM_ARCH_STABLELM,        // StableLM
    LLM_ARCH_QWEN,            // 通义千问
    LLM_ARCH_QWEN2,           // Qwen2
    LLM_ARCH_QWEN2MOE,        // Qwen2 MoE
    LLM_ARCH_QWEN2VL,         // Qwen2 VL
    LLM_ARCH_QWEN3,           // Qwen3
    LLM_ARCH_QWEN3MOE,        // Qwen3 MoE
    LLM_ARCH_QWEN3NEXT,       // Qwen3 Next
    LLM_ARCH_QWEN3VL,         // Qwen3 VL
    LLM_ARCH_QWEN3VLMOE,      // Qwen3 VL MoE
    LLM_ARCH_QWEN35,          // Qwen 3.5
    LLM_ARCH_QWEN35MOE,       // Qwen 3.5 MoE
    LLM_ARCH_PHI2,            // Phi-2
    LLM_ARCH_PHI3,            // Phi-3
    LLM_ARCH_PHIMOE,          // Phi MoE
    LLM_ARCH_PLAMO,           // PLaMo
    LLM_ARCH_PLAMO2,          // PLaMo-2
    LLM_ARCH_PLAMO3,          // PLaMo-3
    LLM_ARCH_CODESHELL,       // CodeShell
    LLM_ARCH_ORION,           // Orion
    LLM_ARCH_INTERNLM2,       // InternLM2
    LLM_ARCH_MINICPM,         // MiniCPM
    LLM_ARCH_MINICPM3,        // MiniCPM3
    LLM_ARCH_GEMMA,           // Gemma
    LLM_ARCH_GEMMA2,          // Gemma 2
    LLM_ARCH_GEMMA3,          // Gemma 3
    LLM_ARCH_GEMMA3N,         // Gemma 3N
    LLM_ARCH_GEMMA4,          // Gemma 4
    LLM_ARCH_GEMMA_EMBEDDING, // Gemma Embedding
    LLM_ARCH_STARCODER2,      // StarCoder2
    LLM_ARCH_MAMBA,           // Mamba
    LLM_ARCH_MAMBA2,          // Mamba2
    LLM_ARCH_JAMBA,           // Jamba
    LLM_ARCH_FALCON_H1,       // Falcon H1
    LLM_ARCH_XVERSE,          // Xverse
    LLM_ARCH_COMMAND_R,       // Command R
    LLM_ARCH_COHERE2,         // Cohere2
    LLM_ARCH_DBRX,            // DBRX
    LLM_ARCH_OLMO,            // OLMo
    LLM_ARCH_OLMO2,           // OLMo 2
    LLM_ARCH_OLMOE,           // OLMoE
    LLM_ARCH_OPENELM,         // OpenELM
    LLM_ARCH_ARCTIC,          // Arctic
    LLM_ARCH_DEEPSEEK,        // DeepSeek
    LLM_ARCH_DEEPSEEK2,       // DeepSeek-V2
    LLM_ARCH_DEEPSEEK2OCR,    // DeepSeek OCR
    LLM_ARCH_CHATGLM,         // ChatGLM
    LLM_ARCH_GLM4,            // GLM-4
    LLM_ARCH_GLM4_MOE,        // GLM-4 MoE
    LLM_ARCH_GLM_DSA,         // GLM-DSA
    LLM_ARCH_BITNET,          // BitNet
    LLM_ARCH_T5,              // T5
    LLM_ARCH_T5ENCODER,       // T5 Encoder
    LLM_ARCH_JAIS,            // Jais
    LLM_ARCH_JAIS2,           // Jais2
    LLM_ARCH_NEMOTRON,        // Nemotron
    LLM_ARCH_NEMOTRON_H,      // Nemotron-H
    LLM_ARCH_NEMOTRON_H_MOE,  // Nemotron-H MoE
    LLM_ARCH_EXAONE,          // EXAONE
    LLM_ARCH_EXAONE4,         // EXAONE 4
    LLM_ARCH_EXAONE_MOE,      // EXAONE MoE
    LLM_ARCH_RWKV6,           // RWKV-v6
    LLM_ARCH_RWKV6QWEN2,      // RWKV6-Qwen2
    LLM_ARCH_RWKV7,           // RWKV-v7
    LLM_ARCH_ARWKV7,          // ARWKV-v7
    LLM_ARCH_GRANITE,         // Granite
    LLM_ARCH_GRANITE_MOE,     // Granite MoE
    LLM_ARCH_GRANITE_HYBRID,  // Granite Hybrid
    LLM_ARCH_CHAMELEON,       // Chameleon
    LLM_ARCH_WAVTOKENIZER_DEC,// WavTokenizer
    LLM_ARCH_PLM,             // PLM
    LLM_ARCH_BAILINGMOE,      // Bailing MoE
    LLM_ARCH_BAILINGMOE2,     // Bailing MoE 2
    LLM_ARCH_DOTS1,           // DOTS-1
    LLM_ARCH_ARCEE,           // Arcee
    LLM_ARCH_AFMOE,           // AF MoE
    LLM_ARCH_ERNIE4_5,        // Ernie 4.5
    LLM_ARCH_ERNIE4_5_MOE,    // Ernie 4.5 MoE
    LLM_ARCH_HUNYUAN_MOE,     // Hunyuan MoE
    LLM_ARCH_HUNYUAN_DENSE,   // Hunyuan Dense
    LLM_ARCH_SMOLLM3,         // SmolLM3
    LLM_ARCH_OPENAI_MOE,      // OpenAI MoE
    LLM_ARCH_LFM2,            // LFM 2
    LLM_ARCH_LFM2MOE,         // LFM 2 MoE
    LLM_ARCH_DREAM,           // Dream
    LLM_ARCH_SMALLTHINKER,    // SmallThinker
    LLM_ARCH_LLADA,           // LLaDA
    LLM_ARCH_LLADA_MOE,       // LLaDA MoE
    LLM_ARCH_SEED_OSS,        // Seed OSS
    LLM_ARCH_GROVEMOE,        // Grove MoE
    LLM_ARCH_APERTUS,         // Apertus
    LLM_ARCH_MINIMAX_M2,      // MiniMax-M2
    LLM_ARCH_COGVLM,          // CogVLM
    LLM_ARCH_RND1,            // RND-1
    LLM_ARCH_PANGU_EMBED,     // Pangu Embedding
    LLM_ARCH_MISTRAL3,        // Mistral 3
    LLM_ARCH_MISTRAL4,        // Mistral 4
    LLM_ARCH_PADDLEOCR,       // PaddleOCR
    LLM_ARCH_MIMO2,           // MIMO-2
    LLM_ARCH_STEP35,          // Step-3.5
    LLM_ARCH_LLAMA_EMBED,     // Llama Embedding
    LLM_ARCH_KIMI_LINEAR,     // Kimi Linear
    LLM_ARCH_MAINCODER,       // MainCoder
    LLM_ARCH_UNKNOWN,
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

**源码位置**：`src/llama-hparams.h` (第36-165行)

```cpp
struct llama_hparams {
    // 模型维度
    uint32_t n_vocab;          // 词表大小
    uint32_t n_ctx_train;      // 训练时的上下文长度
    uint32_t n_embd;           // 嵌入维度（隐藏层大小）
    uint32_t n_layer;          // Transformer层数
    uint32_t n_expert;         // MoE专家数量
    uint32_t n_expert_used;    // 每token激活的专家数

    // 注意力头配置（每层可独立配置）
    std::array<uint32_t, LLAMA_MAX_LAYERS> n_head_arr;    // 每层的Q头数
    std::array<uint32_t, LLAMA_MAX_LAYERS> n_head_kv_arr; // 每层的KV头数
    uint32_t n_embd_head_k_full;  // Full Attention的K头维度
    uint32_t n_embd_head_v_full;  // Full Attention的V头维度
    uint32_t n_embd_head_k_swa;   // SWA的K头维度
    uint32_t n_embd_head_v_swa;   // SWA的V头维度
    uint32_t n_embd_head_k_mla_impl; // MLA实现的K头维度
    uint32_t n_embd_head_v_mla_impl; // MLA实现的V头维度
    uint32_t n_rot_full;       // Full Attention的RoPE维度
    uint32_t n_rot_swa;        // SWA的RoPE维度

    // 前馈网络配置
    std::array<uint32_t, LLAMA_MAX_LAYERS> n_ff_arr;      // 每层FFN维度
    uint32_t n_ff_exp;         // 专家FFN扩展维度
    uint32_t n_ff_shexp;       // 共享专家FFN维度
    uint32_t n_ff_chexp;       // 块共享专家FFN维度
    uint32_t n_expert_shared;  // 共享专家数量
    uint32_t n_expert_groups;  // 专家组数量
    uint32_t n_group_used;     // 每组使用的专家数
    uint32_t n_group_experts;  // 每组的专家数

    // MoE门控配置
    enum llama_expert_gating_func_type {
        LLAMA_EXPERT_GATING_FUNC_TYPE_NONE = 0,
        LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX = 1,
        LLAMA_EXPERT_GATING_FUNC_TYPE_SIGMOID = 2,
        LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX_WEIGHT = 3,
    };
    uint32_t expert_gating_func;   // MoE门控函数类型
    float    expert_group_scale;   // 专家组缩放因子
    float    expert_weights_scale; // 专家权重缩放
    bool     expert_weights_norm;  // 是否归一化专家权重
    uint32_t moe_every_n_layers;   // MoE层间隔
    uint32_t moe_latent_size;      // MoE潜在维度

    // 位置编码
    float    rope_freq_base_train;     // RoPE基数（训练）
    float    rope_freq_base_train_swa; // SWA的RoPE基数
    float    rope_freq_scale_train;    // RoPE缩放因子（训练）
    float    rope_freq_scale_train_swa;// SWA的RoPE缩放因子
    float    rope_attn_factor;         // RoPE注意力因子
    float    yarn_ext_factor;          // YaRN扩展因子
    float    yarn_attn_factor;         // YaRN注意力因子
    float    yarn_beta_fast;           // YaRN快速beta
    float    yarn_beta_slow;           // YaRN慢速beta
    uint32_t n_ctx_orig_yarn;          // YaRN原始上下文大小
    float    rope_yarn_log_mul;        // YaRN对数乘数
    std::array<int, 4> rope_sections;  // RoPE分段配置

    // 滑动窗口注意力(SWA)
    enum llama_swa_type {
        LLAMA_SWA_TYPE_NONE = 0,
        LLAMA_SWA_TYPE_STANDARD = 1,
        LLAMA_SWA_TYPE_CHUNKED = 2,
        LLAMA_SWA_TYPE_SYMMETRIC = 3,
    };
    llama_swa_type swa_type;   // SWA类型
    uint32_t n_swa;            // SWA窗口大小
    std::array<uint32_t, LLAMA_MAX_LAYERS> swa_layers; // 每层是否使用SWA

    // 归一化配置
    float f_norm_eps;          // LayerNorm epsilon
    float f_norm_rms_eps;      // RMSNorm epsilon
    float f_norm_group_eps;    // GroupNorm epsilon
    bool  swin_norm;           // 是否使用Swin归一化
    bool  use_par_res;         // 是否使用并行残差连接

    // 数值稳定性
    float f_attn_logit_softcapping;   // 注意力logit软裁剪
    float f_router_logit_softcapping; // 路由logit软裁剪
    float f_final_logit_softcapping;  // 最终logit软裁剪

    // 状态空间模型(SSM)参数
    uint32_t ssm_d_conv;       // SSM卷积维度
    uint32_t ssm_d_inner;      // SSM内部维度
    uint32_t ssm_d_state;      // SSM状态维度
    uint32_t ssm_dt_rank;      // SSM dt秩
    uint32_t ssm_n_group;      // SSM组数

    // RWKV参数
    uint32_t rescale_every_n_layers; // 每隔多少层重缩放
    uint32_t time_mix_extra_dim;     // time mix额外维度
    uint32_t time_decay_extra_dim;   // time decay额外维度
    uint32_t wkv_head_size;          // WKV头大小
    uint32_t token_shift_count;      // token移位计数
    uint32_t n_lora_decay;           // LoRA decay维度
    uint32_t n_lora_iclr;            // LoRA iclr维度
    uint32_t n_lora_value_res_mix;   // LoRA value残差混合维度
    uint32_t n_lora_gate;            // LoRA gate维度

    // 特殊模型结构
    uint32_t n_layer_dense_lead;     // 前导稠密层数
    uint32_t n_rel_attn_bkts;        // 相对注意力桶数
    uint32_t n_shortconv_l_cache;    // ShortConv缓存长度
    uint32_t nextn_predict_layers;   // NTP预测层数
    uint32_t n_embd_head_kda;        // Kimi Linear KDA维度
    struct llama_hparams_posnet posnet;       // WavTokenizer PosNet参数
    struct llama_hparams_convnext convnext;   // WavTokenizer ConvNeXt参数

    // 标志
    bool vocab_only;           // 仅加载词表
    bool no_alloc;             // 不分配内存
    bool rope_finetuned;       // RoPE是否微调过
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

## 7.6 模型架构实现 (src/models/)

除了架构定义和配置，llama.cpp在 `src/models/` 目录下包含各模型架构的**具体计算图实现**。这些文件负责构建特定模型的前向传播计算图。

### 7.6.1 模型实现文件列表

**源码位置**：`src/models/`

```
src/models/
├── llama.cpp    / llama4.cpp    # Llama系列 (Llama 1/2/3/4)
├── qwen.cpp     / qwen2.cpp      # Qwen系列
├── qwen3.cpp
├── gemma.cpp    / gemma2.cpp     # Gemma系列
├── gemma3.cpp   / gemma4.cpp
├── deepseek.cpp / deepseek2.cpp  # DeepSeek系列
├── falcon.cpp                   # Falcon
├── mpt.cpp                      # MPT
├── gpt2.cpp                     # GPT-2
├── gptneox.cpp                  # GPT-NeoX / Pythia
├── starcoder.cpp / starcoder2.cpp # StarCoder
├── bert.cpp / modern_bert.cpp   # BERT系列
├── eurobert.cpp / nomic_bert.cpp
├── t5.cpp / t5encoder.cpp       # T5系列
├── mamba.cpp / mamba2.cpp       # Mamba状态空间模型
├── rwkv6.cpp / rwkv7.cpp        # RWKV循环模型
├── arwkv7.cpp                   # 异步RWKV
├── jamba.cpp                    # Jamba混合架构
├── clip.cpp                     # CLIP视觉编码
├── phi2.cpp / phi3.cpp / phimoe.cpp # Phi系列
├── qwen2vl.cpp                  # Qwen2-VL多模态
├── olmo.cpp / olmo2.cpp         # OLMo
├── plamo.cpp / plamo2.cpp / plamo3.cpp # PLaMo
├── exaone.cpp                   # EXAONE
├── command_r.cpp / cohere2.cpp  # Cohere系列
├── nemotron.cpp / nemotron_h.cpp # Nemotron
├── minicpm.cpp / minicpm3.cpp  # MiniCPM
└── ... (更多架构实现)
```

### 7.6.2 模型实现的核心职责

每个 `src/models/*.cpp` 文件通常包含：

```cpp
// 以 llama.cpp 为例

class llm_build_llama : public llm_graph_context {
public:
    llm_build_llama(...)
        : llm_graph_context(..., LLM_ARCH_LLAMA) {}

    // 构建前向传播计算图
    llm_graph_result build() {
        // 1. 词嵌入
        // 2. 遍历各层构建Transformer Block
        // 3. 输出层
    }

private:
    // 构建单层Transformer
    ggml_tensor * build_layer(int32_t il) {
        // · Self-Attention (GQA)
        // · FFN (SwiGLU)
        // · 残差连接
    }
};
```

**主要职责**：

1. **计算图构建**：从输入token构建到输出logits的完整计算图
2. **架构特定逻辑**：处理特定架构的独特结构（如位置编码、层归一化变体等）
3. **张量映射**：将架构特定的权重名称映射到统一的图节点

### 7.6.3 典型模型实现对比

| 模型 | 关键特性 | 特殊实现 |
|------|----------|----------|
| **llama.cpp** | RoPE, GQA, SwiGLU | 标准Transformer基准 |
| **qwen2.cpp** | Qwen特有RoPE配置, SwiGLU | 支持Qwen1.5/2.x |
| **qwen3.cpp** | 共享专家/独立专家 | MoE架构支持 |
| **gemma2.cpp** | 滑动窗口注意力 | SWA层特殊处理 |
| **gemma3.cpp** | 局部-全局注意力交替 | 局部/全局层切换 |
| **deepseek2.cpp** | MLA (Multi-head Latent Attention) | 低秩注意力压缩 |
| **mamba.cpp** | SSM (State Space Model) | 无Attention线性时间 |
| **rwkv7.cpp** | RWKV v7时间衰减 | 循环结构实现 |
| **bert.cpp** | 双向Attention, MLM | 编码器专用图 |
| **jamba.cpp** | Attention-Mamba混合 | 层类型交替逻辑 |
| **clip.cpp** | 视觉编码器 | ViT结构实现 |
| **qwen2vl.cpp** | 视觉-语言融合 | 图像patch处理 |

### 7.6.4 添加新架构支持

**步骤1：创建模型实现文件**

```cpp
// src/models/myllm.cpp

class llm_build_myllm : public llm_graph_context {
public:
    llm_build_myllm(...)
        : llm_graph_context(..., LLM_ARCH_MYLLM) {}

    llm_graph_result build() {
        // 实现计算图构建
        ggml_tensor * cur = build_inp_embd();
        
        for (int il = 0; il < n_layer; il++) {
            cur = build_layer(il, cur);
        }
        
        return build_output(cur);
    }

private:
    ggml_tensor * build_layer(int32_t il, ggml_tensor * inp) {
        // 实现单层逻辑
    }
};
```

**步骤2：在 llama-graph.cpp 中注册**

```cpp
// src/llama-graph.cpp
llm_graph_result llama_graph_build(
    llama_context & lctx,
    const llama_batch & batch,
    llm_graph_type gtype) {
    
    switch (lctx.model.arch) {
        case LLM_ARCH_LLAMA:
            return llm_build_llama(lctx, batch).build();
        case LLM_ARCH_MYLLM:
            return llm_build_myllm(lctx, batch).build();
        // ...
    }
}
```

**步骤3：更新 llama-arch.h 和 llama-arch.cpp**

```cpp
// src/llama-arch.h
enum llm_arch {
    // ...
    LLM_ARCH_MYLLM,
    LLM_ARCH_UNKNOWN,
};

// src/llama-arch.cpp
static const std::map<llm_arch, llm_arch_info> LLM_ARCH_INFO = {
    // ...
    {LLM_ARCH_MYLLM, {
        "myllm",
        {
            {LLM_TENSOR_TOKEN_EMBD, "token_embd"},
            {LLM_TENSOR_ATTN_Q, "attn_q"},
            {LLM_TENSOR_ATTN_K, "attn_k"},
            // ... 张量名称映射
        }
    }},
};
```

### 7.6.5 模型实现与架构配置的关系

```
┌─────────────────────────────────────────────────────────────┐
│                    模型支持层次                             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────┐      ┌──────────────┐                   │
│  │ llama-arch.h │      │ llama-arch.cpp│                   │
│  │ · 架构枚举    │─────→│ · 架构配置    │                   │
│  │ · 张量映射    │      │ · 张量名称    │                   │
│  └──────────────┘      └──────┬─────────┘                   │
│                               │                             │
│                               ▼                             │
│  ┌──────────────┐      ┌──────────────┐                   │
│  │ models/ *.cpp │      │ llama-graph  │                   │
│  │ · 计算图实现  │─────→│ · 图构建分发 │                   │
│  │ · 架构特有逻辑│      │ · 统一执行   │                   │
│  └──────────────┘      └──────────────┘                     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**设计哲学**：

1. **配置与实现分离**：`llama-arch.h` 定义"有什么"，`models/*.cpp` 定义"怎么做"
2. **可扩展性**：添加新模型只需增加配置 + 实现文件，不改动核心
3. **代码复用**：继承 `llm_graph_context` 复用通用构建工具

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
| models/*.cpp | 各架构的具体计算图实现 |

---

*本章对应源码版本：master (2026-04-07)*
