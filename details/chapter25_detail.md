# 第25章 实用工具集 —— 模型开发的"百宝箱"

## 学习目标
1. 掌握llama-tokenize的分词调试技巧
2. 理解llama-embedding的批量嵌入生成
3. 学会使用llama-perplexity评估模型质量
4. 了解GGUF文件操作工具的使用方法
5. 能够综合运用工具进行模型开发和调试

---

## 生活类比：工匠的工具箱

想象你是一位模型开发的"工匠"，llama.cpp的工具集就像你的工具箱：

- **llama-tokenize** = 放大镜（仔细观察每个Token的细节）
- **llama-embedding** = 测量仪（获取文本的向量表示）
- **llama-perplexity** = 质检仪（评估模型输出质量）
- **gguf-split/gguf** = 切割刀和标签机（管理和查看模型文件）
- **convert_lora_to_gguf.py** = 适配器转换器（将LoRA适配器转为可用格式）

就像工匠需要各种工具完成精细工作，模型开发者也需要这些工具进行调试、评估和转换。

---

## 源码地图

```
tools/
├── tokenize/tokenize.cpp       # 分词工具
├── embedding/embedding.cpp     # 嵌入生成工具
├── perplexity/perplexity.cpp   # 困惑度计算工具
├── gguf/gguf.cpp               # GGUF信息查看
├── gguf-split/gguf-split.cpp   # GGUF分割/合并
├── imatrix/imatrix.cpp         # 重要性矩阵生成
├── export-lora/                # LoRA导出工具
└── ...

convert_lora_to_gguf.py         # LoRA转换脚本
convert_hf_to_gguf.py           # HF模型转换
convert_llama_ggml_to_gguf.py   # 旧格式升级
```

---

## 25.1 分词工具（tokenize）

### 25.1.1 功能概述

**源码位置**：`tools/tokenize/tokenize.cpp` (第1-100行)

llama-tokenize是一个轻量级工具，用于：
- 查看文本如何被分词
- 调试特殊Token处理
- 验证BOS/EOS行为
- 支持多种输出格式

```cpp
// 使用示例
// ./llama-tokenize -m model.gguf -p "Hello, world!"
// 输出：
// Hello -> 15043
// , -> 29892
//  world -> 318
// ! -> 29991

// 仅输出ID（便于脚本处理）
// ./llama-tokenize -m model.gguf -p "Hello" --ids
// 输出：[15043]
```

### 25.1.2 实现细节

**源码位置**：`tools/tokenize/tokenize.cpp` (第100-300行)

```cpp
int main(int argc, char ** argv) {
    // 解析参数
    std::string model_path;
    std::string prompt;
    bool print_ids_only = false;
    bool add_bos = true;
    bool parse_special = true;
    
    // 加载模型（仅词汇表）
    llama_model_params model_params = llama_model_default_params();
    llama_model * model = llama_load_model_from_file(model_path.c_str(), model_params);
    const llama_vocab * vocab = llama_model_get_vocab(model);
    
    // 分词
    std::vector<llama_token> tokens;
    tokens = common_tokenize(vocab, prompt, add_bos, parse_special);
    
    // 输出结果
    if (print_ids_only) {
        // Python列表格式：[1, 2, 3]
        printf("[");
        for (size_t i = 0; i < tokens.size(); i++) {
            printf("%d%s", tokens[i], i < tokens.size() - 1 ? ", " : "");
        }
        printf("]\n");
    } else {
        // 详细格式：每行显示Token文本和ID
        for (auto token : tokens) {
            std::string piece = common_token_to_piece(vocab, token);
            printf("%s -> %d\n", piece.c_str(), token);
        }
        
        // 统计信息
        printf("\nTotal tokens: %zu\n", tokens.size());
    }
    
    llama_free_model(model);
    return 0;
}
```

### 25.1.3 调试技巧

```bash
# 1. 检查特殊Token
./llama-tokenize -m model.gguf -p "<|im_start|>user<|im_end|>" --ids

# 2. 对比不同parse_special行为
./llama-tokenize -m model.gguf -p "<s>" --no-parse-special  # 作为普通文本
./llama-tokenize -m model.gguf -p "<s>" --parse-special     # 作为特殊Token

# 3. 检查BOS行为
./llama-tokenize -m model.gguf -p "Hello" --no-bos  # 不添加BOS
./llama-tokenize -m model.gguf -p "Hello"           # 自动添加BOS

# 4. 批量处理文件
./llama-tokenize -m model.gguf -f input.txt --show-count
```

---

## 25.2 嵌入生成工具（embedding）

### 25.2.1 功能概述

**源码位置**：`tools/embedding/embedding.cpp` (第1-100行)

llama-embedding用于生成文本的向量表示（embeddings），支持：
- 批量处理多个文本
- 多种池化策略（mean、cls、last）
- 向量归一化
- OpenAI兼容输出格式

```cpp
// 使用示例
// ./llama-embedding -m model.gguf -p "Hello world" -p "Another text"
// 输出：
// 0.123 0.456 0.789 ...  # 第一个文本的嵌入向量
// 0.234 0.567 0.890 ...  # 第二个文本的嵌入向量
```

### 25.2.2 批量嵌入实现

**源码位置**：`tools/embedding/embedding.cpp` (第100-300行)

```cpp
// 批量添加序列到batch
static void batch_add_seq(
    llama_batch & batch,
    const std::vector<int32_t> & tokens,
    llama_seq_id seq_id
) {
    for (size_t i = 0; i < tokens.size(); i++) {
        common_batch_add(batch, tokens[i], i, {seq_id}, true);
    }
}

// 解码并提取嵌入
static void batch_decode(
    llama_context * ctx,
    llama_batch & batch,
    float * output,
    int n_seq,
    int n_embd_out,
    int embd_norm
) {
    // 清除KV缓存（嵌入不需要历史）
    llama_memory_clear(llama_get_memory(ctx), true);
    
    // 运行模型
    if (llama_decode(ctx, batch) < 0) {
        LOG_ERR("Failed to decode\n");
        return;
    }
    
    // 根据池化类型提取嵌入
    enum llama_pooling_type pooling_type = llama_pooling_type(ctx);
    
    for (int i = 0; i < batch.n_tokens; i++) {
        if (!batch.logits[i]) continue;
        
        const float * embd = nullptr;
        if (pooling_type == LLAMA_POOLING_TYPE_NONE) {
            // 获取每个token的嵌入
            embd = llama_get_embeddings_ith(ctx, i);
        } else {
            // 获取序列级嵌入
            embd = llama_get_embeddings_seq(ctx, batch.seq_id[i][0]);
        }
        
        // 归一化并复制到输出
        float * out = output + batch.seq_id[i][0] * n_embd_out;
        common_embd_normalize(embd, out, n_embd_out, embd_norm);
    }
}
```

### 25.2.3 池化策略

```cpp
// 不同池化策略的适用场景：
//
// LLAMA_POOLING_TYPE_NONE:
//   - 输出每个token的嵌入
//   - 用于token级任务（NER、POS标注）
//
// LLAMA_POOLING_TYPE_MEAN:
//   - 对所有token嵌入取平均
//   - 通用文本表示，平滑噪声
//
// LLAMA_POOLING_TYPE_CLS:
//   - 使用[CLS] token的嵌入
//   - BERT风格模型常用
//
// LLAMA_POOLING_TYPE_LAST:
//   - 使用最后一个token的嵌入
//   - 适用于因果语言模型
//
// LLAMA_POOLING_TYPE_RANK:
//   - 用于重排序模型
//   - 输出相关性分数

// 使用示例
./llama-embedding -m model.gguf -p "Text" --pooling mean
./llama-embedding -m model.gguf -p "Text" --pooling cls
```

---

## 25.3 困惑度计算（perplexity）

### 25.3.1 概念介绍

**困惑度（Perplexity, PPL）** 是衡量语言模型性能的重要指标：
- **越低越好**：PPL = 1 表示完美预测
- **计算公式**：PPL = exp(-平均对数似然)
- **实际意义**：模型有多"困惑"，即模型对下一个token有多不确定

```cpp
// 困惑度解释：
// PPL = 100  -> 相当于每次从100个等概率词中选择
// PPL = 10   -> 相当于每次从10个等概率词中选择（更好）
// PPL = 2    -> 相当于每次从2个等概率词中选择（很好）
```

### 25.3.2 实现细节

**源码位置**：`tools/perplexity/perplexity.cpp` (第1-200行)

```cpp
// 计算单个token的log softmax
static results_log_softmax log_softmax(
    int n_vocab,
    const float * logits,
    int tok
) {
    // 数值稳定性：减去最大值
    float max_logit = logits[0];
    for (int i = 1; i < n_vocab; ++i) {
        max_logit = std::max(max_logit, logits[i]);
    }
    
    // 计算softmax分母
    double sum_exp = 0.0;
    for (int i = 0; i < n_vocab; ++i) {
        sum_exp += expf(logits[i] - max_logit);
    }
    
    // 返回目标token的log概率
    return {
        logits[tok] - max_logit - log(sum_exp),  // log_softmax
        logits[tok],                              // logit
        expf(logits[tok] - max_logit) / sum_exp   // probability
    };
}

// 主计算流程
static void compute_perplexity(
    llama_context * ctx,
    const std::vector<llama_token> & tokens,
    results_perplexity & results
) {
    const int n_vocab = llama_vocab_n_tokens(vocab);
    const int n_ctx = llama_n_ctx(ctx);
    
    double nll = 0.0;   // 负对数似然累积
    int count = 0;
    
    // 滑动窗口处理（避免超出上下文）
    for (size_t i = 0; i < tokens.size() - 1; i += n_ctx - 1) {
        size_t len = std::min(tokens.size() - i - 1, (size_t)n_ctx - 1);
        
        // 构建batch
        llama_batch batch = llama_batch_init(len, 0, 1);
        for (size_t j = 0; j < len; j++) {
            common_batch_add(batch, tokens[i + j], j, {0}, true);
        }
        
        // 解码
        llama_decode(ctx, batch);
        
        // 获取logits并计算
        for (size_t j = 0; j < len; j++) {
            const float * logits = llama_get_logits_ith(ctx, j);
            int next_token = tokens[i + j + 1];
            
            auto res = log_softmax(n_vocab, logits, next_token);
            nll += -res.log_softmax;
            count++;
        }
        
        llama_batch_free(batch);
    }
    
    // 计算最终困惑度
    results.ppl_value = exp(nll / count);
}
```

### 25.3.3 使用场景

```bash
# 1. 评估模型质量
./llama-perplexity -m model.gguf -f wiki.test.raw

# 2. 对比不同量化类型
./llama-perplexity -m model-f16.gguf -f test.txt
./llama-perplexity -m model-q4_0.gguf -f test.txt
./llama-perplexity -m model-q4_k_m.gguf -f test.txt

# 3. 使用特定上下文长度
./llama-perplexity -m model.gguf -f test.txt -c 4096

# 4. 输出详细logits（用于分析）
./llama-perplexity -m model.gguf -f test.txt --logits-all
```

---

## 25.4 GGUF文件操作

### 25.4.1 GGUF信息查看

**源码位置**：`tools/gguf/gguf.cpp` (第1-200行)

```cpp
// 显示GGUF文件信息
void print_gguf_info(const std::string & fname) {
    struct gguf_context * ctx = gguf_init_from_file(
        fname.c_str(),
        {true, nullptr, true}  // no_alloc, params, calc_hash
    );
    
    // 显示元数据
    printf("GGUF version: %d\n", gguf_get_version(ctx));
    printf("Alignment: %zu\n", gguf_get_alignment(ctx));
    printf("Number of tensors: %zu\n", gguf_get_n_tensors(ctx));
    printf("Number of metadata: %zu\n", gguf_get_n_kv(ctx));
    
    // 显示张量信息
    printf("\nTensors:\n");
    for (size_t i = 0; i < gguf_get_n_tensors(ctx); i++) {
        const char * name = gguf_get_tensor_name(ctx, i);
        enum ggml_type type = gguf_get_tensor_type(ctx, i);
        const size_t * dims = gguf_get_tensor_dims(ctx, i);
        int n_dims = gguf_get_tensor_n_dims(ctx, i);
        
        printf("  %s: %s [", name, ggml_type_name(type));
        for (int j = 0; j < n_dims; j++) {
            printf("%zu%s", dims[j], j < n_dims - 1 ? ", " : "");
        }
        printf("]\n");
    }
    
    // 显示元数据
    printf("\nMetadata:\n");
    for (size_t i = 0; i < gguf_get_n_kv(ctx); i++) {
        const char * key = gguf_get_key(ctx, i);
        enum gguf_type type = gguf_get_kv_type(ctx, i);
        
        printf("  %s: ", key);
        switch (type) {
            case GGUF_TYPE_UINT32:
                printf("%u\n", gguf_get_val_u32(ctx, i));
                break;
            case GGUF_TYPE_FLOAT32:
                printf("%f\n", gguf_get_val_f32(ctx, i));
                break;
            case GGUF_TYPE_STRING:
                printf("%s\n", gguf_get_val_str(ctx, i));
                break;
            // ... 其他类型
        }
    }
    
    gguf_free(ctx);
}
```

### 25.4.2 GGUF分割与合并

**源码位置**：`tools/gguf-split/gguf-split.cpp` (第1-300行)

```cpp
// 分割策略
enum split_mode {
    MODE_NONE,
    MODE_TENSOR,    // 按张量数量分割
    MODE_SIZE,      // 按文件大小分割
};

// 分割GGUF文件
void split_gguf(
    const std::string & input,
    const std::string & output_prefix,
    split_mode mode,
    size_t max_size,
    int max_tensors
) {
    // 加载源文件
    struct gguf_context * ctx = gguf_init_from_file(input.c_str(), {true});
    
    size_t n_tensors = gguf_get_n_tensors(ctx);
    size_t current_size = 0;
    int current_tensors = 0;
    int split_idx = 0;
    
    struct gguf_context * current_split = nullptr;
    
    for (size_t i = 0; i < n_tensors; i++) {
        const char * name = gguf_get_tensor_name(ctx, i);
        size_t tensor_size = gguf_get_tensor_size(ctx, i);
        
        // 检查是否需要新分割
        bool need_new_split = false;
        if (mode == MODE_SIZE && current_size + tensor_size > max_size) {
            need_new_split = true;
        }
        if (mode == MODE_TENSOR && current_tensors >= max_tensors) {
            need_new_split = true;
        }
        
        if (need_new_split || current_split == nullptr) {
            // 保存当前分割
            if (current_split) {
                std::string fname = format("%s-%05d.gguf", output_prefix, split_idx);
                gguf_write_to_file(current_split, fname.c_str(), true);
                gguf_free(current_split);
                split_idx++;
            }
            
            // 创建新分割
            current_split = gguf_init_empty();
            // 复制元数据
            gguf_copy_metadata(current_split, ctx);
            current_size = 0;
            current_tensors = 0;
        }
        
        // 添加张量到当前分割
        gguf_add_tensor(current_split, ctx, i);
        current_size += tensor_size;
        current_tensors++;
    }
    
    // 保存最后一个分割
    if (current_split) {
        std::string fname = format("%s-%05d.gguf", output_prefix, split_idx);
        gguf_write_to_file(current_split, fname.c_str(), true);
        gguf_free(current_split);
    }
    
    gguf_free(ctx);
}

// 使用示例
// ./gguf-split --split input.gguf output-prefix --split-max-size 2G
// 输出：output-prefix-00000.gguf, output-prefix-00001.gguf, ...
```

---

## 25.5 LoRA转换与导出

### 25.5.1 LoRA转GGUF

**源码位置**：`convert_lora_to_gguf.py` (关键部分)

```python
#!/usr/bin/env python3
"""
将HuggingFace LoRA适配器转换为GGUF格式
支持多种LoRA格式：
- transformers PEFT
- llama.cpp LoRA
- 其他常见格式
"""

import torch
import json
import struct
from pathlib import Path

def convert_lora_to_gguf(
    input_path: str,
    output_path: str,
    base_model: str = None
):
    """
    转换LoRA适配器到GGUF格式
    
    Args:
        input_path: 输入LoRA目录（包含adapter_config.json和adapter_model.bin）
        output_path: 输出GGUF文件路径
        base_model: 基础模型名称（用于验证）
    """
    
    # 加载配置
    config_path = Path(input_path) / "adapter_config.json"
    with open(config_path) as f:
        config = json.load(f)
    
    # 加载权重
    model_path = Path(input_path) / "adapter_model.bin"
    state_dict = torch.load(model_path, map_location="cpu")
    
    # 写入GGUF头部
    with open(output_path, 'wb') as f:
        # GGUF魔数
        f.write(b'GGUF')
        
        # 版本
        f.write(struct.pack('<I', 3))
        
        # 张量数量
        f.write(struct.pack('<Q', len(state_dict)))
        
        # 元数据
        write_metadata(f, {
            'general.architecture': config['base_model_name_or_path'],
            'general.lora.alpha': config.get('lora_alpha', 16),
            'general.lora.r': config.get('r', 8),
        })
        
        # 写入张量
        for name, tensor in state_dict.items():
            # 转换名称格式
            gguf_name = convert_tensor_name(name)
            
            # 写入张量信息
            write_tensor_info(f, gguf_name, tensor)
            
            # 写入数据（FP16）
            tensor_fp16 = tensor.half().numpy()
            f.write(tensor_fp16.tobytes())

# 使用示例
# python convert_lora_to_gguf.py \
#     --input lora-adapter/ \
#     --output adapter.gguf \
#     --base-model llama-7b
```

### 25.5.2 LoRA使用

```bash
# 1. 转换LoRA
python convert_lora_to_gguf.py \
    --input ./lora-adapter \
    --output ./adapter.gguf

# 2. 使用LoRA进行推理
./llama-cli \
    -m base-model.gguf \
    --lora adapter.gguf \
    -p "Your prompt here"

# 3. 合并LoRA到基础模型（永久应用）
./llama-export-lora \
    -m base-model.gguf \
    --lora adapter.gguf \
    -o merged-model.gguf
```

---

## 设计中的取舍

### 为什么工具要分开而不是一个"瑞士军刀"？

| 方案 | 优点 | 缺点 | 选择 |
|-----|------|------|------|
| 单一工具 | 统一接口 | 体积大，加载慢 | 否 |
| 分离工具 | 专注高效，可组合 | 学习成本稍高 | **是** |
| 库+脚本 | 灵活 | 需要运行时环境 | 部分使用 |

**分离工具的优势**：
1. **启动快**：不需要加载无关代码
2. **内存小**：只加载需要的功能
3. **可组合**：通过shell脚本组合使用
4. **易维护**：修改一个工具不影响其他

### 为什么Python用于转换脚本？

```cpp
// C++转换的问题：
// - 需要处理PyTorch格式
// - 依赖Python生态
// - 开发效率低

// Python转换的优势：
// - 直接读取PyTorch checkpoint
// - 使用transformers库
// - 快速迭代
// - 社区熟悉

// 架构决策：
// - 运行时工具：C++（性能关键）
// - 转换脚本：Python（开发效率）
```

---

## 动手练习

### 练习1：分词分析

使用llama-tokenize分析不同模型的分词差异：
```bash
# 对比Llama和GPT-2的分词
./llama-tokenize -m llama-7b.gguf -p "ChatGPT" --ids
./llama-tokenize -m gpt2.gguf -p "ChatGPT" --ids

# 分析多语言分词
./llama-tokenize -m model.gguf -p "你好世界" --ids
./llama-tokenize -m model.gguf -p "こんにちは" --ids
```

### 练习2：嵌入相似度计算

编写脚本计算文本相似度：
```python
#!/usr/bin/env python3
import subprocess
import numpy as np

def get_embedding(text):
    result = subprocess.run(
        ['./llama-embedding', '-m', 'model.gguf', '-p', text],
        capture_output=True, text=True
    )
    return np.array([float(x) for x in result.stdout.strip().split()])

# 计算余弦相似度
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

emb1 = get_embedding("King")
emb2 = get_embedding("Queen")
emb3 = get_embedding("Apple")

print(f"King vs Queen: {cosine_similarity(emb1, emb2)}")
print(f"King vs Apple: {cosine_similarity(emb1, emb3)}")
```

### 练习3：量化质量评估

使用perplexity评估不同量化配置：
```bash
#!/bin/bash

MODEL_BASE="llama-7b"
TEST_FILE="wiki.test.raw"

# 测试不同量化类型
for qtype in Q4_0 Q4_1 Q5_0 Q5_1 Q8_0 Q4_K_M Q5_K_M Q6_K; do
    echo "Testing $qtype..."
    ./llama-quantize ${MODEL_BASE}-f16.gguf ${MODEL_BASE}-${qtype}.gguf ${qtype}
    ppl=$(./llama-perplexity -m ${MODEL_BASE}-${qtype}.gguf -f ${TEST_FILE} 2>&1 | grep "Final perplexity" | awk '{print $3}')
    size=$(ls -lh ${MODEL_BASE}-${qtype}.gguf | awk '{print $5}')
    echo "$qtype: PPL=$ppl, Size=$size"
done
```

---

## 本课小结

| 工具 | 用途 | 典型命令 |
|-----|------|---------|
| llama-tokenize | 分词调试 | `-m model -p "text" --ids` |
| llama-embedding | 嵌入生成 | `-m model -p "text" --pooling mean` |
| llama-perplexity | 质量评估 | `-m model -f test.txt` |
| gguf-split | 文件分割 | `--split input.gguf output --split-max-size 2G` |
| convert_lora_to_gguf.py | LoRA转换 | `--input lora/ --output adapter.gguf` |

---

*本章对应源码版本：master (2026-04-07)*
