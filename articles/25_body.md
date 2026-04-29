# 第25章 实用工具集 —— 模型开发的"百宝箱"

## 学习目标

1. 掌握llama-tokenize的分词调试技巧
2. 理解llama-embedding的批量嵌入生成
3. 学会使用llama-perplexity评估模型质量
4. 了解GGUF文件操作工具的使用方法
5. 能够综合运用工具进行模型开发和调试

---

## 生活类比：工匠的工具箱

想象你是一位模型开发的"工匠"，每天面对的是复杂的神经网络、海量的参数和神秘的黑盒行为。就像木匠需要锯子、刨子、凿子来完成精细的木工活，你也需要一套专门的工具来打磨、调试和评估模型。llama.cpp的工具集就是你的"百宝箱"，每件工具都有其独特的用途。

当你需要仔细观察模型如何处理文本时，llama-tokenize就像一枚高倍放大镜。通过它，你可以看清每一个token的边界，理解为什么"ChatGPT"被分成"Chat"和"GPT"两个token，为什么中文"你好世界"变成了四个token，或者为什么一段代码的缩进被合并为一个空格token。在调试聊天模板或处理特殊字符时，这个放大镜能让你发现肉眼看不见的细节——这正是排查模型行为差异的第一步。llama-embedding则像一台精密的测量仪，它能将任何文本转换成一个高维向量（数学上的"指纹"），然后你可以计算它们之间的距离。"国王"和"女王"的向量距离很近，但"国王"和"苹果"就相距甚远——这种语义关系的量化正是RAG（检索增强生成）系统的核心。

在质检环节，llama-perplexity是你的"模型质检仪"。它告诉你模型对测试数据的预测有多"困惑"——困惑度越低，模型越能准确预测文本。当你将模型从FP16量化到Q4_K_M后，体积缩小了一半多，但质量下降了多少？困惑度给你一个精确的量化答案，帮你做出"这个量化级别是否可接受"的判断。当模型文件太大无法上传网盘或内存不足时，gguf-split就像一把精准的切割刀，将大文件切分成便于传输和存储的小块。而gguf查看工具则像标签机，让你清晰了解模型文件里有什么元数据、每个张量多大、使用什么格式——一目了然。最后，convert_lora_to_gguf.py是一个万能的适配器转换器：在HuggingFace上找到一个好用的PyTorch格式LoRA适配器，通过这个脚本就能转换为llama.cpp生态可直接使用的GGUF格式，让不同生态系统的模型能够互通。

---

## 25.1 分词工具（tokenize）—— 理解模型的"眼睛"

### 25.1.1 功能概述

分词（Tokenization）是大语言模型的第一步，也是最容易被忽视却极其重要的一环。同一个词在不同模型中可能被分成完全不同的token，这直接影响模型的理解和生成。

**为什么分词很重要？**

```
示例：不同模型的分词差异

输入文本： "ChatGPT"

Llama 3分词：
  [Chat] ->  [GPT] 
  Token 1:  "Chat"  (ID:  8514)
  Token 2:  "GPT"   (ID:  37051)

GPT-2分词：
  [Ch] [atG] [PT]
  Token 1:  "Ch"    (ID:  617)
  Token 2:  "atG"   (ID:  318)
  Token 3:  "PT"    (ID:  29991)

这种差异会影响：
- 少样本学习中的示例格式
- 特殊token的处理（如<|im_start|>）
- 代码生成中的缩进和空格
```

**llama-tokenize的使用场景**

**源码位置**：`tools/tokenize/tokenize.cpp` (第1-100行)

```cpp
/**
 * llama-tokenize 使用示例
 * 
 * 这个轻量级工具用于：
 * 1. 查看文本如何被分词
 * 2. 调试特殊Token处理
 * 3. 验证BOS/EOS行为
 * 4. 支持多种输出格式（便于脚本处理）
 */

// 基本用法：显示详细分词结果
// $ ./llama-tokenize -m model.gguf -p "Hello, world!"
// 输出：
// Hello -> 15043
// , -> 29892
//  world -> 318
// ! -> 29991

// 仅输出ID（便于脚本处理）
// $ ./llama-tokenize -m model.gguf -p "Hello" --ids
// 输出：[15043]

// 查看统计信息
// $ ./llama-tokenize -m model.gguf -f long_text.txt --show-count
// 输出：Total tokens: 1534
```

### 25.1.2 实现细节

**源码位置**：`tools/tokenize/tokenize.cpp` (第100-300行)

```cpp
/**
 * 分词工具主函数
 * 
 * 设计原则：轻量、快速、易于脚本集成
 */
int main(int argc, char ** argv) {
    // ========== 参数解析 ==========
    std::string model_path;   // 模型路径（只需要词汇表）
    std::string prompt;       // 要分词的文本
    std::string file_path;    // 或者从文件读取
    
    // 输出选项
    bool print_ids_only = false;   // 只打印ID列表
    bool show_count = false;       // 显示token数量统计
    
    // 分词选项
    bool add_bos = true;           // 是否添加BOS token
    bool parse_special = true;     // 是否解析特殊token（如<|im_start|>）
    
    // 解析命令行参数...
    parse_args(argc, argv, model_path, prompt, file_path,
               print_ids_only, show_count, add_bos, parse_special);
    
    // ========== 加载模型词汇表 ==========
    // 注意：我们只需要词汇表，不需要完整的模型权重
    llama_model_params model_params = llama_model_default_params();
    
    // 加载模型（这会加载GGUF并初始化词汇表）
    llama_model * model = llama_load_model_from_file(
        model_path.c_str(),
        model_params
    );
    
    if (!model) {
        fprintf(stderr, "Error: Failed to load model\n");
        return 1;
    }
    
    // 获取词汇表
    const llama_vocab * vocab = llama_model_get_vocab(model);
    
    // ========== 读取输入文本 ==========
    std::string text;
    if (!file_path.empty()) {
        // 从文件读取
        std::ifstream file(file_path);
        text = std::string(
            (std::istreambuf_iterator<char>(file)),
            std::istreambuf_iterator<char>()
        );
    } else {
        text = prompt;
    }
    
    // ========== 执行分词 ==========
    // 使用common_tokenize，这是llama.cpp的标准分词函数
    std::vector<llama_token> tokens = common_tokenize(
        vocab,
        text,
        add_bos,        // 是否在开头添加BOS
        parse_special   // 是否将<|xxx|>解析为特殊token
    );
    
    // ========== 输出结果 ==========
    if (print_ids_only) {
        // Python列表格式，便于脚本解析
        printf("[");
        for (size_t i = 0; i < tokens.size(); i++) {
            printf("%d", tokens[i]);
            if (i < tokens.size() - 1) {
                printf(", ");
            }
        }
        printf("]\n");
    } else {
        // 详细格式：每行显示Token文本和ID
        for (auto token : tokens) {
            // 将token ID转换回文本片段
            std::string piece = common_token_to_piece(vocab, token);
            
            // 处理不可见字符（如空格、换行）
            std::string display = sanitize_for_display(piece);
            
            printf("%-20s -> %d\n", display.c_str(), token);
        }
        
        // 统计信息
        printf("\n");
        printf("Total tokens: %zu\n", tokens.size());
        printf("BOS: %s\n", add_bos ? "yes" : "no");
        printf("Parse special: %s\n", parse_special ? "yes" : "no");
    }
    
    // 清理
    llama_free_model(model);
    return 0;
}

/**
 * 清理字符串以便显示
 * 将不可见字符转换为可打印表示
 */
std::string sanitize_for_display(const std::string & piece) {
    std::string result;
    for (char c : piece) {
        switch (c) {
            case '\n': result += "\\n"; break;
            case '\t': result += "\\t"; break;
            case '\r': result += "\\r"; break;
            case ' ':  result += "·"; break;  // 显示空格为中间点
            default:
                if (isprint(c)) {
                    result += c;
                } else {
                    // 其他不可见字符显示为十六进制
                    char buf[8];
                    snprintf(buf, sizeof(buf), "\\x%02x", (unsigned char)c);
                    result += buf;
                }
        }
    }
    return result;
}
```

### 25.1.3 调试技巧

分词工具有许多实际应用场景，特别是在调试模型行为时。

**场景1：检查特殊Token处理**

```bash
# 不同模型有不同的特殊token格式
# Llama 3: <|begin_of_text|>, <|eot_id|>
# ChatML: <|im_start|>, <|im_end|>
# GPT-2: <|endoftext|>

# 测试parse_special行为
$ ./llama-tokenize -m llama3.gguf -p "<|begin_of_text|>Hello" --ids
[128000, 9906]  # 128000是BOS token ID

$ ./llama-tokenize -m llama3.gguf -p "<|begin_of_text|>Hello" --no-parse-special --ids
[27, 91, 5969, 97, 31001, 91, 29, 9906]  # 被当作普通文本分词
```

**场景2：验证聊天模板**

```bash
# 查看完整对话会被如何分词
$ ./llama-tokenize -m model.gguf -p \
"<|im_start|>user
Hello!<|im_end|>
<|im_start|>assistant
" --ids

# 这帮助你理解：
# - 模板格式是否正确
# - 是否有意外的token
# - 上下文长度使用情况
```

**场景3：分析多语言分词**

```bash
# 不同语言的分词效率差异很大
$ ./llama-tokenize -m model.gguf -p "Hello world" --show-count
Total tokens: 3  # 英语高效

$ ./llama-tokenize -m model.gguf -p "你好世界" --show-count
Total tokens: 4  # 中文每个字通常1-2个token

$ ./llama-tokenize -m model.gguf -p "こんにちは" --show-count
Total tokens: 5  # 日语类似中文

# 结论：非英语文本通常需要更多token，成本更高
```

**场景4：调试代码生成**

```bash
# 检查代码缩进和空格处理
$ ./llama-tokenize -m model.gguf -p "    def hello():"
# 输出：
# ···· -> 315  (4个空格合并为一个token)
# def -> 7454
# · -> 220     (单个空格)
# hello -> 23786
# (): -> 279

# 这对理解代码模型的行为很重要
```

---

## 25.2 嵌入生成工具（embedding）—— 捕捉语义本质

### 25.2.1 功能概述

嵌入（Embedding）是将离散对象（如单词、句子）映射到连续向量空间的技术。语义相似的文本在向量空间中也接近，这是现代NLP的基础。

**llama-embedding的用途**

**源码位置**：`tools/embedding/embedding.cpp` (第1-100行)

```cpp
/**
 * llama-embedding 使用场景
 * 
 * 1. 语义搜索：找到与查询相关的文档
 * 2. 文本聚类：将相似文档分组
 * 3. 语义相似度：计算两段文本的相关性
 * 4. RAG系统：检索增强生成的核心组件
 * 5. 分类特征：用作文本分类的输入特征
 */

// 基本用法
$ ./llama-embedding -m model.gguf -p "Hello world"
0.123 0.456 0.789 ...  # 4096维或更多维度的向量

// 批量处理
$ ./llama-embedding -m model.gguf \
    -p "First text" \
    -p "Second text" \
    -p "Third text"
0.123 0.456 ...  # 第一个文本
0.234 0.567 ...  # 第二个文本
0.345 0.678 ...  # 第三个文本

// OpenAI兼容格式
$ ./llama-embedding -m model.gguf -p "Hello" --oaicompat
{
  "object": "list",
  "data": [{
    "object": "embedding",
    "embedding": [0.123, 0.456, ...],
    "index": 0
  }]
}
```

### 25.2.2 批量嵌入实现

**源码位置**：`tools/embedding/embedding.cpp` (第100-300行)

```cpp
/**
 * 添加序列到批次
 * 
 * 为批量嵌入准备输入数据。
 * 每个序列分配一个唯一的seq_id用于区分。
 */
static void batch_add_seq(
    llama_batch & batch,
    const std::vector<int32_t> & tokens,
    llama_seq_id seq_id
) {
    for (size_t i = 0; i < tokens.size(); i++) {
        common_batch_add(
            batch,
            tokens[i],           // token ID
            i,                   // 位置
            {seq_id},            // 序列ID（单个）
            true                 // 需要logits/embedding
        );
    }
}

/**
 * 解码批次并提取嵌入
 * 
 * 这是嵌入生成的核心函数。
 * 它执行模型推理并提取指定层的隐藏状态作为嵌入。
 */
static void batch_decode(
    llama_context * ctx,
    llama_batch & batch,
    float * output,           // 输出缓冲区
    int n_seq,                // 序列数量
    int n_embd_out,           // 嵌入维度
    int embd_norm             // 归一化类型
) {
    // ========== 准备 ==========
    // 清除KV缓存（嵌入不需要历史上下文）
    // 每个序列独立处理
    llama_memory_clear(llama_get_memory(ctx), true);
    
    // ========== 执行推理 ==========
    if (llama_decode(ctx, batch) < 0) {
        LOG_ERR("Failed to decode\n");
        return;
    }
    
    // ========== 提取嵌入 ==========
    // 获取池化类型配置
    enum llama_pooling_type pooling_type = llama_pooling_type(ctx);
    
    for (int i = 0; i < batch.n_tokens; i++) {
        // 跳过不需要输出的位置
        if (!batch.logits[i]) continue;
        
        const float * embd = nullptr;
        
        switch (pooling_type) {
            case LLAMA_POOLING_TYPE_NONE:
                // 获取每个token的嵌入
                // 适用于token级任务
                embd = llama_get_embeddings_ith(ctx, i);
                break;
                
            case LLAMA_POOLING_TYPE_MEAN:
            case LLAMA_POOLING_TYPE_CLS:
            case LLAMA_POOLING_TYPE_LAST:
                // 获取序列级嵌入
                // 模型内部已经进行了池化
                embd = llama_get_embeddings_seq(
                    ctx,
                    batch.seq_id[i][0]
                );
                break;
                
            case LLAMA_POOLING_TYPE_RANK:
                // 重排序模型的特殊处理
                embd = llama_get_embeddings_ith(ctx, i);
                break;
        }
        
        if (embd) {
            // 计算输出位置
            float * out = output + batch.seq_id[i][0] * n_embd_out;
            
            // 归一化并复制
            common_embd_normalize(embd, out, n_embd_out, embd_norm);
        }
    }
}

/**
 * 嵌入归一化
 * 
 * 归一化使不同向量的尺度一致，便于比较。
 */
void common_embd_normalize(
    const float * input,
    float * output,
    int n_embd,
    int normalization_type
) {
    switch (normalization_type) {
        case 0:  // 无归一化
            memcpy(output, input, n_embd * sizeof(float));
            break;
            
        case 1:  // L2归一化（最常用）
        {
            float sum_sq = 0.0f;
            for (int i = 0; i < n_embd; i++) {
                sum_sq += input[i] * input[i];
            }
            float norm = sqrtf(sum_sq);
            for (int i = 0; i < n_embd; i++) {
                output[i] = input[i] / norm;
            }
            break;
        }
            
        case 2:  // L1归一化
        {
            float sum_abs = 0.0f;
            for (int i = 0; i < n_embd; i++) {
                sum_abs += fabsf(input[i]);
            }
            for (int i = 0; i < n_embd; i++) {
                output[i] = input[i] / sum_abs;
            }
            break;
        }
    }
}
```

### 25.2.3 池化策略详解

池化决定了如何从变长文本生成固定大小的向量。

```cpp
/**
 * 池化策略对比
 */

// LLAMA_POOLING_TYPE_NONE
// 输入： "Hello world" (3 tokens: [BOS, Hello, world])
// 输出： 3个向量，每个维度n_embd
// 用途： Token级任务（NER、POS标注）
// 缺点： 输出大小不固定，难以比较不同长度文本

// LLAMA_POOLING_TYPE_MEAN
// 输入： "Hello world" (3 tokens)
// 处理： avg([BOS_emb, Hello_emb, world_emb])
// 输出： 1个向量，维度n_embd
// 优点： 平滑噪声，考虑所有token
// 用途： 通用文本表示

// LLAMA_POOLING_TYPE_CLS
// 输入： "Hello world"
// 处理： 取第一个token（[CLS]或BOS）的嵌入
// 输出： 1个向量
// 优点： 简单，BERT风格
// 缺点： 可能丢失后续信息
// 用途： 分类任务

// LLAMA_POOLING_TYPE_LAST
// 输入： "Hello world"
// 处理： 取最后一个token（world）的嵌入
// 输出： 1个向量
// 优点： 适用于因果语言模型（如GPT）
// 理论： 最后一个token汇聚了前面的信息
// 用途： 生成模型的嵌入

// LLAMA_POOLING_TYPE_RANK
// 特殊池化，输出相关性分数
// 用于重排序（reranking）模型

// 使用示例
$ ./llama-embedding -m model.gguf -p "Text" --pooling mean
$ ./llama-embedding -m model.gguf -p "Text" --pooling cls
$ ./llama-embedding -m model.gguf -p "Text" --pooling last
```

**实际应用：语义相似度计算**

```python
#!/usr/bin/env python3
"""
使用llama-embedding计算文本相似度
"""
import subprocess
import numpy as np

def get_embedding(text, model="model.gguf"):
    """调用llama-embedding获取向量"""
    result = subprocess.run(
        ['./llama-embedding', '-m', model, '-p', text, '--pooling', 'mean'],
        capture_output=True,
        text=True
    )
    # 解析输出
    values = [float(x) for x in result.stdout.strip().split()]
    return np.array(values)

def cosine_similarity(a, b):
    """计算余弦相似度"""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# 测试
if __name__ == "__main__":
    # 获取嵌入
    king = get_embedding("King")
    queen = get_embedding("Queen")
    man = get_embedding("Man")
    woman = get_embedding("Woman")
    apple = get_embedding("Apple")
    
    # 计算相似度
    print(f"King vs Queen: {cosine_similarity(king, queen):.4f}")
    print(f"King vs Man:   {cosine_similarity(king, man):.4f}")
    print(f"King vs Apple: {cosine_similarity(king, apple):.4f}")
    
    # 经典的类比：King - Man + Woman ≈ Queen
    analogy = king - man + woman
    print(f"\nAnalogy (King - Man + Woman) vs Queen: {cosine_similarity(analogy, queen):.4f}")
```

---

## 25.3 困惑度计算（perplexity）—— 模型质量的"温度计"

### 25.3.1 概念介绍

困惑度（Perplexity, PPL）是衡量语言模型性能的核心指标。它量化了模型预测下一个token时的"不确定性"。

**困惑度的直观理解**

```cpp
/**
 * 困惑度解释
 * 
 * PPL = exp(-平均对数似然)
 * 
 * 直观理解：
 * - PPL = 100  相当于每次从100个等概率词中选择
 * - PPL = 10   相当于每次从10个等概率词中选择（更好）
 * - PPL = 2    相当于每次从2个等概率词中选择（很好）
 * 
 * 困惑度越低，模型对文本的预测越准确。
 */

// 示例：不同质量的模型在相同测试集上的PPL
// 模型A (好): PPL = 8.5
// 模型B (中): PPL = 12.3
// 模型C (差): PPL = 25.6

// 量化对PPL的影响（Llama-2-7B在WikiText-2上）
// F16:     5.12  (基准)
// Q8_0:    5.14  (+0.02, 几乎无损)
// Q5_K_M:  5.28  (+0.16, 很好)
// Q4_K_M:  5.35  (+0.23, 良好)
// Q4_0:    5.89  (+0.77, 可接受)
// Q3_K_S:  6.45  (+1.33,  noticeable)
```

### 25.3.2 实现细节

**源码位置**：`tools/perplexity/perplexity.cpp` (第1-200行)

```cpp
/**
 * 计算log softmax（数值稳定版本）
 * 
 * 直接计算softmax再取log会有数值稳定性问题，
 * 这个函数使用log-space技巧避免溢出。
 */
static results_log_softmax log_softmax(
    int n_vocab,
    const float * logits,
    int tok
) {
    // 步骤1：找到最大logit（数值稳定性）
    float max_logit = logits[0];
    for (int i = 1; i < n_vocab; ++i) {
        max_logit = std::max(max_logit, logits[i]);
    }
    
    // 步骤2：计算log-sum-exp
    // log(sum(exp(x))) = max(x) + log(sum(exp(x - max(x))))
    double sum_exp = 0.0;
    for (int i = 0; i < n_vocab; ++i) {
        sum_exp += expf(logits[i] - max_logit);
    }
    float log_sum_exp = max_logit + logf(sum_exp);
    
    // 步骤3：目标token的log概率
    float log_softmax_val = logits[tok] - log_sum_exp;
    float prob = expf(log_softmax_val);
    
    return {
        log_softmax_val,   // log概率（用于PPL计算）
        logits[tok],       // 原始logit
        prob               // 概率
    };
}

/**
 * 主困惑度计算函数
 * 
 * 使用滑动窗口处理长文本，避免超出上下文限制。
 */
static void compute_perplexity(
    llama_context * ctx,
    const std::vector<llama_token> & tokens,
    results_perplexity & results
) {
    const int n_vocab = llama_vocab_n_tokens(vocab);
    const int n_ctx = llama_n_ctx(ctx);
    
    // 累积负对数似然
    double nll = 0.0;
    int count = 0;
    
    // 滑动窗口处理
    // 每次处理n_ctx-1个token，最后一个用于预测
    for (size_t i = 0; i < tokens.size() - 1; i += n_ctx - 1) {
        // 计算本次处理的长度
        size_t len = std::min(
            tokens.size() - i - 1,
            (size_t)n_ctx - 1
        );
        
        // 构建批次
        llama_batch batch = llama_batch_init(len, 0, 1);
        for (size_t j = 0; j < len; j++) {
            common_batch_add(
                batch,
                tokens[i + j],  // token
                j,              // 位置
                {0},            // 序列ID
                true            // 需要logits
            );
        }
        
        // 执行解码
        llama_decode(ctx, batch);
        
        // 计算每个位置的困惑度贡献
        for (size_t j = 0; j < len; j++) {
            // 获取第j个位置的logits
            const float * logits = llama_get_logits_ith(ctx, j);
            
            // 下一个token（这是我们要预测的目标）
            int next_token = tokens[i + j + 1];
            
            // 计算log概率
            auto res = log_softmax(n_vocab, logits, next_token);
            
            // 累积
            nll += -res.log_softmax;  // 负对数似然
            count++;
        }
        
        llama_batch_free(batch);
    }
    
    // 计算最终困惑度
    results.ppl_value = exp(nll / count);
    results.nll = nll;
    results.count = count;
}
```

### 25.3.3 使用场景

**场景1：评估模型质量**

```bash
# 基本用法
./llama-perplexity -m model.gguf -f wiki.test.raw

# 输出示例：
# Final perplexity: 8.5234
```

**场景2：对比不同量化类型**

```bash
#!/bin/bash

MODEL_BASE="llama-7b"
TEST_FILE="wiki.test.raw"

echo "Quantization | Perplexity | Size"
echo "-------------|------------|------"

for qtype in F16 Q8_0 Q5_K_M Q4_K_M Q4_0; do
    # 量化
    ./llama-quantize ${MODEL_BASE}-f16.gguf \
        ${MODEL_BASE}-${qtype}.gguf ${qtype}
    
    # 测试困惑度
    ppl=$(./llama-perplexity \
        -m ${MODEL_BASE}-${qtype}.gguf \
        -f ${TEST_FILE} 2>&1 | \
        grep "Final perplexity" | awk '{print $3}')
    
    # 获取大小
    size=$(ls -lh ${MODEL_BASE}-${qtype}.gguf | awk '{print $5}')
    
    echo "${qtype} | ${ppl} | ${size}"
done

# 输出：
# Quantization | Perplexity | Size
# -------------|------------|------
# F16          | 5.1234     | 13G
# Q8_0         | 5.1456     | 6.8G
# Q5_K_M       | 5.2876     | 5.1G
# Q4_K_M       | 5.3567     | 4.4G
# Q4_0         | 5.8923     | 3.8G
```

**场景3：使用imatrix提升质量**

```bash
# 1. 生成重要性矩阵
./llama-imatrix -m model-f16.gguf -f training.txt -o imatrix.dat

# 2. 使用imatrix量化
./llama-quantize --imatrix imatrix.dat \
    model-f16.gguf model-q4km-imat.gguf Q4_K_M

# 3. 对比
./llama-perplexity -m model-q4km.gguf -f test.txt
./llama-perplexity -m model-q4km-imat.gguf -f test.txt
```

---

## 25.4 GGUF文件操作 —— 解剖模型文件

### 25.4.1 GGUF信息查看

**源码位置**：`tools/gguf/gguf.cpp` (第1-200行)

```cpp
/**
 * 显示GGUF文件详细信息
 * 
 * 这个工具就像是模型文件的"X光机"，
 * 让你看清里面有什么、有多大、什么格式。
 */
void print_gguf_info(const std::string & fname) {
    // 加载GGUF文件（不分配张量内存）
    struct gguf_context * ctx = gguf_init_from_file(
        fname.c_str(),
        {
            true,      // no_alloc: 不分配张量内存
            nullptr,   // params
            true       // calc_hash: 计算文件哈希
        }
    );
    
    // ========== 文件信息 ==========
    printf("=== GGUF File Info ===\n\n");
    printf("File: %s\n", fname.c_str());
    printf("GGUF version: %d\n", gguf_get_version(ctx));
    printf("Alignment: %zu bytes\n", gguf_get_alignment(ctx));
    printf("File size: %.2f MB\n", gguf_get_file_size(ctx) / (1024.0 * 1024));
    
    // ========== 张量信息 ==========
    size_t n_tensors = gguf_get_n_tensors(ctx);
    printf("\n=== Tensors (%zu) ===\n", n_tensors);
    
    size_t total_params = 0;
    size_t total_bytes = 0;
    
    // 按类型分组统计
    std::map<ggml_type, int> type_counts;
    std::map<ggml_type, size_t> type_sizes;
    
    for (size_t i = 0; i < n_tensors; i++) {
        const char * name = gguf_get_tensor_name(ctx, i);
        enum ggml_type type = gguf_get_tensor_type(ctx, i);
        size_t n_elements = gguf_get_tensor_nelements(ctx, i);
        size_t tensor_size = gguf_get_tensor_size(ctx, i);
        
        total_params += n_elements;
        total_bytes += tensor_size;
        type_counts[type]++;
        type_sizes[type] += tensor_size;
        
        // 格式化维度显示
        const size_t * dims = gguf_get_tensor_dims(ctx, i);
        int n_dims = gguf_get_tensor_n_dims(ctx, i);
        
        std::string dim_str = "[";
        for (int j = 0; j < n_dims; j++) {
            dim_str += std::to_string(dims[j]);
            if (j < n_dims - 1) dim_str += ", ";
        }
        dim_str += "]";
        
        // 打印张量信息
        printf("  %-40s %6s %-20s %8zu bytes\n",
               name,
               ggml_type_name(type),
               dim_str.c_str(),
               tensor_size
        );
    }
    
    // ========== 统计汇总 ==========
    printf("\n=== Statistics ===\n");
    printf("Total parameters: %.3f B\n", total_params / 1e9);
    printf("Total tensor size: %.2f MB\n", total_bytes / (1024.0 * 1024));
    printf("Effective bits per weight: %.2f\n",
           8.0 * total_bytes / total_params);
    
    printf("\n=== Type Breakdown ===\n");
    for (const auto & [type, count] : type_counts) {
        printf("  %s: %d tensors, %.2f MB\n",
               ggml_type_name(type),
               count,
               type_sizes[type] / (1024.0 * 1024)
        );
    }
    
    // ========== 元数据 ==========
    printf("\n=== Metadata ===\n");
    size_t n_kv = gguf_get_n_kv(ctx);
    for (size_t i = 0; i < n_kv; i++) {
        const char * key = gguf_get_key(ctx, i);
        enum gguf_type type = gguf_get_kv_type(ctx, i);
        
        printf("  %s: ", key);
        switch (type) {
            case GGUF_TYPE_UINT8:
                printf("%u\n", gguf_get_val_u8(ctx, i));
                break;
            case GGUF_TYPE_INT32:
                printf("%d\n", gguf_get_val_i32(ctx, i));
                break;
            case GGUF_TYPE_UINT32:
                printf("%u\n", gguf_get_val_u32(ctx, i));
                break;
            case GGUF_TYPE_FLOAT32:
                printf("%f\n", gguf_get_val_f32(ctx, i));
                break;
            case GGUF_TYPE_STRING:
                printf("\"%s\"\n", gguf_get_val_str(ctx, i));
                break;
            case GGUF_TYPE_ARRAY:
                printf("[array]\n");
                break;
            default:
                printf("[unknown type %d]\n", type);
        }
    }
    
    gguf_free(ctx);
}
```

### 25.4.2 GGUF分割与合并

**源码位置**：`tools/gguf-split/gguf-split.cpp` (第1-300行)

```cpp
/**
 * 分割策略枚举
 */
enum split_mode {
    MODE_NONE,      // 无分割
    MODE_TENSOR,    // 按张量数量分割
    MODE_SIZE,      // 按文件大小分割
};

/**
 * 分割GGUF文件
 * 
 * 将大模型文件切分成多个小文件，便于存储和传输。
 * 
 * @param input 输入文件路径
 * @param output_prefix 输出文件前缀
 * @param mode 分割模式
 * @param max_size 最大文件大小（字节，用于MODE_SIZE）
 * @param max_tensors 最大张量数（用于MODE_TENSOR）
 */
void split_gguf(
    const std::string & input,
    const std::string & output_prefix,
    split_mode mode,
    size_t max_size,
    int max_tensors
) {
    // 加载源文件
    struct gguf_context * ctx = gguf_init_from_file(
        input.c_str(),
        {true}  // no_alloc
    );
    
    size_t n_tensors = gguf_get_n_tensors(ctx);
    size_t current_size = 0;
    int current_tensors = 0;
    int split_idx = 0;
    
    struct gguf_context * current_split = nullptr;
    
    for (size_t i = 0; i < n_tensors; i++) {
        const char * name = gguf_get_tensor_name(ctx, i);
        size_t tensor_size = gguf_get_tensor_size(ctx, i);
        
        // 检查是否需要创建新分割
        bool need_new_split = false;
        
        if (mode == MODE_SIZE && current_size + tensor_size > max_size) {
            // 大小超限
            need_new_split = true;
        }
        if (mode == MODE_TENSOR && current_tensors >= max_tensors) {
            // 张量数超限
            need_new_split = true;
        }
        
        if (need_new_split || current_split == nullptr) {
            // 保存当前分割
            if (current_split) {
                std::string fname = format(
                    "%s-%05d.gguf",
                    output_prefix.c_str(),
                    split_idx
                );
                printf("Writing %s...\n", fname.c_str());
                gguf_write_to_file(current_split, fname.c_str(), true);
                gguf_free(current_split);
                split_idx++;
            }
            
            // 创建新分割
            current_split = gguf_init_empty();
            // 复制元数据（所有分割共享相同的元数据）
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
        std::string fname = format(
            "%s-%05d.gguf",
            output_prefix.c_str(),
            split_idx
        );
        printf("Writing %s...\n", fname.c_str());
        gguf_write_to_file(current_split, fname.c_str(), true);
        gguf_free(current_split);
    }
    
    gguf_free(ctx);
    printf("Split complete: %d files\n", split_idx + 1);
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

LoRA（Low-Rank Adaptation）是一种高效的模型微调方法。
这个脚本将PyTorch格式的LoRA权重转换为llama.cpp可用的GGUF格式。
"""

import torch
import json
import struct
import numpy as np
from pathlib import Path
from typing import Dict, Tuple

def convert_tensor_name(name: str) -> str:
    """
    转换PyTorch张量名称为GGUF格式
    
    PyTorch: base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight
    GGUF:    blk.0.attn_q.lora_a
    """
    # 移除前缀
    name = name.replace("base_model.model.", "")
    
    # 转换层名称
    name = name.replace("model.layers.", "blk.")
    name = name.replace("self_attn.", "attn_")
    name = name.replace("mlp.", "ffn_")
    
    # 转换投影名称
    name = name.replace("q_proj", "q")
    name = name.replace("k_proj", "k")
    name = name.replace("v_proj", "v")
    name = name.replace("o_proj", "o")
    
    # 转换LoRA部分
    name = name.replace("lora_A", "lora_a")
    name = name.replace("lora_B", "lora_b")
    
    return name

def write_metadata(f, metadata: Dict):
    """写入GGUF元数据"""
    # 元数据数量
    f.write(struct.pack('<Q', len(metadata)))
    
    for key, value in metadata.items():
        # 键长度和键
        key_bytes = key.encode('utf-8')
        f.write(struct.pack('<Q', len(key_bytes)))
        f.write(key_bytes)
        
        # 类型和值
        if isinstance(value, str):
            f.write(struct.pack('<I', 8))  # GGUF_TYPE_STRING
            val_bytes = value.encode('utf-8')
            f.write(struct.pack('<Q', len(val_bytes)))
            f.write(val_bytes)
        elif isinstance(value, int):
            f.write(struct.pack('<I', 4))  # GGUF_TYPE_INT32
            f.write(struct.pack('<i', value))
        elif isinstance(value, float):
            f.write(struct.pack('<I', 5))  # GGUF_TYPE_FLOAT32
            f.write(struct.pack('<f', value))

def write_tensor_info(f, name: str, tensor: torch.Tensor):
    """写入张量信息头"""
    # 名称
    name_bytes = name.encode('utf-8')
    f.write(struct.pack('<Q', len(name_bytes)))
    f.write(name_bytes)
    
    # 维度数
    f.write(struct.pack('<I', len(tensor.shape)))
    
    # 维度
    for dim in tensor.shape:
        f.write(struct.pack('<Q', dim))
    
    # 类型（FP16）
    f.write(struct.pack('<I', 1))  # GGML_TYPE_F16

def convert_lora_to_gguf(
    input_path: str,
    output_path: str,
    base_model: str = None
):
    """
    转换LoRA适配器到GGUF格式
    
    Args:
        input_path: 输入LoRA目录
        output_path: 输出GGUF文件
        base_model: 基础模型名称（用于验证）
    """
    input_dir = Path(input_path)
    
    # 加载配置
    config_path = input_dir / "adapter_config.json"
    with open(config_path) as f:
        config = json.load(f)
    
    # 加载权重
    # 支持safetensors和bin格式
    safetensors_path = input_dir / "adapter_model.safetensors"
    bin_path = input_dir / "adapter_model.bin"
    
    if safetensors_path.exists():
        from safetensors.torch import load_file
        state_dict = load_file(safetensors_path)
    else:
        state_dict = torch.load(bin_path, map_location="cpu")
    
    print(f"Loaded {len(state_dict)} tensors")
    
    # 写入GGUF文件
    with open(output_path, 'wb') as f:
        # ===== 头部 =====
        # 魔数
        f.write(b'GGUF')
        
        # 版本（最新是3）
        f.write(struct.pack('<I', 3))
        
        # 张量数量
        f.write(struct.pack('<Q', len(state_dict)))
        
        # ===== 元数据 =====
        metadata = {
            'general.architecture': config.get('base_model_name_or_path', 'unknown'),
            'general.lora.alpha': config.get('lora_alpha', 16),
            'general.lora.r': config.get('r', 8),
            'general.lora.target_modules': ','.join(config.get('target_modules', [])),
            'general.lora.bias': config.get('bias', 'none'),
        }
        write_metadata(f, metadata)
        
        # ===== 张量信息 =====
        # 计算对齐后的数据偏移
        tensor_data_offset = f.tell() + sum(
            8 + len(convert_tensor_name(name)) + 4 + 8 * len(tensor.shape) + 4
            for name, tensor in state_dict.items()
        )
        # 对齐到32字节
        tensor_data_offset = (tensor_data_offset + 31) // 32 * 32
        f.write(struct.pack('<Q', tensor_data_offset))
        
        # 写入每个张量的信息
        for name, tensor in state_dict.items():
            gguf_name = convert_tensor_name(name)
            write_tensor_info(f, gguf_name, tensor)
        
        # 对齐到32字节
        current_pos = f.tell()
        padding = (32 - current_pos % 32) % 32
        f.write(b'\x00' * padding)
        
        # ===== 张量数据 =====
        for name, tensor in state_dict.items():
            # 转换为FP16并写入
            tensor_fp16 = tensor.half().numpy()
            f.write(tensor_fp16.tobytes())
    
    print(f"Converted to {output_path}")

# 命令行接口
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Convert LoRA adapter to GGUF format"
    )
    parser.add_argument("--input", "-i", required=True,
                       help="Input LoRA directory")
    parser.add_argument("--output", "-o", required=True,
                       help="Output GGUF file")
    parser.add_argument("--base-model",
                       help="Base model name for validation")
    
    args = parser.parse_args()
    
    convert_lora_to_gguf(args.input, args.output, args.base_model)
```

### 25.5.2 LoRA使用流程

```bash
# 1. 从HuggingFace下载LoRA适配器
huggingface-cli download user/lora-adapter --local-dir ./lora-adapter

# 2. 转换为GGUF格式
python convert_lora_to_gguf.py \
    --input ./lora-adapter \
    --output ./adapter.gguf \
    --base-model llama-7b

# 3. 使用LoRA进行推理（动态加载）
./llama-cli \
    -m base-model.gguf \
    --lora adapter.gguf \
    --lora-scale 0.8 \
    -p "Your prompt here"

# 4. 或者合并LoRA到基础模型（永久应用）
./llama-export-lora \
    -m base-model.gguf \
    --lora adapter.gguf \
    --lora-scale 0.8 \
    -o merged-model.gguf

# 合并后的模型可以直接使用，无需再指定LoRA
./llama-cli -m merged-model.gguf -p "Your prompt"
```

---

## 设计中的取舍

### 为什么工具要分开而不是一个"瑞士军刀"？

llama.cpp选择了将不同功能拆分为独立工具而非合并成单一的可执行文件，这是Unix哲学"做好一件事"的体现。如果将所有功能合并为一个工具，优点是统一接口、学习成本低——用户只需记住一个命令。但代价也很明显：二进制体积变大，启动时加载许多当前不需要的代码，启动速度从不到100ms增加到超过500ms。更重要的是，不同工具的资源需求差异极大——分词工具只需要加载词汇表（几MB），而困惑度计算需要完整加载模型权重（几GB）并执行推理。单一工具无法针对这种差异进行优化，用户每次运行分词都要等待模型加载吗？那将是一场灾难。

分离工具的优势体现在多个层面。首先是启动速度和内存效率——工具只加载它真正需要的功能，分词工具启动时间不到100ms且只占用几MB内存。其次是可组合性——Unix管道可以将工具串联成强大的工作流，比如将分词输出传给Python脚本进行分析，将嵌入输出重定向到向量数据库。最后是独立演进——添加新功能或修复bug只需修改对应工具，不会影响生态系统中的其他组件。llama.cpp也部分使用了库+脚本的方案（如Python转换脚本），但这主要用于开发效率和生态整合的场景，而非性能敏感的运行时工具。

### 为什么Python用于转换脚本？

llama.cpp的核心推理引擎使用C++以实现最佳性能，但转换脚本选择了Python。这个决策体现了"在正确的地方使用正确的语言"的原则。用C++编写模型格式转换器面临几个难题：PyTorch的safetensors和.bin格式解析复杂，需要大量代码来处理各种张量布局和数据类型；而且为了读取HuggingFace的config.json和tokenizer.json配置，需要在C++中实现JSON解析和推理。Python在这方面有天然优势——可以直接使用torch和safetensors库读取checkpoint，使用transformers库处理模型配置，几行代码就能完成C++需要上百行才能实现的功能。

Python的第二个优势是开发迭代速度。转换脚本需要支持50+种模型架构，每种架构都有独特的配置格式和张量命名规范。用Python开发和调试新架构的支持只需几分钟到几小时，而C++的开发周期会显著更长。第三个优势是社区参与——HuggingFace生态的用户主要使用Python，使用相同的语言编写转换脚本降低了贡献门槛。架构上的整体决策清晰：性能敏感的运行时工具使用C++，而开发效率和生态整合优先的转换脚本使用Python——两者各司其职，通过GGUF格式作为桥接。

---

## 动手练习

### 练习1：分词分析

使用llama-tokenize分析不同模型的分词差异：

```bash
# 1. 对比不同模型的分词
./llama-tokenize -m llama3.gguf -p "ChatGPT" --ids
./llama-tokenize -m llama2.gguf -p "ChatGPT" --ids
./llama-tokenize -m mistral.gguf -p "ChatGPT" --ids

# 2. 分析多语言分词效率
for text in "Hello world" "你好世界" "こんにちは" "Bonjour le monde"; do
    count=$(./llama-tokenize -m model.gguf -p "$text" --show-count 2>&1 | grep "Total" | awk '{print $3}')
    echo "\"$text\": $count tokens"
done

# 3. 检查特殊Token
./llama-tokenize -m model.gguf -p "<|im_start|>user<|im_end|>" --ids
./llama-tokenize -m model.gguf -p "<s>Hello</s>" --ids
```

### 练习2：嵌入相似度计算

编写脚本计算文本相似度：

```python
#!/usr/bin/env python3
"""
使用llama-embedding计算文本相似度
"""
import subprocess
import numpy as np
from typing import List

def get_embedding(text: str, model: str = "model.gguf") -> np.ndarray:
    """获取文本的嵌入向量"""
    result = subprocess.run(
        ['./llama-embedding', '-m', model, '-p', text, '--pooling', 'mean'],
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        raise RuntimeError(f"Embedding failed: {result.stderr}")
    
    values = [float(x) for x in result.stdout.strip().split()]
    return np.array(values)

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """计算余弦相似度"""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def main():
    # 测试文本对
    pairs = [
        ("King", "Queen"),
        ("King", "Man"),
        ("King", "Apple"),
        ("Machine learning", "Artificial intelligence"),
        ("Machine learning", "Banana"),
    ]
    
    print("Computing embeddings...")
    embeddings = {}
    for a, b in pairs:
        if a not in embeddings:
            embeddings[a] = get_embedding(a)
        if b not in embeddings:
            embeddings[b] = get_embedding(b)
    
    print("\nSimilarity Results:")
    print("-" * 60)
    for a, b in pairs:
        sim = cosine_similarity(embeddings[a], embeddings[b])
        print(f"{a:30s} vs {b:30s}: {sim:.4f}")

if __name__ == "__main__":
    main()
```

### 练习3：量化质量评估

使用perplexity评估不同量化配置：

```bash
#!/bin/bash

MODEL_BASE="llama-7b"
TEST_FILE="wiki.test.raw"

echo "=== Quantization Quality Benchmark ==="
echo ""

# 创建结果文件
echo "method,ppl,size_mb" > results.csv

for method in F16 Q8_0 Q5_K_M Q4_K_M Q4_0 Q3_K_M; do
    echo "Testing $method..."
    
    # 量化
    ./llama-quantize ${MODEL_BASE}-f16.gguf \
        ${MODEL_BASE}-${method}.gguf ${method} 2>/dev/null
    
    # 计算困惑度
    ppl=$(./llama-perplexity \
        -m ${MODEL_BASE}-${method}.gguf \
        -f ${TEST_FILE} 2>&1 | \
        grep "Final perplexity" | awk '{print $3}')
    
    # 获取大小
    size_bytes=$(stat -f%z ${MODEL_BASE}-${method}.gguf 2>/dev/null || \
                 stat -c%s ${MODEL_BASE}-${method}.gguf 2>/dev/null)
    size_mb=$(echo "scale=2; $size_bytes / 1024 / 1024" | bc)
    
    echo "$method,$ppl,$size_mb" >> results.csv
    echo "  PPL: $ppl, Size: ${size_mb}MB"
done

echo ""
echo "Results saved to results.csv"
echo ""
column -s, -t results.csv
```

---

## 本课小结

本课深入解析了llama.cpp的辅助工具及其典型用法。llama-tokenize用于分词调试，典型命令为 `-m model -p "text" --ids`，输出Token列表。llama-embedding用于嵌入生成，典型命令为 `-m model -p "text" --pooling mean`，输出向量表示。llama-perplexity用于质量评估，典型命令为 `-m model -f test.txt`，输出PPL分数。gguf工具用于查看文件信息，典型命令为 `gguf-dump model.gguf`，输出元数据。gguf-split用于文件分割，典型命令为 `--split input.gguf output`，输出多个GGUF文件。convert_lora_to_gguf.py用于LoRA转换，典型命令为 `--input lora/ --output adapter.gguf`，输出GGUF格式适配器。

工具选择指南：调试模板问题推荐使用 llama-tokenize；构建RAG系统推荐使用 llama-embedding；评估量化质量推荐使用 llama-perplexity；管理大模型文件推荐使用 gguf-split；使用社区LoRA推荐使用 convert_lora_to_gguf.py。

本章我们一起学习了以下概念：

| 概念 | 解释 |
|------|------|
| llama-tokenize | 分词调试工具，展示文本如何被切分为token，支持多种输出格式和BOS/特殊token控制 |
| llama-embedding | 嵌入生成工具，支持多种池化策略（mean/cls/last），用于语义搜索和RAG系统 |
| llama-perplexity | 困惑度评估工具，通过滑动窗口计算模型对文本的预测质量，量化质量的关键指标 |
| gguf-split | GGUF文件分割工具，按大小或张量数将大模型切分，便于存储和传输 |
| log_softmax | 数值稳定的softmax对数计算，通过减去最大值避免溢出，是困惑度计算的基础 |
| 池化策略 | 从变长token序列生成固定大小向量的方法，不同策略适用于不同任务类型 |

下一章中，我们将学习模型转换与生态——从HuggingFace到GGUF的完整转换流程，涵盖架构检测、张量映射、量化和gguf-py库的使用。

---

## 关联阅读

- **第12章**：GGUF文件格式详解
- **第19章**：LoRA适配器完整解析
- **第24章**：llama-quantize和llama-bench使用
- **LLaMA.cpp Wiki**：`docs/tools.md`

---

*本章对应源码版本：master (2026-04-07)*
*本章作者：llama.cpp社区*
*最后更新：2026-04-08*
