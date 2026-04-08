# 第13章 分词器架构（llama_vocab） —— 文本与Token的"翻译官"

大语言模型无法直接处理原始文本，需要通过分词器（Tokenizer）将文本转换为数字token序列。llama.cpp的`llama_vocab`模块支持多种分词算法（BPE、SentencePiece等），是模型与文本世界之间的桥梁。

## 学习目标

1. 理解`llama_vocab`的核心结构和设计
2. 掌握BPE、SentencePiece等分词算法的实现
3. 了解特殊Token的管理和使用
4. 掌握tokenize和detokenize的完整流程

## 生活类比：多语言翻译局的"编码专家"

想象`llama_vocab`是一家**精通百种语言的翻译局**：

- **词表（Vocabulary）** = 翻译字典
  - 每个token是一个"词汇条目"
  - 包含文本内容、分数（频率）、属性
  - 不同模型有不同大小的字典（32K到200K不等）

- **分词（Tokenization）** = 编码过程
  - 将人类可读的文本 → 模型可理解的token ID
  - 类似摩斯电码编码
  - "Hello" → [15496, 11]（可能对应"Hell"+"o"）

- **反分词（Detokenization）** = 解码过程
  - 将token ID序列 → 人类可读的文本
  - 类似摩斯电码解码
  - [15496, 11] → "Hello,"

- **预分词（Pre-tokenization）** = 文本预处理
  - 按正则表达式切分文本
  - 处理Unicode、空格、标点等

就像翻译局需要精确处理每种语言的特性，分词器需要处理不同模型的词表格式。

---

## 13.1 词表结构设计

### 13.1.1 Token数据结构

**源码位置**：`src/llama-vocab.h`（第68-72行）

```cpp
struct llama_vocab {
    struct token_data {
        std::string      text;   // token文本内容
        float            score;  // 分数（BPE合并优先级）
        llama_token_attr attr;   // 属性（正常/控制/未知等）
    };

    // 核心查询方法
    llama_token text_to_token(const std::string & text) const;
    const token_data & get_token_data(llama_token id) const;
    
    // 特殊token获取
    llama_token token_bos() const;  // 开始符
    llama_token token_eos() const;  // 结束符
    llama_token token_unk() const;  // 未知词
};

这段代码定义了llama_vocab词表结构，包含token数据结构（文本、分数、属性）和核心查询方法（text_to_token将文本转为token，get_token_data获取token信息），以及获取特殊token（BOS/EOS/UNK）的便捷方法。

**Token属性定义**：
```cpp
enum llama_token_attr {
    LLAMA_TOKEN_ATTR_UNDEFINED    = 0,
    LLAMA_TOKEN_ATTR_NORMAL       = 1 << 0,  // 普通token
    LLAMA_TOKEN_ATTR_UNKNOWN      = 1 << 1,  // 未知token
    LLAMA_TOKEN_ATTR_CONTROL      = 1 << 2,  // 控制token（如<EOS>）
    LLAMA_TOKEN_ATTR_USER_DEFINED = 1 << 3,
    LLAMA_TOKEN_ATTR_BYTE         = 1 << 4,  // 字节token（用于BPE）
    LLAMA_TOKEN_ATTR_UNUSED       = 1 << 5,
};

这段代码定义了token的属性枚举，包括普通token、未知token、控制token（如EOS）、用户自定义token、字节token（用于BPE字节回退）等属性标志，用于区分不同类型token的特殊处理方式。

### 13.1.2 预分词类型

**源码位置**：`src/llama-vocab.h`（第10-62行）

```cpp
enum llama_vocab_pre_type {
    LLAMA_VOCAB_PRE_TYPE_DEFAULT         = 0,   // 默认
    LLAMA_VOCAB_PRE_TYPE_LLAMA3          = 1,   // Llama3风格
    LLAMA_VOCAB_PRE_TYPE_DEEPSEEK_LLM    = 2,   // DeepSeek
    LLAMA_VOCAB_PRE_TYPE_FALCON          = 4,   // Falcon
    LLAMA_VOCAB_PRE_TYPE_GPT2            = 7,   // GPT-2
    LLAMA_VOCAB_PRE_TYPE_QWEN2           = 11,  // Qwen2
    LLAMA_VOCAB_PRE_TYPE_KIMI_K2         = 37,  // Kimi K2
    // ... 更多类型
};

这段代码定义了预分词类型枚举，不同模型使用不同的预分词策略，如LLAMA3风格、DeepSeek、Falcon、GPT-2、Qwen2、Kimi K2等，每种类型对应特定的正则表达式切分规则。

**不同预分词类型的区别**：

| 类型 | 特点 | 代表模型 |
|-----|------|---------|
| GPT2 | 原始BPE，字节级 | GPT-2, 早期模型 |
| LLAMA3 | 改进的BPE，更好的多语言 | Llama3 |
| QWEN2 | 中文优化 | Qwen系列 |
| KIMI_K2 | 汉字单独切分 | Kimi K2 |

---

## 13.2 分词算法实现

### 13.2.1 BPE（Byte-Pair Encoding）

**算法原理**：
1. 从字符级开始
2. 统计最频繁的字符对
3. 合并成新token
4. 重复直到词表大小达标

**源码位置**：`src/llama-vocab.cpp`（第1000-1500行）

```cpp
// BPE分词核心
std::vector<llama_token> llama_vocab::tokenize_bpe(
        const std::string & text,
        bool add_special) const {

    // ① 预分词：按正则切分
    auto words = pre_tokenize(text);

    std::vector<llama_token> result;

    for (const auto & word : words) {
        // ② 字节编码：将UTF-8转为字节token
        auto bytes = byte_encode(word);

        // ③ BPE合并
        while (bytes.size() > 1) {
            // 查找最佳合并对
            auto best_pair = find_best_merge_pair(bytes);
            if (best_pair.first == -1) break;

            // 执行合并
            bytes = merge_pair(bytes, best_pair);
        }

        // ④ 添加到结果
        for (const auto & b : bytes) {
            result.push_back(byte_to_token(b));
        }
    }

    return result;
}

// 查找最佳合并对
std::pair<int, int> find_best_merge_pair(
        const std::vector<uint8_t> & bytes) {
    std::pair<int, int> best = {-1, -1};
    float best_score = -INFINITY;

    for (size_t i = 0; i < bytes.size() - 1; i++) {
        // 查找这对字节在BPE合并表中的分数
        auto pair = std::make_pair(bytes[i], bytes[i+1]);
        auto it = bpe_merges.find(pair);

        if (it != bpe_merges.end() && it->second > best_score) {
            best_score = it->second;
            best = {i, i+1};
        }
    }

    return best;
}
```

这段代码实现了BPE分词的核心算法。首先进行预分词将文本切分为单词，然后对每个单词进行字节编码，接着迭代查找最佳字节对（分数最高的合并对）并执行合并，直到无法继续合并为止，最终将字节序列转换为token ID。

**BPE分词示例**：
```
输入: "hello"

步骤1: 字节编码
  h(68) e(65) l(6C) l(6C) o(6F)
  → [<68>, <65>, <6C>, <6C>, <6F>]

步骤2: 查找合并对（假设BPE表中有）
  <6C>+<6C> = "ll" (分数最高)
  
步骤3: 合并
  [<68>, <65>, "ll", <6F>]

步骤4: 继续合并（假设）
  <68>+<65> = "he"
  → ["he", "ll", <6F>]

步骤5: 最终合并
  "he"+"ll" = "hell"
  → ["hell", <6F>]

输出tokens: ["hell", "o"]
```

### 13.2.2 SentencePiece（SPM）

**算法原理**：
1. 从初始词表（所有字符）开始
2. 使用EM算法训练
3. 基于Viterbi算法解码
4. 输出最可能的token序列

**源码位置**：`src/llama-vocab.cpp`（第2000-2500行）

```cpp
// SPM分词核心
std::vector<llama_token> llama_vocab::tokenize_spm(
        const std::string & text,
        bool add_special) const {

    // ① 规范化（NFC/NFKC）
    auto normalized = normalize(text);

    // ② Viterbi解码
    // dp[i] = 到位置i的最佳分割
    std::vector<float> dp(normalized.size() + 1, -INFINITY);
    std::vector<int> prev(normalized.size() + 1, -1);
    dp[0] = 0;

    for (size_t i = 0; i < normalized.size(); i++) {
        if (dp[i] == -INFINITY) continue;

        // 尝试所有可能的子串
        for (size_t j = i + 1; j <= normalized.size() && j <= i + max_token_len; j++) {
            auto sub = normalized.substr(i, j - i);
            auto token = text_to_token(sub);

            if (token != LLAMA_TOKEN_NULL) {
                float score = get_token_data(token).score;
                if (dp[i] + score > dp[j]) {
                    dp[j] = dp[i] + score;
                    prev[j] = token;
                }
            }
        }
    }

    // ③ 回溯得到token序列
    std::vector<llama_token> result;
    int pos = normalized.size();
    while (pos > 0) {
        result.push_back(prev[pos]);
        pos -= get_token_data(prev[pos]).text.size();
    }

    std::reverse(result.begin(), result.end());
    return result;
}

这段代码实现了SentencePiece分词算法，使用Viterbi动态规划算法进行解码。首先规范化文本，然后计算每个位置的最佳分割分数（dp数组），记录前驱token（prev数组），最后回溯得到最优token序列。

### 13.2.3 预分词处理

**源码位置**：`src/unicode.cpp`（第946-1138行）

```cpp
// 预分词：使用正则表达式切分文本
std::vector<std::string> unicode_regex_split(
        const std::string & text,
        const std::vector<std::string> & regex_exprs,
        bool byte_encode) {

    // ① 将文本转为codepoints
    const auto cpts = unicode_cpts_from_utf8(text);

    // ② 应用每个正则表达式
    std::vector<size_t> bpe_offsets = { cpts.size() };

    for (const auto & regex_expr : regex_exprs) {
        // 尝试使用优化的自定义实现
        auto tmp = unicode_regex_split_custom(text, regex_expr, bpe_offsets);

        if (!tmp.empty()) {
            bpe_offsets = std::move(tmp);
            continue;
        }

        // 回退到std::regex
        bpe_offsets = unicode_regex_split_stl(text, regex_expr, bpe_offsets);
    }

    // ③ 将offsets转为字符串
    std::vector<std::string> bpe_words;
    size_t start = 0;
    for (size_t & offset : bpe_offsets) {
        bpe_words.emplace_back();
        for (size_t i = start; i < start + offset; i++) {
            bpe_words.back() += unicode_cpt_to_utf8(cpts[i]);
        }
        start += offset;
    }

    return bpe_words;
}

这段代码实现了基于正则表达式的预分词处理。首先将UTF-8文本转换为codepoints数组，然后依次应用每个正则表达式进行切分，优先使用优化的自定义实现，如果不支持则回退到std::regex，最后将切分结果转换回字符串列表。

**常见预分词正则**：

| 模型 | 正则表达式 | 特点 |
|-----|-----------|------|
| GPT-2 | `'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+` | 处理缩写 |
| Llama3 | `(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}|...` | 改进数字处理 |
| Kimi K2 | `[\p{Han}]+|...` | 汉字单独切分 |

---

## 13.3 特殊Token管理

### 13.3.1 特殊Token类型

**源码位置**：`src/llama-vocab.h`（第109-128行）

```cpp
struct llama_vocab {
    // 基本特殊token
    llama_token token_bos() const;   // Beginning of Sequence
    llama_token token_eos() const;   // End of Sequence
    llama_token token_eot() const;   // End of Turn
    llama_token token_unk() const;   // Unknown
    llama_token token_pad() const;   // Padding

    // 对话相关
    llama_token token_nl() const;    // New Line
    llama_token token_sep() const;   // Separator

    // FIM (Fill-In-the-Middle)代码补全
    llama_token token_fim_pre() const;  // 前缀标记
    llama_token token_fim_suf() const;  // 后缀标记
    llama_token token_fim_mid() const;  // 中间标记
};

这段代码展示了llama_vocab中特殊token的获取方法，包括基本特殊token（BOS/EOS/EOT/UNK/PAD）、对话相关token（换行符、分隔符）以及FIM代码补全token（前缀/后缀/中间标记）。

### 13.3.2 特殊Token添加策略

**源码位置**：`src/llama-vocab.cpp`

```cpp
int32_t llama_vocab::tokenize(
               const char * text,
               int32_t   text_len,
           llama_token * tokens,
               int32_t   n_tokens_max,
                  bool   add_special,    // 是否添加特殊token
                  bool   parse_special   // 是否解析文本中的特殊token
               ) const {

    // ① 添加BOS（如果配置要求）
    if (add_special && get_add_bos()) {
        tokens[n_tokens++] = token_bos();
    }

    // ② 主分词逻辑
    auto raw_tokens = tokenize_internal(text, text_len, parse_special);

    // ③ 添加EOS（如果配置要求）
    if (add_special && get_add_eos()) {
        raw_tokens.push_back(token_eos());
    }

    // ④ 复制到输出
    for (size_t i = 0; i < raw_tokens.size() && n_tokens < n_tokens_max; i++) {
        tokens[n_tokens++] = raw_tokens[i];
    }

    return n_tokens;
}
```

这段代码实现了完整的tokenize函数，处理特殊token的添加逻辑。首先根据配置添加BOS token，然后执行主分词逻辑，再根据配置添加EOS token，最后将结果复制到输出缓冲区，返回实际生成的token数量。

**不同模型的特殊Token策略**：

| 模型 | BOS | EOS | 其他特点 |
|-----|-----|-----|---------|
| Llama2 | 是 | 否 | 有特殊的INST标记 |
| Llama3 | 是 | 是 | 使用<\|eot_id\|> |
| GPT-2 | 否 | 否 | 无特殊处理 |
| Qwen | 是 | 是 | 支持im_start/end |

---

## 13.4 反分词（Detokenization）

### 13.4.1 基本反分词

**源码位置**：`src/llama-vocab.h`（第161-182行）

```cpp
struct llama_vocab {
    // 单个token转文本
    int32_t token_to_piece(
              llama_token   token,
                     char * buf,
                  int32_t   length,
                  int32_t   lstrip,     // 左侧剥离空格数
                     bool   special      // 是否包含特殊token
                     ) const;

    // 批量反分词
    std::string detokenize(
            const std::vector<llama_token> & tokens,
                                      bool   special) const;
};
```

### 13.4.2 反分词实现

```cpp
std::string llama_vocab::detokenize(
        const std::vector<llama_token> & tokens,
        bool special) const {

    std::string result;

    for (const auto & token : tokens) {
        // 跳过特殊token（如果不需要）
        if (!special && is_control(token)) {
            continue;
        }

        // 获取token文本
        const auto & data = get_token_data(token);
        std::string piece = data.text;

        // 处理BPE字节token
        if (is_byte(token)) {
            piece = token_to_byte(token);
        }

        // 处理SentencePiece空格标记
        if (get_type() == LLAMA_VOCAB_TYPE_SPM) {
            // SPM用"▁"表示词首空格
            if (!piece.empty() && piece[0] == '\xe2\x96\x81') {
                piece[0] = ' ';
            }
        }

        result += piece;
    }

    return result;
}
```

### 13.4.3 反分词示例

```
Tokens: [15496, 11, 616, 1438, 318, 13779]

步骤1: 逐个转换
  15496 → "hello"
  11    → ","
  616   → " how"
  1438  → " are"
  318   → " you"
  13779 → "?"

步骤2: 拼接
  "hello" + "," + " how" + " are" + " you" + "?"

步骤3: 处理SPM空格
  "hello, how are you?"

输出: "hello, how are you?"
```

---

## 13.5 Unicode支持

### 13.5.1 Unicode字符属性

**源码位置**：`src/unicode.h`（第8-90行）

```cpp
struct unicode_cpt_flags {
    enum {
        UNDEFINED       = 0x0001,
        NUMBER          = 0x0002,  // \p{N} - 数字
        LETTER          = 0x0004,  // \p{L} - 字母
        SEPARATOR       = 0x0008,  // \p{Z} - 分隔符
        ACCENT_MARK     = 0x0010,  // \p{M} - 重音标记
        PUNCTUATION     = 0x0020,  // \p{P} - 标点
        SYMBOL          = 0x0040,  // \p{S} - 符号
        CONTROL         = 0x0080,  // \p{C} - 控制字符
        WHITESPACE      = 0x0100,
        LOWERCASE       = 0x0200,
        UPPERCASE       = 0x0400,
        NFD             = 0x0800,  // 规范化形式D
    };

    // 位域存储
    uint16_t is_undefined   : 1;
    uint16_t is_number      : 1;
    uint16_t is_letter      : 1;
    // ... 更多字段
};

// 查询字符属性
unicode_cpt_flags unicode_cpt_flags_from_cpt(uint32_t cpt);
```

### 13.5.2 UTF-8编解码

**源码位置**：`src/unicode.cpp`（第16-65行）

```cpp
// UTF-8编码长度查询
size_t unicode_len_utf8(char src) {
    const size_t lookup[] = { 1, 1, 1, 1, 1, 1, 1, 1, 
                              1, 1, 1, 1, 2, 2, 3, 4 };
    uint8_t highbits = static_cast<uint8_t>(src) >> 4;
    return lookup[highbits];
}

// UTF-8解码
uint32_t unicode_cpt_from_utf8(const std::string & utf8, size_t & offset) {
    // 单字节 (0xxxxxxx)
    if (!(utf8[offset + 0] & 0x80)) {
        auto result = utf8[offset + 0];
        offset += 1;
        return result;
    }

    // 双字节 (110xxxxx 10xxxxxx)
    if (!(utf8[offset + 0] & 0x20)) {
        auto result = ((utf8[offset + 0] & 0x1f) << 6) | 
                       (utf8[offset + 1] & 0x3f);
        offset += 2;
        return result;
    }

    // 三字节 (1110xxxx 10xxxxxx 10xxxxxx)
    if (!(utf8[offset + 0] & 0x10)) {
        auto result = ((utf8[offset + 0] & 0x0f) << 12) | 
                      ((utf8[offset + 1] & 0x3f) << 6) | 
                       (utf8[offset + 2] & 0x3f);
        offset += 3;
        return result;
    }

    // 四字节 (11110xxx 10xxxxxx 10xxxxxx 10xxxxxx)
    // ... 类似处理
}
```

---

## 13.6 设计中的取舍

### 为什么需要多种分词算法？

**BPE的优点**：
- 简单高效
- 可重现性好
- 适合英语等空格分隔语言

**SPM的优点**：
- 语言无关（不需要预分词）
- 直接处理原始文本
- 适合日语、中文等无空格语言

**llama.cpp的选择**：
- 同时支持BPE和SPM
- 根据模型配置自动选择
- 统一接口隐藏实现差异

### 预分词的性能优化

**问题**：std::regex在处理Unicode时很慢

**解决方案**：
1. **自定义实现**：针对常见正则写专用代码
2. **字符折叠**：将Unicode字符映射到单字节处理
3. **缓存优化**：预计算字符属性表

---

## 13.7 动手练习

### 练习1：理解BPE合并

给定BPE合并表：
```
("h", "e"): 0.5
("he", "l"): 0.4
("l", "l"): 0.6
("hel", "l"): 0.3
("hell", "o"): 0.7
```

将"hello"分词，写出每一步合并过程。

**答案**：
```
初始: [h, e, l, l, o]
步骤1: 合并(l,l)→ll (分数最高0.6)
  → [h, e, ll, o]
步骤2: 合并(h,e)→he (分数0.5)
  → [he, ll, o]
步骤3: 合并(he,ll)→hell (分数0.3)
  → [hell, o]
步骤4: 合并(hell,o)→hello (分数0.7)
  → [hello]

最终tokens: [hello]
```

### 练习2：计算词表内存占用

给定词表配置：
- vocab_size = 128000
- 平均token长度 = 8字节
- 每个token存储text、score、attr

估算总内存占用。

**答案**：
```
text: 128000 * 8 = 1 MB
score: 128000 * 4 = 512 KB
attr: 128000 * 4 = 512 KB
hash表开销: ~2 MB
总计: ~4-5 MB
```

---

## 13.8 本章小结

| 概念 | 一句话总结 |
|------|-----------|
| **llama_vocab** | 词表管理类，封装tokenize/detokenize |
| **BPE** | 字节对编码，迭代合并最频繁字符对 |
| **SPM** | SentencePiece，基于Viterbi解码 |
| **预分词** | 正则切分，处理Unicode和特殊字符 |
| **特殊token** | BOS/EOS/UNK等控制token |
| **Unicode支持** | UTF-8编解码，字符属性查询 |

**下一步预告**：

在第14章，我们将探索采样策略——理解如何让模型生成多样化且高质量的文本。
