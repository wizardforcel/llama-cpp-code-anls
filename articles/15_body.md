# 第15章 Unicode与文本处理 —— 多语言支持的"幕后功臣"

当AI助手用流利的日语回复你，或者准确理解一句阿拉伯语问候时，你可曾想过这些字符是如何在计算机中表示和处理的？从古老的象形文字到现代的emoji表情，Unicode标准为全球所有书写系统提供了统一的编码方案。而UTF-8作为互联网的事实标准，以巧妙的变长编码平衡了空间效率和兼容性。llama.cpp的Unicode处理系统就是支撑多语言能力的幕后功臣，它像一位精通全球语言的专家，确保每一种文字都能被正确理解和处理。

## 学习目标

1. 理解Unicode在llama.cpp中的核心作用
2. 掌握UTF-8编解码实现原理
3. 了解Unicode字符属性查询系统
4. 理解正则表达式在预分词中的应用

## 生活类比：国际会议的"同声传译团队"

想象Unicode处理系统是**国际会议的同声传译团队**：

- **UTF-8编码** = 多语言翻译手册
  - 英文（ASCII）：1字节，快速简洁
  - 欧洲语言：2字节，扩展字符
  - 中日韩：3字节，复杂表意文字
  - 特殊符号：4字节，全覆盖

- **Codepoint** = 通用概念编号
  - 每个字符有唯一的"身份证号"
  - 'A' = U+0041
  - '中' = U+4E2D
  - '😀' = U+1F600

- **Unicode属性** = 语言学家分析
  - 这是字母？数字？标点？
  - 是否空白？是否大写？
  - 用于智能分词和文本处理

- **正则表达式** = 翻译规则手册
  - 按语言特性切分文本
  - 不同语言有不同规则

就像传译团队需要处理各种语言的细微差别，Unicode系统需要正确处理全球各种文字的编码和特性。一个音节文字的发音规则、一个表意文字的部首结构、一个emoji的组合序列——所有这些都需要精确的处理。

## 源码地图

```
src/unicode.h
├── unicode_cpt_flags          # Unicode字符属性结构
│   ├── NUMBER, LETTER, PUNCTUATION  # 基本类别
│   ├── WHITESPACE, LOWERCASE, UPPERCASE  # 辅助属性
│   └── NFD (规范化形式D)      # 字符规范化
├── unicode_len_utf8()         # UTF-8编码长度查询
├── unicode_cpt_to_utf8()      # Codepoint转UTF-8
├── unicode_cpt_from_utf8()    # UTF-8解码
└── unicode_regex_split()      # Unicode正则切分

src/unicode.cpp
├── UTF-8编解码实现 (第16-65行, 第818-845行)
├── Unicode属性查询 (第116-146行, 第877-890行)
├── 正则切分实现 (第196-1138行)
│   ├── unicode_regex_split_custom_gpt2()   # GPT-2风格
│   ├── unicode_regex_split_custom_llama3() # Llama3风格
│   └── unicode_regex_split_custom_kimi_k2() # Kimi K2风格
└── 字符工具函数 (第892-944行)
    ├── unicode_byte_to_utf8()   # 字节转UTF-8
    ├── unicode_tolower()        # 转小写
    └── unicode_cpt_is_han()     # 判断是否汉字
```

## 15.1 UTF-8编解码

### 15.1.1 UTF-8编码原理

UTF-8（8-bit Unicode Transformation Format）是一种变长字符编码，由Ken Thompson和Rob Pike在1992年设计。它的设计非常巧妙：

1. **向后兼容ASCII**：所有ASCII字符（U+0000-U+007F）使用单字节编码，与ASCII完全一致
2. **自同步**：可以通过首字节判断字符长度，无需从开头扫描
3. **紧凑**：常用字符（如拉丁字母、CJK）使用较少字节

UTF-8根据字符的Unicode码点（codepoint）选择1-4字节编码：

| 码点范围 | 字节数 | 编码格式 | 实际可用位数 |
|---------|-------|---------|-------------|
| U+0000 - U+007F | 1 | `0xxxxxxx` | 7位 |
| U+0080 - U+07FF | 2 | `110xxxxx 10xxxxxx` | 11位 |
| U+0800 - U+FFFF | 3 | `1110xxxx 10xxxxxx 10xxxxxx` | 16位 |
| U+10000 - U+10FFFF | 4 | `11110xxx 10xxxxxx 10xxxxxx 10xxxxxx` | 21位 |

**编码规则解析**：

- **单字节**（ASCII）：最高位为0，剩余7位存储值
- **多字节**：首字节以N个1加1个0开头，表示N字节序列；后续字节都以10开头

这种设计使得：
1. 可以从任意字节开始解析（自同步）
2. 可以向前/向后遍历字符
3. 错误的字节序列容易检测

### 15.1.2 编码长度查询

**源码位置**：`src/unicode.cpp`（第16-20行）

```cpp
// 通过首字节高4位判断UTF-8序列长度
size_t unicode_len_utf8(char src) {
    const size_t lookup[] = { 
        1, 1, 1, 1, 1, 1, 1, 1,  // 0xxx xxxx: 1字节 (ASCII)
        1, 1, 1, 1,             // 10xx xxxx: 续字节(不应该出现在开头)
        2, 2,                   // 110x xxxx: 2字节
        3,                      // 1110 xxxx: 3字节
        4                       // 1111 0xxx: 4字节
    };
    uint8_t highbits = static_cast<uint8_t>(src) >> 4;
    return lookup[highbits];
}
```

**查找表示例**：
```
首字节    高4位    查表索引    长度    说明
0xxxxxxx  0000-0111  0-7      1      ASCII字符
10xxxxxx  1000-1011  8-11     1      续字节（不应在开头）
110xxxxx  1100-1101  12-13    2      2字节序列
1110xxxx  1110       14       3      3字节序列
11110xxx  1111       15       4      4字节序列
```

这个函数使用查找表而非条件判断，原因有三：
1. **性能**：查表是O(1)操作，比多个if-else更快
2. **简洁**：16个元素的数组比多分支代码清晰
3. **分支预测友好**：避免分支预测失败的开销

**为什么续字节返回1？**

如果续字节（10xxxxxx）意外出现在字符串开头，函数返回1。这是一种容错设计——错误地返回1比越界读取更安全。调用者应该确保传入的是有效UTF-8字符串的首字节。

### 15.1.3 UTF-8解码

**源码位置**：`src/unicode.cpp`（第30-65行）

```cpp
uint32_t unicode_cpt_from_utf8(const std::string & utf8, size_t & offset) {
    assert(offset < utf8.size());

    // 单字节: 0xxxxxxx
    if (!(utf8[offset + 0] & 0x80)) {
        auto result = utf8[offset + 0];
        offset += 1;
        return result;
    }

    // 双字节: 110xxxxx 10xxxxxx
    if (!(utf8[offset + 0] & 0x20)) {
        // 验证续字节格式
        if (offset + 1 >= utf8.size() || 
            !((utf8[offset + 1] & 0xc0) == 0x80)) {
            throw std::invalid_argument("invalid character");
        }
        // 提取有效位
        auto result = ((utf8[offset + 0] & 0x1f) << 6) | 
                       (utf8[offset + 1] & 0x3f);
        offset += 2;
        return result;
    }

    // 三字节: 1110xxxx 10xxxxxx 10xxxxxx
    if (!(utf8[offset + 0] & 0x10)) {
        if (offset + 2 >= utf8.size() || 
            !((utf8[offset + 1] & 0xc0) == 0x80) || 
            !((utf8[offset + 2] & 0xc0) == 0x80)) {
            throw std::invalid_argument("invalid character");
        }
        auto result = ((utf8[offset + 0] & 0x0f) << 12) | 
                      ((utf8[offset + 1] & 0x3f) << 6) | 
                       (utf8[offset + 2] & 0x3f);
        offset += 3;
        return result;
    }

    // 四字节: 11110xxx 10xxxxxx 10xxxxxx 10xxxxxx
    if (!(utf8[offset + 0] & 0x08)) {
        if (offset + 3 >= utf8.size() || 
            !((utf8[offset + 1] & 0xc0) == 0x80) || 
            !((utf8[offset + 2] & 0xc0) == 0x80) || 
            !((utf8[offset + 3] & 0xc0) == 0x80)) {
            throw std::invalid_argument("invalid character");
        }
        auto result = ((utf8[offset + 0] & 0x07) << 18) | 
                      ((utf8[offset + 1] & 0x3f) << 12) | 
                      ((utf8[offset + 2] & 0x3f) << 6) | 
                       (utf8[offset + 3] & 0x3f);
        offset += 4;
        return result;
    }

    throw std::invalid_argument("failed to convert utf8 to codepoint");
}
```

**解码流程详解**：

函数通过检查首字节的高位模式来判断编码长度，然后提取有效位：

1. **单字节**（第35-38行）：
   - 检查：`!(byte & 0x80)` → 最高位为0
   - 直接返回字节值

2. **双字节**（第40-49行）：
   - 检查：`!(byte & 0x20)` → 110xxxxx模式
   - 验证：有第二个字节且是续字节格式（10xxxxxx）
   - 提取：`(byte1 & 0x1f) << 6 | (byte2 & 0x3f)`

3. **三字节**（第51-61行）：
   - 检查：`!(byte & 0x10)` → 1110xxxx模式
   - 验证：有两个续字节
   - 提取：`(byte1 & 0x0f) << 12 | (byte2 & 0x3f) << 6 | (byte3 & 0x3f)`

4. **四字节**（第63-75行）：
   - 检查：`!(byte & 0x08)` → 11110xxx模式
   - 验证：有三个续字节
   - 提取：类似的位运算

**解码示例**（汉字"中" = U+4E2D）：

```
输入: UTF-8字节序列 [0xE4, 0xB8, 0xAD]

步骤1: 检查首字节 0xE4 = 11100100
        - 0xE4 & 0x80 = 0x80 ≠ 0 → 不是单字节
        - 0xE4 & 0x20 = 0x20 ≠ 0 → 不是双字节
        - 0xE4 & 0x10 = 0x00 = 0 → 是三字节！

步骤2: 验证续字节
        - 0xB8 = 10111000 (10xxxxxx ✓)
        - 0xAD = 10101101 (10xxxxxx ✓)

步骤3: 提取有效位
        - 字节1: 11100100 & 0x0F = 00000100
        - 字节2: 10111000 & 0x3F = 00111000
        - 字节3: 10101101 & 0x3F = 00101101

步骤4: 组合
        00000100 00111000 00101101
        = 0100 111000 101101
        = 0x4E2D

输出: Codepoint U+4E2D (汉字"中")
```

### 15.1.4 UTF-8编码

**源码位置**：`src/unicode.cpp`（第818-845行）

```cpp
std::string unicode_cpt_to_utf8(uint32_t cpt) {
    std::string result;

    // 1字节: U+0000 - U+007F
    if (cpt <= 0x7f) {
        result.push_back(cpt);
        return result;
    }

    // 2字节: U+0080 - U+07FF
    if (cpt <= 0x7ff) {
        result.push_back(0xc0 | ((cpt >> 6) & 0x1f));
        result.push_back(0x80 | (cpt & 0x3f));
        return result;
    }

    // 3字节: U+0800 - U+FFFF
    if (cpt <= 0xffff) {
        result.push_back(0xe0 | ((cpt >> 12) & 0x0f));
        result.push_back(0x80 | ((cpt >> 6) & 0x3f));
        result.push_back(0x80 | (cpt & 0x3f));
        return result;
    }

    // 4字节: U+10000 - U+10FFFF
    if (cpt <= 0x10ffff) {
        result.push_back(0xf0 | ((cpt >> 18) & 0x07));
        result.push_back(0x80 | ((cpt >> 12) & 0x3f));
        result.push_back(0x80 | ((cpt >> 6) & 0x3f));
        result.push_back(0x80 | (cpt & 0x3f));
        return result;
    }

    throw std::invalid_argument("invalid codepoint");
}
```

编码是解码的逆过程：
1. 根据码点范围确定字节数
2. 将码点拆分为对应的位段
3. 按格式组装字节

**编码示例**（U+4E2D → "中"）：

```
输入: Codepoint 0x4E2D = 0100 1110 0010 1101

步骤1: 判断范围
        0x4E2D > 0x07FF 且 <= 0xFFFF → 3字节

步骤2: 拆分为位段
        0100 1110 0010 1101
        → 0100 (高4位)
        → 111000 (中6位)
        → 101101 (低6位)

步骤3: 按格式组装
        字节1: 1110 0000 | 0000 0100 = 1110 0100 = 0xE4
        字节2: 1000 0000 | 0011 1000 = 1011 1000 = 0xB8
        字节3: 1000 0000 | 0010 1101 = 1010 1101 = 0xAD

输出: [0xE4, 0xB8, 0xAD] = "中"
```

## 15.2 Unicode字符属性

### 15.2.1 属性结构定义

**源码位置**：`src/unicode.h`（第8-90行）

```cpp
struct unicode_cpt_flags {
    enum {
        UNDEFINED       = 0x0001,  // 未定义
        NUMBER          = 0x0002,  // \p{N} - 数字
        LETTER          = 0x0004,  // \p{L} - 字母
        SEPARATOR       = 0x0008,  // \p{Z} - 分隔符
        ACCENT_MARK     = 0x0010,  // \p{M} - 重音标记
        PUNCTUATION     = 0x0020,  // \p{P} - 标点
        SYMBOL          = 0x0040,  // \p{S} - 符号
        CONTROL         = 0x0080,  // \p{C} - 控制字符
        MASK_CATEGORIES = 0x00FF,  // 类别掩码
        WHITESPACE      = 0x0100,  // 空白字符
        LOWERCASE       = 0x0200,  // 小写
        UPPERCASE       = 0x0400,  // 大写
        NFD             = 0x0800,  // 规范化形式D
    };

    // 位域存储（小端优化）
    uint16_t is_undefined   : 1;
    uint16_t is_number      : 1;
    uint16_t is_letter      : 1;
    uint16_t is_separator   : 1;
    uint16_t is_accent_mark : 1;
    uint16_t is_punctuation : 1;
    uint16_t is_symbol      : 1;
    uint16_t is_control     : 1;
    uint16_t is_whitespace  : 1;
    uint16_t is_lowercase   : 1;
    uint16_t is_uppercase   : 1;
    uint16_t is_nfd         : 1;

    // 从uint16解码
    inline unicode_cpt_flags(const uint16_t flags = 0) {
        #if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
            *reinterpret_cast<uint16_t*>(this) = flags;
        #else
            // 大端：逐位解析
            is_undefined = (flags & UNDEFINED) ? 1 : 0;
            // ... 其他字段
        #endif
    }
};
```

**结构体设计分析**：

这个结构体使用C++位域（bit-field）来紧凑存储12个布尔属性：
- 8个基本类别（对应Unicode通用类别）
- 4个辅助属性（空白、大小写、NFD）

**为什么使用位域？**

1. **空间效率**：12个布尔值原本需要12字节，位域只需2字节
2. **缓存友好**：更小的内存占用意味着更好的缓存命中率
3. **快速比较**：整个结构体可以作为uint16_t一次性比较

**大小端处理**：

在小端系统上，直接将uint16_t复制到位域内存是最快的。在大端系统上，需要逐位解析。这种优化对性能至关重要，因为字符属性查询在分词时会被频繁调用。

### 15.2.2 属性查询实现

**源码位置**：`src/unicode.cpp`（第116-146行）

```cpp
static std::vector<unicode_cpt_flags> unicode_cpt_flags_array() {
    // 初始化所有字符为UNDEFINED
    std::vector<unicode_cpt_flags> cpt_flags(MAX_CODEPOINTS, 
                                             unicode_cpt_flags::UNDEFINED);

    // 从范围表填充基本类别
    // unicode_ranges_flags: 预定义的字符范围表
    for (size_t i = 1; i < unicode_ranges_flags.size(); ++i) {
        const auto range_ini = unicode_ranges_flags.begin()[i-1];
        const auto range_end = unicode_ranges_flags.begin()[i];
        for (uint32_t cpt = range_ini.first; cpt < range_end.first; ++cpt) {
            cpt_flags[cpt] = range_ini.second;
        }
    }

    // 标记空白字符
    for (auto cpt : unicode_set_whitespace) {
        cpt_flags[cpt].is_whitespace = true;
    }

    // 标记小写字符
    for (auto p : unicode_map_lowercase) {
        cpt_flags[p.second].is_lowercase = true;
    }

    // 标记大写字符
    for (auto p : unicode_map_lowercase) {
        cpt_flags[p.second].is_uppercase = true;
    }

    // 标记NFD字符
    for (auto &range : unicode_ranges_nfd) {
        cpt_flags[range.nfd].is_nfd = true;
    }

    return cpt_flags;
}

// 查询接口
unicode_cpt_flags unicode_cpt_flags_from_cpt(const uint32_t cpt) {
    static const unicode_cpt_flags undef(unicode_cpt_flags::UNDEFINED);
    static const auto cpt_flags = unicode_cpt_flags_array();
    return cpt < cpt_flags.size() ? cpt_flags[cpt] : undef;
}
```

**实现策略分析**：

1. **预计算**：在启动时构建完整的属性表，查询时O(1)访问
2. **范围压缩**：Unicode字符按范围分组存储，减少内存占用
3. **静态初始化**：使用static变量确保只初始化一次

**为什么预计算而不是实时查询？**

Unicode属性数据量很大（数万个字符），如果在分词时实时查询Unicode数据库，性能会严重下降。预计算将整个数据库压缩为一个数组，查询仅需一次数组访问。

**属性查询示例**：

```cpp
// 查询 'A' 的属性
auto flags = unicode_cpt_flags_from_cpt('A');
flags.is_letter      // true  - 是字母
flags.is_uppercase   // true  - 是大写
flags.is_number      // false - 不是数字
flags.is_whitespace  // false - 不是空白

// 查询 '中' 的属性
auto flags = unicode_cpt_flags_from_cpt(0x4E2D);
flags.is_letter      // true  - CJK统一表意文字属于字母类
flags.is_number      // false
```

### 15.2.3 汉字检测

**源码位置**：`src/unicode.cpp`（第914-944行）

```cpp
bool unicode_cpt_is_han(uint32_t cpt) {
    // CJK Unified Ideographs (最常用汉字)
    if (cpt >= 0x4E00 && cpt <= 0x9FFF) return true;

    // CJK Extension A (罕见汉字)
    if (cpt >= 0x3400 && cpt <= 0x4DBF) return true;

    // CJK Extension B-F (非常罕见)
    if (cpt >= 0x20000 && cpt <= 0x2A6DF) return true;  // B
    if (cpt >= 0x2A700 && cpt <= 0x2B73F) return true;  // C
    if (cpt >= 0x2B740 && cpt <= 0x2B81F) return true;  // D
    if (cpt >= 0x2B820 && cpt <= 0x2CEAF) return true;  // E
    if (cpt >= 0x2CEB0 && cpt <= 0x2EBEF) return true;  // F

    // CJK Compatibility Ideographs
    if (cpt >= 0xF900 && cpt <= 0xFAFF) return true;
    if (cpt >= 0x2F800 && cpt <= 0x2FA1F) return true;

    return false;
}
```

汉字在Unicode中分布在多个区块：
- **U+4E00-U+9FFF**：CJK统一表意文字（约20,000个常用汉字）
- **U+3400-U+4DBF**：扩展A（罕见汉字）
- **U+20000+**：扩展B-F（古籍、方言用字）
- **兼容区**：与东亚其他编码标准的兼容字符

这个函数在Kimi K2等模型的预分词中被使用，用于识别汉字序列。

## 15.3 正则表达式预分词

### 15.3.1 预分词概述

预分词（Pre-tokenization）是将原始文本切分为"单词"或"token候选"的过程。不同模型使用不同的正则规则：

- **GPT-2**：基于空格和标点的简单切分
- **Llama3**：改进的数字处理，1-3位一组
- **Kimi K2**：针对中文优化的切分，汉字单独处理

**源码位置**：`src/unicode.cpp`（第946-1138行）

```cpp
std::vector<std::string> unicode_regex_split(
        const std::string & text,
        const std::vector<std::string> & regex_exprs,
        bool byte_encode) {

    // ① 将文本转为codepoints
    const auto cpts = unicode_cpts_from_utf8(text);

    // ② 应用每个正则表达式
    std::vector<size_t> bpe_offsets = { cpts.size() };

    for (const auto & regex_expr : regex_exprs) {
        // 优先使用自定义实现
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

    // ④ 字节编码（如果需要）
    if (byte_encode) {
        return unicode_byte_encoding_process(bpe_words);
    }

    return bpe_words;
}
```

**流程解析**：

1. **UTF-8转Codepoints**（第8行）：统一使用codepoints处理，避免UTF-8多字节的复杂性
2. **逐级切分**（第11-21行）：可以应用多个正则，每个正则在上一个结果基础上继续切分
3. **自定义优先**（第14-18行）：优先使用针对特定正则优化的自定义实现
4. **Codepoints转字符串**（第24-30行）：将切分结果转回UTF-8字符串
5. **字节编码**（第33-35行）：BPE模型需要将字符串转为字节表示

### 15.3.2 GPT-2风格预分词

**源码位置**：`src/unicode.cpp`（第214-330行）

```cpp
// GPT-2系统正则:
// 's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
static std::vector<size_t> unicode_regex_split_custom_gpt2(
        const std::string & text, 
        const std::vector<size_t> & offsets) {

    const auto cpts = unicode_cpts_from_utf8(text);
    std::vector<size_t> bpe_offsets;

    size_t start = 0;
    for (auto offset : offsets) {
        const size_t offset_ini = start;
        const size_t offset_end = start + offset;
        start = offset_end;

        // 辅助lambda
        auto _get_flags = [&](const size_t pos) -> unicode_cpt_flags {
            return (offset_ini <= pos && pos < offset_end) ? 
                   unicode_cpt_flags_from_cpt(cpts[pos]) : unicode_cpt_flags{};
        };

        size_t _prev_end = offset_ini;
        auto _add_token = [&](const size_t end) -> size_t {
            size_t len = end - _prev_end;
            if (len > 0) bpe_offsets.push_back(len);
            _prev_end = end;
            return len;
        };

        for (size_t pos = offset_ini; pos < offset_end; ) {
            const uint32_t cpt = cpts[pos];
            const auto flags = _get_flags(pos);

            // 规则1: 缩写词 ('s, 't, 're, 've, 'm, 'll, 'd)
            if (cpt == '\'' && pos+1 < offset_end) {
                uint32_t cpt_next = _get_cpt(pos+1);
                if (cpt_next == 's' || cpt_next == 't' || 
                    cpt_next == 'm' || cpt_next == 'd') {
                    pos += _add_token(pos+2);
                    continue;
                }
                if (pos+2 < offset_end) {
                    uint32_t cpt_next_next = _get_cpt(pos+2);
                    if ((cpt_next == 'r' && cpt_next_next == 'e') ||
                        (cpt_next == 'v' && cpt_next_next == 'e') ||
                        (cpt_next == 'l' && cpt_next_next == 'l')) {
                        pos += _add_token(pos+3);
                        continue;
                    }
                }
            }

            // 规则2: 可选空格 + 字母序列 ( ?\p{L}+)
            auto flags2 = (cpt == ' ' ? _get_flags(pos+1) : flags);
            if (flags2.is_letter) {
                pos += (cpt == ' ');  // 跳过前导空格
                while (_get_flags(pos).is_letter) pos++;
                _add_token(pos);
                continue;
            }

            // 规则3: 可选空格 + 数字序列 ( ?\p{N}+)
            if (flags2.is_number) {
                pos += (cpt == ' ');
                while (_get_flags(pos).is_number) pos++;
                _add_token(pos);
                continue;
            }

            // 规则4: 可选空格 + 非字母数字空白 ( ?[^\s\p{L}\p{N}]+)
            if (!(flags2.is_whitespace | flags2.is_letter | flags2.is_number) 
                && flags2.as_uint()) {
                pos += (cpt == ' ');
                while (!(flags2.is_whitespace | flags2.is_letter | flags2.is_number) 
                       && flags2.as_uint()) {
                    flags2 = _get_flags(++pos);
                }
                _add_token(pos);
                continue;
            }

            // 规则5: 多个空白，但后面还有非空白 (\s+(?!\S))
            size_t num_whitespaces = 0;
            while (_get_flags(pos+num_whitespaces).is_whitespace) {
                num_whitespaces++;
            }
            if (num_whitespaces > 1 && _get_cpt(pos+num_whitespaces) != OUT_OF_RANGE) {
                pos += num_whitespaces - 1;
                _add_token(pos);
                continue;
            }

            // 规则6: 一般空白序列 (\s+)
            if (num_whitespaces > 0) {
                pos += num_whitespaces;
                _add_token(pos);
                continue;
            }

            // 无匹配，单字符推进
            _add_token(++pos);
        }
    }

    return bpe_offsets;
}
```

**GPT-2正则规则解析**：

| 规则 | 模式 | 说明 |
|------|------|------|
| 缩写词 | `'s\|'t\|'re\|'ve\|'m\|'ll\|'d` | 英语缩写形式 |
| 字母词 | ` ?\p{L}+` | 可选空格+字母序列 |
| 数字 | ` ?\p{N}+` | 可选空格+数字序列 |
| 符号 | ` ?[^\s\p{L}\p{N}]+` | 可选空格+非字母数字空白 |
| 修剪空白 | `\s+(?!\S)` | 多个空白但后面有非空白 |
| 空白 | `\s+` | 一般空白序列 |

**为什么要预分词？**

BPE（Byte Pair Encoding）算法在合并token时需要先识别"单词"边界。预分词确保：
1. 不同语言有合理的切分粒度
2. 数字、标点被正确处理
3. 缩写词等特殊形式被保留

### 15.3.3 Llama3风格预分词

**源码位置**：`src/unicode.cpp`（第332-471行）

Llama3改进了数字处理，将数字切分为1-3位一组：

```cpp
// 规则: \p{N}{1,3} (数字1-3位)
if (flags.is_number) {
    size_t ini = pos;
    while (_get_flags(pos).is_number) {
        if (++pos - ini >= 3) {  // 每3位切分
            _add_token(pos);
            ini = pos;
        }
    }
    _add_token(pos);
    continue;
}
```

**数字切分对比**：

```
输入: "123456789"

GPT-2切分: ["123456789"]  # 整个数字
Llama3切分: ["123", "456", "789"]  # 每3位
```

这种切分方式的好处：
1. **平衡词表大小**：不需要为每个可能的数字组合创建token
2. **保留局部模式**：3位一组对应千分位，人类阅读习惯
3. **减少OOV**：任意长度数字都能表示

### 15.3.4 Kimi K2风格预分词

**源码位置**：`src/llama-vocab.cpp`（第512-680行）

Kimi K2对中文有特殊处理，将汉字单独切分：

```cpp
// Pattern 1: [\p{Han}]+ (汉字序列)
if (unicode_cpt_is_han(cpt)) {
    while (unicode_cpt_is_han(_get_cpt(pos))) {
        pos++;
    }
    _add_token(pos);
    continue;
}

// Pattern 2-3: 非汉字字母词
// [^\r\n\p{L}\p{N}]?[\p{Lu}...]*[\p{Ll}...]+...
```

**切分对比**：

```
输入: "Hello世界123"

GPT-2切分:
["Hello", "世界", "123"]

Llama3切分:
["Hello", "世界", "1", "2", "3"] 或 ["Hello", "世界", "12", "3"]

Kimi K2切分:
["Hello", "世", "界", "123"]  # 汉字单独切分
```

Kimi K2的策略：
1. **汉字单独切分**：每个汉字是一个独立token，因为汉字数量巨大且语义相对独立
2. **其他文字按规则切分**：英文、数字使用类似GPT-2的规则
3. **混合文本优化**：中英文混合时各有合适的切分粒度

## 15.4 字节编码处理

### 15.4.1 字节到UTF-8映射

BPE（Byte Pair Encoding）使用字节级token，需要将字节映射为可显示的UTF-8字符。

**源码位置**：`src/unicode.cpp`（第148-170行）

```cpp
static std::unordered_map<uint8_t, std::string> unicode_byte_to_utf8_map() {
    std::unordered_map<uint8_t, std::string> map;

    // 可打印ASCII字符直接映射
    for (int ch = 0x21; ch <= 0x7E; ++ch) {  // '!' 到 '~'
        map[ch] = unicode_cpt_to_utf8(ch);
    }

    // 扩展ASCII可打印字符
    for (int ch = 0xA1; ch <= 0xAC; ++ch) {  // '¡' 到 '¬'
        map[ch] = unicode_cpt_to_utf8(ch);
    }
    for (int ch = 0xAE; ch <= 0xFF; ++ch) {  // '®' 到 'ÿ'
        map[ch] = unicode_cpt_to_utf8(ch);
    }

    // 控制字符映射到U+0100-U+017F范围
    auto n = 0;
    for (int ch = 0; ch < 256; ++ch) {
        if (map.find(ch)) {
            map[ch] = unicode_cpt_to_utf8(256 + n);
            ++n;
        }
    }

    return map;
}
```

**字节映射策略**：

1. **可打印字符直接映射**：0x21-0x7E（ASCII可打印）、0xA1-0xAC、0xAE-0xFF（扩展ASCII）
2. **控制字符映射到U+0100+**：换行、空格、NULL等控制字符映射到拉丁扩展-A区

**字节映射示例**：
```
字节值    UTF-8表示    显示字符    说明
0x41      'A'          A           ASCII字母
0x0A      U+010A       Ċ           换行符(\n)
0x20      U+0120       Ġ           空格
0x00      U+0100       Ā           NULL
```

这种映射允许BPE将所有字节（包括不可打印的控制字符）表示为可见的Unicode字符，便于调试和可视化。

### 15.4.2 字节编码处理流程

```cpp
// 将BPE分词结果进行字节编码
static std::vector<std::string> unicode_byte_encoding_process(
        const std::vector<std::string> & bpe_words) {
    std::vector<std::string> bpe_encoded_words;

    for (const auto & word : bpe_words) {
        // 将UTF-8转为codepoints
        auto utf_word = unicode_cpts_from_utf8(word);

        // 转回UTF-8（规范化）
        std::string text_utf;
        for (size_t i = 0; i < utf_word.size(); ++i) {
            text_utf += unicode_cpt_to_utf8(utf_word[i]);
        }

        // 每个字节转为对应的UTF-8字符
        std::string encoded_token;
        for (char & c : text_utf) {
            encoded_token += unicode_byte_to_utf8(c);
        }

        bpe_encoded_words.emplace_back(encoded_token);
    }

    return bpe_encoded_words;
}
```

**为什么要字节编码？**

1. **BPE基础**：BPE算法从字节级开始合并，需要字节表示
2. **字符集无关**：可以处理任意编码的文本
3. **OOV处理**：任何字符都能表示为字节序列

## 15.5 设计中的取舍

### 为什么使用自定义正则而不是std::regex？

**std::regex的问题**：

1. **不支持Unicode属性**：无法直接使用`\p{L}`（所有字母）这类模式
2. **性能问题**：处理多字节UTF-8字符时性能差
3. **平台差异**：不同编译器的实现行为不一致

**llama.cpp的解决方案**：

1. **针对常见正则写专用解析器**：为GPT-2、Llama3等常用正则有优化实现
2. **字符折叠技术**：将Unicode字符映射到单字节范围，用标准正则处理
3. **回退机制**：自定义不支持时回退到std::regex

### 字符折叠技术详解

**原理**：将Unicode字符按类别映射到单字节范围，然后使用标准正则引擎。

```cpp
// Unicode类别映射到单字节
\p{N} (数字)  → 0xD1
\p{L} (字母)  → 0xD2
\p{P} (标点)  → 0xD3
\p{Z} (空白)  → 0x0B (垂直制表符)

// 示例
原始: "Hello 世界 123"
折叠: "\xD2\xD2\xD2\xD2\xD2\x0B\xD2\xD2\x0B\xD1\xD1\xD1"
```

**字符折叠步骤**：

1. 将UTF-8文本转为codepoints
2. 查询每个codepoint的Unicode类别
3. 将类别映射到单字节值
4. 在折叠后的文本上应用标准正则
5. 将匹配位置映射回原始文本

这种技术使得llama.cpp可以：
- 使用高效的正则引擎（如RE2）
- 避免引入大型Unicode正则库
- 保持跨平台一致性

## 15.6 动手练习

### 练习1：UTF-8编解码

将以下codepoint转为UTF-8：
1. U+0041 ('A')
2. U+00E9 ('é')
3. U+4E2D ('中')
4. U+1F600 (😀)

**答案**：
```
1. U+0041 ≤ 0x7F → 1字节
   0x41

2. U+00E9 (0000 1110 1001) ≤ 0x7FF → 2字节
   1100 0011  1010 1001
   = 0xC3 0xA9

3. U+4E2D (0100 1110 0010 1101) ≤ 0xFFFF → 3字节
   1110 0100  1011 1000  1010 1101
   = 0xE4 0xB8 0xAD

4. U+1F600 (0001 1111 0110 0000 0000) → 4字节
   1111 0000  1001 1111  1001 1000  1000 0000
   = 0xF0 0x9F 0x98 0x80
```

### 练习2：分析预分词正则

阅读 `src/unicode.cpp` 第332-471行，回答：
1. Llama3与GPT-2的正则有何不同？
2. 数字是如何被切分的？
3. 换行符如何处理？

**参考答案**：

1. **主要区别**：
   - Llama3对数字使用`\p{N}{1,3}`，每1-3位切分
   - GPT-2对整个数字序列不切分

2. **数字切分逻辑**：
   - 遇到数字字符时开始计数
   - 每累积3个数字就切分一次
   - 剩余不足3位的也作为一组

3. **换行符处理**：
   - 作为空白字符处理
   - 多个空白（包括换行）可能被合并
   - `\s+(?!\S)`规则保留结尾空白

### 练习3：理解字符折叠

阅读 `src/unicode.cpp` 第989-1118行，解释：
1. 什么是字符折叠？
2. 为什么要折叠？
3. 如何处理Unicode类别？

**参考答案**：

1. **字符折叠**：将Unicode字符映射到单字节表示，保留类别信息但缩小字符集

2. **折叠目的**：
   - 使用标准正则引擎处理Unicode文本
   - 避免大型Unicode正则库依赖
   - 提高匹配性能

3. **类别处理**：
   - 每个Unicode通用类别映射到一个单字节值
   - 正则匹配在折叠后的文本上进行
   - 匹配结果再映射回原始文本位置

## 15.7 本章小结

| 概念 | 一句话总结 |
|------|-----------|
| **UTF-8** | 变长Unicode编码，1-4字节，向后兼容ASCII |
| **Codepoint** | Unicode字符的唯一编号，如U+4E2D |
| **unicode_cpt_flags** | 字符属性结构，使用位域紧凑存储12个属性 |
| **预分词** | 用正则将文本切分为单词，不同模型有不同规则 |
| **字节编码** | 将字节映射为可显示UTF-8字符，便于BPE处理 |
| **字符折叠** | 将Unicode映射到单字节范围，加速正则处理 |

**Unicode处理的设计哲学**：

1. **预计算优先**：字符属性表预先生成，查询O(1)
2. **变长编码原生支持**：全程使用UTF-8，只在必要时转codepoint
3. **针对优化**：为常用预分词正则写专用实现
4. **跨平台一致**：避免依赖平台相关的Unicode库

Unicode处理是llama.cpp多语言能力的基石。从UTF-8编解码到字符属性查询，从预分词到字节编码，每个环节都经过精心设计，确保全球各种文字都能被准确处理。就像一位精通百种语言的大师，llama.cpp能够理解和生成世界上绝大多数人类语言。

**下一步预告**：

在第16章，我们将探索采样算法——理解模型如何从概率分布中选择下一个token，以及温度、Top-K、Top-P等参数如何影响生成结果。
