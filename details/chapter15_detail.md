# 第15章 Unicode与文本处理 —— 多语言支持的"幕后功臣"

## 学习目标
1. 理解Unicode在llama.cpp中的核心作用
2. 掌握UTF-8编解码实现原理
3. 了解Unicode字符属性查询系统
4. 理解正则表达式在预分词中的应用

---

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

就像传译团队需要处理各种语言的细微差别，Unicode系统需要正确处理全球各种文字的编码和特性。

---

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

---

## 15.1 UTF-8编解码

### 15.1.1 UTF-8编码原理

UTF-8是一种变长编码，根据字符的Unicode码点选择1-4字节：

| 码点范围 | 字节数 | 编码格式 |
|---------|-------|---------|
| U+0000 - U+007F | 1 | `0xxxxxxx` |
| U+0080 - U+07FF | 2 | `110xxxxx 10xxxxxx` |
| U+0800 - U+FFFF | 3 | `1110xxxx 10xxxxxx 10xxxxxx` |
| U+10000 - U+10FFFF | 4 | `11110xxx 10xxxxxx 10xxxxxx 10xxxxxx` |

### 15.1.2 编码长度查询

**源码位置**：`src/unicode.cpp` (第16-20行)

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
首字节    高4位    查表索引    长度
0xxxxxxx  0000-0111  0-7      1
10xxxxxx  1000-1011  8-11     1 (错误续字节)
110xxxxx  1100-1101  12-13    2
1110xxxx  1110       14       3
11110xxx  1111       15       4
```

### 15.1.3 UTF-8解码

**源码位置**：`src/unicode.cpp` (第30-65行)

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

**解码流程图解**：
```
输入: UTF-8字节序列 [0xE4, 0xB8, 0xAD] ("中")

步骤1: 检查首字节 0xE4 = 11100100
        - 高4位 = 1110 → 3字节序列

步骤2: 验证续字节
        - 0xB8 = 10111000 (10xxxxxx ✓)
        - 0xAD = 10101101 (10xxxxxx ✓)

步骤3: 提取有效位
        - 字节1: 1110xxxx → 取低4位: 0100
        - 字节2: 10xxxxxx → 取低6位: 111000
        - 字节3: 10xxxxxx → 取低6位: 101101

步骤4: 组合
        - 0100 111000 101101 = 0x4E2D = U+4E2D

输出: Codepoint U+4E2D (汉字"中")
```

### 15.1.4 UTF-8编码

**源码位置**：`src/unicode.cpp` (第818-845行)

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

---

## 15.2 Unicode字符属性

### 15.2.1 属性结构定义

**源码位置**：`src/unicode.h` (第8-90行)

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

### 15.2.2 属性查询实现

**源码位置**：`src/unicode.cpp` (第116-146行)

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
    for (auto p : unicode_map_uppercase) {
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

**属性查询示例**：
```cpp
// 查询 'A' 的属性
auto flags = unicode_cpt_flags_from_cpt('A');
flags.is_letter      // true
flags.is_uppercase   // true
flags.is_number      // false
flags.is_whitespace  // false

// 查询 '中' 的属性
auto flags = unicode_cpt_flags_from_cpt(0x4E2D);
flags.is_letter      // true
flags.is_number      // false
```

### 15.2.3 汉字检测

**源码位置**：`src/unicode.cpp` (第914-944行)

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

---

## 15.3 正则表达式预分词

### 15.3.1 预分词概述

预分词是将文本切分为"单词"的过程，不同模型使用不同的正则规则。

**源码位置**：`src/unicode.cpp` (第946-1138行)

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

### 15.3.2 GPT-2风格预分词

**源码位置**：`src/unicode.cpp` (第214-330行)

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

### 15.3.3 Llama3风格预分词

**源码位置**：`src/unicode.cpp` (第332-471行)

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

### 15.3.4 Kimi K2风格预分词

**源码位置**：`src/llama-vocab.cpp` (第512-680行)

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

---

## 15.4 字节编码处理

### 15.4.1 字节到UTF-8映射

BPE使用字节级token，需要将字节映射为可显示的UTF-8字符。

**源码位置**：`src/unicode.cpp` (第148-170行)

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
        if (map.find(ch) == map.end()) {
            map[ch] = unicode_cpt_to_utf8(256 + n);
            ++n;
        }
    }

    return map;
}
```

**字节映射示例**：
```
字节值    UTF-8表示    显示
0x41      'A'          A
0x0A      U+010A       Ċ  (换行符)
0x20      U+0120       Ġ  (空格)
0x00      U+0100       Ā  (NULL)
```

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

---

## 设计中的取舍

### 为什么使用自定义正则而不是std::regex？

**std::regex问题**：
- 不支持Unicode属性（如\p{L}）
- 处理多字节字符性能差
- 不同平台行为不一致

**llama.cpp方案**：
- 针对常见正则写专用解析器
- 使用字符折叠技术加速
- 跨平台一致性

### 字符折叠技术

**原理**：将Unicode字符映射到单字节范围，用标准正则处理。

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

---

## 动手练习

### 练习1：UTF-8编解码
将以下codepoint转为UTF-8：
1. U+0041 ('A')
2. U+00E9 ('é')
3. U+4E2D ('中')
4. U+1F600 (😀)

**答案**：
```
1. 0x41
2. 0xC3 0xA9
3. 0xE4 0xB8 0xAD
4. 0xF0 0x9F 0x98 0x80
```

### 练习2：分析预分词正则
阅读 `src/unicode.cpp` 第332-471行，回答：
1. Llama3与GPT-2的正则有何不同？
2. 数字是如何被切分的？
3. 换行符如何处理？

### 练习3：理解字符折叠
阅读 `src/unicode.cpp` 第989-1118行，解释：
1. 什么是字符折叠？
2. 为什么要折叠？
3. 如何处理Unicode类别？

---

## 本课小结

| 概念 | 一句话总结 |
|------|-----------|
| UTF-8 | 变长Unicode编码，1-4字节 |
| Codepoint | Unicode字符的唯一编号 |
| unicode_cpt_flags | 字符属性查询结构 |
| 预分词 | 用正则将文本切分为单词 |
| 字节编码 | 将字节映射为可显示UTF-8字符 |
| 字符折叠 | 将Unicode映射到单字节加速处理 |

---

*本章对应源码版本：master (2026-04-07)*
