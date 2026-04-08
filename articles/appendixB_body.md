# 附录B：GBNF语法参考 —— 约束生成的"语法大全"

## 学习目标

1. 掌握GBNF语法的基本规则和结构
2. 理解GBNF与标准BNF的区别
3. 学会编写常见数据格式的GBNF语法
4. 能够在项目中正确应用语法约束

---

## 生活类比：交通信号灯

想象GBNF语法是城市道路的"交通规则"——红灯停、绿灯行、黄灯等待。没有规则，车辆会随意穿行，造成混乱；有了规则，每辆车都知道何时该走、何时该停。同样，没有语法约束的LLM就像一个即兴演讲者，可能说出任何内容；而GBNF语法就是给这位演讲者一份"演讲提纲"，确保他说出的每一句话都符合预期格式。

---

## B.1 GBNF 语法概述

### B.1.1 什么是GBNF

GBNF（GGML BNF）是llama.cpp使用的语法约束格式，基于扩展的巴科斯-瑙尔范式（EBNF）。它允许开发者定义严格的输出格式规则，确保模型生成符合预期的结构化数据。

```
┌─────────────────────────────────────────────────────────────┐
│                    语法约束生成流程                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   输入文本 ──→ 语法解析器 ──→ 语法规则集合                     │
│                    │                                        │
│                    ↓                                        │
│              候选Token过滤                                   │
│                    │                                        │
│   模型输出 ←── 合法Token ←── 语法验证                        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**GBNF与标准BNF的区别：**

| 特性 | 标准BNF | GBNF |
|------|---------|------|
| 字符集 | ASCII | 完整Unicode支持 |
| 量词 | 有限 | `*`, `+`, `?`, `{m,n}` |
| Token引用 | 不支持 | 支持 `<token>` 或 `<[id]>` |
| 注释 | 不支持 | 支持 `#` 注释 |
| 贪婪匹配 | 固定 | 可配置 |

---

### B.1.2 基本语法规则

**规则定义格式：**

```gbnf
rule-name ::= expression
```

**命名规范：**
- 使用小写字母、数字和连字符
- 以字母开头
- 例如：`json-object`, `identifier`, `number`

**示例：**

```gbnf
# 定义一个简单的数字规则
number ::= [0-9]+

# 定义一个标识符规则
identifier ::= [a-zA-Z_] [a-zA-Z0-9_]*

# 定义一个字符串规则
string ::= "\"" ([^"\\] | "\\" .)* "\""
```

---

## B.2 GBNF 语法元素详解

### B.2.1 终结符

**字符字面量：**

```gbnf
# 精确匹配单个字符
letter-a ::= "a"

# 匹配转义字符
newline ::= "\n"
tab ::= "\t"
quote ::= "\""
backslash ::= "\\"

# Unicode字符
unicode-char ::= "\u0041"    # 大写A
unicode-emoji ::= "\U0001F600"  # 😀
```

**支持的转义序列：**

| 转义序列 | 含义 | 示例 |
|----------|------|------|
| `\n` | 换行 | `"hello\nworld"` |
| `\r` | 回车 | `"\r\n"` |
| `\t` | 制表符 | `"col1\tcol2"` |
| `\\` | 反斜杠 | `"C:\\path"` |
| `\"` | 双引号 | `"say \"hi\""` |
| `\xHH` | 2位十六进制 | `"\x41"` (A) |
| `\uHHHH` | 4位Unicode | `"\u0041"` (A) |
| `\UHHHHHHHH` | 8位Unicode | `"\U0001F600"` (😀) |

**字符类：**

```gbnf
# 字符范围
lowercase ::= [a-z]
uppercase ::= [A-Z]
digit ::= [0-9]
hex-digit ::= [0-9a-fA-F]

# 字符集合
vowel ::= [aeiouAEIOU]
word-char ::= [a-zA-Z0-9_]

# 否定字符类（匹配不在集合中的字符）
non-digit ::= [^0-9]
non-space ::= [^ \t\n\r]

# 任意字符
any-char ::= .
```

**字符类优先级：**

```gbnf
# 正确：范围在集合内
range ::= [a-z0-9]      # 匹配a-z或0-9

# 正确：否定后接范围
neg-range ::= [^a-z]    # 匹配非小写字母

# 错误：范围边界不明确
invalid ::= [z-a]       # 错误！范围顺序颠倒
```

---

### B.2.2 非终结符（规则引用）

```gbnf
# 定义基础规则
digit ::= [0-9]
letter ::= [a-zA-Z]

# 在其他规则中引用
alphanumeric ::= letter | digit
identifier ::= letter alphanumeric*
```

**引用注意事项：**
- 所有引用的规则必须已定义或预定义
- 支持前向引用（但建议先定义后引用）
- 规则名区分大小写

---

### B.2.3 量词

| 量词 | 含义 | 等价形式 | 示例 |
|------|------|----------|------|
| `*` | 零次或多次 | `{0,}` | `a*` 匹配 "", "a", "aa" |
| `+` | 一次或多次 | `{1,}` | `a+` 匹配 "a", "aa" |
| `?` | 零次或一次 | `{0,1}` | `a?` 匹配 "", "a" |
| `{n}` | 恰好n次 | - | `a{3}` 匹配 "aaa" |
| `{m,}` | 至少m次 | - | `a{2,}` 匹配 "aa", "aaa"... |
| `{m,n}` | m到n次 | - | `a{1,3}` 匹配 "a", "aa", "aaa" |

**量词使用示例：**

```gbnf
# 整数（至少一位数字）
integer ::= [0-9]+

# 可选的正负号
signed-int ::= [+-]? integer

# 精确长度的十六进制颜色
color ::= "#" [0-9a-fA-F]{6}

# 1到3位的小数部分
decimal-part ::= "." [0-9]{1,3}

# 浮点数
float ::= signed-int decimal-part?

# 至少3个字母的单词
word ::= [a-zA-Z]{3,}

# 8-16位密码
password ::= [!-~]{8,16}
```

---

### B.2.4 选择（Alternation）

```gbnf
# 使用 | 表示"或"
bool ::= "true" | "false"

# 多种数字格式
number ::= integer | float | scientific

# 不同的空白字符
whitespace ::= " " | "\t" | "\n" | "\r"

# 三种引号字符串
quoted-string ::= double-quoted | single-quoted | backtick

double-quoted ::= "\"" [^"]* "\""
single-quoted ::= "'" [^']* "'"
backtick ::= "`" [^`]* "`"
```

**选择优先级：**
- 选择是左结合的
- 较长的匹配优先于较短的匹配（贪婪匹配）
- 使用分组明确优先级

---

### B.2.5 分组

```gbnf
# 使用括号进行分组
optional-prefix ::= ("Mr." | "Ms." | "Dr.")? " " name

# 分组与量词结合
repeated-group ::= ("ab" | "cd")+
# 匹配: "ab", "cd", "abcd", "abab", "cdcd", "abcdab"...

# 嵌套分组
complex ::= (("a" | "b")+ | ("c" | "d")*)
```

**分组的作用：**
1. 改变运算优先级
2. 对一组元素应用量词
3. 提高可读性

---

## B.3 内置规则

### B.3.1 标准内置规则

llama.cpp 提供以下内置规则，可直接使用：

```gbnf
# 基本类型
root ::= object | array

# JSON相关
object ::= "{" (pair ("," pair)*)? "}"
pair ::= string ":" value
array ::= "[" (value ("," value)*)? "]"
value ::= object | array | string | number | "true" | "false" | "null"

# 字符串
string ::= "\"" char* "\""
char ::= [^"\\] | "\\" (["\\/bfnrt] | "u" [0-9a-fA-F]{4})

# 数字
number ::= [0-9]+ ("." [0-9]+)? ([eE] [+-]? [0-9]+)?

# 空白
ws ::= [ \t\n\r]*
```

### B.3.2 常用模式库

**数字模式：**

```gbnf
# 整数
int ::= [0-9]+

# 带符号整数
signed-int ::= [+-]? int

# 浮点数
float ::= signed-int ("." int)? ([eE] signed-int)?

# 十六进制
hex ::= "0x" [0-9a-fA-F]+

# 二进制
binary ::= "0b" [01]+

# 八进制
octal ::= "0o" [0-7]+

# 科学计数法
scientific ::= signed-int ([eE] signed-int)
```

**字符串模式：**

```gbnf
# 简单字符串（无双引号）
simple-string ::= [a-zA-Z0-9_]+

# 带引号的字符串（支持转义）
quoted-string ::= "\"" ([^"\\] | "\\" .)* "\""

# 单引号字符串
single-quoted ::= "'" ([^'\\] | "\\" .)* "'"

# 多行字符串（三引号）
multiline-string ::= "\"\"\"" ([^"] | "\"" [^"] | "\"\"" [^"])* "\"\"\""
```

**标识符模式：**

```gbnf
# 驼峰命名（camelCase）
camelCase ::= [a-z] [a-zA-Z0-9]*

# 帕斯卡命名（PascalCase）
PascalCase ::= [A-Z] [a-zA-Z0-9]*

# 蛇形命名（snake_case）
snake_case ::= [a-z] [a-z0-9]* ("_" [a-z0-9]+)*

# 全大写下划线（CONSTANT_NAME）
UPPER_SNAKE ::= [A-Z] [A-Z0-9]* ("_" [A-Z0-9]+)*

# 短横线命名（kebab-case）
kebab-case ::= [a-z] [a-z0-9]* ("-" [a-z0-9]+)*
```

---

## B.4 高级特性

### B.4.1 Token引用

GBNF支持直接引用模型的Token，使用 `<token>` 或 `<[id]>` 语法：

```gbnf
# 引用特定token（通过ID）
special-token ::= <[100]>

# 引用特定token（通过文本）
# 假设词表中有 "<|endoftext|>" 这个token
end-token ::= <|endoftext|>

# 否定token
not-special ::= ![12345]
```

**Token引用示例：**

```gbnf
# 确保生成以特定token结尾的文本
custom-output ::= sentence* <|endoftext|>
sentence ::= [a-zA-Z ]+ "." " "

# 匹配特定分隔符
code-block ::= "```" language "\n" code "\n```"
language ::= "python" | "javascript" | "cpp"
code ::= [^`]+ | "`" [^`]+
```

---

### B.4.2 递归规则

GBNF支持递归定义，可用于匹配嵌套结构：

```gbnf
# 嵌套括号
nested-parens ::= "(" (nested-parens | [^()])* ")"
# 匹配: "()", "(())", "(()())", "(a(b)c)"

# 嵌套JSON对象
nested-object ::= "{" (pair ("," pair)*)? "}"
pair ::= string ":" (value | nested-object)
value ::= string | number | "true" | "false" | "null" | nested-object | array

# 算术表达式
expr ::= term (("+" | "-") term)*
term ::= factor (("*" | "/") factor)*
factor ::= number | "(" expr ")"
# 匹配: "1+2", "(1+2)*3", "a*(b+c)"
```

**重要限制：不支持左递归**

```gbnf
# ❌ 错误：左递归（无法解析）
expr ::= expr "+" term | term

# ✅ 正确：右递归
expr ::= term ("+" term)*
```

---

## B.5 实用语法示例

### B.5.1 JSON Schema 约束

**完整JSON对象：**

```gbnf
# JSON根规则
root ::= object | array

# 对象
object ::= "{" ws (pair ("," ws pair)*)? ws "}"
pair ::= string ws ":" ws value

# 数组
array ::= "[" ws (value ("," ws value)*)? ws "]"

# 值类型
value ::= object | array | string | number | "true" | "false" | "null"

# 字符串
string ::= "\"" char* "\""
char ::= [^"\\\x00-\x1F] | "\\" (["\\/bfnrt] | "u" hex{4})
hex ::= [0-9a-fA-F]

# 数字
number ::= int frac? exp?
int ::= "-"? ([0-9] | [1-9] [0-9]*)
frac ::= "." [0-9]+
exp ::= [eE] [+-]? [0-9]+

# 空白
ws ::= [ \t\n\r]*
```

**特定JSON结构：**

```gbnf
# 用户对象
user ::= "{" ws 
    "\"name\"" ws ":" ws string ws "," ws
    "\"age\"" ws ":" ws number ws 
"}"

# 带类型的值
typed-value ::= "{" ws 
    "\"type\"" ws ":" ws ("\"string\"" | "\"number\"" | "\"boolean\"") ws "," ws
    "\"value\"" ws ":" ws (string | number | boolean) ws
"}"
boolean ::= "true" | "false"
```

---

### B.5.2 代码生成约束

**Python函数定义：**

```gbnf
# Python函数
python-func ::= "def " identifier "(" params "):" ws body

params ::= param ("," ws param)* | ""

param ::= identifier (":" ws type-annot)?

type-annot ::= identifier | "List[" identifier "]" | "Dict[" identifier "," ws identifier "]"

body ::= (ws stmt "\n")+

stmt ::= "    " (assignment | return-stmt | func-call)

assignment ::= identifier "=" expr

return-stmt ::= "return " expr

func-call ::= identifier "(" args ")"

args ::= expr ("," ws expr)* | ""

expr ::= identifier | number | string | func-call

identifier ::= [a-zA-Z_] [a-zA-Z0-9_]*
```

**SQL查询：**

```gbnf
# 简单SQL SELECT
select-stmt ::= "SELECT " columns " FROM " table where-clause?

columns ::= "*" | column ("," ws column)*

column ::= identifier | table "." identifier

table ::= identifier (" AS " identifier)?

where-clause ::= " WHERE " condition

condition ::= column op value | condition " AND " condition

op ::= "=" | "!=" | "<" | "<=" | ">" | ">=" | "LIKE"

value ::= number | string | "NULL"

identifier ::= [a-zA-Z_] [a-zA-Z0-9_]*
```

---

### B.5.3 结构化数据格式

**CSV行：**

```gbnf
# CSV行
csv-row ::= field ("," field)*

field ::= quoted-field | unquoted-field

quoted-field ::= "\"" ([^"] | "\"\"")* "\""
# 支持转义的双引号

unquoted-field ::= [^,\n\r]*
```

**YAML子集：**

```gbnf
# YAML-like结构
yaml-value ::= scalar | list | mapping

scalar ::= string | number | boolean | "null"

list ::= "-" ws yaml-value ("\n" "-" ws yaml-value)*

mapping ::= pair ("\n" pair)*

pair ::= key ":" ws yaml-value

key ::= [a-zA-Z_] [a-zA-Z0-9_]*

boolean ::= "true" | "false" | "yes" | "no"
```

---

### B.5.4 自然语言约束

**特定格式回复：**

```gbnf
# 结构化回复
response ::= greeting ws body ws closing

greeting ::= "Dear " name ",\n\n"
name ::= [A-Z][a-z]* (" " [A-Z][a-z]*)?

body ::= paragraph ("\n\n" paragraph)*
paragraph ::= sentence+
sentence ::= [A-Z][^\.!?]*[\.!?] " "

closing ::= "\n\n" "Sincerely," "\n" signature
signature ::= [A-Z][a-z]* " " [A-Z][a-z]*
```

**选择题答案：**

```gbnf
# 选择题答案格式
answer ::= "Answer: " choice (" Explanation: " explanation)?

choice ::= "A" | "B" | "C" | "D"

explanation ::= [^.]+ "."
```

---

## B.6 在代码中使用 GBNF

### B.6.1 C API 使用

```c
#include "llama.h"

// 定义GBNF语法
const char * grammar_str = R"(
    root ::= object
    object ::= "{" ws pair ("," ws pair)* ws "}"
    pair ::= string ws ":" ws value
    value ::= string | number | "true" | "false" | "null"
    string ::= "\"" [a-zA-Z0-9]* "\""
    number ::= [0-9]+
    ws ::= [ \t\n\r]*
)";

// 创建语法采样器
struct llama_sampler * grammar_sampler = llama_sampler_init_grammar(
    vocab,           // 词汇表
    grammar_str,     // 语法字符串
    "root"           // 根规则名
);

// 添加到采样链
llama_sampler_chain_add(smpl, grammar_sampler);
```

### B.6.2 常见错误处理

```c
// 检查语法是否有效
struct llama_sampler * smpl = llama_sampler_init_grammar(vocab, grammar_str, "root");
if (smpl == NULL) {
    fprintf(stderr, "Failed to parse grammar\n");
    // 处理错误
}

// 常见错误：
// 1. 未定义的规则引用
// 2. 左递归
// 3. 无效的正则表达式（懒加载模式）
// 4. 不存在的token引用
```

### B.6.3 性能优化

**语法设计优化建议：**

1. **避免过度复杂的递归**：深度递归会降低性能
2. **限制重复次数**：使用 `{m,n}` 代替 `*` 和 `+` 当可能时
3. **简化字符类**：`[a-z]` 比 `[^...]` 的否定类更快
4. **预编译语法**：避免在循环中重复创建语法采样器

```c
// ✅ 好的做法：复用语法采样器
struct llama_sampler * create_grammar_sampler(const llama_vocab * vocab) {
    static const char * grammar = R"(
        root ::= [a-z]+
    )";
    
    return llama_sampler_init_grammar(vocab, grammar, "root");
}

// ❌ 避免：在每次生成时重新创建
for (int i = 0; i < n_iterations; i++) {
    // 不要这样做！
    struct llama_sampler * smpl = llama_sampler_init_grammar(vocab, grammar, "root");
    // ...
    llama_sampler_free(smpl);
}
```

---

## B.7 调试 GBNF 语法

### B.7.1 语法验证

```c
// 启用调试输出
setenv("LLAMA_LOG_LEVEL", "DEBUG", 1);

// 测试语法解析
struct llama_sampler * smpl = llama_sampler_init_grammar(vocab, grammar, "root");
if (smpl) {
    printf("Grammar parsed successfully\n");
    llama_sampler_free(smpl);
}
```

### B.7.2 常见问题

| 问题 | 原因 | 解决方案 |
|------|------|----------|
| 语法解析失败 | 未定义的规则 | 检查所有引用的规则是否已定义 |
| 无限递归 | 左递归 | 重写为右递归形式 |
| 性能低下 | 复杂递归或大量选择 | 简化语法，限制重复次数 |
| 约束不生效 | 语法根规则不匹配 | 确保根规则名称正确 |
| 意外token被拒绝 | 字符类过于严格 | 扩展字符类范围 |

### B.7.3 调试技巧

```gbnf
# 使用简单测试用例
# 原语法（复杂）
complex-json ::= object | array

# 调试版本（简化）
debug-root ::= "{" ws "\"test\"" ws ":" ws number ws "}"
number ::= [0-9]+
ws ::= " "*
```

---

## B.8 GBNF 与 JSON Schema 转换

llama.cpp 提供从 JSON Schema 自动生成 GBNF 的功能：

```c
#include "common/json-schema-to-grammar.h"

// JSON Schema
const char * json_schema = R"({
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer"}
    },
    "required": ["name", "age"]
})";

// 转换为GBNF
std::string grammar = json_schema_to_grammar(json_schema);

// 使用生成的语法
struct llama_sampler * smpl = llama_sampler_init_grammar(
    vocab, 
    grammar.c_str(), 
    "root"
);
```

**支持的 JSON Schema 特性：**

- 基本类型：string, number, integer, boolean, null
- 复合类型：object, array
- 约束：minLength, maxLength, pattern, enum
- 结构：properties, required, items, additionalProperties

**限制：**

- 不支持：$ref 引用外部定义
- 不支持：oneOf, anyOf, allOf（转换为简单选择）
- 不支持：条件验证（if/then/else）

---

## 本课小结

| GBNF元素 | 语法 | 示例 |
|----------|------|------|
| 字符类 | `[a-z]`, `[^0-9]` | `[a-zA-Z_]` |
| 量词 | `*`, `+`, `?`, `{m,n}` | `[0-9]+` |
| 选择 | `\|` | `"true" \| "false"` |
| 分组 | `()` | `("ab"\|"cd")+` |
| 字符串 | `"..."` | `"hello\nworld"` |
| Token引用 | `<token>` | `<\|endoftext\|>` |
| 注释 | `#` | `# 这是注释` |

**GBNF速查表：**

```gbnf
# 基础结构
root ::= object | array | value

# 数字
number ::= [+-]? [0-9]+ ("." [0-9]+)? ([eE] [+-]? [0-9]+)?

# 字符串
string ::= "\"" ([^"\\] | "\\" .)* "\""

# 标识符
identifier ::= [a-zA-Z_] [a-zA-Z0-9_]*

# 空白
ws ::= [ \t\n\r]*
```

---

*本附录对应源码版本：master (2026-04-07)*

