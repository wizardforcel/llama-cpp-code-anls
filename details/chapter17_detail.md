# 第17章 语法约束生成（llama_grammar）—— 让输出"规规矩矩"的魔法

## 1. 学习目标

通过本章学习，你将能够：
- 理解GBNF（GGML BNF）语法约束的基本原理
- 掌握GBNF语法规则的编写方法
- 深入理解语法解析器的实现机制
- 学会使用约束采样实现JSON、代码等结构化输出
- 能够将JSON Schema转换为GBNF语法

## 2. 生活类比：填空题与选择题

想象你在做一份试卷：

- **无约束生成**（自由写作）：开放性问题，可以任意发挥，但可能离题
- **语法约束**（填空题）：有明确的格式要求，必须在指定位置填入合适的内容
- **GBNF语法**（题目模板）：定义了填空的规则和可选项

例如，生成JSON就像做一道有严格格式的填空题：
```
姓名：____  年龄：____  邮箱：____
```
语法约束确保模型不会填错位置或格式。

## 3. 源码地图

```
src/llama-grammar.h            # 语法约束头文件
  ├── llama_gretype                    # 语法元素类型枚举
  ├── llama_grammar_element            # 语法元素结构
  ├── llama_grammar_candidate          # 候选token结构
  ├── llama_grammar_rule               # 语法规则
  ├── llama_grammar_parser             # 语法解析器
  └── llama_grammar                    # 语法约束主体

src/llama-grammar.cpp          # 语法约束实现（约1500行）
  ├── parse_rule()                     # 解析语法规则
  ├── parse_sequence()                 # 解析序列
  ├── parse_alternates()               # 解析交替
  ├── llama_grammar_accept()           # 接受字符
  ├── llama_grammar_apply()            # 应用约束
  └── llama_grammar_reject_candidates() # 过滤候选

common/json-schema-to-grammar.h/cpp  # JSON Schema转换
  └── json_schema_to_grammar()         # Schema转GBNF

src/llama-sampler.cpp          # 语法采样器
  └── llama_sampler_init_grammar()     # 语法采样器初始化
```

## 4. 详细章节内容

### 4.1 语法约束概述

#### 4.1.1 什么是GBNF

GBNF（GGML BNF）是llama.cpp使用的语法描述语言，基于EBNF（Extended Backus-Naur Form）扩展：

```ebnf
# 基本语法元素
root ::= expression                    # 规则定义
expr1 ::= "literal"                    # 字符串字面量
expr2 ::= [a-zA-Z]                     # 字符范围
expr3 ::= item+                        # 一次或多次
expr4 ::= item*                        # 零次或多次
expr5 ::= item?                        # 零次或一次
expr6 ::= alt1 | alt2 | alt3           # 多选一
expr7 ::= "(" group ")"                # 分组
```

**源码位置**：`src/llama-grammar.cpp:434-661`

#### 4.1.2 语法元素类型

**源码位置**：`src/llama-grammar.h:12-45`

```cpp
enum llama_gretype {
    LLAMA_GRETYPE_END            = 0,  // 规则结束
    LLAMA_GRETYPE_ALT            = 1,  // 交替定义开始
    LLAMA_GRETYPE_RULE_REF       = 2,  // 规则引用
    LLAMA_GRETYPE_CHAR           = 3,  // 字符匹配
    LLAMA_GRETYPE_CHAR_NOT       = 4,  // 反向字符匹配
    LLAMA_GRETYPE_CHAR_RNG_UPPER = 5,  // 字符范围上限
    LLAMA_GRETYPE_CHAR_ALT       = 6,  // 字符交替
    LLAMA_GRETYPE_CHAR_ANY       = 7,  // 任意字符
    LLAMA_GRETYPE_TOKEN          = 8,  // Token匹配
    LLAMA_GRETYPE_TOKEN_NOT      = 9,  // 反向Token匹配
};
```

**图解语法树**：
```
GBNF: number ::= "-"? [0-9]+ ("." [0-9]+)?

解析树:
              number
                 │
    ┌────────────┼────────────┐
    │            │            │
optional("-")  digits    optional
    │            │       ┌────┴────┐
   "-"       [0-9]+     "."    [0-9]+
                │                 │
              one+               one+
```

### 4.2 语法解析器实现

#### 4.2.1 解析器结构

**源码位置**：`src/llama-grammar.h:86-117`

```cpp
struct llama_grammar_parser {
    const llama_vocab * vocab;
    std::map<std::string, uint32_t> symbol_ids;  // 符号名到ID映射
    llama_grammar_rules rules;                    // 解析后的规则集
    
    // 解析入口
    bool parse(const char * src);
    
    // 解析单个规则
    const char * parse_rule(const char * src);
    
    // 解析交替（|）
    const char * parse_alternates(
        const char * src,
        const std::string & rule_name,
        uint32_t rule_id,
        bool is_nested);
    
    // 解析序列
    const char * parse_sequence(
        const char * src,
        const std::string & rule_name,
        llama_grammar_rule & rule,
        bool is_nested);
};
```

#### 4.2.2 规则解析流程

**源码位置**：`src/llama-grammar.cpp:663-719`

```cpp
const char * llama_grammar_parser::parse_rule(const char * src) {
    // 1. 解析规则名
    const char * name_end = parse_name(src);
    uint32_t rule_id = get_symbol_id(src, name_end - src);
    
    // 2. 期望 ::= 分隔符
    pos = parse_space(name_end, false);
    if (!(pos[0] == ':' && pos[1] == ':' && pos[2] == '=')) {
        throw std::runtime_error("expecting ::=");
    }
    
    // 3. 解析规则体
    pos = parse_space(pos + 3, true);
    pos = parse_alternates(pos, name, rule_id, false);
    
    return parse_space(pos, true);
}
```

**解析过程图解**：
```
输入: "number ::= \"-\"? [0-9]+"

步骤1: 识别规则名 "number"
        ↓
步骤2: 验证 ::= 分隔符
        ↓
步骤3: 解析规则体
        ├── \"-\"? → optional("-")
        ├── [0-9]+ → one_or_more(digit)
        └── 组合成序列
        ↓
输出: llama_grammar_rule 对象
```

#### 4.2.3 重复量词处理

**源码位置**：`src/llama-grammar.cpp:462-527`

```cpp
// 处理 *, +, ?, {m,n} 量词
auto handle_repetitions = [&](uint64_t min_times, uint64_t max_times) {
    // S{m,n} --> S S S (m times) S'(n-m)
    // S'     ::= S S' |
    // 
    // S*     --> S{0,}
    // S+     --> S{1,}
    // S?     --> S{0,1}
    
    // 展开重复为递归规则
    if (min_times == 0) {
        rule.resize(last_sym_start);
    } else {
        // 重复min_times-1次
        for (uint64_t i = 1; i < min_times; i++) {
            rule.insert(rule.end(), prev_rule.begin(), prev_rule.end());
        }
    }
    
    // 创建递归规则处理可选部分
    uint32_t rec_rule_id = generate_symbol_id(rule_name);
    rec_rule.push_back({LLAMA_GRETYPE_RULE_REF, no_max ? rec_rule_id : last_rec_rule_id});
    rec_rule.push_back({LLAMA_GRETYPE_ALT, 0});
    rec_rule.push_back({LLAMA_GRETYPE_END, 0});
    add_rule(rec_rule_id, rec_rule);
};
```

### 4.3 约束采样流程

#### 4.3.1 下推自动机（PDA）

**源码位置**：`src/llama-grammar.cpp:851-934`

```cpp
// 将语法栈转换为N个可能的栈，全部指向字符范围
static void llama_grammar_advance_stack(
        const llama_grammar_rules  & rules,
        const llama_grammar_stack  & stack,
        llama_grammar_stacks & new_stacks) {
    
    std::vector<llama_grammar_stack> todo;
    todo.push_back(stack);
    
    while (!todo.empty()) {
        llama_grammar_stack curr_stack = std::move(todo.back());
        todo.pop_back();
        
        if (curr_stack.empty()) {
            new_stacks.emplace_back(std::move(curr_stack));
            continue;
        }
        
        const llama_grammar_element * pos = curr_stack.back();
        
        switch (pos->type) {
        case LLAMA_GRETYPE_RULE_REF: {
            // 展开规则引用
            const size_t rule_id = static_cast<size_t>(pos->value);
            const llama_grammar_element * subpos = rules[rule_id].data();
            do {
                llama_grammar_stack next_stack(curr_stack.begin(), curr_stack.end() - 1);
                if (!llama_grammar_is_end_of_sequence(pos + 1)) {
                    next_stack.push_back(pos + 1);
                }
                if (!llama_grammar_is_end_of_sequence(subpos)) {
                    next_stack.push_back(subpos);
                }
                todo.push_back(std::move(next_stack));
                // 处理交替定义
                while (!llama_grammar_is_end_of_sequence(subpos)) {
                    subpos++;
                }
                if (subpos->type == LLAMA_GRETYPE_ALT) {
                    subpos++;
                } else {
                    break;
                }
            } while (true);
            break;
        }
        case LLAMA_GRETYPE_CHAR:
        case LLAMA_GRETYPE_CHAR_NOT:
            // 到达终结符，加入结果
            new_stacks.emplace_back(std::move(curr_stack));
            break;
        }
    }
}
```

**PDA状态转换图解**：
```
语法: expr ::= number | "(" expr ")"
       number ::= [0-9]+

初始栈: [expr]
           ↓
展开:    [number] 或 ["(", expr, ")"]
           ↓
继续:    [[0-9]] 或 ...
           ↓
接受字符后更新栈状态
```

#### 4.3.2 Token候选过滤

**源码位置**：`src/llama-grammar.cpp:1053-1122`

```cpp
llama_grammar_candidates llama_grammar_reject_candidates_for_stack(
        const llama_grammar_rules      & rules,
        const llama_grammar_stack      & stack,
        const llama_grammar_candidates & candidates) {
    
    llama_grammar_candidates rejects;
    rejects.reserve(candidates.size());
    
    if (stack.empty()) {
        // 栈为空，拒绝所有非空候选
        for (const auto & tok : candidates) {
            if (*tok.code_points != 0 || tok.partial_utf8.n_remain != 0) {
                rejects.push_back(tok);
            }
        }
        return rejects;
    }
    
    const llama_grammar_element * stack_pos = stack.back();
    
    for (const auto & tok : candidates) {
        if (*tok.code_points == 0) {
            // Token已完全匹配，检查部分UTF-8序列
            if (tok.partial_utf8.n_remain != 0 &&
                    !llama_grammar_match_partial_char(stack_pos, tok.partial_utf8)) {
                rejects.push_back(tok);
            }
        } else if (llama_grammar_match_char(stack_pos, *tok.code_points).first) {
            // 字符匹配成功，继续检查后续字符
            next_candidates.push_back({ tok.index, tok.code_points + 1, tok.partial_utf8, tok.id });
        } else {
            // 字符不匹配，拒绝
            rejects.push_back(tok);
        }
    }
    
    return rejects;
}
```

**过滤流程图解**：
```
输入候选: ["{", "a", "1", "\"", ...]
语法期望: object开始，需要"{"
           ↓
逐个检查候选:
  "{" → 匹配 ✓
  "a" → 不匹配 ✗
  "1" → 不匹配 ✗
  "\"" → 不匹配 ✗
           ↓
输出: 只允许"{"，其他设为 -inf
```

#### 4.3.3 增量解析

**源码位置**：`src/llama-grammar.cpp:1016-1051`

```cpp
void llama_grammar_accept(struct llama_grammar * grammar, uint32_t chr) {
    llama_grammar_stacks stacks_new;
    stacks_new.reserve(grammar->stacks.size());
    
    for (const auto & stack : grammar->stacks) {
        llama_grammar_accept_chr(*grammar, stack, chr, stacks_new);
    }
    
    grammar->stacks = std::move(stacks_new);
}

static void llama_grammar_accept_chr(
        struct llama_grammar       & grammar,
        const llama_grammar_stack  & stack,
              uint32_t               chr,
              llama_grammar_stacks & new_stacks) {
    if (stack.empty()) return;
    
    const llama_grammar_element * pos = stack.back();
    
    auto match = llama_grammar_match_char(pos, chr);
    if (match.first) {
        // 字符匹配成功，更新栈
        llama_grammar_stack new_stack(stack.begin(), stack.end() - 1);
        if (!llama_grammar_is_end_of_sequence(match.second)) {
            new_stack.push_back(match.second);
        }
        llama_grammar_advance_stack(grammar.rules, new_stack, new_stacks);
    }
}
```

### 4.4 JSON Schema约束

#### 4.4.1 Schema到GBNF转换

**源码位置**：`common/json-schema-to-grammar.cpp`

```cpp
std::string json_schema_to_grammar(const json & schema) {
    // 1. 解析Schema结构
    // 2. 生成对应的GBNF规则
    // 3. 处理嵌套对象和数组
    
    std::string grammar = R"(
root ::= object

object ::= "{" ws object-content "}" ws

object-content ::= 
    | string ":" value ("," ws string ":" value)*
    | ""

array ::= "[" ws array-content "]" ws

array-content ::=
    | value ("," ws value)*
    | ""

value ::= object | array | string | number | boolean | null

string ::= "\"" char* "\"" ws

number ::= ("-")? int frac? exp? ws

boolean ::= ("true" | "false") ws

null ::= "null" ws

ws ::= [ \t\n\r]*
)";
    
    return grammar;
}
```

#### 4.4.2 使用示例

**定义JSON Schema**：
```json
{
  "type": "object",
  "properties": {
    "name": {"type": "string"},
    "age": {"type": "integer", "minimum": 0},
    "email": {"type": "string", "format": "email"}
  },
  "required": ["name", "age"]
}
```

**转换后的GBNF**：
```ebnf
root ::= object

object ::= "{" ws object-content "}" ws

object-content ::=
    | "\"name\"" ws ":" ws string 
      ("," ws "\"age\"" ws ":" ws integer)?
      ("," ws "\"email\"" ws ":" ws string)?
    | "\"age\"" ws ":" ws integer
      ("," ws "\"name\"" ws ":" ws string)?
      ("," ws "\"email\"" ws ":" ws string)?

string ::= "\"" char* "\""

integer ::= "-"? ([0-9] [0-9]*)

ws ::= [ \t\n\r]*
```

**C++代码使用**：
```cpp
// 从Schema生成语法
std::string grammar = json_schema_to_grammar(schema);

// 创建语法采样器
llama_sampler * smpl = llama_sampler_init_grammar(
    vocab,
    grammar.c_str(),
    "root"  // 起始规则
);

// 使用采样器生成
llama_token token = llama_sampler_sample(smpl, ctx, 0);
```

### 4.5 懒加载语法（Lazy Grammar）

**源码位置**：`src/llama-grammar.h:142-149`

```cpp
struct llama_grammar {
    // 懒加载语法等待触发词或token后才激活约束
    bool lazy = false;
    bool awaiting_trigger = false;
    std::string trigger_buffer;
    std::vector<token_pos> trigger_buffer_positions;
    std::vector<llama_token> trigger_tokens;
    std::vector<llama_grammar_trigger_pattern> trigger_patterns;
};
```

**使用场景**：
- 工具调用：等待`<function_call>`标签后激活JSON语法
- 代码生成：等待```python后激活Python语法

**代码示例**：
```cpp
// 定义触发token
llama_token trigger_tokens[] = {function_call_token_id};

// 创建懒加载语法采样器
llama_sampler * smpl = llama_sampler_init_grammar_lazy(
    vocab,
    json_grammar,      // JSON语法
    "root",
    nullptr, 0,        // 无触发词
    trigger_tokens, 1  // 触发token
);
```

## 5. 设计中的取舍

### 5.1 性能与表达力

| 特性 | 优点 | 缺点 |
|------|------|------|
| 完整GBNF | 表达力强 | 解析开销大 |
| 简化语法 | 解析快 | 表达能力受限 |
| 预编译语法 | 运行时快 | 内存占用增加 |

### 5.2 约束严格度

- **严格约束**：保证输出格式正确，但可能降低生成质量
- **宽松约束**：允许更多灵活性，但需要后处理验证

### 5.3 增量解析vs批量解析

- **增量解析**（当前实现）：
  - 优点：内存效率高，适合流式生成
  - 缺点：状态管理复杂

- **批量解析**（替代方案）：
  - 优点：实现简单
  - 缺点：需要预先生成完整序列

## 6. 动手练习

### 练习1：编写自定义GBNF语法

编写一个语法约束生成简单的算术表达式：
```ebnf
expr ::= term (("+" | "-") term)*
term ::= factor (("*" | "/") factor)*
factor ::= number | "(" expr ")"
number ::= [0-9]+ ("." [0-9]+)?
```

### 练习2：调试语法解析

使用llama-cli测试语法约束：
```bash
# 创建语法文件
cat > json.gbnf << 'EOF'
root ::= object
object ::= "{" ws pair ("," ws pair)* "}" ws
pair ::= string ":" ws value
string ::= "\"" [a-zA-Z]+ "\""
value ::= string | number
number ::= [0-9]+
ws ::= [ \t\n]*
EOF

# 使用语法约束生成
./llama-cli -m model.gguf --grammar-file json.gbnf -p "Generate JSON:"
```

### 练习3：实现JSON验证器

```cpp
// 验证生成的JSON是否符合Schema
bool validate_json(const std::string & json, const json & schema) {
    // 解析JSON
    auto parsed = json::parse(json);
    
    // 验证类型
    if (schema["type"] == "object") {
        // 验证对象结构
        for (auto &[prop, subschema] : schema["properties"].items()) {
            if (!parsed.contains(prop)) {
                if (schema["required"].contains(prop)) {
                    return false;  // 缺少必需字段
                }
                continue;
            }
            // 递归验证子schema
            if (!validate_json(parsed[prop].dump(), subschema)) {
                return false;
            }
        }
    }
    return true;
}
```

## 7. 本课小结

本章我们深入学习了llama.cpp的语法约束系统：

1. **GBNF语法**：基于EBNF的语法描述语言，支持字符、Token、重复、交替等
2. **解析器实现**：递归下降解析器将GBNF转换为内部规则表示
3. **约束采样**：使用下推自动机跟踪解析状态，过滤不符合语法的Token
4. **JSON Schema**：自动转换JSON Schema为GBNF，实现结构化输出
5. **懒加载**：支持触发词/Token机制，延迟激活语法约束

**关键源码文件**：`src/llama-grammar.cpp`、`src/llama-grammar.h`、`common/json-schema-to-grammar.cpp`

**下一步**：学习投机解码、前瞻解码等高级生成技术，进一步提升推理速度。
