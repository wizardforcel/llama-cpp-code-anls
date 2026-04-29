# 第17章 语法约束生成 —— 让输出"规规矩矩"的魔法

当你要求AI模型生成JSON数据时，最担心什么？格式错误、缺少引号、类型不匹配……传统的大模型生成是"自由发挥"模式，输出虽然流畅却可能不符合严格的格式要求。语法约束系统就像一位严格的格式检查员，它能够在生成过程中实时过滤不符合规则的token，确保最终输出完全符合预定义的语法规范。无论是JSON、SQL还是自定义的领域特定语言，GBNF语法约束都能让模型的输出"规规矩矩"。

## 学习目标

1. 理解GBNF（GGML BNF）语法约束的基本原理
2. 掌握GBNF语法规则的编写方法
3. 深入理解语法解析器的实现机制
4. 学会使用约束采样实现JSON、代码等结构化输出
5. 能够将JSON Schema转换为GBNF语法

## 生活类比：填空题与选择题

想象你在做一份试卷。无约束生成就像自由写作题——你可以任意发挥，可能写出优美的散文，但它不一定是你需要的那种格式。而语法约束则像填空题：每个空白处都有明确的格式要求，必须在指定位置填入合适的内容，就像填写表格时每个字段都有严格的规则。GBNF 语法就是那份题目模板，它定义了填空的规则和可选项，告诉模型：这里需要一个数字，那里需要一个字符串，括号必须成对出现。

例如，生成 JSON 就像做一道有严格格式的填空题：姓名、年龄、邮箱各有各的格式，模型不能把字符串填到数字的位置，也不能忘记闭合引号或括号。

这种约束并不是限制模型的创造力，而是确保输出在结构上是正确的。模型在符合格式的框架内仍然可以发挥语言能力——就像填空题中的每个空白处，你依然可以填入最贴切的答案。

## 源码地图

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

## 17.1 语法约束概述

### 17.1.1 什么是GBNF

GBNF（GGML BNF）是llama.cpp使用的语法描述语言，基于EBNF（Extended Backus-Naur Form）扩展。它允许你定义生成文本的精确格式规则。

**基本语法元素**：
```ebnf
root ::= expression                    # 规则定义
expr1 ::= "literal"                    # 字符串字面量
expr2 ::= [a-zA-Z]                     # 字符范围
expr3 ::= item+                        # 一次或多次
expr4 ::= item*                        # 零次或多次
expr5 ::= item?                        # 零次或一次
expr6 ::= alt1 | alt2 | alt3           # 多选一
expr7 ::= "(" group ")"                # 分组
```

**实际示例**：
```ebnf
# 定义一个数字：可选负号 + 一个或多个数字 + 可选小数部分
number ::= "-"? [0-9]+ ("." [0-9]+)?

# 定义一个字符串：双引号 + 零个或多个字符 + 双引号
string ::= "\"" char* "\""

# 定义字符：除双引号外的任意字符
char ::= [^"]
```

### 17.1.2 语法元素类型

**源码位置**：`src/llama-grammar.h`（第12-45行）

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

**语法树图解**：
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

这个语法树展示了`number`规则的结构：
- 可选的负号（`-`）
- 一个或多个数字（`[0-9]+`）
- 可选的小数部分（`.`后跟一个或多个数字）

### 17.1.3 为什么需要语法约束

**问题场景**：

1. **JSON生成**：模型可能生成`{name: "John"}`（缺少引号）或`{"name": "John",}`（多余逗号）
2. **SQL生成**：模型可能生成语法错误的SQL语句
3. **代码生成**：缩进、括号匹配等格式问题

**传统解决方案的局限**：

- **后处理验证**：生成后验证并重新生成，效率低下
- **提示工程**：通过提示词要求格式，但不保证结果
- **微调模型**：成本高，灵活性差

**语法约束的优势**：

- **实时过滤**：在生成过程中实时约束，保证100%格式正确
- **零样本**：无需微调，即插即用
- **灵活性**：可随时更换语法规则

## 17.2 语法解析器实现

### 17.2.1 解析器结构

**源码位置**：`src/llama-grammar.h`（第86-117行）

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

**解析器工作流程**：

1. **词法分析**：将GBNF字符串分解为token
2. **语法分析**：根据EBNF语法规则构建解析树
3. **规则生成**：将解析树转换为内部规则表示

### 17.2.2 规则解析流程

**源码位置**：`src/llama-grammar.cpp`（第663-719行）

```cpp
const char * llama_grammar_parser::parse_rule(const char * src) {
    // 1. 解析规则名
    const char * name_end = parse_name(src);
    std::string name(src, name_end - src);
    uint32_t rule_id = get_symbol_id(name);
    
    // 2. 期望 ::= 分隔符
    const char * pos = parse_space(name_end, false);
    if (!(pos[0] == ':' && pos[1] == ':' && pos[2] == '=')) {
        throw std::runtime_error("expecting ::=");
    }
    
    // 3. 解析规则体
    pos = parse_space(pos + 3, true);
    pos = parse_alternates(pos, name, rule_id, false);
    
    // 4. 添加规则到规则集
    add_rule(rule_id, rules[rule_id]);
    
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

### 17.2.3 重复量词处理

**源码位置**：`src/llama-grammar.cpp`（第462-527行）

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
    if (max_times > min_times || !has_max) {
        uint32_t rec_rule_id = generate_symbol_id(rule_name);
        llama_grammar_rule rec_rule;
        rec_rule.insert(rec_rule.end(), prev_rule.begin(), prev_rule.end());
        rec_rule.push_back({LLAMA_GRETYPE_RULE_REF, 
                           no_max ? rec_rule_id : last_rec_rule_id});
        rec_rule.push_back({LLAMA_GRETYPE_ALT, 0});
        rec_rule.push_back({LLAMA_GRETYPE_END, 0});
        add_rule(rec_rule_id, rec_rule);
        
        if (min_times == 0) {
            rule.push_back({LLAMA_GRETYPE_RULE_REF, rec_rule_id});
        }
    }
};
```

**量词转换原理**：

GBNF中的重复量词（`*`, `+`, `?`）被转换为递归规则：

```
A*  →  A_star
A_star ::= A A_star |     // 递归：A后跟A_star，或空

A+  →  A A_star           // 至少一个A，后跟零个或多个A

A?  →  A_optional
A_optional ::= A |        // A或空
```

这种转换的原因：下推自动机更容易处理递归规则而非重复量词。

## 17.3 约束采样流程

### 17.3.1 下推自动机（PDA）

语法约束的核心是一个下推自动机（Pushdown Automaton），它跟踪当前的解析状态。

**源码位置**：`src/llama-grammar.cpp`（第851-934行）

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
                llama_grammar_stack next_stack(curr_stack.begin(), 
                                               curr_stack.end() - 1);
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

**为什么使用下推自动机？**

1. **表达力强**：可以处理嵌套结构（如JSON对象嵌套）
2. **增量处理**：可以逐字符处理，适合流式生成
3. **状态清晰**：栈明确记录了当前期望的语法结构

### 17.3.2 Token候选过滤

**源码位置**：`src/llama-grammar.cpp`（第1053-1122行）

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
    llama_grammar_candidates next_candidates;
    
    for (const auto & tok : candidates) {
        if (*tok.code_points == 0) {
            // Token已完全匹配，检查部分UTF-8序列
            if (tok.partial_utf8.n_remain != 0 &&
                    !llama_grammar_match_partial_char(stack_pos, tok.partial_utf8)) {
                rejects.push_back(tok);
            }
        } else if (llama_grammar_match_char(stack_pos, *tok.code_points).first) {
            // 字符匹配成功，继续检查后续字符
            next_candidates.push_back({
                tok.index, 
                tok.code_points + 1, 
                tok.partial_utf8, 
                tok.id
            });
        } else {
            // 字符不匹配，拒绝
            rejects.push_back(tok);
        }
    }
    
    // 递归处理下一个字符
    if (!next_candidates.empty()) {
        auto next_rejects = llama_grammar_reject_candidates_for_stack(
            rules, stack, next_candidates);
        rejects.insert(rejects.end(), next_rejects.begin(), next_rejects.end());
    }
    
    return rejects;
}
```

**过滤流程图解**：
```
输入候选: ["{", "a", "1", """, ...]
当前栈顶: object开始，期望"{"
           ↓
逐个检查候选:
  "{" → 匹配 ✓ (允许)
  "a" → 不匹配 ✗ (拒绝)
  "1" → 不匹配 ✗ (拒绝)
  "\"" → 不匹配 ✗ (拒绝)
           ↓
输出: 只允许"{"，其他token的logit设为 -inf
```

**为什么拒绝而非选择？**

实现上选择"拒绝"而非"允许"有两个原因：
1. **效率**：通常大部分候选都会被拒绝，记录拒绝列表更短
2. **安全**：默认拒绝可以避免意外允许不符合语法的token

### 17.3.3 增量解析

**源码位置**：`src/llama-grammar.cpp`（第1016-1051行）

```cpp
void llama_grammar_accept(struct llama_grammar * grammar, uint32_t chr) {
    llama_grammar_stacks stacks_new;
    stacks_new.reserve(grammar->stacks.size());
    
    for (const auto & stack : grammar->stacks) {
        llama_grammar_accept_chr(*grammar, stack, chr, stacks_new);
    }
    
    grammar->stacks = std::move(stacks_new);
    
    // 检查是否有有效栈
    if (grammar->stacks.empty()) {
        throw std::runtime_error("grammar error: no valid stack after accepting char");
    }
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

**增量解析流程**：

1. 模型生成一个字符
2. 调用`llama_grammar_accept`更新解析状态
3. 语法栈根据字符进行状态转换
4. 下一个token的候选根据新栈状态进行过滤

这种增量方式使得语法约束可以与流式生成无缝集成。

## 17.4 JSON Schema约束

### 17.4.1 Schema到GBNF转换

**源码位置**：`common/json-schema-to-grammar.cpp`

```cpp
std::string json_schema_to_grammar(const json & schema) {
    std::string grammar = R"(
root ::= object

object ::= "{" ws object-content "}" ws

object-content ::= 
    | string ":" ws value ("," ws string ":" value)*
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

**转换原理**：

1. **对象转换**：`{"type": "object", "properties": {...}}` → `object`规则
2. **数组转换**：`{"type": "array"}` → `array`规则
3. **基本类型**：`string`, `number`, `boolean` → 对应规则
4. **嵌套处理**：递归处理嵌套对象和数组

### 17.4.2 使用示例

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
json schema = json::parse(R"({
  "type": "object",
  "properties": {
    "name": {"type": "string"},
    "age": {"type": "integer"}
  }
})");

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

## 17.5 懒加载语法（Lazy Grammar）

### 17.5.1 懒加载语法结构

**源码位置**：`src/llama-grammar.h`（第142-149行）

```cpp
struct llama_grammar {
    // 懒加载语法等待触发词或token后才激活约束
    bool lazy = false;
    bool awaiting_trigger = false;
    std::string trigger_buffer;
    std::vector<token_pos> trigger_buffer_positions;
    std::vector<llama_token> trigger_tokens;
    std::vector<llama_grammar_trigger_pattern> trigger_patterns;
    
    llama_grammar_rules rules;
    llama_grammar_stacks stacks;
};
```

### 17.5.2 使用场景

**工具调用场景**：
```
用户: 查询北京的天气
模型: 我来为您查询北京的天气<function_call>{"name": "get_weather", "args": {"city": "北京"}}</function_call>
```

在这个场景中，语法约束只在`<function_call>`标签后激活，确保函数参数是有效的JSON。

**代码生成场景**：
````
用户: 写一个Python函数计算阶乘
模型: ```python
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)
```
````

语法约束在```python后激活，确保生成有效的Python代码。

### 17.5.3 代码示例

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

**懒加载工作流程**：

1. 正常生成直到遇到触发token
2. 激活语法约束
3. 后续生成必须符合语法规则
4. 语法完成后可自动或手动解除约束

## 17.6 设计中的取舍

### 为什么选择 GBNF 语法？

GBNF 在表达力和实现复杂度之间找到了平衡。完整的 EBNF 语法虽然表达力强、能描述任意上下文无关文法，但解析开销大，会拖慢 token 生成速度。简化语法则解析快，适合简单的格式约束，但表达能力受限。GBNF 选择了中间路线——支持大部分常用语法元素（规则引用、字符类、重复量词、交替），同时保持解析器足够轻量，使其可以在每次 token 采样时高效运行。预编译语法可以进一步优化运行时性能，但需要额外的内存来存储编译后的规则表示。

### 为什么选择严格约束而不是宽松约束？

llama.cpp 选择严格约束路线：保证输出格式 100% 正确，实时过滤不符合语法的 token。这种设计虽然可能在极少数情况下迫使模型选择次优 token 从而略微降低流畅度，但换来的是结构化输出的确定性。宽松约束虽然允许更多灵活性，但需要后处理验证——如果验证失败还得重新生成，这在批处理和 API 场景下代价更高。严格约束下，可以在语法设计上提供足够的灵活性（如可选字段、重复次数范围等），让模型在合规框架内仍有发挥空间。

### 为什么采用增量解析？

增量解析每次只处理一个字符，内存效率高，天然适配流式生成场景。llama.cpp 的核心使用场景就是逐 token 的流式输出，因此下推自动机每次接受一个字符后更新栈状态，即时判断下一个位置允许哪些字符。批量解析虽然实现更简单——先生成完整序列再验证——但不符合流式生成的实时性要求。增量解析的代价是状态管理更复杂（需要维护多个可能的语法栈），但这个代价完全值得。

## 17.7 动手练习

### 练习1：编写自定义GBNF语法

编写一个语法约束生成简单的算术表达式：

```ebnf
# 表达式：项（加减 项）*
expr ::= term (("+" | "-") term)*

# 项：因子（乘除 因子）*
term ::= factor (("*" | "/") factor)*

# 因子：数字或括号表达式
factor ::= number | "(" expr ")"

# 数字：整数或小数
number ::= [0-9]+ ("." [0-9]+)?

# 空白字符
ws ::= [ \t\n]*
```

**测试用例**：
- 有效：`1+2`, `(3+4)*5`, `1.5+2.5`
- 无效：`1++2`, `*3`, `(1+2`

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

### 练习3：设计SQL语法约束

设计一个GBNF语法，约束模型生成有效的SELECT语句：

```ebnf
select ::= "SELECT" ws column_list ws "FROM" ws table_name 
           (ws "WHERE" ws condition)?

column_list ::= "*" | column ("," ws column)*
column ::= [a-zA-Z_][a-zA-Z0-9_]*
table_name ::= [a-zA-Z_][a-zA-Z0-9_]*
condition ::= column ws operator ws value
operator ::= "=" | "!=" | "<" | ">" | "<=" | ">="
value ::= number | "'" [^']* "'"
number ::= [0-9]+
ws ::= [ \t]+
```

## 17.8 本章小结

本章深入解析了语法约束系统。GBNF是llama.cpp的语法描述语言，基于EBNF扩展设计。语法解析器采用递归下降算法，将GBNF语法转换为内部规则表示。下推自动机用于跟踪解析状态，使用栈结构处理嵌套结构。候选过滤机制根据语法规则拒绝不符合的token候选。增量解析技术逐字符更新解析状态，支持流式生成场景。JSON Schema转换功能可以自动将JSON Schema转换为GBNF语法。懒加载语法机制等待触发词出现后激活约束，支持工具调用等场景。

语法约束采用"边生成边检查"的模式。传统方法生成后验证，错误率高且需要重新生成；语法约束在生成过程中实时过滤，保证100%格式正确。这就像在填空时就检查答案格式，而不是交卷后才发现格式错误。

本章我们一起学习了以下概念：

| 概念 | 解释 |
|------|------|
| GBNF | GGML BNF 语法描述语言，基于 EBNF 扩展，定义生成文本的精确格式规则 |
| 下推自动机 | 使用栈跟踪解析状态，支持嵌套结构（如 JSON 对象嵌套）的增量处理 |
| 候选过滤 | 根据语法规则拒绝不符合的 token，将 logit 设为 -inf 以排除 |
| 增量解析 | 逐字符更新解析状态，与流式生成无缝集成 |
| 懒加载语法 | 等待触发词或 token 出现后才激活约束，适用于工具调用等场景 |
| JSON Schema 转换 | 自动将 JSON Schema 转换为 GBNF 语法，简化结构化输出的配置 |

**下一章预告**：

下一章中，我们将学习高级生成技术，理解投机解码、前瞻解码和查找解码等加速方法，让模型生成更快更高效。
