# 第14章 聊天模板系统 —— 对话格式的"智能裁缝"

## 学习目标
1. 理解聊天模板的作用和设计原理
2. 掌握llama-chat的模板匹配机制
3. 了解多种对话格式的实现差异
4. 能自定义和应用聊天模板

---

## 生活类比：高级定制服装店的"格式裁缝"

想象llama-chat是一家**专门定制对话格式的服装店**：

- **聊天模板** = 服装款式模板
  - ChatML款式：`<|im_start|>user\n...<|im_end|>`
  - Llama2款式：`[INST] ... [/INST]`
  - Llama3款式：`<|start_header_id|>user<|end_header_id|>`
  - 每种模型都有自己的"款式"

- **模板检测** = 款式识别系统
  - 通过特征识别模板类型
  - 如看到`<|im_start|>`就知道是ChatML
  - 如看到`[INST]`就知道是Llama2风格

- **模板应用** = 裁剪缝制
  - 将用户消息按模板格式组装
  - 添加必要的标记和分隔符
  - 生成模型可理解的提示文本

就像裁缝需要根据客户选择的面料和款式制作服装，聊天模板需要根据模型类型格式化对话。

---

## 源码地图

```
src/llama-chat.h
├── llm_chat_template          # 聊天模板类型枚举
│   ├── LLM_CHAT_TEMPLATE_CHATML
│   ├── LLM_CHAT_TEMPLATE_LLAMA_2
│   ├── LLM_CHAT_TEMPLATE_LLAMA_3
│   ├── LLM_CHAT_TEMPLATE_MISTRAL_V1/V3/V7
│   └── ... 更多模板
├── llm_chat_template_from_str()    # 字符串转模板类型
├── llm_chat_detect_template()      # 自动检测模板
└── llm_chat_apply_template()       # 应用模板

src/llama-chat.cpp
├── LLM_CHAT_TEMPLATES map     # 模板名称映射表
├── llm_chat_detect_template() # 模板检测实现（第88-236行）
└── llm_chat_apply_template()  # 模板应用实现（第240-928行）

common/chat-parser.cpp
└── 聊天消息解析辅助函数
```

---

## 14.1 聊天模板概述

### 14.1.1 为什么需要聊天模板

**问题背景**：
- 不同模型训练时使用的对话格式不同
- 直接使用原始文本会导致模型理解错误
- 需要在推理时复现训练时的格式

**示例对比**：

**用户输入**：
```json
[
  {"role": "system", "content": "You are a helpful assistant."},
  {"role": "user", "content": "Hello!"},
  {"role": "assistant", "content": "Hi there!"}
]
```

**Llama2格式**：
```
<s>[INST] <<SYS>>
You are a helpful assistant.
<</SYS>>

Hello! [/INST] Hi there! </s>
```

**ChatML格式**：
```
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
Hello!<|im_end|>
<|im_start|>assistant
Hi there!<|im_end|>
```

**Llama3格式**：
```
<|start_header_id|>system<|end_header_id|>

You are a helpful assistant.<|eot_id|>
<|start_header_id|>user<|end_header_id|>

Hello!<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>

Hi there!<|eot_id|>
```

### 14.1.2 模板类型枚举

**源码位置**：`src/llama-chat.h` (第7-63行)

```cpp
enum llm_chat_template {
    LLM_CHAT_TEMPLATE_CHATML,           // ChatML格式
    LLM_CHAT_TEMPLATE_LLAMA_2,          // Llama2基础版
    LLM_CHAT_TEMPLATE_LLAMA_2_SYS,      // Llama2带系统消息
    LLM_CHAT_TEMPLATE_LLAMA_2_SYS_BOS,  // Llama2带BOS
    LLM_CHAT_TEMPLATE_MISTRAL_V1,       // Mistral v1
    LLM_CHAT_TEMPLATE_MISTRAL_V3,       // Mistral v3
    LLM_CHAT_TEMPLATE_MISTRAL_V7,       // Mistral v7
    LLM_CHAT_TEMPLATE_PHI_3,            // Phi-3
    LLM_CHAT_TEMPLATE_PHI_4,            // Phi-4
    LLM_CHAT_TEMPLATE_GEMMA,            // Gemma
    LLM_CHAT_TEMPLATE_LLAMA_3,          // Llama3
    LLM_CHAT_TEMPLATE_CHATGLM_3,        // ChatGLM3
    LLM_CHAT_TEMPLATE_CHATGLM_4,        // ChatGLM4
    LLM_CHAT_TEMPLATE_DEEPSEEK,         // DeepSeek
    LLM_CHAT_TEMPLATE_DEEPSEEK_2,       // DeepSeek-V2
    LLM_CHAT_TEMPLATE_DEEPSEEK_3,       // DeepSeek-V3
    LLM_CHAT_TEMPLATE_COMMAND_R,        // Command-R
    LLM_CHAT_TEMPLATE_QWEN2,            // Qwen2
    // ... 更多模板
    LLM_CHAT_TEMPLATE_UNKNOWN,          // 未知模板
};
```

---

## 14.2 模板检测机制

### 14.2.1 检测函数实现

**源码位置**：`src/llama-chat.cpp` (第88-236行)

```cpp
llm_chat_template llm_chat_detect_template(const std::string & tmpl) {
    // 先尝试直接匹配模板名称
    try {
        return llm_chat_template_from_str(tmpl);
    } catch (const std::out_of_range & ) {
        // 不是直接名称，继续检测
    }

    // 辅助lambda：检查模板是否包含某字符串
    auto tmpl_contains = [&tmpl](const char * haystack) -> bool {
        return tmpl.find(haystack) != std::string::npos;
    };

    // ChatML检测
    if (tmpl_contains("<|im_start|>")) {
        if (tmpl_contains("<|im_sep|>")) {
            return LLM_CHAT_TEMPLATE_PHI_4;
        } else if (tmpl_contains("<end_of_utterance>")) {
            return LLM_CHAT_TEMPLATE_SMOLVLM;
        }
        return LLM_CHAT_TEMPLATE_CHATML;
    }

    // Mistral检测
    else if (tmpl.find("mistral") == 0 || tmpl_contains("[INST]")) {
        if (tmpl_contains("[SYSTEM_PROMPT]")) {
            return LLM_CHAT_TEMPLATE_MISTRAL_V7;
        } else if (tmpl_contains("' [INST] ' + system_message") ||
                   tmpl_contains("[AVAILABLE_TOOLS]")) {
            // 官方Mistral模板
            if (tmpl_contains(" [INST]")) {
                return LLM_CHAT_TEMPLATE_MISTRAL_V1;
            } else if (tmpl_contains("\"[INST]\"")) {
                return LLM_CHAT_TEMPLATE_MISTRAL_V3_TEKKEN;
            }
            return LLM_CHAT_TEMPLATE_MISTRAL_V3;
        } else {
            // Llama2风格变体
            bool support_system = tmpl_contains("<<SYS>>");
            bool add_bos_inside = tmpl_contains("bos_token + '[INST]");
            bool strip_message = tmpl_contains("content.strip()");
            
            if (strip_message) return LLM_CHAT_TEMPLATE_LLAMA_2_SYS_STRIP;
            if (add_bos_inside) return LLM_CHAT_TEMPLATE_LLAMA_2_SYS_BOS;
            if (support_system) return LLM_CHAT_TEMPLATE_LLAMA_2_SYS;
            return LLM_CHAT_TEMPLATE_LLAMA_2;
        }
    }

    // Llama3检测
    else if (tmpl_contains("<|start_header_id|>") && 
             tmpl_contains("<|end_header_id|>")) {
        return LLM_CHAT_TEMPLATE_LLAMA_3;
    }

    // DeepSeek检测
    else if (tmpl_contains("<｜User｜>") && 
             tmpl_contains("<｜Assistant｜>")) {
        return LLM_CHAT_TEMPLATE_DEEPSEEK_3;
    }

    // ... 更多检测规则

    return LLM_CHAT_TEMPLATE_UNKNOWN;
}
```

### 14.2.2 检测规则图解

```
┌─────────────────────────────────────────────────────────────────┐
│                     模板检测流程                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  输入: tokenizer_config.json中的chat_template字符串             │
│                                                                 │
│  ① 直接名称匹配?                                                │
│     "chatml" → LLM_CHAT_TEMPLATE_CHATML                         │
│     "llama2" → LLM_CHAT_TEMPLATE_LLAMA_2                        │
│                                                                 │
│  ② 特征检测（按优先级）                                          │
│                                                                 │
│     包含"<|im_start|>"?                                          │
│     ├─ 是 → 包含"<|im_sep|>"? → Phi-4                           │
│     │       否 → ChatML                                         │
│     │                                                           │
│     否 → 包含"[INST]"?                                          │
│     ├─ 是 → 包含"[SYSTEM_PROMPT]"? → Mistral-V7                 │
│     │       包含"[AVAILABLE_TOOLS]"? → Mistral-V3               │
│     │       包含"<<SYS>>"? → Llama2-Sys                        │
│     │       否则 → Llama2                                       │
│     │                                                           │
│     否 → 包含"<|start_header_id|>"? → Llama3                    │
│     │                                                           │
│     否 → 包含"<｜User｜>"? → DeepSeek-3                          │
│     │                                                           │
│     否 → ... 更多规则                                           │
│                                                                 │
│  ③ 无匹配 → LLM_CHAT_TEMPLATE_UNKNOWN                           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 14.3 模板应用实现

### 14.3.1 主应用函数

**源码位置**：`src/llama-chat.cpp` (第240-260行)

```cpp
int32_t llm_chat_apply_template(
    llm_chat_template tmpl,                           // 模板类型
    const std::vector<const llama_chat_message *> & chat,  // 消息列表
    std::string & dest,                               // 输出目标
    bool add_ass) {                                   // 是否添加assistant提示

    std::stringstream ss;

    // 根据模板类型选择格式化逻辑
    switch (tmpl) {
        case LLM_CHAT_TEMPLATE_CHATML:
            // ChatML格式化
            for (auto message : chat) {
                ss << "<|im_start|>" << message->role 
                   << "\n" << message->content << "<|im_end|>\n";
            }
            if (add_ass) {
                ss << "<|im_start|>assistant\n";
            }
            break;

        case LLM_CHAT_TEMPLATE_LLAMA_3:
            // Llama3格式化
            for (auto message : chat) {
                ss << "<|start_header_id|>" << message->role 
                   << "<|end_header_id|>\n\n" << trim(message->content) 
                   << "<|eot_id|>";
            }
            if (add_ass) {
                ss << "<|start_header_id|>assistant<|end_header_id|>\n\n";
            }
            break;

        // ... 更多模板实现
    }

    dest = ss.str();
    return dest.size();
}
```

### 14.3.2 具体模板实现

**ChatML模板**（第246-253行）：
```cpp
case LLM_CHAT_TEMPLATE_CHATML:
    for (auto message : chat) {
        ss << "<|im_start|>" << message->role 
           << "\n" << message->content << "<|im_end|>\n";
    }
    if (add_ass) {
        ss << "<|im_start|>assistant\n";
    }
    break;
```

**Llama2模板**（第295-331行）：
```cpp
case LLM_CHAT_TEMPLATE_LLAMA_2:
case LLM_CHAT_TEMPLATE_LLAMA_2_SYS:
case LLM_CHAT_TEMPLATE_LLAMA_2_SYS_BOS:
case LLM_CHAT_TEMPLATE_LLAMA_2_SYS_STRIP: {
    bool support_system = tmpl != LLM_CHAT_TEMPLATE_LLAMA_2;
    bool add_bos_inside = tmpl == LLM_CHAT_TEMPLATE_LLAMA_2_SYS_BOS;
    bool strip_message = tmpl == LLM_CHAT_TEMPLATE_LLAMA_2_SYS_STRIP;

    bool is_inside_turn = true;
    ss << "[INST] ";

    for (auto message : chat) {
        std::string content = strip_message ? trim(message->content) : message->content;
        std::string role(message->role);

        if (!is_inside_turn) {
            is_inside_turn = true;
            ss << (add_bos_inside ? "<s>[INST] " : "[INST] ");
        }

        if (role == "system") {
            if (support_system) {
                ss << "<<SYS>>\n" << content << "\n<</SYS>>\n\n";
            } else {
                ss << content << "\n";
            }
        } else if (role == "user") {
            ss << content << " [/INST]";
        } else {
            ss << content << "</s>";
            is_inside_turn = false;
        }
    }
    break;
}
```

**Llama3模板**（第481-489行）：
```cpp
case LLM_CHAT_TEMPLATE_LLAMA_3:
    for (auto message : chat) {
        std::string role(message->role);
        ss << "<|start_header_id|>" << role 
           << "<|end_header_id|>\n\n" << trim(message->content) 
           << "<|eot_id|>";
    }
    if (add_ass) {
        ss << "<|start_header_id|>assistant<|end_header_id|>\n\n";
    }
    break;
```

### 14.3.3 模板对比表

| 模板 | 格式特点 | 系统消息支持 | 特殊标记 |
|-----|---------|-------------|---------|
| ChatML | 简单直观 | 是 | `<|im_start/end|>` |
| Llama2 | 指令风格 | 可选 | `[INST]`, `<<SYS>>` |
| Llama3 | 结构化 | 是 | `<|start_header_id|>`, `<|eot_id|>` |
| Mistral-V7 | 系统提示区 | 是 | `[SYSTEM_PROMPT]` |
| Phi-3 | 类似ChatML | 是 | `<|role|>`, `<|end|>` |
| Gemma | 回合制 | 否（合并到user） | `<start_of_turn>` |
| DeepSeek-3 | 中文友好 | 是 | `<｜User｜>`, `<｜Assistant｜>` |

---

## 14.4 复杂模板示例

### 14.4.1 Mistral V7模板

**源码位置**：`src/llama-chat.cpp` (第254-269行)

```cpp
case LLM_CHAT_TEMPLATE_MISTRAL_V7:
case LLM_CHAT_TEMPLATE_MISTRAL_V7_TEKKEN: {
    const char * trailing_space = tmpl == LLM_CHAT_TEMPLATE_MISTRAL_V7 ? " " : "";
    for (auto message : chat) {
        std::string role(message->role);
        std::string content(message->content);
        if (role == "system") {
            ss << "[SYSTEM_PROMPT]" << trailing_space 
               << content << "[/SYSTEM_PROMPT]";
        } else if (role == "user") {
            ss << "[INST]" << trailing_space 
               << content << "[/INST]";
        } else {
            ss << trailing_space << content << "</s>";
        }
    }
    break;
}
```

**输出示例**：
```
[SYSTEM_PROMPT] You are a helpful assistant.[/SYSTEM_PROMPT]
[INST] Hello! [/INST] Hi there!</s>
[INST] How are you? [/INST]
```

### 14.4.2 DeepSeek-V3模板

**源码位置**：`src/llama-chat.cpp` (第544-558行)

```cpp
case LLM_CHAT_TEMPLATE_DEEPSEEK_3:
    for (auto message : chat) {
        std::string role(message->role);
        if (role == "system") {
            ss << message->content << "\n\n";
        } else if (role == "user") {
            ss << LU8("<｜User｜>") << message->content;
        } else if (role == "assistant") {
            ss << LU8("<｜Assistant｜>") 
               << message->content 
               << LU8("<｜end▁of▁sentence｜>");
        }
    }
    if (add_ass) {
        ss << LU8("<｜Assistant｜>");
    }
    break;
```

**输出示例**：
```
You are a helpful assistant.

<｜User｜>Hello!<｜Assistant｜>Hi there!<｜end▁of▁sentence｜><｜User｜>How are you?<｜Assistant｜>
```

### 14.4.3 Gemma模板（无系统消息）

**源码位置**：`src/llama-chat.cpp` (第375-396行)

```cpp
case LLM_CHAT_TEMPLATE_GEMMA:
    std::string system_prompt = "";
    for (auto message : chat) {
        std::string role(message->role);
        if (role == "system") {
            // Gemma没有系统消息，合并到第一个user
            system_prompt += trim(message->content);
            continue;
        }
        // Gemma中"assistant"叫"model"
        role = role == "assistant" ? "model" : message->role;
        ss << "<start_of_turn>" << role << "\n";
        if (!system_prompt.empty() && role != "model") {
            ss << system_prompt << "\n\n";
            system_prompt = "";
        }
        ss << trim(message->content) << "<end_of_turn>\n";
    }
    if (add_ass) {
        ss << "<start_of_turn>model\n";
    }
    break;
```

---

## 14.5 工具调用模板

### 14.5.1 支持工具的模板

**Granite 4.0模板**（第640-653行）：
```cpp
case LLM_CHAT_TEMPLATE_GRANITE_4_0:
    for (const auto & message : chat) {
        std::string role(message->role);
        if (role == "assistant_tool_call") {
            ss << "<|start_of_role|>assistant<|end_of_role|><|tool_call|>";
        } else {
            ss << "<|start_of_role|>" << role << "<|end_of_role|>";
        }
        ss << message->content << "<|end_of_text|>\n";
    }
    if (add_ass) {
        ss << "<|start_of_role|>assistant<|end_of_role|>";
    }
    break;
```

### 14.5.2 模板中的角色扩展

现代聊天模板支持的角色：
- `system`：系统提示
- `user`：用户输入
- `assistant`：助手回复
- `tool`：工具调用结果
- `assistant_tool_call`：助手请求调用工具

---

## 设计中的取舍

### 为什么不用Jinja2解析？

**Jinja2方案**：
- 优点：完整支持所有模板特性
- 缺点：需要引入Jinja2库，增加依赖

**llama.cpp方案（启发式匹配）**：
- 优点：零依赖，轻量级
- 缺点：不支持复杂模板逻辑

**权衡**：
- 大多数聊天模板结构简单，启发式足够
- 复杂模板可以通过自定义代码支持
- 性能和可移植性优先

### 如何处理模板变体？

**问题**：同一模型家族有多个模板变体（如Llama2有4种）

**解决方案**：
1. 使用枚举区分变体
2. 在apply_template中使用switch-case处理
3. 共享大部分逻辑，只变更有差异的部分

---

## 动手练习

### 练习1：理解模板转换
给定消息：
```json
[
  {"role": "system", "content": "Be helpful."},
  {"role": "user", "content": "Hi"},
  {"role": "assistant", "content": "Hello!"}
]
```

写出以下模板的输出：
1. ChatML
2. Llama3
3. Mistral-V7

### 练习2：分析模板检测
阅读 `src/llama-chat.cpp` 第88-236行，回答：
1. 如何区分Mistral-V1和Mistral-V3？
2. 如何检测Llama2的变体？
3. 如果模板无法识别会怎样？

### 练习3：设计新模板
假设有一个新模型使用以下格式：
```
[SYSTEM]system_content[/SYSTEM]
[USER]user_content[/USER]
[ASSISTANT]assistant_content[/ASSISTANT]
```

为这个模板：
1. 添加新的枚举值
2. 实现apply_template逻辑
3. 添加检测规则

---

## 本课小结

| 概念 | 一句话总结 |
|------|-----------|
| llm_chat_template | 聊天模板类型枚举 |
| llm_chat_detect_template | 通过特征检测模板类型 |
| llm_chat_apply_template | 将消息格式化为模型输入 |
| ChatML | 通用模板，简单直观 |
| Llama3 | 结构化模板，使用header_id |
| add_ass | 是否添加assistant生成提示 |
| LU8宏 | 处理UTF-8字符串字面量 |

---

*本章对应源码版本：master (2026-04-07)*
