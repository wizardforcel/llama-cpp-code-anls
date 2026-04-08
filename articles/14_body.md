# 第14章 聊天模板系统 —— 对话格式的"智能裁缝"

当你与AI助手愉快交谈时，可曾想过这些对话是如何被格式化成模型能理解的格式的？不同的模型就像来自不同国家的客人，有着各自独特的"语言习惯"。ChatML用尖括号标记对话角色，Llama2用方括号包裹指令，Llama3使用特殊的header标记。聊天模板系统就是那位精通各种"方言"的智能裁缝，将标准化的对话消息裁剪成每个模型专属的对话格式。

## 学习目标

1. 理解聊天模板的作用和设计原理
2. 掌握llama-chat的模板匹配机制
3. 了解多种对话格式的实现差异
4. 能自定义和应用聊天模板

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

就像裁缝需要根据客户选择的面料和款式制作服装，聊天模板需要根据模型类型格式化对话。一件衣服如果款式不对，再贵的面料也穿不出门；同样，如果对话格式不对，再强大的模型也会产生混乱的输出。

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

## 14.1 聊天模板概述

### 14.1.1 为什么需要聊天模板

**问题背景**：
在大型语言模型的训练过程中，不同厂商采用了不同的对话格式。这种差异源于各自的设计理念和使用场景：

- **ChatML**（OpenAI风格）：使用XML风格的标记，清晰直观，易于解析
- **Llama2**（Meta风格）：基于指令微调，强调指令和响应的区分
- **Llama3**（Meta新版）：更结构化的标记系统，支持更复杂的对话流程
- **Mistral**：在Llama2基础上优化，增加了系统提示支持

如果推理时不使用正确的格式，模型就像在听一门不熟悉的外语——它可能理解一些词汇，但整体的语法结构和预期回应方式都会产生偏差。

**示例对比**：

假设有以下对话消息：
```json
[
  {"role": "system", "content": "You are a helpful assistant."},
  {"role": "user", "content": "Hello!"},
  {"role": "assistant", "content": "Hi there!"}
]
```

不同模型需要的输入格式截然不同：

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

可以看到，同样的对话内容在三种格式中的呈现方式完全不同。Llama2使用`[INST]`和`<<SYS>>`标记，ChatML使用`<|im_start|>`和`<|im_end|>`，而Llama3使用`<|start_header_id|>`和`<|eot_id|>`。聊天模板系统的任务就是自动完成这种转换。

### 14.1.2 模板类型枚举

**源码位置**：`src/llama-chat.h`（第7-63行）

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
    LLM_CHAT_TEMPLATE_GRANITE_4_0,      // Granite 4.0（支持工具调用）
    // ... 更多模板
    LLM_CHAT_TEMPLATE_UNKNOWN,          // 未知模板
};

这段代码定义了llama.cpp支持的所有聊天模板类型枚举，包括ChatML、Llama2（多种变体）、Mistral（多版本）、Phi、Gemma、Llama3、ChatGLM、DeepSeek、Qwen2等多种模板格式，以及未知模板作为兜底。

这个枚举定义了llama.cpp支持的所有聊天模板类型。可以看到，即使是同一个模型家族（如Llama2、Llama3、Mistral），也可能有多个变体。这是因为：

1. **版本演进**：模型版本更新时格式可能有细微变化
2. **功能差异**：是否支持系统消息、工具调用等特性
3. **训练细节**：不同的训练配置可能产生不同的格式要求

**为什么有这么多模板？**

就像服装有季节款、经典款、限量版一样，聊天模板也有多个版本：
- **经典款**：LLM_CHAT_TEMPLATE_LLAMA_2（基础Llama2格式）
- **升级版**：LLM_CHAT_TEMPLATE_LLAMA_2_SYS（支持系统提示）
- **特别版**：LLM_CHAT_TEMPLATE_MISTRAL_V7（Mistral最新格式）

每种模板都对应着特定的使用场景和模型配置，正确选择模板是获得良好对话体验的关键。

## 14.2 模板检测机制

### 14.2.1 检测函数实现

**源码位置**：`src/llama-chat.cpp`（第88-236行）

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

这段代码实现了聊天模板的自动检测功能。首先尝试将输入作为模板名称直接匹配；如果失败，则通过查找特征字符串（如"<|im_start|>"表示ChatML，"[INST]"表示Mistral/Llama2风格）来识别模板类型，返回对应的模板枚举值。

**检测流程详解**：

1. **直接名称匹配**（第91-95行）：
   首先尝试将输入作为模板名称直接匹配。如果用户传入"chatml"，直接返回对应的枚举值。

2. **特征检测**（第98行开始）：
   如果直接匹配失败，通过查找特征字符串来识别模板类型。

**为什么使用字符串包含检测？**

因为HuggingFace模型通常以Jinja2模板字符串的形式存储chat_template。完整的模板可能非常复杂，包含条件判断、循环等逻辑。但每种模板都有其独特的标记特征，通过查找这些特征就能准确识别模板类型。

例如：
- 看到`<|im_start|>`就知道是ChatML风格
- 看到`[INST]`就知道是Mistral/Llama2风格
- 看到`<|start_header_id|>`就知道是Llama3风格

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

这个检测流程是一个决策树，通过一系列条件判断逐步缩小可能的模板范围。设计这种分层检测的原因是：

1. **效率优先**：最常见的模板先检测（ChatML、Mistral）
2. **精确度**：多个特征组合使用（如Llama3需要同时包含start_header_id和end_header_id）
3. **容错性**：即使无法精确匹配，也能返回一个最接近的模板

## 14.3 模板应用实现

### 14.3.1 主应用函数

**源码位置**：`src/llama-chat.cpp`（第240-260行）

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

这段代码实现了聊天模板的主应用函数，根据模板类型选择对应的格式化逻辑。函数遍历消息列表，为每条消息添加相应的标记（如ChatML的<|im_start|>role\ncontent<|im_end|>），如果add_ass为true则在末尾添加assistant标记提示模型生成回复。

**函数参数解析**：

| 参数 | 类型 | 说明 |
|------|------|------|
| `tmpl` | `llm_chat_template` | 要应用的模板类型枚举 |
| `chat` | `vector<const llama_chat_message *>` | 对话消息列表 |
| `dest` | `string &` | 输出字符串引用，函数将结果写入这里 |
| `add_ass` | `bool` | 是否在末尾添加assistant标记，提示模型开始生成 |

**为什么需要add_ass参数？**

在对话生成场景中，我们需要提示模型"该你回复了"。add_ass=true时，模板会在最后添加assistant的开头标记，让模型知道接下来应该生成assistant的回复。例如：

- ChatML：添加`<|im_start|>assistant\n`
- Llama3：添加`<|start_header_id|>assistant<|end_header_id|>\n\n`

这就像在说："我说完了，该你了。"

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
}

这段代码实现了ChatML模板的格式化逻辑。为每条消息添加<|im_start|>标记和角色名，换行后添加消息内容，最后以<|im_end|>标记结束。如果add_ass为true，则在末尾添加assistant起始标记以提示模型生成回复。

ChatML是最直观的模板格式。每条消息由三部分组成：
1. `<|im_start|>` + 角色名（如"user"、"assistant"）
2. 换行符 + 消息内容
3. `<|im_end|>` + 换行符

这种格式清晰易读，也方便解析。每个角色和消息内容都有明确的边界标记。

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

这段代码实现了Llama2模板的格式化逻辑，支持多种变体（基础版、带系统消息、带BOS、去除空白）。使用[INST]和[/INST]包裹用户输入，<<SYS>>包裹系统提示，根据模板变体设置相应的标志位控制输出格式。

Llama2的格式更为复杂，需要处理多种变体：

1. **support_system**：基础Llama2不支持系统消息，变体支持
2. **add_bos_inside**：是否在指令前添加BOS标记
3. **strip_message**：是否去除消息内容的空白字符

Llama2使用`[INST]`和`[/INST]`包裹用户输入，`<<SYS>>`包裹系统提示。这种格式源于Llama2的指令微调训练方式。

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
}

这段代码实现了Llama3模板的格式化逻辑。每条消息使用<|start_header_id|>标记头部开始，包含角色名，以<|end_header_id|>标记头部结束，后跟两个换行符和修剪后的消息内容，最后以<|eot_id|>标记回合结束。如add_ass为true则添加assistant头部提示生成。

Llama3的格式更加结构化：
- `<|start_header_id|>`标记头部开始
- 角色名（system/user/assistant）
- `<|end_header_id|>`标记头部结束
- 两个换行符
- 消息内容
- `<|eot_id|>`标记结束（End Of Turn）

这种格式的设计考虑了更复杂的对话场景，如多轮对话、工具调用等。

### 14.3.3 模板对比表

| 模板 | 格式特点 | 系统消息支持 | 特殊标记 | 适用场景 |
|-----|---------|-------------|---------|---------|
| ChatML | 简单直观 | 是 | `<|im_start/end|>` | 通用对话 |
| Llama2 | 指令风格 | 可选 | `[INST]`, `<<SYS>>` | 指令遵循 |
| Llama3 | 结构化 | 是 | `<|start_header_id|>`, `<|eot_id|>` | 复杂对话 |
| Mistral-V7 | 系统提示区 | 是 | `[SYSTEM_PROMPT]` | 长系统提示 |
| Phi-3 | 类似ChatML | 是 | `<|role|>`, `<|end|>` | 轻量级模型 |
| Gemma | 回合制 | 否（合并到user） | `<start_of_turn>` | 谷歌模型 |
| DeepSeek-3 | 中文友好 | 是 | `<｜User｜>`, `<｜Assistant｜>` | 中文对话 |

## 14.4 复杂模板示例

### 14.4.1 Mistral V7模板

**源码位置**：`src/llama-chat.cpp`（第254-269行）

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

这段代码实现了Mistral V7模板的格式化逻辑，区分标准版和Tekken版（通过trailing_space控制）。系统消息用[SYSTEM_PROMPT]包裹，用户消息用[INST]包裹，助手回复后直接添加</s>标记结束。

**输出示例**：
```
[SYSTEM_PROMPT] You are a helpful assistant.[/SYSTEM_PROMPT]
[INST] Hello! [/INST] Hi there!</s>
[INST] How are you? [/INST]
```

Mistral V7引入了专门的`[SYSTEM_PROMPT]`区域来放置系统提示，这在处理长系统提示时特别有用。系统将提示与对话内容分离，使模型能更好地理解指令边界。

### 14.4.2 DeepSeek-V3模板

**源码位置**：`src/llama-chat.cpp`（第544-558行）

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
}

这段代码实现了DeepSeek-V3模板的格式化逻辑。系统消息直接输出内容加换行；用户消息用<｜User｜>标记包裹；助手消息用<｜Assistant｜>标记开始，以<｜end▁of▁sentence｜>标记结束。LU8宏用于处理UTF-8字符串字面量。

**输出示例**：
```
You are a helpful assistant.

<｜User｜>Hello!<｜Assistant｜>Hi there!<｜end▁of▁sentence｜><｜User｜>How are you?<｜Assistant｜>
```

DeepSeek-V3使用全角符号`<｜`和`｜>`作为标记，这种设计对中文处理更友好。`LU8`宏用于处理UTF-8字符串字面量。

### 14.4.3 Gemma模板（无系统消息）

**源码位置**：`src/llama-chat.cpp`（第375-396行）

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

Gemma模板需要特殊处理系统消息，因为Gemma原生不支持系统角色。实现策略是将系统提示合并到第一个用户消息前。此外，Gemma将"assistant"角色命名为"model"，这需要在转换时做映射。

**Gemma格式示例**：
```
<start_of_turn>user
You are a helpful assistant.

Hello!<end_of_turn>
<start_of_turn>model
Hi there!<end_of_turn>
<start_of_turn>model
```

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

现代聊天模板不仅需要处理基本的user/assistant角色，还需要支持工具调用场景：
- `tool`：工具返回的结果
- `assistant_tool_call`：助手请求调用工具

Granite 4.0使用特殊的`<|tool_call|>`标记来标识工具调用请求，模型看到这个标记就知道接下来应该输出工具调用参数。

### 14.5.2 模板中的角色扩展

现代聊天模板支持的角色：
- `system`：系统提示，定义助手行为
- `user`：用户输入
- `assistant`：助手回复
- `tool`：工具调用结果
- `assistant_tool_call`：助手请求调用工具

这种多角色设计支持Agent系统的工作流程：
1. User提出请求
2. Assistant决定需要调用工具
3. 执行工具，返回结果
4. Assistant基于工具结果回复

## 14.6 设计中的取舍

### 为什么不用Jinja2解析？

**Jinja2方案**：
- 优点：完整支持所有模板特性，包括条件判断、循环、过滤器
- 缺点：需要引入Jinja2库，增加依赖，增加二进制体积

**llama.cpp方案（启发式匹配）**：
- 优点：零依赖，轻量级，执行效率高
- 缺点：不支持复杂模板逻辑，需要为每种模板硬编码实现

**权衡**：
- 大多数聊天模板结构简单，启发式检测足够准确
- 复杂模板可以通过自定义代码支持
- 性能和可移植性是llama.cpp的核心目标，优先选择轻量级方案

### 如何处理模板变体？

**问题**：同一模型家族有多个模板变体（如Llama2有4种变体）

**解决方案**：
1. 使用枚举区分变体（LLM_CHAT_TEMPLATE_LLAMA_2、LLM_CHAT_TEMPLATE_LLAMA_2_SYS等）
2. 在apply_template中使用switch-case处理，让相关变体共享大部分逻辑
3. 通过bool标志区分变体特性（如support_system、add_bos_inside）

这种设计保持了代码的DRY原则（Don't Repeat Yourself），同时提供了足够的灵活性来支持各种变体。

## 14.7 动手练习

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

**参考答案**：

1. **ChatML**：
```
<|im_start|>system
Be helpful.<|im_end|>
<|im_start|>user
Hi<|im_end|>
<|im_start|>assistant
Hello!<|im_end|>
```

2. **Llama3**：
```
<|start_header_id|>system<|end_header_id|>

Be helpful.<|eot_id|><|start_header_id|>user<|end_header_id|>

Hi<|eot_id|><|start_header_id|>assistant<|end_header_id|>

Hello!<|eot_id|>
```

3. **Mistral-V7**：
```
[SYSTEM_PROMPT] Be helpful.[/SYSTEM_PROMPT][INST] Hi [/INST] Hello!</s>
```

### 练习2：分析模板检测

阅读 `src/llama-chat.cpp` 第88-236行，回答：
1. 如何区分Mistral-V1和Mistral-V3？
2. 如何检测Llama2的变体？
3. 如果模板无法识别会怎样？

**参考答案**：

1. **区分Mistral-V1和V3**：通过检查模板字符串中的空格处理。V1使用" [INST]"（前导空格），V3使用"\"[INST]\""（引号包裹）。

2. **检测Llama2变体**：通过检查三个特征：
   - 是否包含`<<SYS>>`（支持系统消息）
   - 是否包含`bos_token + '[INST]`（BOS在内部）
   - 是否包含`content.strip()`（去除空白）

3. **无法识别时**：返回`LLM_CHAT_TEMPLATE_UNKNOWN`，调用者需要决定如何处理（通常是使用默认模板或报错）。

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

**参考答案**：

```cpp
// 1. 在llama-chat.h中添加枚举值
enum llm_chat_template {
    // ... 其他模板
    LLM_CHAT_TEMPLATE_CUSTOM_BRACKET,  // 新模板
    LLM_CHAT_TEMPLATE_UNKNOWN,
};

// 2. 在apply_template中添加实现
case LLM_CHAT_TEMPLATE_CUSTOM_BRACKET:
    for (auto message : chat) {
        std::string role(message->role);
        if (role == "system") {
            ss << "[SYSTEM]" << message->content << "[/SYSTEM]\n";
        } else if (role == "user") {
            ss << "[USER]" << message->content << "[/USER]\n";
        } else if (role == "assistant") {
            ss << "[ASSISTANT]" << message->content << "[/ASSISTANT]\n";
        }
    }
    if (add_ass) {
        ss << "[ASSISTANT]";
    }
    break;

// 3. 在detect_template中添加检测
if (tmpl_contains("[SYSTEM]") && tmpl_contains("[USER]") && tmpl_contains("[ASSISTANT]")) {
    return LLM_CHAT_TEMPLATE_CUSTOM_BRACKET;
}
```

## 14.8 本章小结

本章深入解析了聊天模板系统。`llm_chat_template` 是聊天模板类型枚举，定义了所有支持的对话格式。`llm_chat_detect_template` 通过特征检测模板类型，支持从 Jinja2 模板字符串自动识别。`llm_chat_apply_template` 将消息列表格式化为模型可理解的提示文本。ChatML 是通用模板格式，使用 `<|im_start|>` 和 `<|im_end|>` 标记消息边界。Llama3 采用结构化模板，使用 `<|start_header_id|>` 和 `<|eot_id|>` 标记。add_ass 参数控制是否在末尾添加 assistant 标记，用于提示模型开始生成回复。LU8 宏用于处理 UTF-8 字符串字面量，在中文模板标记中特别有用。

聊天模板系统的设计哲学包括：零依赖设计，不引入 Jinja2 等外部库，保持 llama.cpp 的轻量级特性；启发式检测机制，通过特征字符串匹配实现高效的模板识别；精确实现策略，为每种模板硬编码格式化逻辑，确保输出正确；灵活扩展能力，通过枚举和 switch-case 支持新模板的添加。就像裁缝需要了解每种面料的特性才能做出合身的衣服，聊天模板系统需要深入理解每种模型的对话格式才能产生正确的提示。这种对细节的关注是 llama.cpp 能够支持数百种模型的关键。

**下一步预告**：

在第15章，我们将探索Unicode文本处理——理解llama.cpp如何处理多语言文本、特殊字符和复杂的编码问题。
