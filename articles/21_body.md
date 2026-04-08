# 第21章 多模态架构（MTMD） —— 让模型"看懂"图像

人类认知世界的方式是多模态的——我们同时通过视觉、听觉、触觉等感官获取信息。当我们看到一张图片时，不仅能识别图中的物体，还能理解场景、情感和故事。传统的大语言模型只能处理文本，但现代多模态模型（如LLaVA、MiniCPM-V）将视觉能力融入语言模型，实现了真正的"看图说话"。llama.cpp的MTMD（Multi-Modal Text-Image Decoder）架构正是支持这类模型的核心，它像一座桥梁，连接了视觉世界和语言世界。

## 学习目标

1. 理解多模态模型的工作原理，掌握视觉-语言融合的基本概念
2. 深入MTMD架构的实现细节
3. 掌握CLIP视觉编码器的源码结构和图像处理流程
4. 学会使用llama.cpp的MTMD API开发图文混合应用
5. 了解多模态推理的性能优化技巧

## 生活类比：人类的"看图说话"能力

想象你正在向一位朋友描述一张照片：

- **你的眼睛** = CLIP视觉编码器（将图像转换为大脑能理解的"视觉语言"）
- **你的大脑** = 大语言模型（理解和推理视觉信息）
- **你的嘴巴** = 文本生成器（将理解转化为语言输出）
- **图像分块** = 观察照片的细节（先看整体缩略图，再看局部细节）
- **投影层** = 翻译官（将视觉特征翻译成文本模型能理解的格式）

就像人类需要眼睛、大脑和语言能力的协同工作才能"看图说话"，多模态模型也需要视觉编码器、投影层和语言模型的紧密配合。如果眼睛看到的不能正确传达给大脑，或者大脑理解了却无法表达，整个系统就无法正常工作。

## 源码地图

```
tools/mtmd/
├── mtmd.h                  # MTMD API头文件（第1-300行）
├── mtmd.cpp                # MTMD核心实现（第1-1500行）
│   ├── mtmd_context        # 多模态上下文管理
│   ├── mtmd_input_chunks   # 输入分块处理
│   └── mtmd_image_tokens   # 图像Token生成
├── mtmd-image.h/cpp        # 图像预处理
├── mtmd-audio.h/cpp        # 音频预处理
├── mtmd-helper.h/cpp       # 辅助函数
├── clip.h                  # CLIP模型接口
├── clip.cpp                # CLIP视觉编码器实现
├── clip-impl.h             # CLIP内部实现
└── clip-graph.h            # CLIP计算图构建

tools/cli/cli.cpp           # CLI多模态支持
```

## 21.1 多模态模型概述

### 21.1.1 什么是多模态？

**多模态（Multimodal）** 指模型能够同时处理和理解多种类型的数据（文本、图像、音频等）。在llama.cpp中，MTMD架构支持视觉-语言模型（如LLaVA、MiniCPM-V等）。

**核心组件图解**：
```
┌─────────────────────────────────────────────────────────────┐
│                    多模态推理流程                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐              │
│  │  图像输入 │───▶│  CLIP    │───▶│  投影层  │              │
│  │  (PNG/   │    │  编码器  │    │          │              │
│  │   JPG)   │    │          │    │          │              │
│  └──────────┘    └──────────┘    └────┬─────┘              │
│                                       │                     │
│  ┌──────────┐                        ▼                      │
│  │  文本输入 │──────────────────▶ ┌──────────┐              │
│  │  (Prompt)│                    │  语言模型 │              │
│  └──────────┘                    │  (LLM)   │              │
│                                  └────┬─────┘              │
│                                       │                     │
│                                       ▼                     │
│                                  ┌──────────┐              │
│                                  │  文本输出 │              │
│                                  │ (回答)   │              │
│                                  └──────────┘              │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

多模态模型的关键挑战在于：如何将像素信息转换为语言模型能理解的形式？CLIP视觉编码器解决了这个问题，它将图像编码为与文本嵌入同一空间的特征向量。

### 21.1.2 MTMD架构设计

**源码位置**：`tools/mtmd/mtmd.h`（第1-100行）

```cpp
// MTMD上下文参数
struct mtmd_context_params {
    bool use_gpu;           // 是否使用GPU加速视觉编码
    bool print_timings;     // 是否打印时间统计
    int n_threads;          // 线程数
    const char * image_marker;   // 图像标记（如"<image>"）
    const char * media_marker;   // 媒体标记（统一标记）
    enum llama_flash_attn_type flash_attn_type;  // Flash Attention类型
    bool warmup;            // 是否预热
    int image_min_tokens;   // 图像最小Token数
    int image_max_tokens;   // 图像最大Token数
    ggml_abort_callback cb_eval;  // 评估回调
    void * cb_eval_user_data;
};
```

**关键设计决策**：

1. **统一媒体标记**：使用`<__media__>`作为统一的图像/音频标记，简化输入处理
2. **可插拔编码器**：支持视觉编码器（CLIP）和音频编码器，架构灵活
3. **分块处理**：支持大图像的分块编码（llava-uhd风格），平衡细节与效率

**为什么需要分块？**

单张大图像（如1024x1024）直接编码会产生大量token（约1024个），超出模型上下文限制。分块处理将大图像切分为多个小块，每块单独编码，既保留了细节又控制了token数量。

## 21.2 CLIP视觉编码器

### 21.2.1 CLIP模型结构

**CLIP（Contrastive Language-Image Pre-training）** 是OpenAI开发的视觉-语言模型，llama.cpp使用其视觉部分作为编码器。

**源码位置**：`tools/mtmd/clip.h`（第1-150行）

```cpp
// CLIP上下文
struct clip_ctx;

// 图像编码结果
struct clip_image_f32_batch;

// CLIP上下文参数
struct clip_context_params {
    bool use_gpu;
    clip_flash_attn_type flash_attn_type;
    int image_min_tokens;
    int image_max_tokens;
    bool warmup;
    ggml_abort_callback cb_eval;
    void * cb_eval_user_data;
};

// 初始化CLIP模型
struct clip_init_result {
    struct clip_ctx * ctx_v;  // 视觉上下文
    struct clip_ctx * ctx_a;  // 音频上下文（可选）
};

clip_init_result clip_init(
    const char * fname,               // mmproj文件路径
    const clip_context_params & params
);
```

CLIP视觉编码器的核心是一个Vision Transformer（ViT），它将图像切分为patch序列，通过Transformer编码为特征向量。

### 21.2.2 图像预处理流程

**源码位置**：`tools/mtmd/mtmd-image.cpp`（第1-500行）

```cpp
// 图像预处理步骤：
// 1. 解码图像（PNG/JPG）
// 2. 调整大小（如336x336）
// 3. 归一化（使用ImageNet均值/标准差）
// 4. 分块（对于大图像）
// 5. 转换为模型输入格式

struct mtmd_image_preprocessor {
    // 预处理单张图像
    mtmd_image_tokens_ptr preprocess(
        const mtmd_bitmap & bitmap,
        const std::string & id
    );
    
    // 分块策略
    std::vector<mtmd_bitmap> slice_image(
        const mtmd_bitmap & bitmap,
        int n_slice
    );
};
```

**图像分块图解**（llava-uhd风格）：
```
原始图像 (1024x768)
┌──────────────────────────────┐
│                              │
│                              │
│                              │
│                              │
└──────────────────────────────┘

分块处理（2x2网格）：
┌──────────┬──────────┐
│  块 0    │  块 1    │
│ (512x384)│ (512x384)│
├──────────┼──────────┤
│  块 2    │  块 3    │
│ (512x384)│ (512x384)│
└──────────┴──────────┘

缩略图（全局视图）：
┌──────────┐
│  缩略图  │
│(336x336) │
└──────────┘

最终Token序列：
[缩略图Token] + [块0 Token] + [块1 Token] + [块2 Token] + [块3 Token]
```

这种处理方式的好处：
1. **保留全局信息**：缩略图提供整体上下文
2. **保留局部细节**：分块提供高分辨率细节
3. **灵活适应不同尺寸**：根据图像大小自动选择分块策略

### 21.2.3 视觉特征提取

**源码位置**：`tools/mtmd/clip.cpp`（第1-500行）

```cpp
// CLIP视觉编码器前向传播
static void clip_encode_image_internal(
    clip_context & ctx,
    const clip_image_f32_batch & imgs,
    float * output,
    int n_threads
) {
    // 1. 构建计算图
    ggml_cgraph * gf = clip_image_build_graph(ctx, imgs);
    
    // 2. 分配计算缓冲区
    ggml_backend_buffer_t buf_compute = 
        ggml_backend_alloc_buffer(ctx.backend, compute_size);
    
    // 3. 执行计算
    ggml_backend_graph_compute(ctx.backend, gf);
    
    // 4. 提取输出特征
    memcpy(output, output_tensor->data, output_size);
}
```

CLIP视觉编码器的输出是一个固定维度的特征向量（通常是1024或2048维），这个向量编码了图像的语义信息，可以被语言模型理解。

## 21.3 MTMD核心实现

### 21.3.1 输入分块处理

**源码位置**：`tools/mtmd/mtmd.cpp`（第100-300行）

```cpp
// 输入分块类型
enum mtmd_input_chunk_type {
    MTMD_INPUT_CHUNK_TYPE_TEXT,   // 文本块
    MTMD_INPUT_CHUNK_TYPE_IMAGE,  // 图像块
    MTMD_INPUT_CHUNK_TYPE_AUDIO,  // 音频块
};

// 单个输入分块
struct mtmd_input_chunk {
    mtmd_input_chunk_type type;
    std::vector<llama_token> tokens_text;  // 文本Token
    mtmd_image_tokens_ptr tokens_image;    // 图像Token
    mtmd_audio_tokens_ptr tokens_audio;    // 音频Token
};

// 输入分块集合
struct mtmd_input_chunks {
    std::vector<mtmd_input_chunk> entries;
};
```

**输入处理流程**：
```cpp
// 将混合输入（文本+图像标记）解析为分块
mtmd_input_chunks mtmd_tokenize_input(
    const char * text,  // 如 "描述这张图片: <__media__>"
    const std::vector<mtmd_bitmap> & images
) {
    mtmd_input_chunks chunks;
    
    // 1. 按媒体标记分割文本
    std::vector<std::string> parts = split_by_marker(text, "<__media__>");
    
    // 2. 交替添加文本块和图像块
    for (size_t i = 0; i < parts.size(); i++) {
        // 添加文本块
        if (!parts[i].empty()) {
            mtmd_input_chunk chunk;
            chunk.type = MTMD_INPUT_CHUNK_TYPE_TEXT;
            chunk.tokens_text = tokenize(parts[i]);
            chunks.entries.push_back(chunk);
        }
        
        // 添加图像块
        if (i < images.size()) {
            mtmd_input_chunk chunk;
            chunk.type = MTMD_INPUT_CHUNK_TYPE_IMAGE;
            chunk.tokens_image = preprocess_image(images[i]);
            chunks.entries.push_back(chunk);
        }
    }
    
    return chunks;
}
```

**分块处理示例**：
```
输入:
text = "描述这张图片: <__media__> 它有什么特点？"
images = [image1.jpg]

输出分块:
[
  {type: TEXT, tokens_text: ["描述", "这", "张", "图片", ":"]},
  {type: IMAGE, tokens_image: image1_tokens},
  {type: TEXT, tokens_text: ["它", "有", "什么", "特点", "？"]}
]
```

### 21.3.2 图像Token生成

**源码位置**：`tools/mtmd/mtmd.cpp`（第300-600行）

```cpp
// 图像Token结构
struct mtmd_image_tokens {
    uint32_t nx;  // x方向Token数
    uint32_t ny;  // y方向Token数
    bool use_mrope_pos;  // 是否使用M-RoPE位置编码
    
    uint32_t n_tokens() const { return nx * ny; }
    
    clip_image_f32_batch batch_f32;  // 预处理后的图像块
    std::string id;  // 可选ID（用于KV缓存跟踪）
};

// 将图像编码为嵌入向量
std::vector<float> mtmd_encode_image(
    mtmd_context * ctx,
    const mtmd_image_tokens * image_tokens
) {
    const int n_embd = clip_n_mmproj_embd(ctx->ctx_v);
    const int n_tokens = image_tokens->n_tokens();
    
    std::vector<float> embd(n_tokens * n_embd);
    
    // 调用CLIP编码图像
    clip_encode_image_internal(
        ctx->ctx_v,
        image_tokens->batch_f32,
        embd.data(),
        ctx->n_threads
    );
    
    return embd;
}
```

**图像Token与文本Token的区别**：

- **文本Token**：通过词嵌入表查找，每个token对应一个向量
- **图像Token**：通过CLIP编码生成，每个图像块对应一个向量

两者在语言模型看来都是"输入向量"，可以统一处理。

### 21.3.3 分块模板系统

**源码位置**：`tools/mtmd/mtmd.cpp`（第600-900行）

不同多模态模型使用不同的分块格式，MTMD通过模板系统支持这些差异：

```cpp
// 分块模板类型
enum mtmd_slice_tmpl {
    MTMD_SLICE_TMPL_NONE,           // 无分块（原始LLaVA）
    MTMD_SLICE_TMPL_MINICPMV_2_5,   // MiniCPM-V 2.5
    MTMD_SLICE_TMPL_MINICPMV_2_6,   // MiniCPM-V 2.6
    MTMD_SLICE_TMPL_LLAMA4,         // LLaMA 4
    MTMD_SLICE_TMPL_IDEFICS3,       // Idefics3
    MTMD_SLICE_TMPL_LFM2,           // LFM2
};

// 模板应用示例（LLaMA 4风格）：
// 输入：缩略图 + 4个分块
// 输出：
//   <ov_img_start> [缩略图嵌入] <ov_img_end>
//   <slices_start>
//   <sli_img_start> [块0嵌入] <sli_img_end>
//   <sli_img_start> [块1嵌入] <sli_img_end>
//   ...
//   <slices_end>
```

**为什么需要模板？**

不同模型的训练数据使用了不同的图像标记格式：
- LLaVA使用`<image>`
- MiniCPM-V使用特殊的开始/结束标记
- LLaMA 4使用`<ov_img_start>`等标记

模板系统让MTMD可以适配各种模型，而无需修改模型本身。

## 21.4 多模态推理实现

### 21.4.1 完整推理流程

**源码位置**：`tools/mtmd/mtmd-cli.cpp`（第1-500行）

```cpp
// 多模态推理主流程
int main(int argc, char ** argv) {
    // 1. 初始化参数
    mtmd_context_params ctx_params = mtmd_context_params_default();
    ctx_params.use_gpu = true;
    ctx_params.n_threads = 4;
    
    // 2. 创建MTMD上下文
    mtmd_context * ctx = new mtmd_context(
        mmproj_path,      // CLIP模型路径
        text_model,       // 语言模型
        ctx_params
    );
    
    // 3. 加载图像
    mtmd_bitmap bitmap = load_image(image_path);
    
    // 4. 构建输入
    std::string prompt = "描述这张图片: <__media__>";
    mtmd_input_chunks chunks = mtmd_tokenize_input(prompt, {bitmap});
    
    // 5. 编码图像为嵌入向量
    for (auto & chunk : chunks.entries) {
        if (chunk.type == MTMD_INPUT_CHUNK_TYPE_IMAGE) {
            chunk.image_embd = mtmd_encode_image(ctx, chunk.tokens_image.get());
        }
    }
    
    // 6. 构建llama_batch并解码
    llama_batch batch = build_batch_from_chunks(chunks);
    llama_decode(ctx_llama, batch);
    
    // 7. 采样生成回复
    // ...
}
```

### 21.4.2 批处理与KV缓存

**源码位置**：`tools/mtmd/mtmd.cpp`（第900-1200行）

```cpp
// 多模态批处理
void mtmd_decode_batches(
    mtmd_context * ctx,
    llama_context * ctx_llama,
    const mtmd_input_chunks & chunks
) {
    for (const auto & chunk : chunks.entries) {
        switch (chunk.type) {
            case MTMD_INPUT_CHUNK_TYPE_TEXT: {
                // 文本Token直接解码
                llama_batch batch = make_batch_text(chunk.tokens_text);
                llama_decode(ctx_llama, batch);
                break;
            }
            case MTMD_INPUT_CHUNK_TYPE_IMAGE: {
                // 图像嵌入通过embd输入
                llama_batch batch = make_batch_embd(chunk.image_embd);
                llama_decode(ctx_llama, batch);
                break;
            }
            // ...
        }
    }
}
```

**KV缓存优化**：
- 图像特征占用大量KV缓存空间（一块图像可能对应数百个token）
- 使用`id`字段跟踪图像，支持缓存复用
- 相同图像的后续查询可直接复用缓存，无需重新编码

## 21.5 性能优化

### 21.5.1 GPU加速

```cpp
// 启用GPU加速视觉编码
mtmd_context_params params = mtmd_context_params_default();
params.use_gpu = true;  // CLIP编码在GPU上执行

// 效果：
// - CPU: 图像编码 ~500ms
// - GPU: 图像编码 ~50ms（10倍加速）
```

视觉编码是计算密集型任务，GPU加速效果显著。

### 21.5.2 Flash Attention

```cpp
// 启用Flash Attention减少内存占用
params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;

// 效果：
// - 长序列图像（多分块）推理更稳定
// - 内存使用减少30-50%
```

多模态场景下，图像token数量可能很大，Flash Attention能有效控制内存使用。

### 21.5.3 图像尺寸选择

```cpp
// 限制图像Token数
params.image_max_tokens = 256;  // 最多256个Token

// 策略：
// - 小图像（336x336）：~144 tokens
// - 大图像分块：每块144 tokens
// - 根据显存调整限制
```

**图像尺寸建议**：
- 快速预览：336x336（144 tokens）
- 标准质量：448x448（256 tokens）
- 高分辨率：分块处理（512+ tokens）

## 21.6 设计中的取舍

### 21.6.1 为什么MTMD要分离视觉编码和语言模型？

| 方案 | 优点 | 缺点 | MTMD选择 |
|-----|------|------|---------|
| 端到端联合模型 | 可能更优性能 | 灵活性差，难以替换 | 否 |
| 分离式（当前） | 灵活组合，独立优化 | 需要投影层对齐 | **是** |
| 纯文本模型外挂 | 简单 | 性能差 | 否 |

**分离式架构的优势**：
1. **模块化**：可独立更新CLIP或语言模型
2. **灵活性**：同一视觉编码器可配不同语言模型
3. **效率**：视觉特征可缓存复用

### 21.6.2 为什么使用分块（Slice/Tiles）处理大图像？

```
单张大图像（1024x1024）：
- 直接编码：~1024 tokens（过多，占用KV缓存）
- 可能超出模型上下文限制

分块处理（4块 512x512）：
- 缩略图 + 4块 = ~720 tokens
- 保留细节的同时控制Token数
- 支持高分辨率图像理解
```

## 21.7 动手练习

### 练习1：图像预处理实验

使用 `tools/mtmd/mtmd-cli.cpp` 运行多模态推理：
```bash
./llama-mtmd-cli \
    -m llama-model.gguf \
    --mmproj mmproj.gguf \
    -p "描述这张图片: <__media__>" \
    --image test.jpg
```

观察输出并分析：
- 图像编码耗时
- 生成的Token数
- 内存占用

### 练习2：实现自定义分块策略

基于以下框架，实现一个简单的2x2分块策略：

```cpp
std::vector<mtmd_bitmap> slice_image_2x2(const mtmd_bitmap & original) {
    std::vector<mtmd_bitmap> slices;
    
    int w = original.width;
    int h = original.height;
    
    // 每块尺寸
    int slice_w = w / 2;
    int slice_h = h / 2;
    
    // 裁剪为4块
    for (int y = 0; y < 2; y++) {
        for (int x = 0; x < 2; x++) {
            mtmd_bitmap slice;
            slice.width = slice_w;
            slice.height = slice_h;
            // 复制像素数据...
            slices.push_back(slice);
        }
    }
    
    return slices;
}
```

### 练习3：多图像推理

```cpp
// 构建多图像输入
std::string prompt = "对比这两张图片: <__media__> 和 <__media__>";
std::vector<mtmd_bitmap> images = {image1, image2};

mtmd_input_chunks chunks = mtmd_tokenize_input(prompt.c_str(), images);

// 编码所有图像
for (auto & chunk : chunks.entries) {
    if (chunk.type == MTMD_INPUT_CHUNK_TYPE_IMAGE) {
        chunk.image_embd = mtmd_encode_image(ctx, chunk.tokens_image.get());
    }
}

// 解码并生成
// ...
```

## 21.8 本章小结

本章深入解析了多模态支持架构。MTMD是多模态文本-图像解码器架构。CLIP作为视觉编码器，将图像转换为特征向量。投影层用于对齐视觉特征和文本嵌入空间（在mmproj文件中实现）。图像分块技术将大图像切分处理，在细节和效率之间取得平衡。`<__media__>` 是统一的媒体标记，用于标识图像位置。M-RoPE是多模态位置编码，专门处理2D图像位置信息。KV缓存复用机制对相同图像缓存视觉特征，加速后续查询。

多模态架构的核心思想包括：统一表示，将不同模态转换为统一的向量表示；分块处理，灵活适应不同尺寸和分辨率的输入；模块化设计，视觉编码器和语言模型可以独立优化；缓存优化，视觉特征可复用，减少重复计算。

**下一步预告**：

在第22章，我们将探索llama.cpp的Common Utilities——了解CLI工具背后的共享基础设施，包括参数解析、下载管理、控制台交互等实用功能。
