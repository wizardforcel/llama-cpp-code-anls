# 第30章 完整项目实战 —— 把知识转化为"生产力"

## 1. 学习目标

- 综合运用所学知识构建完整应用
- 掌握RAG（检索增强生成）系统的实现
- 学习代码助手插件的开发流程
- 理解多模态应用的架构设计
- 培养独立解决复杂问题的能力

## 2. 生活类比：建筑师的作品

想象你是一位建筑师，从学习材料特性（GGML张量）、结构设计（模型架构）、到施工技术（性能优化），最终需要交付完整的建筑作品。本章的四个项目就是四座不同风格的建筑：智能客服是现代化写字楼（系统集成）、代码助手是精密实验室（准确性要求高）、知识库是大型图书馆（检索效率关键）、多模态应用是艺术中心（跨领域融合）。

## 3. 源码地图

| 文件路径 | 职责 | 核心内容 |
|---------|------|---------|
| `examples/retrieval/retrieval.cpp` | RAG检索示例 | 文档分块、嵌入生成、相似度搜索 |
| `examples/embedding/embedding.cpp` | 嵌入生成 | 批量文本嵌入 |
| `examples/server/server.cpp` | 服务端 | API服务、WebSocket |
| `examples/llava/` | 多模态示例 | CLIP视觉编码、图文融合 |
| `common/chat.cpp` | 聊天处理 | 消息格式化、模板应用 |

## 4. 详细章节内容

### 4.1 项目一：智能客服系统

#### 4.1.1 系统架构

```
智能客服系统架构

┌─────────────────────────────────────────────────────────────┐
│                        用户界面层                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │ Web聊天界面 │  │ 微信小程序 │  │    企业微信集成     │  │
│  └──────┬──────┘  └──────┬──────┘  └──────────┬──────────┘  │
└─────────┼────────────────┼────────────────────┼─────────────┘
          │                │                    │
          └────────────────┴────────────────────┘
                           │
┌──────────────────────────┼──────────────────────────────────┐
│                     API网关层                               │
│  ┌───────────────────────┼──────────────────────────────┐  │
│  │    负载均衡    │    限流器    │    认证授权         │  │
│  └───────────────────────┼──────────────────────────────┘  │
└──────────────────────────┼──────────────────────────────────┘
                           │
┌──────────────────────────┼──────────────────────────────────┐
│                     业务服务层                              │
│  ┌───────────────────────┼──────────────────────────────┐  │
│  │  ┌─────────────┐     │     ┌─────────────┐          │  │
│  │  │  会话管理   │◄────┼────►│  意图识别   │          │  │
│  │  └──────┬──────┘     │     └──────┬──────┘          │  │
│  │         │            │            │                  │  │
│  │  ┌──────▼──────┐     │     ┌──────▼──────┐          │  │
│  │  │  上下文管理 │◄────┼────►│  知识检索   │          │  │
│  │  └──────┬──────┘     │     └──────┬──────┘          │  │
│  │         │            │            │                  │  │
│  │  ┌──────▼────────────┼────────────▼──────┐          │  │
│  │  │           llama.cpp推理引擎            │          │  │
│  │  └────────────────────────────────────────┘          │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                           │
┌──────────────────────────┼──────────────────────────────────┐
│                     数据存储层                              │
│  ┌─────────────┐  ┌──────┴──────┐  ┌─────────────────────┐  │
│  │  向量数据库 │  │  关系数据库 │  │    文档存储         │  │
│  │ (Milvus)   │  │  (PostgreSQL)│  │   (MinIO/S3)       │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

#### 4.1.2 核心实现

```cpp
// 智能客服核心类
class CustomerServiceBot {
private:
    llama_model* model;
    llama_context* ctx;
    std::vector<Chunk> knowledge_base;
    
public:
    // 初始化
    bool init(const std::string& model_path) {
        llama_model_params mparams = llama_model_default_params();
        model = llama_model_load_from_file(model_path.c_str(), mparams);
        if (!model) return false;
        
        llama_context_params cparams = llama_context_default_params();
        cparams.n_ctx = 8192;
        cparams.n_batch = 512;
        ctx = llama_new_context_with_model(model, cparams);
        
        return ctx != nullptr;
    }
    
    // 加载知识库
    void loadKnowledgeBase(const std::vector<std::string>& documents) {
        for (const auto& doc : documents) {
            auto chunks = splitIntoChunks(doc, 512);
            for (const auto& chunk : chunks) {
                Chunk c;
                c.text = chunk;
                c.embedding = generateEmbedding(chunk);
                knowledge_base.push_back(c);
            }
        }
    }
    
    // 生成回复
    std::string generateResponse(const std::string& query) {
        // 1. 检索相关知识
        auto query_emb = generateEmbedding(query);
        auto relevant_chunks = retrieveRelevantChunks(query_emb, 3);
        
        // 2. 构建提示词
        std::string prompt = buildPrompt(query, relevant_chunks);
        
        // 3. 生成回复
        return generateText(prompt);
    }
    
private:
    std::vector<float> generateEmbedding(const std::string& text) {
        // 使用嵌入模型生成向量
        auto tokens = common_tokenize(ctx, text, true, false);
        
        llama_batch batch = llama_batch_init(tokens.size(), 0, 1);
        for (size_t i = 0; i < tokens.size(); i++) {
            common_batch_add(batch, tokens[i], i, {0}, false);
        }
        batch.logits[batch.n_tokens - 1] = true;
        
        llama_decode(ctx, batch);
        
        const float* emb = llama_get_embeddings_ith(ctx, batch.n_tokens - 1);
        std::vector<float> result(emb, emb + llama_model_n_embd(model));
        
        llama_batch_free(batch);
        return result;
    }
    
    std::vector<Chunk> retrieveRelevantChunks(
        const std::vector<float>& query_emb, 
        int top_k
    ) {
        std::vector<std::pair<float, Chunk>> scores;
        
        for (const auto& chunk : knowledge_base) {
            float sim = cosineSimilarity(query_emb, chunk.embedding);
            scores.push_back({sim, chunk});
        }
        
        // 排序取Top-K
        std::sort(scores.begin(), scores.end(), 
            [](auto& a, auto& b) { return a.first > b.first; });
        
        std::vector<Chunk> result;
        for (int i = 0; i < std::min(top_k, (int)scores.size()); i++) {
            result.push_back(scores[i].second);
        }
        
        return result;
    }
    
    std::string buildPrompt(
        const std::string& query, 
        const std::vector<Chunk>& chunks
    ) {
        std::string prompt = 
            "你是一个专业的客服助手。请根据以下知识回答问题。\n\n";
        
        prompt += "知识：\n";
        for (const auto& chunk : chunks) {
            prompt += chunk.text + "\n";
        }
        
        prompt += "\n用户问题：" + query + "\n";
        prompt += "回答：";
        
        return prompt;
    }
};
```

### 4.2 项目二：代码助手插件

#### 4.2.1 VS Code插件架构

```
VS Code插件架构

┌─────────────────────────────────────────┐
│           VS Code Extension             │
│  ┌─────────────────────────────────┐   │
│  │      Extension Host (Node.js)   │   │
│  │  ┌───────────┐  ┌───────────┐  │   │
│  │  │ 命令注册  │  │ 代码补全  │  │   │
│  │  │  (F1)    │  │  (Ctrl+Space)│   │
│  │  └─────┬─────┘  └─────┬─────┘  │   │
│  │        │              │        │   │
│  │  ┌─────▼──────────────▼─────┐  │   │
│  │  │     Language Client      │  │   │
│  │  │   (LSP协议通信)          │  │   │
│  │  └───────────┬──────────────┘  │   │
│  └──────────────┼─────────────────┘   │
└─────────────────┼─────────────────────┘
                  │ WebSocket/StdIO
┌─────────────────▼─────────────────────┐
│         Language Server               │
│  ┌───────────────────────────────┐   │
│  │    llama.cpp推理后端          │   │
│  │  ┌─────────┐  ┌───────────┐  │   │
│  │  │ 代码分析 │  │ 补全生成  │  │   │
│  │  │ (AST)   │  │ (FIM)    │  │   │
│  │  └────┬────┘  └─────┬─────┘  │   │
│  │       │             │        │   │
│  │  ┌────▼─────────────▼────┐   │   │
│  │  │   Fill-In-Middle     │   │   │
│  │  │   代码补全模型        │   │   │
│  │  └────────────────────────┘   │   │
│  └───────────────────────────────────┘   │
└─────────────────────────────────────────┘
```

#### 4.2.2 FIM (Fill-In-Middle)实现

```cpp
// 代码补全核心
class CodeCompletionEngine {
private:
    llama_model* model;
    llama_context* ctx;
    
    // FIM特殊token
    std::string fim_prefix = "<fim_prefix>";
    std::string fim_suffix = "<fim_suffix>";
    std::string fim_middle = "<fim_middle>";
    
public:
    // 生成代码补全
    std::string complete(
        const std::string& prefix,
        const std::string& suffix,
        int max_tokens = 64
    ) {
        // 构建FIM提示
        std::string prompt = fim_prefix + prefix + 
                            fim_suffix + suffix + 
                            fim_middle;
        
        // Tokenize
        auto tokens = common_tokenize(ctx, prompt, true, false);
        
        // 确保添加EOS
        if (llama_vocab_eos(llama_model_get_vocab(model)) >= 0) {
            tokens.push_back(llama_vocab_eos(llama_model_get_vocab(model)));
        }
        
        // 评估前缀
        llama_batch batch = llama_batch_init(tokens.size() + max_tokens, 0, 1);
        for (size_t i = 0; i < tokens.size(); i++) {
            common_batch_add(batch, tokens[i], i, {0}, false);
        }
        batch.logits[batch.n_tokens - 1] = true;
        
        llama_decode(ctx, batch);
        
        // 生成补全
        std::string completion;
        llama_sampler* sampler = createCodeSampler();
        
        for (int i = 0; i < max_tokens; i++) {
            llama_token new_token = llama_sampler_sample(sampler, ctx, -1);
            
            if (llama_vocab_is_eog(llama_model_get_vocab(model), new_token)) {
                break;
            }
            
            completion += common_token_to_piece(ctx, new_token);
            
            // 检查是否生成了停止序列
            if (shouldStop(completion)) {
                break;
            }
            
            // 继续生成
            common_batch_clear(batch);
            common_batch_add(batch, new_token, tokens.size() + i, {0}, true);
            llama_decode(ctx, batch);
        }
        
        llama_batch_free(batch);
        llama_sampler_free(sampler);
        
        return completion;
    }
    
private:
    llama_sampler* createCodeSampler() {
        llama_sampler* sampler = llama_sampler_chain_init({});
        
        // 温度
        llama_sampler_chain_add(sampler, 
            llama_sampler_init_temp(0.2f));
        
        // Top-P
        llama_sampler_chain_add(sampler, 
            llama_sampler_init_top_p(0.95f, 1));
        
        // 重复惩罚
        llama_sampler_chain_add(sampler, 
            llama_sampler_init_penalties(64, 1.0f, 1.1f, 1.0f));
        
        // 分布采样
        llama_sampler_chain_add(sampler, 
            llama_sampler_init_dist(LLAMA_DEFAULT_SEED));
        
        return sampler;
    }
    
    bool shouldStop(const std::string& text) {
        // 停止序列检测
        static const std::vector<std::string> stop_sequences = {
            "\n\n",      // 空行
            "\nfunc ",   // 新函数
            "\nclass ",  // 新类
            "\ndef ",    // Python函数
        };
        
        for (const auto& seq : stop_sequences) {
            if (text.size() >= seq.size() && 
                text.compare(text.size() - seq.size(), seq.size(), seq) == 0) {
                return true;
            }
        }
        return false;
    }
};
```

### 4.3 项目三：私有知识库（RAG）

#### 4.3.1 系统架构

```
RAG知识库系统

┌─────────────────────────────────────────────────────────────┐
│                        文档处理流程                          │
│                                                              │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐ │
│  │ 文档上传 │──►│ 格式解析 │──►│ 文本提取 │──►│ 内容分块 │ │
│  │ (PDF/Doc)│   │          │   │          │   │          │ │
│  └──────────┘   └──────────┘   └──────────┘   └────┬─────┘ │
│                                                     │       │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐        │       │
│  │ 向量存储 │◄──│ 嵌入生成 │◄──│ 块处理   │◄───────┘       │
│  │          │   │          │   │          │                │
│  └──────────┘   └──────────┘   └──────────┘                │
└─────────────────────────────────────────────────────────────┘
                           │
┌──────────────────────────┼──────────────────────────────────┐
│                     查询流程                                │
│                                                              │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐ │
│  │ 用户查询 │──►│ 查询重写 │──►│ 嵌入生成 │──►│ 向量检索 │ │
│  │          │   │          │   │          │   │          │ │
│  └──────────┘   └──────────┘   └──────────┘   └────┬─────┘ │
│                                                     │       │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐        │       │
│  │ 结果展示 │◄──│ 答案生成 │◄──│ 重排序   │◄───────┘       │
│  │          │   │          │   │          │                │
│  └──────────┘   └──────────┘   └──────────┘                │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

#### 4.3.2 基于retrieval.cpp的实现

```cpp
// 基于examples/retrieval/retrieval.cpp的扩展

class KnowledgeBaseRAG {
private:
    llama_model* embed_model;    // 嵌入模型
    llama_model* gen_model;      // 生成模型
    llama_context* embed_ctx;
    llama_context* gen_ctx;
    
    struct Document {
        std::string id;
        std::string title;
        std::vector<Chunk> chunks;
    };
    
    struct Chunk {
        std::string text;
        std::vector<float> embedding;
        std::string doc_id;
        int chunk_idx;
    };
    
    std::vector<Chunk> index;      // 向量索引
    
public:
    // 添加文档
    void addDocument(const Document& doc) {
        for (auto& chunk : doc.chunks) {
            chunk.embedding = computeEmbedding(chunk.text);
            chunk.doc_id = doc.id;
            index.push_back(chunk);
        }
    }
    
    // 查询
    QueryResult query(const std::string& question, int top_k = 5) {
        // 1. 生成查询嵌入
        auto query_emb = computeEmbedding(question);
        
        // 2. 检索相关块
        auto candidates = vectorSearch(query_emb, top_k * 2);
        
        // 3. 重排序
        auto top_chunks = rerank(question, candidates, top_k);
        
        // 4. 生成答案
        std::string answer = generateAnswer(question, top_chunks);
        
        return {
            .answer = answer,
            .sources = top_chunks,
            .confidence = computeConfidence(top_chunks)
        };
    }
    
private:
    std::vector<float> computeEmbedding(const std::string& text) {
        // 使用BGE或类似的嵌入模型
        auto tokens = common_tokenize(embed_ctx, text, true, false);
        
        // 添加EOS
        if (llama_vocab_eos(llama_model_get_vocab(embed_model)) >= 0) {
            tokens.push_back(llama_vocab_eos(llama_model_get_vocab(embed_model)));
        }
        
        // 批量编码
        llama_batch batch = llama_batch_init(tokens.size(), 0, 1);
        for (size_t i = 0; i < tokens.size(); i++) {
            common_batch_add(batch, tokens[i], i, {0}, false);
        }
        
        // 使用最后一个token的嵌入作为句子表示
        batch.logits[batch.n_tokens - 1] = true;
        
        llama_decode(embed_ctx, batch);
        
        const float* emb = llama_get_embeddings_ith(embed_ctx, batch.n_tokens - 1);
        int n_embd = llama_model_n_embd(embed_model);
        
        // 归一化
        std::vector<float> result(n_embd);
        common_embd_normalize(emb, result.data(), n_embd, 2);
        
        llama_batch_free(batch);
        return result;
    }
    
    std::vector<Chunk> vectorSearch(
        const std::vector<float>& query_emb, 
        int n_results
    ) {
        std::vector<std::pair<float, Chunk>> scores;
        
        for (const auto& chunk : index) {
            float sim = common_embd_similarity_cos(
                query_emb.data(), 
                chunk.embedding.data(), 
                query_emb.size()
            );
            scores.push_back({sim, chunk});
        }
        
        // 排序
        std::partial_sort(scores.begin(), 
            scores.begin() + std::min(n_results, (int)scores.size()),
            scores.end(),
            [](auto& a, auto& b) { return a.first > b.first; });
        
        std::vector<Chunk> results;
        for (int i = 0; i < std::min(n_results, (int)scores.size()); i++) {
            results.push_back(scores[i].second);
        }
        
        return results;
    }
    
    std::string generateAnswer(
        const std::string& question,
        const std::vector<Chunk>& chunks
    ) {
        // 构建RAG提示词
        std::string prompt = "基于以下参考资料回答问题。\n\n";
        prompt += "参考资料：\n";
        
        for (int i = 0; i < chunks.size(); i++) {
            prompt += "[" + std::to_string(i + 1) + "] " + 
                     chunks[i].text + "\n\n";
        }
        
        prompt += "问题：" + question + "\n";
        prompt += "答案：";
        
        // 生成
        return generateText(prompt, 512);
    }
};
```

### 4.4 项目四：多模态应用

#### 4.4.1 多模态架构

```
多模态AI助手架构

┌─────────────────────────────────────────────────────────────┐
│                        输入处理                              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │   文本输入  │  │   图像输入  │  │    语音输入         │  │
│  │  (Tokenize) │  │  (CLIP编码) │  │  (Whisper转录)      │  │
│  └──────┬──────┘  └──────┬──────┘  └──────────┬──────────┘  │
│         │                │                    │             │
│         └────────────────┼────────────────────┘             │
│                          │                                  │
│  ┌───────────────────────▼──────────────────────────────┐  │
│  │              多模态投影层 (Projection)                │  │
│  │         将图像/音频特征映射到文本嵌入空间             │  │
│  └───────────────────────┬──────────────────────────────┘  │
│                          │                                  │
└──────────────────────────┼──────────────────────────────────┘
                           │
┌──────────────────────────┼──────────────────────────────────┐
│                     融合推理层                              │
│  ┌───────────────────────▼──────────────────────────────┐  │
│  │              多模态Transformer                        │  │
│  │    ┌─────────┐  ┌─────────┐  ┌─────────┐            │  │
│  │    │ 文本注意力 │  │ 图像注意力 │  │ 跨模态注意力 │            │  │
│  │    └────┬────┘  └────┬────┘  └────┬────┘            │  │
│  │         └─────────────┼─────────────┘                │  │
│  │                       │                              │  │
│  │              ┌────────▼────────┐                     │  │
│  │              │   融合表示      │                     │  │
│  │              └────────┬────────┘                     │  │
│  └───────────────────────┼──────────────────────────────┘  │
│                          │                                  │
│  ┌───────────────────────▼──────────────────────────────┐  │
│  │              输出生成                                   │  │
│  │    ┌─────────┐  ┌─────────┐  ┌─────────┐            │  │
│  │    │ 文本回答 │  │ 图像描述 │  │ 语音合成 │            │  │
│  │    └─────────┘  └─────────┘  └─────────┘            │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

#### 4.4.2 基于LLaVA的实现

```cpp
// 基于examples/llava/的多模态实现

class MultimodalAssistant {
private:
    // CLIP视觉编码器
    clip_ctx* clip_ctx;
    
    // LLM
    llama_model* llm_model;
    llama_context* llm_ctx;
    
    // 投影层
    struct projector_weights {
        ggml_tensor* mm_projector_w;
        ggml_tensor* mm_projector_b;
    } proj_weights;
    
public:
    // 处理图像+文本查询
    std::string process(
        const std::string& image_path,
        const std::string& question
    ) {
        // 1. 加载并预处理图像
        clip_image_u8 img;
        clip_image_load_from_file(image_path.c_str(), &img);
        
        // 2. CLIP编码图像
        float* image_features = new float[clip_n_mmproj_embd(clip_ctx)];
        clip_image_encode(clip_ctx, img.nx, img.ny, img.data, image_features);
        
        // 3. 投影到LLM嵌入空间
        std::vector<float> projected = projectImageFeatures(image_features);
        
        // 4. 构建多模态提示
        std::string prompt = "<image>\n" + question;
        
        // 5. Tokenize文本部分
        auto tokens = common_tokenize(llm_ctx, prompt, true, false);
        
        // 6. 找到<image>位置，插入图像嵌入
        int image_token_pos = findImageTokenPosition(tokens);
        
        // 7. 多模态推理
        return generateMultimodalResponse(tokens, projected, image_token_pos);
    }
    
private:
    std::vector<float> projectImageFeatures(const float* clip_features) {
        int n_embd = llama_model_n_embd(llm_model);
        int n_clip_embd = clip_n_mmproj_embd(clip_ctx);
        
        std::vector<float> result(n_embd);
        
        // 简单的线性投影: result = clip_features @ W + b
        for (int i = 0; i < n_embd; i++) {
            float sum = 0;
            for (int j = 0; j < n_clip_embd; j++) {
                sum += clip_features[j] * 
                       ((float*)proj_weights.mm_projector_w->data)[j * n_embd + i];
            }
            result[i] = sum + ((float*)proj_weights.mm_projector_b->data)[i];
        }
        
        return result;
    }
    
    std::string generateMultimodalResponse(
        const std::vector<llama_token>& tokens,
        const std::vector<float>& image_embed,
        int image_pos
    ) {
        // 构建批次：文本token + 图像嵌入
        llama_batch batch = llama_batch_init(tokens.size(), 0, 1);
        
        for (int i = 0; i < tokens.size(); i++) {
            if (i == image_pos) {
                // 插入图像嵌入（作为特殊的"token"）
                // 实际实现需要修改llama_decode支持多模态输入
            } else {
                common_batch_add(batch, tokens[i], i, {0}, false);
            }
        }
        
        // 解码并生成
        llama_decode(llm_ctx, batch);
        
        // ... 生成回复
        
        llama_batch_free(batch);
        return "";
    }
};
```

## 5. 设计中的取舍

### 5.1 RAG vs 微调

| 方案 | 优点 | 缺点 | 适用场景 |
|-----|------|------|---------|
| RAG | 实时更新、可解释 | 检索质量依赖 | 动态知识 |
| 微调 | 内化知识、速度快 | 训练成本高 | 固定知识 |
| 混合 | 两者结合 | 系统复杂 | 生产推荐 |

### 5.2 延迟vs质量

```
代码助手场景:

方案A: 本地小模型
  - 延迟: <100ms
  - 质量: 中等
  - 适用: 实时代码补全

方案B: 云端大模型
  - 延迟: 500ms-2s
  - 质量: 高
  - 适用: 复杂代码生成

方案C: 混合策略
  - 简单补全: 本地模型
  - 复杂生成: 云端模型
  
推荐: 方案C
```

### 5.3 精度vs成本

| 模型大小 | 显存需求 | 质量 | 成本/1K tokens |
|---------|---------|------|----------------|
| 7B | 8GB | 良好 | $0.001 |
| 13B | 16GB | 较好 | $0.002 |
| 70B | 80GB | 优秀 | $0.008 |

## 6. 动手练习

### 练习1：构建最小RAG系统

```cpp
// minimal_rag.cpp
#include "llama.h"
#include <vector>
#include <cstring>
#include <algorithm>

int main() {
    // 1. 加载嵌入模型
    llama_model_params mparams = llama_model_default_params();
    llama_model* embed_model = llama_model_load_from_file(
        "bge-base-en-v1.5-f16.gguf", mparams);
    
    llama_context_params cparams = llama_context_default_params();
    llama_context* ctx = llama_new_context_with_model(embed_model, cparams);
    
    // 2. 索引文档
    std::vector<std::string> docs = {
        "llama.cpp is a C++ library for LLM inference.",
        "GGML is a tensor library for machine learning.",
        "GGUF is a file format for storing models."
    };
    
    // 3. 查询
    std::string query = "What is GGML?";
    
    // 4. 简单关键词匹配（实际应使用向量相似度）
    std::vector<std::pair<int, std::string>> scores;
    for (const auto& doc : docs) {
        int score = 0;
        if (doc.find("GGML") != std::string::npos) score += 10;
        scores.push_back({score, doc});
    }
    
    std::sort(scores.begin(), scores.end(), 
        [](auto& a, auto& b) { return a.first > b.first; });
    
    printf("Query: %s\n", query.c_str());
    printf("Top result: %s\n", scores[0].second.c_str());
    
    // 清理
    llama_free(ctx);
    llama_model_free(embed_model);
    
    return 0;
}
```

### 练习2：代码补全服务

```python
#!/usr/bin/env python3
"""简易代码补全服务"""
from flask import Flask, request, jsonify
from llama_cpp import Llama

app = Flask(__name__)

# 加载代码模型
model = Llama(
    model_path="codellama-7b-instruct.Q4_K_M.gguf",
    n_ctx=4096,
    n_gpu_layers=999
)

@app.route('/complete', methods=['POST'])
def complete():
    data = request.json
    prefix = data.get('prefix', '')
    suffix = data.get('suffix', '')
    
    # FIM格式
    prompt = f"<PRE> {prefix} <SUF>{suffix} <MID>"
    
    response = model(prompt, max_tokens=64, temperature=0.2)
    completion = response['choices'][0]['text']
    
    return jsonify({
        'completion': completion,
        'model': 'codellama-7b'
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

### 练习3：多模态聊天机器人

```python
#!/usr/bin/env python3
"""多模态聊天示例"""
import sys
from llama_cpp import Llama
from PIL import Image

class MultimodalChat:
    def __init__(self):
        # 加载多模态模型
        self.model = Llama(
            model_path="llava-v1.5-7b-Q4_K_M.gguf",
            clip_model_path="mmproj-v1.5-7b-f16.gguf",
            n_ctx=4096
        )
    
    def chat(self, image_path, question):
        response = self.model.create_chat_completion(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": image_path}},
                        {"type": "text", "text": question}
                    ]
                }
            ]
        )
        return response['choices'][0]['message']['content']

if __name__ == '__main__':
    chat = MultimodalChat()
    answer = chat.chat("image.jpg", "What do you see in this image?")
    print(answer)
```

## 7. 本课小结

- **智能客服**：RAG架构结合向量检索和文本生成，实现知识问答
- **代码助手**：FIM模式适合代码补全，需要专门的代码模型
- **知识库**：文档分块、嵌入生成、向量检索是RAG的核心组件
- **多模态**：CLIP视觉编码+LLM文本生成，实现图文理解

**项目开发流程：**
1. 需求分析：明确功能、性能、部署环境
2. 架构设计：选择合适的组件和交互方式
3. 原型开发：快速验证核心功能
4. 性能优化：量化、批处理、缓存
5. 生产部署：监控、日志、容错

**学习路径建议：**
- 初级：完成练习1，理解RAG基本原理
- 中级：实现练习2，集成到IDE
- 高级：完成练习3，探索多模态应用
