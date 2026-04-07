# 第20章 控制向量与风格迁移 —— 引导模型"换个风格说话"

## 1. 学习目标

通过本章学习，你将能够：
- 理解控制向量（Control Vector）的基本原理
- 掌握控制向量的生成方法和使用场景
- 学会使用llama_adapter_cvec API进行风格控制
- 了解控制向量与LoRA的区别与联系
- 能够实现角色扮演、情感控制、写作风格迁移等应用

## 2. 生活类比：调音台与滤镜

想象你有一个声音调音台：

- **基础模型**：原始人声
- **控制向量**：调音台上的各个旋钮（低音、高音、混响等）
- **正向样本**："开心"的语音片段 → 提取"开心"向量
- **负向样本**："悲伤"的语音片段 → 提取"悲伤"向量
- **向量运算**：开心向量 - 悲伤向量 = "更开心"的控制向量
- **应用控制**：将控制向量加到隐藏层，改变模型"语气"

**与LoRA的区别**：
- LoRA像换一个人来唱（改变权重矩阵）
- 控制向量像给同一个人加特效（改变激活值）

## 3. 源码地图

```
src/llama-adapter.h            # 控制向量头文件
  ├── llama_adapter_cvec               # 控制向量适配器
  ├── tensor_for()                     # 获取层张量
  └── apply_to()                       # 应用到张量

src/llama-adapter.cpp          # 控制向量实现
  ├── llama_adapter_cvec::init()       # 初始化控制向量
  ├── llama_adapter_cvec::apply()      # 应用控制向量
  └── llama_adapter_cvec::apply_to()   # 层级别应用

tools/cvector-generator/       # 控制向量生成工具
  ├── cvector-generator.cpp            # 主程序（约500行）
  ├── pca.hpp                          # PCA降维
  └── mean.hpp                         # 均值计算

include/llama.h                # C API声明
  ├── llama_control_vector_load()      # 加载控制向量
  ├── llama_control_vector_apply()     # 应用控制向量
  └── llama_adapter_cvec_ptr           # 智能指针

common/common.h/cpp            # 控制向量加载封装
  └── common_control_vector_load()     # 通用加载函数
```

## 4. 详细章节内容

### 4.1 控制向量原理

#### 4.1.1 什么是控制向量

控制向量是一种通过修改模型隐藏层激活值来控制生成行为的技术：

```
标准前向传播:
h_out = LayerNorm(h_in + Attention(h_in) + FFN(h_in))

应用控制向量后:
h_out = LayerNorm(h_in + Attention(h_in) + FFN(h_in) + cvec)

其中 cvec 是控制向量，形状为 [n_embd]
```

**图解控制向量作用位置**：
```
Transformer层结构:
┌─────────────────────────────────────────────┐
│  Input x (shape: [n_tokens, n_embd])        │
│     ↓                                       │
│  RMSNorm(x)                                 │
│     ↓                                       │
│  Attention                                  │
│     ↓                                       │
│  Residual: x + Attention(x)                 │
│     ↓                                       │
│  RMSNorm                                    │
│     ↓                                       │
│  FFN                                        │
│     ↓                                       │
│  Residual: x + FFN(x)                       │
│     ↓                                       │
│  + Control Vector ← [1, n_embd]            │
│     ↓                                       │
│  Output                                     │
└─────────────────────────────────────────────┘
```

#### 4.1.2 控制向量的生成原理

**对比学习思想**：
```
正向样本（期望行为）: P_1, P_2, ..., P_n
负向样本（相反行为）: N_1, N_2, ..., N_n

步骤:
1. 对每个正向样本，提取隐藏层激活: h_P = Model(P)
2. 对每个负向样本，提取隐藏层激活: h_N = Model(N)
3. 计算差异: diff = h_P - h_N
4. 降维/平均得到控制向量: cvec = PCA(diff) 或 Mean(diff)
```

**图解生成流程**：
```
正向prompt: "I am very happy today!" 
负向prompt: "I am very sad today."

        ┌─────────────────┐
        │   正向推理       │
        │  提取隐藏层      │
        │  h_pos [n_embd] │
        └────────┬────────┘
                 │
                 ↓
        ┌─────────────────┐
        │   负向推理       │
        │  提取隐藏层      │
        │  h_neg [n_embd] │
        └────────┬────────┘
                 │
                 ↓
        ┌─────────────────┐
        │   计算差异       │
        │ diff = h_pos -  │
        │        h_neg    │
        └────────┬────────┘
                 │
                 ↓
        ┌─────────────────┐
        │   多组平均/PCA   │
        │  cvec = mean    │
        │       (diffs)   │
        └────────┬────────┘
                 │
                 ↓
        控制向量文件 (.gguf)
```

### 4.2 控制向量生成工具

#### 4.2.1 生成器架构

**源码位置**：`tools/cvector-generator/cvector-generator.cpp:56-175`

```cpp
// 回调数据结构：保存隐藏层输出
struct callback_data {
    ggml_context * ctx_ggml = nullptr;
    int n_layers = 0;
    int n_tokens = 0;
    bool is_eval_pos = true;  // 当前是正向还是负向
    
    // 每层保存一个张量 [n_tokens, n_embd]
    std::vector<struct ggml_tensor *> v_pos;  // 正向隐藏层
    std::vector<struct ggml_tensor *> v_neg;  // 负向隐藏层
    std::vector<struct ggml_tensor *> v_diff_filtered;
    
    // 保存张量到对应向量
    void save_tensor_for_layer(struct ggml_tensor * t) {
        // 创建张量副本
        struct ggml_tensor * t_layer = ggml_new_tensor_2d(
            ctx_ggml, t->type, t->ne[0], t->ne[1]);
        t_layer->data = malloc(n_bytes);
        ggml_backend_tensor_get(t, t_layer->data, 0, n_bytes);
        
        if (is_eval_pos) {
            v_pos.push_back(t_layer);
        } else {
            v_neg.push_back(t_layer);
        }
    }
    
    // 计算差异并过滤零行
    std::vector<struct ggml_tensor *> calc_diff() {
        for (int il = 0; il < v_pos.size(); il++) {
            float * a = (float *) v_pos[il]->data;
            float * b = (float *) v_neg[il]->data;
            size_t n_elem = ggml_nelements(v_pos[il]);
            
            // 逐元素相减
            for (size_t j = 0; j < n_elem; j++) {
                a[j] -= b[j];
            }
            
            // 过滤全零行
            auto diff_filtered = filter_nonzero_rows(v_pos[il]);
            v_diff_filtered.push_back(diff_filtered);
        }
        return v_diff_filtered;
    }
};
```

#### 4.2.2 训练上下文

**源码位置**：`tools/cvector-generator/cvector-generator.cpp:181-269`

```cpp
struct train_context {
    ggml_context * ctx_ggml;
    int n_embd;
    int n_layers;
    
    // 正负样本对
    std::vector<std::string> positive_entries;
    std::vector<std::string> negative_entries;
    
    // 差异张量 [n_samples * n_tokens, n_embd]
    std::vector<struct ggml_tensor *> v_diff;
    
    // 最终控制向量 [n_embd] per layer
    std::vector<struct ggml_tensor *> v_final;
    
    // 合并差异张量
    void concat_diff_tmp(const std::vector<struct ggml_tensor *> & diff_filtered) {
        for (int il = 0; il < n_layers - 1; il++) {
            auto t = diff_filtered[il];
            auto & diff_tmp = v_diff_tmp[il];
            size_t curr_size = diff_tmp.size();
            diff_tmp.resize(curr_size + ggml_nbytes(t));
            memcpy(diff_tmp.data() + curr_size, t->data, ggml_nbytes(t));
        }
    }
    
    // 构建差异张量
    void build_v_diff(bool transpose) {
        for (int il = 0; il < n_layers - 1; il++) {
            auto & diff_tmp = v_diff_tmp[il];
            int n_elem = diff_tmp.size() / sizeof(float);
            int n_rows = n_elem / n_embd;
            
            // 转置为 [n_rows, n_embd]
            struct ggml_tensor * diff = ggml_new_tensor_2d(
                ctx_ggml, GGML_TYPE_F32, n_rows, n_embd);
            diff->data = malloc(ggml_nbytes(diff));
            
            // 复制并转置数据
            float * arr = (float *) diff_tmp.data();
            for (int ir = 0; ir < n_rows; ++ir) {
                for (int ic = 0; ic < n_embd; ++ic) {
                    float f = arr[ir*n_embd + ic];
                    ggml_set_f32_nd(diff, ir, ic, 0, 0, f);
                }
            }
            v_diff.push_back(diff);
        }
    }
};
```

#### 4.2.3 PCA降维

**源码位置**：`tools/cvector-generator/pca.hpp`

```cpp
namespace PCA {

struct pca_params {
    int n_threads;
    int n_batch;
    int n_iterations;
};

void run_pca(
    const pca_params & params,
    const std::vector<struct ggml_tensor *> & v_diff,  // 输入差异矩阵
    std::vector<struct ggml_tensor *> & v_final        // 输出控制向量
) {
    // PCA算法步骤:
    // 1. 中心化数据: X = X - mean(X)
    // 2. 计算协方差矩阵: C = X^T * X / (n-1)
    // 3. 特征值分解: C = V * D * V^T
    // 4. 取第一主成分: cvec = V[:, 0]
    
    for (int il = 0; il < v_diff.size(); il++) {
        struct ggml_tensor * diff = v_diff[il];
        int n_rows = diff->ne[1];  // 样本数
        int n_cols = diff->ne[0];  // n_embd
        
        // 计算均值
        std::vector<float> mean(n_cols, 0);
        for (int i = 0; i < n_rows; i++) {
            for (int j = 0; j < n_cols; j++) {
                mean[j] += ggml_get_f32_nd(diff, j, i, 0, 0);
            }
        }
        for (int j = 0; j < n_cols; j++) {
            mean[j] /= n_rows;
        }
        
        // 中心化
        for (int i = 0; i < n_rows; i++) {
            for (int j = 0; j < n_cols; j++) {
                float val = ggml_get_f32_nd(diff, j, i, 0, 0);
                ggml_set_f32_nd(diff, j, i, 0, 0, val - mean[j]);
            }
        }
        
        // 幂迭代计算第一主成分
        std::vector<float> vec(n_cols, 0);
        // ... 迭代计算 ...
        
        // 保存结果
        struct ggml_tensor * final_vec = v_final[il];
        memcpy(final_vec->data, vec.data(), n_cols * sizeof(float));
    }
}

} // namespace PCA
```

#### 4.2.4 均值方法

**源码位置**：`tools/cvector-generator/mean.hpp`

```cpp
namespace mean {

void run(
    const std::vector<struct ggml_tensor *> & v_diff,
    std::vector<struct ggml_tensor *> & v_final
) {
    // 简单平均法：对所有差异向量取平均
    for (int il = 0; il < v_diff.size(); il++) {
        struct ggml_tensor * diff = v_diff[il];
        int n_rows = diff->ne[1];
        int n_cols = diff->ne[0];
        
        struct ggml_tensor * final_vec = v_final[il];
        
        for (int j = 0; j < n_cols; j++) {
            float sum = 0;
            for (int i = 0; i < n_rows; i++) {
                sum += ggml_get_f32_nd(diff, j, i, 0, 0);
            }
            float avg = sum / n_rows;
            ggml_set_f32_nd(final_vec, j, 0, 0, 0, avg);
        }
    }
}

} // namespace mean
```

### 4.3 实时风格控制

#### 4.3.1 控制向量结构

**源码位置**：`src/llama-adapter.h:14-42`

```cpp
struct llama_adapter_cvec {
    // 获取指定层的控制向量张量
    ggml_tensor * tensor_for(int il) const;
    
    // 应用控制向量到张量
    ggml_tensor * apply_to(ggml_context * ctx, ggml_tensor * cur, int il) const;
    
    // 加载控制向量数据
    bool apply(
        const llama_model & model,
        const float * data,      // 原始控制向量数据
        size_t len,              // 数据长度
        int32_t n_embd,          // 嵌入维度
        int32_t il_start,        // 起始层
        int32_t il_end           // 结束层
    );
    
private:
    int32_t layer_start = -1;    // 生效起始层
    int32_t layer_end = -1;      // 生效结束层
    
    std::vector<ggml_context_ptr> ctxs;
    std::vector<ggml_backend_buffer_ptr> bufs;
    std::vector<ggml_tensor *> tensors;  // 每层一个控制向量
};
```

#### 4.3.2 应用到隐藏层

**源码位置**：`src/llama-adapter.cpp:13-29`

```cpp
ggml_tensor * llama_adapter_cvec::apply_to(
        ggml_context * ctx, 
        ggml_tensor * cur,  // 当前隐藏层激活 [n_tokens, n_embd]
        int il              // 层索引
) const {
    // 获取该层的控制向量
    ggml_tensor * layer_dir = tensor_for(il);
    
    if (layer_dir != nullptr) {
        // 广播加法: cur + layer_dir
        // layer_dir [1, n_embd] 广播到 [n_tokens, n_embd]
        cur = ggml_add(ctx, cur, layer_dir);
    }
    
    return cur;
}
```

**图解广播加法**：
```
当前激活 cur:          控制向量 layer_dir:
┌─────────────────┐    ┌─────────────────┐
│ 0.1  0.2  0.3  │    │ 0.5  0.1 -0.2  │
│ 0.2  0.1  0.4  │ +  └─────────────────┘
│ 0.3  0.3  0.1  │         ↓ 广播
│ ...            │    ┌─────────────────┐
└─────────────────┘    │ 0.5  0.1 -0.2  │
 [n_tokens, n_embd]    │ 0.5  0.1 -0.2  │
                       │ 0.5  0.1 -0.2  │
                       │ ...            │
                       └─────────────────┘
                        [n_tokens, n_embd]
                              ↓
                       ┌─────────────────┐
                       │ 0.6  0.3  0.1  │
                       │ 0.7  0.2  0.2  │
                       │ 0.8  0.4 -0.1  │
                       │ ...            │
                       └─────────────────┘
```

#### 4.3.3 层范围控制

**源码位置**：`src/llama-adapter.cpp:94-134`

```cpp
bool llama_adapter_cvec::apply(
        const llama_model & model,
        const float * data,
        size_t len,
        int32_t n_embd,
        int32_t il_start,
        int32_t il_end) {
    
    // 禁用控制向量
    if (data == nullptr) {
        layer_start = -1;
        layer_end = -1;
        return true;
    }
    
    // 验证维度
    if (n_embd != (int) model.hparams.n_embd) {
        LLAMA_LOG_ERROR("control vector n_embd does not match model");
        return false;
    }
    
    // 初始化张量
    if (tensors.empty()) {
        if (!init(model)) {
            return false;
        }
    }
    
    layer_start = il_start;
    layer_end = il_end;
    
    // 复制数据到各层
    for (size_t il = 1; il < model.hparams.n_layer; il++) {
        const size_t off = n_embd * (il - 1);
        if (off + n_embd <= len) {
            ggml_backend_tensor_set(
                tensors[il], 
                data + off, 
                0, 
                n_embd * ggml_element_size(tensors[il])
            );
        }
    }
    
    return true;
}
```

### 4.4 应用案例

#### 4.4.1 角色扮演控制

**生成角色控制向量**：
```bash
# 准备正负样本文件
# positive.txt: 角色台词（如："我是蝙蝠侠，黑暗骑士..."）
# negative.txt: 普通对话（如："我是一个普通人..."）

# 生成控制向量
./cvector-generator \
    -m llama-3.gguf \
    --cvector-positive-file positive.txt \
    --cvector-negative-file negative.txt \
    --method pca \
    -o batman.gguf
```

**使用控制向量**：
```cpp
// 加载控制向量
llama_adapter_cvec_ptr cvec = common_control_vector_load(
    model, 
    "batman.gguf",
    0,     // il_start: 从第0层开始
    -1     // il_end: 到最后层
);

// 设置强度
float strength = 1.5f;  // >1 增强效果, <1 减弱效果

// 推理时应用
// 控制向量会自动加到各层隐藏状态
```

#### 4.4.2 情感控制

**情感向量生成示例**：
```
正向（开心）: 
- "今天天气真好！"
- "收到礼物太开心了！"
- "考试通过了，耶！"

负向（悲伤）:
- "今天又是阴天..."
- "失去了重要的东西"
- "考试没通过，难过"

差异方向 = 开心 - 悲伤 = "更开心"的控制向量
```

**强度调节**：
```cpp
// 弱效果
float strength = 0.5f;

// 标准效果
float strength = 1.0f;

// 强效果
float strength = 2.0f;

// 反向效果（悲伤）
float strength = -1.0f;
```

#### 4.4.3 写作风格迁移

**风格控制示例**：
```
正式风格向量:
正向: 法律文件、学术论文
负向: 日常对话、网络用语

诗意风格向量:
正向: 诗歌、散文
负向: 技术文档、说明书

简洁风格向量:
正向: 摘要、简报
负向: 长篇小说、详细描述
```

**多风格组合**：
```cpp
// 加载多个控制向量
auto formal = common_control_vector_load(model, "formal.gguf", 0, -1);
auto poetic = common_control_vector_load(model, "poetic.gguf", 0, -1);

// 组合应用（加权求和）
// 效果 = base + 0.7 * formal + 0.5 * poetic
```

### 4.5 控制向量vs LoRA

#### 4.5.1 技术对比

| 特性 | 控制向量 | LoRA |
|------|----------|------|
| **修改对象** | 隐藏层激活 | 权重矩阵 |
| **参数量** | n_layer × n_embd | r × (d + k) |
| **计算开销** | 低（加法） | 中（矩阵乘法） |
| **效果强度** | 可调节（scale） | 可调节（scale） |
| **适用场景** | 风格、情感、角色 | 能力、知识、任务 |
| **可叠加性** | 好（向量加法） | 好（权重加法） |
| **训练数据** | 需要正负样本对 | 需要输入输出对 |

#### 4.5.2 选择建议

**使用控制向量**：
- 需要快速切换风格/情感
- 希望实时调节强度
- 修改模型的"语气"而非"能力"

**使用LoRA**：
- 需要学习新任务/知识
- 修改模型的特定能力
- 长期固定使用

**组合使用**：
```cpp
// LoRA提供代码能力
llama_set_adapter_lora(ctx, coding_lora, 1.0f);

// 控制向量提供正式语气
llama_adapter_cvec_apply(ctx, formal_cvec, 0.8f);

// 结果: 能写代码，且语气正式
```

## 5. 设计中的取舍

### 5.1 降维方法选择

| 方法 | 优点 | 缺点 |
|------|------|------|
| **PCA** | 找到最大方差方向，去噪 | 计算复杂，需要迭代 |
| **Mean** | 简单快速 | 可能包含噪声方向 |
| **Median** | 鲁棒性好 | 丢失一些信息 |

### 5.2 层范围选择

- **浅层（0-10）**：影响词法、语法
- **中层（10-20）**：影响语义、风格
- **深层（20+）**：影响整体连贯性
- **全部层**：最强效果，但可能不稳定

### 5.3 样本数量与质量

- **样本数量**：通常10-50对样本足够
- **样本质量**：正负样本对比要鲜明
- **样本多样性**：覆盖目标行为的不同表现

## 6. 动手练习

### 练习1：生成情感控制向量

```bash
# 1. 准备样本文件
cat > happy.txt << EOF
I am so excited about this!
What a wonderful day!
This is the best news ever!
EOF

cat > sad.txt << EOF
I am disappointed by this.
What a terrible day.
This is the worst news ever.
EOF

# 2. 生成控制向量
./cvector-generator \
    -m llama-3.gguf \
    --cvector-positive-file happy.txt \
    --cvector-negative-file sad.txt \
    --method mean \
    -o happiness.gguf

# 3. 测试
./llama-cli -m llama-3.gguf --control-vector happiness.gguf -p "I got a promotion"
```

### 练习2：分析控制向量效果

```cpp
// 比较不同强度的效果
for (float strength : {0.0f, 0.5f, 1.0f, 1.5f, 2.0f}) {
    printf("\n=== Strength: %.1f ===\n", strength);
    
    // 应用控制向量
    llama_adapter_cvec_apply(ctx, cvec, strength);
    
    // 生成文本
    std::string prompt = "The weather today is";
    // ... 生成逻辑 ...
}
```

### 练习3：层范围实验

```cpp
// 测试不同层范围的效果
struct LayerRange {
    int start;
    int end;
    const char* name;
} ranges[] = {
    {0, 10, "Shallow layers"},
    {10, 20, "Middle layers"},
    {20, 32, "Deep layers"},
    {0, 32, "All layers"},
};

for (const auto& range : ranges) {
    printf("\n=== %s ===\n", range.name);
    
    // 只在指定层范围应用
    llama_adapter_cvec_apply_range(ctx, cvec, range.start, range.end);
    
    // 生成并评估
}
```

## 7. 本课小结

本章我们深入学习了llama.cpp的控制向量技术：

1. **控制向量原理**：
   - 通过修改隐藏层激活值控制生成行为
   - 对比学习：正向样本 - 负向样本 = 控制方向
   - 广播加法应用到每层：[n_tokens, n_embd] + [1, n_embd]

2. **生成工具**：
   - `cvector-generator`：从正负样本对生成控制向量
   - PCA降维：提取主成分方向
   - 均值方法：简单平均差异向量

3. **实时控制**：
   - `llama_adapter_cvec`结构管理控制向量
   - 层范围控制：可选择性应用到特定层
   - 强度调节：scale参数控制效果强弱

4. **应用场景**：
   - 角色扮演：通过角色台词样本生成角色向量
   - 情感控制：调节生成文本的情感倾向
   - 风格迁移：正式、诗意、简洁等写作风格

5. **与LoRA对比**：
   - 控制向量改激活，LoRA改权重
   - 控制向量适合风格，LoRA适合能力
   - 两者可组合使用

**关键源码文件**：`src/llama-adapter.cpp`、`tools/cvector-generator/cvector-generator.cpp`

**下一步**：学习多模态支持，让模型能够"看懂"图像。
