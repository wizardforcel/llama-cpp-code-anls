# 第20章 控制向量与风格迁移 —— 引导模型"换个风格说话"

当你想让AI模型扮演某个特定角色、表达某种情感，或者采用特定的写作风格时，有哪些方法？全参数微调成本太高，提示工程效果有限，LoRA需要训练。控制向量（Control Vector）技术提供了一种轻量级的解决方案——通过简单的向量运算，就能实时改变模型的"语气"和"性格"。就像给声音加上不同的音效滤镜，控制向量让同一个基础模型能够呈现出千变万化的风格。

## 学习目标

1. 理解控制向量（Control Vector）的基本原理
2. 掌握控制向量的生成方法和使用场景
3. 学会使用llama_adapter_cvec API进行风格控制
4. 了解控制向量与LoRA的区别与联系
5. 能够实现角色扮演、情感控制、写作风格迁移等应用

## 生活类比：调音台与滤镜

想象你有一个专业的声音调音台，基础模型就是调音台接入的原始人声——未经任何处理，保留着歌手的本色嗓音。控制向量则对应调音台上的各个旋钮：低音增强让声音更浑厚，高音提升让声音更明亮，混响效果增加空间感，回声延迟赋予悠远的意境。每个旋钮控制一种特定的声音属性，就像每个控制向量编码一种特定的风格特征。

正向样本和负向样本是提取这些控制方向的关键。你收集一段"开心"的语音片段和一段"悲伤"的语音片段，前者代表目标风格在隐藏空间中的位置，后者代表相反风格的位置。两者的差异方向——开心向量减去悲伤向量——就定义了"更开心"的控制方向，放大了两种情感之间的特征差异。当你将这个控制向量加到模型隐藏层时，就像调节调音台上的旋钮，模型的"语气"沿着这个方向发生偏移。

控制向量与LoRA有着本质区别。LoRA像是换了一个人来唱歌——它通过微调权重矩阵来改变模型的"声带结构"，是结构性的改变。控制向量则像是给同一个人的声音添加音效——它只改变激活值的数值大小，是表现性的调整。前者改变了模型"能做什么"，后者只改变了模型"怎么做"。

## 源码地图

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

## 20.1 控制向量原理

### 20.1.1 什么是控制向量

控制向量是一种通过修改模型隐藏层激活值来控制生成行为的技术。与LoRA修改权重不同，控制向量在推理时动态调整隐藏状态。

```
标准前向传播:
h_out = LayerNorm(h_in + Attention(h_in) + FFN(h_in))

应用控制向量后:
h_out = LayerNorm(h_in + Attention(h_in) + FFN(h_in) + cvec)

其中 cvec 是控制向量，形状为 [1, n_embd] 或 [n_embd]
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

控制向量通常在残差连接后添加，这样可以直接影响下一层的输入。

### 20.1.2 控制向量的生成原理

控制向量的生成基于对比学习的思想：

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

为什么这种方法有效？因为神经网络的隐藏层编码了丰富的语义信息。正向和负向样本在隐藏空间中的差异方向，就代表了"目标属性"的方向。沿着这个方向调整激活值，就能增强或减弱相应的属性。

## 20.2 控制向量生成工具

### 20.2.1 生成器架构

**源码位置**：`tools/cvector-generator/cvector-generator.cpp`（第56-175行）

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
        size_t n_bytes = ggml_nbytes(t);
        
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
        for (int il = 0; il < (int) v_pos.size(); il++) {
            float * a = (float *) v_pos[il]->data;
            float * b = (float *) v_neg[il]->data;
            size_t n_elem = ggml_nelements(v_pos[il]);
            
            // 逐元素相减
            for (size_t j = 0; j < n_elem; j++) {
                a[j] -= b[j];
            }
            
            // 过滤全零行（如padding位置）
            auto diff_filtered = filter_nonzero_rows(v_pos[il]);
            v_diff_filtered.push_back(diff_filtered);
        }
        return v_diff_filtered;
    }
};
```

### 20.2.2 训练上下文

**源码位置**：`tools/cvector-generator/cvector-generator.cpp`（第181-269行）

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
    
    // 临时存储差异数据
    std::vector<std::vector<uint8_t>> v_diff_tmp;
    
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
            int n_elem = (int) diff_tmp.size() / sizeof(float);
            int n_rows = n_elem / n_embd;
            
            // 创建张量 [n_rows, n_embd]
            struct ggml_tensor * diff = ggml_new_tensor_2d(
                ctx_ggml, GGML_TYPE_F32, n_embd, n_rows);
            diff->data = malloc(ggml_nbytes(diff));
            
            // 复制数据
            memcpy(diff->data, diff_tmp.data(), diff_tmp.size());
            v_diff.push_back(diff);
        }
    }
};
```

### 20.2.3 PCA降维

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
    
    for (size_t il = 0; il < v_diff.size(); il++) {
        struct ggml_tensor * diff = v_diff[il];
        int n_rows = (int) diff->ne[1];  // 样本数
        int n_cols = (int) diff->ne[0];  // n_embd
        
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
        // ... 迭代计算主成分 ...
        
        // 保存结果
        struct ggml_tensor * final_vec = v_final[il];
        memcpy(final_vec->data, vec.data(), n_cols * sizeof(float));
    }
}

} // namespace PCA
```

PCA（主成分分析）的作用是找到差异矩阵中方差最大的方向，这个方向通常代表了最显著的控制方向。

### 20.2.4 均值方法

**源码位置**：`tools/cvector-generator/mean.hpp`

```cpp
namespace mean {

void run(
    const std::vector<struct ggml_tensor *> & v_diff,
    std::vector<struct ggml_tensor *> & v_final
) {
    // 简单平均法：对所有差异向量取平均
    for (size_t il = 0; il < v_diff.size(); il++) {
        struct ggml_tensor * diff = v_diff[il];
        int n_rows = (int) diff->ne[1];
        int n_cols = (int) diff->ne[0];
        
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

均值方法更简单直接，适合样本质量高、噪声少的情况。

## 20.3 实时风格控制

### 20.3.1 控制向量结构

**源码位置**：`src/llama-adapter.h`（第14-42行）

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
        int32_t il_start,        // 生效起始层
        int32_t il_end           // 生效结束层
    );
    
private:
    int32_t layer_start = -1;    // 生效起始层
    int32_t layer_end = -1;      // 生效结束层
    
    std::vector<ggml_context_ptr> ctxs;
    std::vector<ggml_backend_buffer_ptr> bufs;
    std::vector<ggml_tensor *> tensors;  // 每层一个控制向量
};
```

### 20.3.2 应用到隐藏层

**源码位置**：`src/llama-adapter.cpp`（第13-29行）

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

广播机制使得同一个控制向量可以应用到序列中的所有token位置。

### 20.3.3 层范围控制

**源码位置**：`src/llama-adapter.cpp`（第94-134行）

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
                n_embd * sizeof(float)
            );
        }
    }
    
    return true;
}
```

层范围控制的用途：
- **浅层（0-10）**：主要影响词法、语法层面的风格
- **中层（10-20）**：主要影响语义、表达方式
- **深层（20+）**：主要影响整体连贯性和高层特征

## 20.4 应用案例

### 20.4.1 角色扮演控制

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

// 推理时自动应用
// 控制向量会加到各层隐藏状态
```

### 20.4.2 情感控制

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
// 弱效果（轻微影响）
float strength = 0.5f;

// 标准效果
float strength = 1.0f;

// 强效果（强烈情感）
float strength = 2.0f;

// 反向效果（悲伤）
float strength = -1.0f;
```

### 20.4.3 写作风格迁移

**风格控制示例**：
```
正式风格向量:
正向: 法律文件、学术论文样本
负向: 日常对话、网络用语样本

诗意风格向量:
正向: 诗歌、散文样本
负向: 技术文档、说明书样本

简洁风格向量:
正向: 摘要、简报样本
负向: 长篇小说、详细描述样本
```

**多风格组合**：
```cpp
// 加载多个控制向量
auto formal = common_control_vector_load(model, "formal.gguf", 0, -1);
auto poetic = common_control_vector_load(model, "poetic.gguf", 0, -1);

// 组合应用（加权求和）
// 效果 = base + 0.7 * formal + 0.5 * poetic
// 既正式又有诗意
```

## 20.5 控制向量vs LoRA

### 20.5.1 技术对比

| 特性 | 控制向量 | LoRA |
|------|----------|------|
| **修改对象** | 隐藏层激活 | 权重矩阵 |
| **参数量** | n_layer × n_embd | r × (d + k) |
| **计算开销** | 低（加法） | 中（矩阵乘法） |
| **效果强度** | 可调节（scale） | 可调节（scale） |
| **适用场景** | 风格、情感、角色 | 能力、知识、任务 |
| **可叠加性** | 好（向量加法） | 好（权重加法） |
| **训练数据** | 需要正负样本对 | 需要输入输出对 |

### 20.5.2 选择建议

**使用控制向量**：
- 需要快速切换风格/情感/角色
- 希望实时调节强度
- 修改模型的"语气"而非"能力"
- 不想重新训练任何参数

**使用LoRA**：
- 需要学习新任务/知识
- 修改模型的特定能力
- 长期固定使用
- 可以接受少量训练

**组合使用**：
```cpp
// LoRA提供代码能力
llama_set_adapter_lora(ctx, coding_lora, 1.0f);

// 控制向量提供正式语气
llama_adapter_cvec_apply(ctx, formal_cvec, 0.8f);

// 结果: 能写代码，且语气正式
```

这种组合非常强大：LoRA负责"能做什么"，控制向量负责"怎么做"。

## 20.6 设计中的取舍

### 为什么选择PCA降维而不是简单的均值？

控制向量生成中一个关键的设计选择是降维方法。PCA（主成分分析）通过提取差异矩阵中方差最大的方向来找到控制方向，这个方向通常代表了最显著、最一致的控制效果。PCA的去噪效果好，能够过滤掉样本中的随机噪声，提取出真正有意义的特征变化方向。但其代价是计算更复杂，需要迭代求解特征向量。

均值方法（Mean）则简单直接得多——对所有差异向量取平均即可得到一个控制方向。这种方法计算速度极快，适合样本质量高、噪声少的场景，但如果样本中包含无关的噪声方向，均值法无法有效过滤它们。中位数方法（Median）在鲁棒性上更优，对异常值不敏感，即在少数样本质量极差的情况下也能给出合理的结果，但可能丢失一些有用的信息。llama.cpp提供了PCA和Mean两种方法的选择，让用户根据样本质量和计算资源来决定使用哪种方式。

### 控制向量应该应用到哪些层？

控制向量的层范围选择直接决定了它影响模型行为的层面。浅层（0-10层）主要处理词法和语法层面的特征，将控制向量应用到这里会影响模型的措辞选择和句式结构——比如让模型更倾向于使用正式词汇还是口语化表达。中层（10-20层）编码了更丰富的语义信息和风格特征，应用控制向量到这里会改变模型的表达方式和整体语气——这是大多数风格控制任务的主要作用范围。深层（20+层）负责高层语义连贯性和逻辑一致性，这里的变化会影响模型生成内容的整体结构和逻辑。应用到所有层会产生最强的效果，但也可能带来不稳定——因为每一层的隐藏空间编码的信息类型不同，统一的方向偏移在某些层可能产生不协调的效果。实践中通常从0-8层或全部层开始实验，根据生成效果调整，这也是llama.cpp在API中提供il_start和il_end参数的初衷。

### 需要多少样本才能生成好的控制向量？

控制向量的质量很大程度上取决于正负样本的选择。通常10-50对正负样本就足够生成效果良好的控制向量，更多样本并不总是更好——当样本数量过大时，可能引入噪声和矛盾的信号，反而弱化了核心控制方向。样本的质量比数量更重要：正负样本之间的对比要足够鲜明，两者在目标属性上的差异要清晰可辨。例如，生成"正式风格"控制向量时，正样本应来自严格的学术论文和法律文本，负样本应来自随意的网络聊天和口语对话，两者之间的差距越大，提取出的控制方向就越精确。样本的多样性同样关键——需要覆盖目标行为的不同表现形式，避免控制向量过度拟合某一种特定的表达方式。如果所有正样本都使用了相同的句式，生成的控制向量可能只是鼓励模型重复那种句式，而不是真正掌握了"正式风格"。

## 20.7 动手练习

### 练习1：生成情感控制向量

```bash
# 1. 准备样本文件
cat > happy.txt << EOF
I am so excited about this!
What a wonderful day!
This is the best news ever!
I love this so much!
EOF

cat > sad.txt << EOF
I am disappointed by this.
What a terrible day.
This is the worst news ever.
I hate this so much.
EOF

# 2. 生成控制向量
./cvector-generator \
    -m llama-3.gguf \
    --cvector-positive-file happy.txt \
    --cvector-negative-file sad.txt \
    --method mean \
    -o happiness.gguf

# 3. 测试
./llama-cli -m llama-3.gguf --control-vector happiness.gguf \
    -p "I got a promotion"
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
    
    // 生成并评估效果差异
}
```

## 20.8 本章小结

本章深入解析了控制向量技术。控制向量是一种通过修改隐藏层激活值来控制生成风格的轻量级技术。对比学习通过正向样本减去向负样本提取控制方向。PCA降维用于提取差异矩阵的主成分方向。广播加法机制将控制向量 [1, n_embd] 加到激活值 [n_tokens, n_embd] 上。层范围控制允许选择性地将控制应用到浅层、中层或深层。强度调节通过scale参数控制效果强弱，甚至可以使用负值实现反向效果。

控制向量的核心优势包括：零训练需求，无需训练只需准备样本对；实时性，推理过程中可动态调节；可叠加性，多个控制向量可以组合使用；可逆性，移除控制向量即可恢复原模型行为。控制向量与LoRA有协作关系：控制向量负责"风格"和"表现"，LoRA负责"能力"和"知识"，两者结合可实现更精细的模型控制。

本章我们一起学习了以下概念：

| 概念 | 解释 |
|------|------|
| 控制向量（cvec） | 通过修改隐藏层激活值来控制模型生成风格的轻量级向量，在残差连接后添加 |
| 对比学习 | 通过正向样本与负向样本的隐藏状态差异提取控制方向的核心方法 |
| PCA降维 | 从差异矩阵中提取最大方差方向作为控制向量，去噪效果优于简单均值 |
| 广播加法 | 控制向量[1, n_embd]广播加到激活值[n_tokens, n_embd]上，对所有token位置统一生效 |
| 层范围控制 | 通过il_start和il_end参数选择性地将控制应用到浅层、中层或深层 |
| 强度调节 | 通过scale参数控制效果强弱，正值增强目标方向，负值实现反向效果 |

下一章中，我们将学习多模态支持——让llama.cpp不仅能够处理文本，还能理解图像，实现图文对话能力。
