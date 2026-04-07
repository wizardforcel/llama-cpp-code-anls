# 第19章 LoRA适配器支持（llama_adapter）—— 模型微调的"即插即用"方案

## 1. 学习目标

通过本章学习，你将能够：
- 理解LoRA（低秩适应）技术的核心原理
- 掌握llama.cpp中LoRA适配器的加载与合并机制
- 学会使用llama_adapter相关API进行推理时适配器切换
- 了解GGUF LoRA格式与HuggingFace格式的转换
- 能够实现多适配器组合与动态权重调整

## 2. 生活类比：乐高积木与插件系统

想象你有一个基础乐高城堡（基础模型）：

- **全参数微调**：把城堡拆了重建——效果好但费时费力
- **LoRA适配器**：给城堡添加可拆卸的扩展模块——快速、灵活
- **多适配器组合**：同时安装塔楼扩展包 + 吊桥扩展包
- **动态切换**：上午用骑士主题装饰，下午换成龙主题
- **GGUF格式**：标准化的接口规格，确保兼容性

**核心优势**：只训练1%的参数，实现90%的全参数微调效果。

## 3. 源码地图

```
src/llama-adapter.h            # 适配器头文件
  ├── llama_adapter_cvec               # 控制向量适配器
  ├── llama_adapter_lora_weight        # LoRA权重结构
  ├── llama_adapter_lora               # LoRA适配器主体
  └── llama_adapter_loras              # 适配器集合

src/llama-adapter.cpp          # 适配器实现（约500行）
  ├── llama_adapter_lora_init()        # 初始化LoRA适配器
  ├── llama_adapter_lora_weight::get_scale()  # 计算缩放因子
  └── llama_adapter_cvec::apply()      # 应用控制向量

convert_lora_to_gguf.py        # LoRA转换工具（约500行）
  ├── LoraTorchTensor                  # LoRA张量封装
  ├── get_base_tensor_name()           # 获取基础张量名
  └── main()                           # 转换主流程

include/llama.h                # C API声明
  ├── llama_adapter_lora_init()        # 初始化适配器
  ├── llama_adapter_lora_free()        # 释放适配器
  ├── llama_set_adapter_lora()         # 设置适配器
  └── llama_rm_adapter_lora()          # 移除适配器
```

## 4. 详细章节内容

### 4.1 LoRA原理回顾

#### 4.1.1 低秩适应的数学原理

**原始权重更新**：
```
W_new = W_original + ΔW

传统微调：ΔW 是完整矩阵，参数量 = d × k
```

**LoRA分解**：
```
ΔW = B × A

其中：
- B: d × r 矩阵
- A: r × k 矩阵
- r << min(d, k) （秩，通常8-64）

参数量 = d×r + r×k = r×(d+k) << d×k
```

**图解**：
```
原始权重矩阵 W (d×k):
┌─────────────────────────────┐
│                             │
│        d × k 参数           │
│                             │
└─────────────────────────────┘

LoRA分解:
                    ┌─────────┐
┌─────────┐        │   A     │
│    B    │   ×    │  (r×k)  │
│  (d×r)  │        └─────────┘
└─────────┘              ↓
                    低秩矩阵
                    
总参数: d×r + r×k = r×(d+k)

当 r=16, d=4096, k=4096:
原始: 16,777,216 参数
LoRA: 131,072 参数 (0.78%)
```

#### 4.1.2 前向传播计算

**源码位置**：`src/llama-adapter.h:48-61`

```cpp
struct llama_adapter_lora_weight {
    ggml_tensor * a = nullptr;  // lora_A (r × k)
    ggml_tensor * b = nullptr;  // lora_B (d × r)
    
    // 计算实际缩放因子
    float get_scale(float alpha, float adapter_scale) const {
        const float rank = (float) b->ne[0];  // r
        const float scale = alpha ? adapter_scale * alpha / rank : adapter_scale;
        return scale;
    }
};
```

**计算流程**：
```
输入 x
    ↓
原始输出: h = W × x
    ↓
LoRA分支: 
  h_lora = B × A × x × scale
         = B × (A × x) × scale
    ↓
合并输出: y = h + h_lora
         = W×x + B×A×x×scale
         = (W + B×A×scale) × x
```

**图解计算图**：
```
        ┌─────────────┐
  x ───→│    W        │────┐
        │  (原始权重)  │    │
        └─────────────┘    │    ┌─────────┐
                           ├───→│   Add   │──→ output
        ┌─────────────┐    │    └─────────┘
  x ───→│     A       │──┐ │
        │   (r×k)     │  │ │
        └─────────────┘  │ │
                       ┌─┘ │
                       ↓   │
        ┌─────────────┐    │
        │     B       │────┘
        │   (d×r)     │
        └─────────────┘
              │
              ↓ scale
```

### 4.2 适配器加载机制

#### 4.2.1 GGUF LoRA文件解析

**源码位置**：`src/llama-adapter.cpp:149-240`

```cpp
static void llama_adapter_lora_init_impl(
        llama_model & model, 
        const char * path_lora, 
        llama_adapter_lora & adapter) {
    
    // 1. 加载GGUF文件
    gguf_context_ptr ctx_gguf { gguf_init_from_file(path_lora, meta_gguf_params) };
    ggml_context_ptr ctx { ctx_init };
    
    // 2. 验证元数据
    {
        auto general_type = get_kv_str(llm_kv(LLM_KV_GENERAL_TYPE));
        if (general_type != "adapter") {
            throw std::runtime_error("expect general.type to be 'adapter'");
        }
        
        auto adapter_type = get_kv_str(llm_kv(LLM_KV_ADAPTER_TYPE));
        if (adapter_type != "lora") {
            throw std::runtime_error("expect adapter.type to be 'lora'");
        }
        
        // 检查架构匹配
        auto general_arch = llm_arch_from_string(general_arch_str);
        if (general_arch != model.arch) {
            throw std::runtime_error("model arch and LoRA arch mismatch");
        }
        
        adapter.alpha = get_kv_f32(llm_kv(LLM_KV_ADAPTER_LORA_ALPHA));
    }
    
    // 3. 解析张量
    // ...
}
```

**GGUF LoRA文件结构**：
```
GGUF LoRA文件:
├── Header
│   ├── magic: 'GGUF'
│   ├── version: 3
│   └── tensor_count: N
│
├── Metadata
│   ├── general.type = "adapter"
│   ├── general.architecture = "llama"
│   ├── adapter.type = "lora"
│   └── adapter.lora.alpha = 16.0
│
├── Tensor Info
│   ├── tok_embeddings.lora_a (shape: [r, vocab])
│   ├── tok_embeddings.lora_b (shape: [dim, r])
│   ├── layers.0.attention.wq.lora_a (shape: [r, dim])
│   ├── layers.0.attention.wq.lora_b (shape: [dim, r])
│   └── ... (每个目标张量对应一对lora_a/lora_b)
│
└── Tensor Data
    └── 原始二进制数据
```

#### 4.2.2 权重映射与配对

**源码位置**：`src/llama-adapter.cpp:265-295`

```cpp
// 将lora_a和lora_b配对
std::map<std::string, llama_adapter_lora_weight> ab_map;

for (ggml_tensor * cur = ggml_get_first_tensor(ctx.get()); 
     cur; 
     cur = ggml_get_next_tensor(ctx.get(), cur)) {
    
    std::string name(cur->name);
    
    if (str_endswith(name, ".lora_a")) {
        replace_all(name, ".lora_a", "");
        if (ab_map.find(name) == ab_map.end()) {
            ab_map[name] = llama_adapter_lora_weight(cur, nullptr);
        } else {
            ab_map[name].a = cur;
        }
    } else if (str_endswith(name, ".lora_b")) {
        replace_all(name, ".lora_b", "");
        if (ab_map.find(name) == ab_map.end()) {
            ab_map[name] = llama_adapter_lora_weight(nullptr, cur);
        } else {
            ab_map[name].b = cur;
        }
    }
}
```

**张量命名映射**：
```
HuggingFace LoRA命名 → GGUF命名

base_model.model.model.embed_tokens.lora_A.weight
    → tok_embeddings.lora_a

base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight
    → layers.0.attention.wq.lora_a

base_model.model.model.layers.0.mlp.gate_proj.lora_A.weight
    → layers.0.feed_forward.w1.lora_a

base_model.model.model.layers.0.mlp.up_proj.lora_A.weight
    → layers.0.feed_forward.w3.lora_a

base_model.model.model.layers.0.mlp.down_proj.lora_A.weight
    → layers.0.feed_forward.w2.lora_a
```

#### 4.2.3 内存分配与加载

**源码位置**：`src/llama-adapter.cpp:319-412`

```cpp
// 为每个目标张量创建对应的LoRA张量
for (auto & it : ab_map) {
    const std::string & name = it.first;
    llama_adapter_lora_weight & w = it.second;
    
    // 查找对应的基础模型张量
    const auto * model_tensor = model.get_tensor(name.c_str());
    if (!model_tensor) {
        throw std::runtime_error("LoRA tensor does not exist in base model");
    }
    
    // 获取buffer类型（CPU/GPU）
    auto * buft = ggml_backend_buffer_get_type(model_tensor->buffer);
    
    // 创建张量副本
    ggml_context * dev_ctx = ctx_for_buft(buft);
    ggml_tensor * tensor_a = ggml_dup_tensor(dev_ctx, w.a);
    ggml_tensor * tensor_b = ggml_dup_tensor(dev_ctx, w.b);
    
    // 验证形状
    if (model_tensor->ne[0] != w.a->ne[0] || model_tensor->ne[1] != w.b->ne[1]) {
        throw std::runtime_error("tensor shape mismatch");
    }
    if (w.a->ne[1] != w.b->ne[0]) {
        throw std::runtime_error("lora_a tensor is not transposed");
    }
    
    adapter.ab_map[name] = llama_adapter_lora_weight(tensor_a, tensor_b);
}

// 分配缓冲区并加载数据
for (auto & it : ctx_map) {
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx_dev, buft);
    // 从文件读取数据到buffer
}

// 注册适配器到模型
model.loras.insert(&adapter);
```

### 4.3 推理时适配器应用

#### 4.3.1 适配器查询接口

**源码位置**：`src/llama-adapter.cpp:138-147`

```cpp
llama_adapter_lora_weight * llama_adapter_lora::get_weight(ggml_tensor * w) {
    const std::string name(w->name);
    
    const auto pos = ab_map.find(name);
    if (pos != ab_map.end()) {
        return &pos->second;
    }
    
    return nullptr;  // 该张量没有LoRA适配
}
```

#### 4.3.2 计算图构建时的适配器集成

**图解适配器集成到Transformer层**：
```
标准Transformer层前向传播:
┌─────────────────────────────────────────────┐
│  Input x                                    │
│     ↓                                       │
│  RMSNorm                                    │
│     ↓                                       │
│  Attention                                  │
│    ├── wq (with LoRA: wq + B_q×A_q×scale)  │
│    ├── wk (with LoRA: wk + B_k×A_k×scale)  │
│    ├── wv (with LoRA: wv + B_v×A_v×scale)  │
│    └── wo (with LoRA: wo + B_o×A_o×scale)  │
│     ↓                                       │
│  Residual Connection                        │
│     ↓                                       │
│  RMSNorm                                    │
│     ↓                                       │
│  FFN                                        │
│    ├── w1 (with LoRA: w1 + B_1×A_1×scale)  │
│    ├── w2 (with LoRA: w2 + B_2×A_2×scale)  │
│    └── w3 (with LoRA: w3 + B_3×A_3×scale)  │
│     ↓                                       │
│  Residual Connection                        │
│     ↓                                       │
│  Output                                     │
└─────────────────────────────────────────────┘
```

#### 4.3.3 多适配器管理

**源码位置**：`src/llama-adapter.h:90-91`

```cpp
using llama_adapter_loras = std::unordered_map<llama_adapter_lora *, float>;
// key: 适配器指针, value: 缩放系数
```

**多适配器组合示例**：
```cpp
// 加载两个适配器
llama_adapter_lora * lora_coding = llama_adapter_lora_init(model, "coding-lora.gguf");
llama_adapter_lora * lora_english = llama_adapter_lora_init(model, "english-lora.gguf");

// 设置适配器权重
llama_set_adapter_lora(ctx, lora_coding, 0.8f);   // 80% 代码能力
llama_set_adapter_lora(ctx, lora_english, 0.6f);  // 60% 英文优化

// 推理时两个适配器同时生效
// 效果 = base + 0.8 × ΔW_coding + 0.6 × ΔW_english
```

**图解多适配器组合**：
```
单适配器:
Output = W_base × x + scale × B × A × x

多适配器:
Output = W_base × x 
       + scale_1 × B_1 × A_1 × x
       + scale_2 × B_2 × A_2 × x
       + ...
       
等价于:
Output = (W_base + Σ(scale_i × B_i × A_i)) × x
```

### 4.4 适配器导出工具

#### 4.4.1 转换流程概览

**源码位置**：`convert_lora_to_gguf.py:301-503`

```python
def main():
    # 1. 加载LoRA权重
    if input_model.suffix == '.safetensors':
        from safetensors.torch import load_file
        lora_model = load_file(input_model, device="cpu")
    else:
        lora_model = torch.load(input_model, map_location="cpu")
    
    # 2. 加载基础模型配置
    hparams = ModelBase.load_hparams(dir_base_model)
    
    # 3. 创建LoraModel类
    class LoraModel(model_class):
        def get_tensors(self):
            # 配对lora_A和lora_B
            tensor_map = {}
            for name, tensor in lora_model.items():
                base_name = get_base_tensor_name(name)
                is_lora_a = ".lora_A.weight" in name
                is_lora_b = ".lora_B.weight" in name
                
                if base_name in tensor_map:
                    if is_lora_a:
                        tensor_map[base_name].A = tensor
                    else:
                        tensor_map[base_name].B = tensor
                else:
                    if is_lora_a:
                        tensor_map[base_name] = PartialLoraTensor(A=tensor)
                    else:
                        tensor_map[base_name] = PartialLoraTensor(B=tensor)
            
            # 返回LoraTorchTensor
            for name, tensor in tensor_map.items():
                yield (name, LoraTorchTensor(tensor.A, tensor.B))
    
    # 4. 导出GGUF
    model_instance = LoraModel(...)
    model_instance.write()
```

#### 4.4.2 LoraTorchTensor封装

**源码位置**：`convert_lora_to_gguf.py:40-235`

```python
class LoraTorchTensor:
    """封装LoRA张量对(A, B)，支持切片、reshape等操作"""
    
    def __init__(self, A: Tensor, B: Tensor):
        assert A.shape[-2] == B.shape[-1]  # 验证秩匹配
        self._lora_A = A
        self._lora_B = B
        self._rank = B.shape[-1]
    
    @property
    def shape(self) -> tuple[int, ...]:
        # 最终形状由B的前几维和A的最后一维决定
        return (*self._lora_B.shape[:-1], self._lora_A.shape[-1])
    
    def reshape(self, *shape: int) -> LoraTorchTensor:
        # 对A和B分别reshape
        shape_A = (*(1 for _ in new_shape[:-2]), self._rank, orig_shape[-1])
        shape_B = (*new_shape[:-1], self._rank)
        return LoraTorchTensor(
            self._lora_A.reshape(shape_A),
            self._lora_B.reshape(shape_B)
        )
    
    def permute(self, *dims: int) -> LoraTorchTensor:
        # 处理维度置换
        if dims[-1] == -1:
            return LoraTorchTensor(self._lora_A, self._lora_B.permute(*dims))
        # ...
    
    def get_lora_A_B(self) -> tuple[Tensor, Tensor]:
        return (self._lora_A, self._lora_B)
```

#### 4.4.3 命令行使用

```bash
# 基本转换
python convert_lora_to_gguf.py /path/to/lora_adapter \
    --base /path/to/base_model \
    --outfile output.gguf

# 指定输出格式
python convert_lora_to_gguf.py /path/to/lora_adapter \
    --outtype f16 \
    --outfile output-f16.gguf

# 从HuggingFace加载基础模型配置
python convert_lora_to_gguf.py /path/to/lora_adapter \
    --base-model-id meta-llama/Llama-2-7b-hf \
    --outfile output.gguf
```

### 4.5 激活型LoRA（aLoRA）

**源码位置**：`src/llama-adapter.h:78` 和 `src/llama-adapter.cpp:221-238`

```cpp
struct llama_adapter_lora {
    // 激活型LoRA：通过特定token序列触发
    std::vector<llama_token> alora_invocation_tokens;
};

// 解析aLoRA调用序列
const auto & key = llm_kv(LLM_KV_ADAPTER_ALORA_INVOCATION_TOKENS);
const int kid = gguf_find_key(ctx_gguf.get(), key.c_str());
if (kid >= 0) {
    const size_t seq_len = gguf_get_arr_n(ctx_gguf.get(), kid);
    const void * data = gguf_get_arr_data(ctx_gguf.get(), kid);
    adapter.alora_invocation_tokens.resize(seq_len);
    std::copy(
        (const llama_token *)data,
        (const llama_token *)data + seq_len,
        adapter.alora_invocation_tokens.begin());
}
```

**aLoRA工作原理**：
```
普通LoRA: 始终生效
aLoRA: 检测到调用token后才生效

示例:
基础模型: 通用对话模型
aLoRA: 代码专家
调用token: [CODE_MODE]

对话:
用户: 你好
模型: 你好！有什么可以帮助你的吗？（基础模型）

用户: [CODE_MODE] 写一个快排
模型: ```python
def quicksort(arr):
    ...（激活代码LoRA）
```
```

## 5. 设计中的取舍

### 5.1 计算与存储的权衡

| 方案 | 存储开销 | 计算开销 | 灵活性 |
|------|----------|----------|--------|
| 全参数微调 | 100% | 100% | 低 |
| LoRA (r=16) | 0.78% | ~105% | 高 |
| LoRA (r=64) | 3.1% | ~110% | 高 |
| 适配器合并 | 100% | 100% | 无 |

### 5.2 适配器合并vs动态加载

**合并（Merge）**：
- 优点：推理时无额外开销
- 缺点：失去基础模型，无法动态调整

**动态加载**：
- 优点：灵活切换、组合多个适配器
- 缺点：额外内存和计算开销

### 5.3 秩（rank）的选择

- **r=4-8**: 简单任务，极少参数
- **r=16-32**: 平衡选择（推荐）
- **r=64-128**: 复杂任务，接近全参数效果
- **r>256**: 收益递减，不推荐

## 6. 动手练习

### 练习1：加载并使用LoRA适配器

```cpp
#include "llama.h"

int main() {
    // 加载基础模型
    llama_model_params model_params = llama_model_default_params();
    llama_model * model = llama_load_model("base.gguf", model_params);
    
    // 加载LoRA适配器
    llama_adapter_lora * lora = llama_adapter_lora_init(model, "lora.gguf");
    if (!lora) {
        fprintf(stderr, "Failed to load LoRA\n");
        return 1;
    }
    
    // 创建上下文
    llama_context_params ctx_params = llama_context_default_params();
    llama_context * ctx = llama_new_context_with_model(model, ctx_params);
    
    // 设置LoRA适配器（scale=1.0）
    llama_set_adapter_lora(ctx, lora, 1.0f);
    
    // 推理...
    
    // 清理
    llama_rm_adapter_lora(ctx, lora);
    llama_adapter_lora_free(lora);
    llama_free(ctx);
    llama_free_model(model);
    
    return 0;
}
```

### 练习2：转换HuggingFace LoRA

```bash
# 下载HuggingFace LoRA
huggingface-cli download user/repo --local-dir ./lora

# 转换为GGUF
python convert_lora_to_gguf.py ./lora \
    --base-model-id meta-llama/Llama-2-7b-hf \
    --outtype f16 \
    --outfile ./lora.gguf

# 验证转换结果
./llama-cli -m base.gguf --lora lora.gguf -p "Test prompt"
```

### 练习3：多适配器插值

```cpp
// 创建风格混合效果
llama_adapter_lora * lora_formal = llama_adapter_lora_init(model, "formal.gguf");
llama_adapter_lora * lora_casual = llama_adapter_lora_init(model, "casual.gguf");

// 测试不同插值比例
for (float alpha = 0.0f; alpha <= 1.0f; alpha += 0.25f) {
    llama_clear_adapter_lora(ctx);  // 清除之前的适配器
    
    // 插值: formal × alpha + casual × (1-alpha)
    llama_set_adapter_lora(ctx, lora_formal, alpha);
    llama_set_adapter_lora(ctx, lora_casual, 1.0f - alpha);
    
    printf("\n=== Alpha = %.2f ===\n", alpha);
    // 生成文本...
}
```

## 7. 本课小结

本章我们深入学习了llama.cpp的LoRA适配器支持：

1. **LoRA原理**：
   - 低秩分解：ΔW = B × A
   - 参数量减少：从d×k到r×(d+k)
   - 缩放因子：scale = alpha / rank

2. **适配器加载**：
   - GGUF格式验证（general.type="adapter"）
   - 张量配对（lora_a + lora_b）
   - 内存分配与数据加载

3. **推理应用**：
   - 动态查询适配器权重
   - 计算图集成
   - 多适配器组合（加权求和）

4. **转换工具**：
   - `convert_lora_to_gguf.py`使用
   - LoraTorchTensor封装
   - 命名映射规则

5. **高级特性**：
   - aLoRA：token触发的条件适配器
   - 多适配器插值：平滑切换风格

**关键源码文件**：`src/llama-adapter.cpp`、`src/llama-adapter.h`、`convert_lora_to_gguf.py`

**下一步**：学习控制向量技术，通过向量运算实现模型风格的实时控制。
