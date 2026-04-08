# 第8章 模型加载与管理 —— 模型文件的"智能管家"

大语言模型的参数规模庞大，如何高效地加载和管理这些参数是推理性能的关键。llama.cpp 提供了灵活的模型加载机制，支持内存映射（mmap）、延迟加载等优化技术。本章将深入解析 GGUF 文件格式和模型加载系统。

## 学习目标

1. 理解 GGUF 文件格式的四段式结构
2. 掌握模型加载流程与内存映射技术
3. 了解模型量化转换的实现
4. 掌握三种内存管理策略

## 生活类比：跨国企业的档案管理系统

想象 llama.cpp 是一位**精通档案管理的智能管家**：

- **GGUF 文件** = 一座标准化的智能仓库
  - **Header** = 仓库总览图（魔数、版本、结构说明）
  - **Tensor Info** = 货物清单（每个箱子的规格标签）
  - **Alignment Padding** = 货架对齐区（方便叉车搬运）
  - **Tensor Data** = 实际货物存放区
  - **Metadata** = 货物档案卡（模型参数、特殊标记）
  
- **内存映射（mmap）** = 智能借阅系统
  - 不搬货，只开窗口查看
  - 需要时才真正把货物搬进工作区
  
- **延迟加载** = 按需取货
  - 用多少取多少，不浪费空间
  
- **内存管理策略** = 不同的仓库管理模式
  - **常规模式** = 全部搬入临时库房（简单直接）
  - **混合模式** = 常用品放库房，大件 mmap（平衡之选）
  - **循环模式** = 轮流使用有限的货架空间（极致压缩）

就像优秀的管家需要高效管理档案，llama.cpp 需要高效管理模型数据的加载和内存使用。

---

## 8.1 GGUF 文件格式详解

### 8.1.1 GGUF 设计目标

**为什么需要 GGUF？**

| 格式 | 问题 | GGUF 解决方案 |
|-----|------|--------------|
| GGML | 太简单，无法存储元数据 | 完整的 K-V 元数据系统 |
| PyTorch | 依赖 Python，体积大 | C 实现，零依赖 |
| SafeTensors | 不支持量化 | 原生支持多种量化格式 |
| ONNX | 复杂，不适合 LLM | 针对 LLM 优化设计 |

**GGUF 的核心优势**：
1. **自包含**：元数据 + 权重在一个文件
2. **可扩展**：K-V 元数据系统支持任意属性
3. **对齐友好**：支持内存映射，直接加载
4. **跨平台**：纯 C 实现，无依赖

### 8.1.2 四段式结构

**整体结构**：

```
┌─────────────────────────────────────────────────────┐
│                    GGUF 文件                         │
├─────────────────────────────────────────────────────┤
│  Header (固定大小，24 字节)                           │
│  - Magic (4 字节): "GGUF"                            │
│  - Version (4 字节): 文件格式版本                      │
│  - tensor_count (8 字节): 张量数量                    │
│  - metadata_kv_count (8 字节): 元数据项数              │
├─────────────────────────────────────────────────────┤
│  Tensor Info (变长，每个张量一个条目)                  │
│  - 名称长度 + 名称                                    │
│  - 维度数 + 维度数组                                   │
│  - 类型 + 偏移量                                      │
├─────────────────────────────────────────────────────┤
│  Alignment Padding (0-255 字节填充)                   │
│  - 对齐到 256 字节边界                                 │
├─────────────────────────────────────────────────────┤
│  Tensor Data (数据区)                                │
│  - 实际的权重数据                                     │
│  - 按 Tensor Info 中的偏移量存放                        │
├─────────────────────────────────────────────────────┤
│  Metadata K-V (键值对区)                             │
│  - 模型参数、特殊 token、版本信息等                     │
└─────────────────────────────────────────────────────┘
```

### 8.1.3 Header 解析

**源码位置**：`ggml/src/gguf.c`（第 200-300 行）

```c
// GGUF Header 结构
struct gguf_header {
    uint32_t magic;              // "GGUF" = 0x46475547 (小端序)
    uint32_t version;            // 当前版本 3
    uint64_t tensor_count;       // 张量数量
    uint64_t metadata_kv_count;  // 元数据键值对数量
};

这段代码定义了GGUF文件的头部结构。包含魔数(用于文件格式识别)、版本号、张量数量和元数据键值对数量。头部固定24字节，是解析GGUF文件的入口点。

// Header 读取
bool gguf_read_header(FILE * file, struct gguf_header * header) {
    // ① 读取魔数
    if (fread(&header->magic, sizeof(header->magic), 1, file) != 1) {
        return false;
    }

    // ② 验证魔数
    if (header->magic != GGUF_MAGIC) {
        fprintf(stderr, "Invalid GGUF magic: %08x\n", header->magic);
        return false;
    }

    // ③ 读取版本
    if (fread(&header->version, sizeof(header->version), 1, file) != 1) {
        return false;
    }

    // ④ 版本兼容性检查
    if (header->version < GGUF_MIN_VERSION ||
        header->version > GGUF_MAX_VERSION) {
        fprintf(stderr, "Unsupported GGUF version: %d\n", header->version);
        return false;
    }

    // ⑤ 读取张量和元数据计数
    fread(&header->tensor_count, sizeof(header->tensor_count), 1, file);
    fread(&header->metadata_kv_count, sizeof(header->metadata_kv_count), 1, file);

    return true;
}

这段代码实现了GGUF文件头的读取和验证。首先读取并验证魔数(0x46475547)，然后读取版本号进行兼容性检查，最后读取张量数量和元数据计数。任何步骤失败都会返回false，确保文件格式正确。
```

### 8.1.4 Tensor Info 解析

**源码位置**：`ggml/src/gguf.c`（第 300-450 行）

```c
// 单个张量信息
struct gguf_tensor_info {
    // 名称
    uint64_t namelen;
    char * name;

    // 维度
    uint32_t n_dims;
    uint64_t ne[GGUF_MAX_DIMS];  // 各维度大小

    // 类型和位置
    enum ggml_type type;         // GGML_TYPE_F32, GGML_TYPE_Q4_0 等
    uint64_t offset;             // 相对于 data 区的偏移

    // 计算得到的信息
    size_t size;                 // 总字节数
    void * data;                 // 加载后的数据指针
};

这段代码定义了GGUF文件中单个张量的元信息结构。包含张量名称、维度信息、数据类型、在数据区的偏移量及计算得到的总大小。这些信息存储在文件头部，用于快速定位和数据验证。

// 读取单个张量信息
bool gguf_read_tensor_info(FILE * file, struct gguf_tensor_info * info) {
    // ① 读取名称长度和名称
    fread(&info->namelen, sizeof(info->namelen), 1, file);
    info->name = malloc(info->namelen + 1);
    fread(info->name, 1, info->namelen, file);
    info->name[info->namelen] = '\0';

    // ② 读取维度信息
    fread(&info->n_dims, sizeof(info->n_dims), 1, file);
    fread(info->ne, sizeof(uint64_t), info->n_dims, file);

    // ③ 读取类型
    uint32_t type_id;
    fread(&type_id, sizeof(type_id), 1, file);
    info->type = (enum ggml_type)type_id;

    // ④ 读取偏移量
    fread(&info->offset, sizeof(info->offset), 1, file);

    // ⑤ 计算总大小
    info->size = ggml_type_size(info->type);
    for (uint32_t i = 0; i < info->n_dims; i++) {
        info->size *= info->ne[i];
    }

    return true;
}

这段代码实现了GGUF张量信息的读取。依次读取名称长度和名称字符串、维度数量和各维度大小、数据类型及在数据区的偏移量，最后根据类型和维度计算张量总字节数，完成单个张量元数据的解析。
```

### 8.1.5 Metadata 系统

**源码位置**：`ggml/include/gguf.h`（第 100-200 行）

```c
// 元数据值类型
enum gguf_metadata_value_type {
    GGUF_METADATA_TYPE_UINT8 = 0,
    GGUF_METADATA_TYPE_INT8 = 1,
    GGUF_METADATA_TYPE_UINT16 = 2,
    GGUF_METADATA_TYPE_INT16 = 3,
    GGUF_METADATA_TYPE_UINT32 = 4,
    GGUF_METADATA_TYPE_INT32 = 5,
    GGUF_METADATA_TYPE_FLOAT32 = 6,
    GGUF_METADATA_TYPE_BOOL = 7,
    GGUF_METADATA_TYPE_STRING = 8,
    GGUF_METADATA_TYPE_ARRAY = 9,
    GGUF_METADATA_TYPE_UINT64 = 10,
    GGUF_METADATA_TYPE_INT64 = 11,
    GGUF_METADATA_TYPE_FLOAT64 = 12,
};

// 常见的元数据键（标准化命名）
#define LLM_KV_GENERAL_ARCHITECTURE "general.architecture"
#define LLM_KV_GENERAL_NAME         "general.name"
#define LLM_KV_GENERAL_DESCRIPTION  "general.description"
#define LLM_KV_CONTEXT_LENGTH       "{arch}.context_length"
#define LLM_KV_EMBEDDING_LENGTH     "{arch}.embedding_length"
#define LLM_KV_ATTENTION_HEAD_COUNT "{arch}.attention.head_count"
#define LLM_KV_ROPE_DIMENSION_COUNT "{arch}.rope.dimension_count"
#define LLM_KV_ROPE_FREQ_BASE       "{arch}.rope.freq_base"
```

**示例元数据**（Llama2 7B）：

```json
{
  "general.architecture": "llama",
  "general.name": "LLaMA v2",
  "llama.context_length": 4096,
  "llama.embedding_length": 4096,
  "llama.block_count": 32,
  "llama.feed_forward_length": 11008,
  "llama.attention.head_count": 32,
  "llama.attention.head_count_kv": 32,
  "llama.attention.layer_norm_rms_epsilon": 1e-06,
  "llama.rope.dimension_count": 128,
  "llama.rope.freq_base": 10000.0,
  "tokenizer.ggml.model": "llama",
  "tokenizer.ggml.tokens": ["<s>", "▁the", "▁to", ...],
  "tokenizer.ggml.scores": [0.0, -0.1, -0.2, ...],
  "tokenizer.ggml.token_type": [1, 1, 1, ...]
}
```

---

## 8.2 模型加载流程

### 8.2.1 llama_model_loader 核心流程

**源码位置**：`src/llama-model-loader.cpp`（第 1-500 行）

```cpp
class llama_model_loader {
public:
    struct gguf_context * gguf_ctx;   // GGUF 上下文
    std::string path_model;           // 模型路径
    std::unique_ptr<llama_mmap> mapping;  // 内存映射

    // 主加载函数
    void load(llama_model & model, const llama_model_params & params) {
        // ① 打开并解析 GGUF 文件
        gguf_ctx = gguf_init_from_file(path_model.c_str(), {
            .no_alloc = params.use_mmap,  // 使用 mmap 时不分配内存
        });

        // ② 加载超参数
        load_hparams(model);

        // ③ 加载词汇表
        load_vocab(model);

        // ④ 加载张量
        load_tensors(model, params);

        // ⑤ 验证模型完整性
        validate_model(model);
    }

private:
    void load_hparams(llama_model & model) {
        // 从 metadata 读取架构
        std::string arch_name = gguf_get_val_str(
            gguf_ctx,
            gguf_find_key(gguf_ctx, "general.architecture")
        );
        model.arch = llm_arch_from_string(arch_name);

        // 读取各超参数
        model.hparams.n_vocab = get_u32(LLM_KV_VOCAB_SIZE);
        model.hparams.n_ctx_train = get_u32(LLM_KV_CONTEXT_LENGTH);
        model.hparams.n_embd = get_u32(LLM_KV_EMBEDDING_LENGTH);
        model.hparams.n_layer = get_u32(LLM_KV_BLOCK_COUNT);
        // ... 更多参数
    }

    void load_tensors(llama_model & model, const llama_model_params & params) {
        const int n_tensors = gguf_get_n_tensors(gguf_ctx);

        for (int i = 0; i < n_tensors; i++) {
            const char * name = gguf_get_tensor_name(gguf_ctx, i);
            size_t offset = gguf_get_data_offset(gguf_ctx) + 
                           gguf_get_tensor_offset(gguf_ctx, i);

            // 创建 ggml_tensor
            struct ggml_tensor * tensor = create_tensor_for_name(name);

            // 设置数据
            if (params.use_mmap) {
                // 内存映射：tensor 数据指向 mmap 区域
                tensor->data = (char *)mapping->addr + offset;
            } else {
                // 常规加载：从文件读取到分配的内存
                read_tensor_data(i, tensor->data);
            }

            model.tensors[name] = tensor;
        }
    }
};
```

### 8.2.2 内存映射（mmap）技术

**源码位置**：`src/llama-mmap.h/cpp`（第 1-200 行）

```cpp
// 内存映射实现
class llama_mmap {
public:
    void * addr;      // 映射后的内存地址
    size_t size;      // 映射大小

    // 构造函数执行 mmap
    llama_mmap(const char * path, size_t prefetch = 0) {
        int fd = open(path, O_RDONLY);
        if (fd < 0) {
            throw std::runtime_error("Failed to open file");
        }

        struct stat st;
        fstat(fd, &st);
        size = st.st_size;

        // 执行 mmap：将文件映射到进程地址空间
        addr = mmap(NULL, size, PROT_READ, MAP_PRIVATE | MAP_POPULATE, fd, 0);
        if (addr == MAP_FAILED) {
            close(fd);
            throw std::runtime_error("mmap failed");
        }

        close(fd);

        // 预读取（可选）：建议内核预加载指定范围
        if (prefetch > 0) {
            prefetch_pages(prefetch);
        }
    }

    // 析构函数执行 munmap
    ~llama_mmap() {
        if (addr) {
            munmap(addr, size);
        }
    }

private:
    void prefetch_pages(size_t nbytes) {
        // 使用 madvise 建议内核预读取
        madvise(addr, std::min(nbytes, size), MADV_WILLNEED);
    }
};
```

**mmap 的优势**：

1. **零拷贝**：数据直接从磁盘缓存映射到进程空间，无需 read() 拷贝
2. **按需加载**：只加载实际访问的页面（lazy loading）
3. **内存节省**：多进程共享同一份物理内存
4. **快速启动**：无需等待全部数据读取，立即可用

**mmap vs 常规加载对比**：

| 特性 | 常规加载 | mmap |
|------|---------|------|
| 启动速度 | 慢（需读取全部） | 快（立即可用） |
| 内存占用 | 高（两份数据） | 低（共享缓存） |
| 多进程 | 各有一份 | 共享物理页 |
| 随机访问 | 已加载，快 | 可能缺页，慢 |
| 适用场景 | 小模型，充足内存 | 大模型，内存紧张 |

---

## 8.3 内存管理策略

### 8.3.1 三种内存管理方案对比

| 策略 | 内存占用 | 加载速度 | 适用场景 |
|-----|---------|---------|---------|
| 常规（default）| 高 | 慢 | 小模型，充足内存 |
| 混合（hybrid）| 中 | 中 | 大模型，mmap 辅助 |
| 映射（mmap）| 低 | 快 | 超大规模模型 |

### 8.3.2 常规内存管理

```cpp
class llama_memory_default : public llama_memory {
public:
    // 分配张量内存
    void alloc_tensor(struct ggml_tensor * tensor) override {
        size_t size = ggml_nbytes(tensor);
        tensor->data = malloc(size);
        if (!tensor->data) {
            throw std::bad_alloc();
        }
    }

    // 释放张量内存
    void free_tensor(struct ggml_tensor * tensor) override {
        free(tensor->data);
        tensor->data = nullptr;
    }

    // 特点：每个张量单独分配，简单直接
    // 缺点：容易产生内存碎片
};
```

### 8.3.3 混合内存管理

```cpp
class llama_memory_hybrid : public llama_memory {
public:
    // 策略：大权重用 mmap，激活值用 malloc
    void alloc_tensor(struct ggml_tensor * tensor) override {
        if (is_weight_tensor(tensor)) {
            // 权重：使用 mmap 区域（只读，共享）
            tensor->data = get_mmap_ptr(tensor);
        } else {
            // 激活值：动态分配（读写，私有）
            tensor->data = malloc(ggml_nbytes(tensor));
        }
    }

private:
    bool is_weight_tensor(struct ggml_tensor * tensor) {
        // 权重张量的命名特征
        return strstr(tensor->name, "weight") != nullptr ||
               strstr(tensor->name, "bias") != nullptr;
    }
};
```

---

## 8.4 设计中的取舍

### 为什么 GGUF 使用 K-V 元数据而不是固定结构？

| 方案 | 优点 | 缺点 | GGUF 选择 |
|-----|------|------|-----------|
| **K-V 结构** | 灵活可扩展 | 解析稍复杂 | ✅ 是 |
| **固定结构** | 访问快 | 扩展困难 | ❌ |

**K-V 结构的优势**：
1. 新模型架构可以添加新的元数据字段，无需修改格式
2. 向后兼容：旧 loader 遇到不认识的关键字可以忽略
3. 自描述：文件本身包含所有必要信息

### 为什么 Tensor Data 在 Metadata 之前？

**GGUF 结构**：
```
Header -> Tensor Info -> Alignment -> Tensor Data -> Metadata
```

**原因**：
1. **内存对齐**：Tensor Data 需要按 256 字节对齐，放在前面易于计算偏移
2. **顺序访问**：加载时先读 Tensor Info，然后直接跳到 Data 区
3. **Metadata 可变**：Metadata 长度可能变化，放在最后不影响 Tensor 位置

---

## 8.5 动手练习

### 练习 1：解析 GGUF 文件

使用 Python 读取一个 GGUF 文件，打印：

```python
import struct

def read_gguf_header(filename):
    with open(filename, 'rb') as f:
        magic = f.read(4)
        version = struct.unpack('<I', f.read(4))[0]
        tensor_count = struct.unpack('<Q', f.read(8))[0]
        metadata_count = struct.unpack('<Q', f.read(8))[0]
        
        print(f"Magic: {magic}")
        print(f"Version: {version}")
        print(f"Tensor count: {tensor_count}")
        print(f"Metadata count: {metadata_count}")

read_gguf_header("model.gguf")
```

### 练习 2：计算内存占用

给定一个 7B 模型（Q4_K_M 量化），计算：

1. **GGUF 文件大小**：约 3.8GB（参数）+ 词汇表 + 元数据
2. **mmap 加载时的实际内存占用**：仅加载访问的部分，通常 4-5GB
3. **常规加载时的内存占用**：3.8GB（模型）+ 激活值缓冲区 ~1GB = ~5GB

### 练习 3：阅读模型加载代码

阅读 `src/llama-model-loader.cpp` 第 1-500 行，回答：

1. `use_mmap` 参数如何影响加载流程？
   - `use_mmap=true`：tensor->data 指向 mmap 区域
   - `use_mmap=false`：从文件读取数据到新分配的内存

2. 张量是如何与 mmap 区域关联的？
   - 通过 `tensor->data = mapping->addr + offset`

3. 加载过程中的错误处理机制是什么？
   - 使用异常（C++）和断言验证形状、类型等

---

## 8.6 本章小结

| 概念 | 一句话总结 |
|------|-----------|
| **GGUF** | 自包含模型格式，Header + Tensor Info + Alignment + Data + Metadata |
| **Magic** | 文件标识 "GGUF" = 0x46475547，用于格式识别 |
| **mmap** | 内存映射，零拷贝，按需加载，节省内存 |
| **延迟加载** | 只加载当前需要的张量数据，减少启动时间 |
| **混合内存** | 权重 mmap + 激活动态分配，平衡速度与内存 |
| **K-V 元数据** | 键值对结构，灵活可扩展的模型信息存储 |

**下一步预告**：

在理解了模型加载之后，我们将在第 9 章深入计算图构建——理解 Transformer 层如何被动态组装成计算图。
