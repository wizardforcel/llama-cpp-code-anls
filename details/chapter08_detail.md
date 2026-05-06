# 第8章 模型加载与管理 —— 模型文件的"智能管家"

## 学习目标
1. 理解GGUF文件格式的四段式结构
2. 掌握模型加载流程与内存映射技术
3. 了解模型量化转换的实现
4. 掌握三种内存管理策略

---

## 生活类比：跨国企业的档案管理系统

想象llama.cpp是一位**精通档案管理的智能管家**：

- **GGUF文件** = 一座标准化的智能仓库
  - **Header** = 仓库总览图（魔数、版本、结构说明）
  - **Tensor Info** = 货物清单（每个箱子的规格标签）
  - **Alignment Padding** = 货架对齐区（方便叉车搬运）
  - **Tensor Data** = 实际货物存放区
  - **Metadata** = 货物档案卡（模型参数、特殊标记）
- **内存映射（mmap）** = 智能借阅系统（不搬货，只开窗口看）
- **延迟加载** = 按需取货（用多少取多少）
- **内存管理策略** = 不同的仓库管理模式
  - **常规模式** = 全部搬入临时库房
  - **混合模式** = 常用品放库房，大件 mmap
  - **循环模式** = 轮流使用有限的货架空间

就像优秀的管家需要高效管理档案，llama.cpp需要高效管理模型数据的加载和内存使用。

---

## 源码地图

```
ggml/include/gguf.h
├── struct gguf_context      # GGUF上下文
│   ├── header               # 文件头
│   ├── tensor_infos         # 张量信息数组
│   └── kv_data              # 元数据键值对
├── gguf_init_from_file()    # 从文件加载
├── gguf_get_tensor_info()   # 获取张量信息
├── gguf_get_val_str()       # 获取元数据字符串
├── gguf_get_val_u32()       # 获取元数据u32
├── gguf_get_val_f32()       # 获取元数据f32
└── gguf_get_val_arr()       # 获取元数据数组

ggml/src/gguf.c
├── GGUF文件解析（第100-500行）
├── Header读取（第200-300行）
│   └── gguf_read_header()
├── Tensor Info解析（第300-400行）
│   └── gguf_read_tensor_info()
└── Metadata读取（第400-500行）
    └── gguf_read_kv()

src/llama-model-loader.h/cpp
├── llama_model_loader       # 模型加载器类
│   ├── gguf_ctx             # GGUF上下文
│   ├── path_model           # 模型路径
│   └── mapping_addr         # 内存映射地址
├── load()                   # 主加载函数（第1-500行）
│   ├── gguf_init_from_file()
│   ├── load_hparams()       # 加载超参数
│   ├── load_vocab()         # 加载词表
│   └── load_tensors()       # 加载张量
├── 张量验证（第500-1000行）
│   └── validate_tensor()
└── 内存映射（第1000-1500行）
    └── llama_mmap

src/llama-model.h/cpp
├── llama_model              # 模型结构
│   ├── hparams              # 超参数
│   ├── vocab                # 词表
│   ├── tensors              # 张量数组
│   ├── layers               # 层数组
│   └── backends             # 后端数组
├── llama_layer              # 层结构
│   ├── attn_norm            # 注意力归一化
│   ├── wq/wk/wv/wo          # 注意力权重
│   ├── ffn_norm             # FFN归一化
│   └── wgate/wup/wdown      # FFN权重
└── llama_cparams            # 计算参数

src/llama-memory.h/cpp
├── llama_memory             # 内存管理基类
├── llama_memory_default     # 常规内存管理
├── llama_memory_hybrid      # 混合内存管理
└── llama_memory_recurrent   # 循环模型内存

src/llama-mmap.h/cpp
├── llama_mmap               # 内存映射类
│   ├── addr                 # 映射地址
│   ├── size                 # 映射大小
│   ├── llama_mmap()         # 构造函数
│   └── ~llama_mmap()        # 析构函数
└── llama_mlock              # 内存锁定类
```

---

## 8.1 GGUF文件格式详解

### 8.1.1 GGUF设计目标

**为什么需要GGUF？**
- **GGML格式**的问题：太简单，无法存储元数据
- **PyTorch格式**的问题：依赖Python，体积大
- **GGUF**的优势：
  - 自包含（元数据+权重）
  - 可扩展（K-V元数据系统）
  - 对齐友好（支持内存映射）
  - 跨平台（C语言实现）

### 8.1.2 四段式结构

**整体结构**：
```
┌─────────────────────────────────────────────────────┐
│                    GGUF文件                          │
├─────────────────────────────────────────────────────┤
│  Header (固定大小)                                   │
│  - Magic (4字节): "GGUF"                            │
│  - Version (4字节): 文件格式版本                      │
│  - tensor_count (8字节): 张量数量                     │
│  - metadata_kv_count (8字节): 元数据项数              │
├─────────────────────────────────────────────────────┤
│  Tensor Info (变长)                                  │
│  - 每个张量: 名称长度+名称+维度数+维度数组+类型+偏移量 │
├─────────────────────────────────────────────────────┤
│  Alignment Padding (填充)                            │
│  - 对齐到256字节边界                                  │
├─────────────────────────────────────────────────────┤
│  Tensor Data (数据区)                                │
│  - 实际的权重数据，按Tensor Info中的偏移量存放         │
├─────────────────────────────────────────────────────┤
│  Metadata K-V (键值对)                               │
│  - 模型参数、特殊token、版本信息等                     │
└─────────────────────────────────────────────────────┘
```

### 8.1.3 Header解析

**源码位置**：`ggml/src/gguf.c` (第200-300行)

```c
// GGUF Header结构
struct gguf_header {
    uint32_t magic;              // "GGUF" = 0x46475547 (小端)
    uint32_t version;            // 当前版本3
    uint64_t tensor_count;       // 张量数量
    uint64_t metadata_kv_count;  // 元数据键值对数量
};

// Header读取
bool gguf_read_header(FILE * file, struct gguf_header * header) {
    // 读取魔数
    if (fread(&header->magic, sizeof(header->magic), 1, file) != 1) {
        return false;
    }

    // 验证魔数
    if (header->magic != GGUF_MAGIC) {
        fprintf(stderr, "Invalid GGUF magic: %08x\n", header->magic);
        return false;
    }

    // 读取版本
    if (fread(&header->version, sizeof(header->version), 1, file) != 1) {
        return false;
    }

    // 版本兼容性检查
    if (header->version < GGUF_MIN_VERSION ||
        header->version > GGUF_MAX_VERSION) {
        fprintf(stderr, "Unsupported GGUF version: %d\n", header->version);
        return false;
    }

    // 读取张量和元数据计数
    fread(&header->tensor_count, sizeof(header->tensor_count), 1, file);
    fread(&header->metadata_kv_count, sizeof(header->metadata_kv_count), 1, file);

    return true;
}
```

### 8.1.4 Tensor Info解析

**源码位置**：`ggml/src/gguf.c` (第300-450行)

```c
// 单个张量信息
struct gguf_tensor_info {
    // 名称
    uint64_t namelen;
    char * name;

    // 维度
    uint32_t n_dims;
    uint64_t ne[GGUF_MAX_DIMS];

    // 类型和位置
    enum ggml_type type;
    uint64_t offset;  // 相对于data区的偏移

    // 计算得到的信息
    size_t size;      // 总字节数
    void * data;      // 加载后的数据指针
};

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
```

### 8.1.5 Metadata系统

**源码位置**：`ggml/include/gguf.h` (第100-200行)

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

// 常见的元数据键
#define LLM_KV_GENERAL_ARCHITECTURE "general.architecture"
#define LLM_KV_GENERAL_NAME         "general.name"
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
  "tokenizer.ggml.token_type": [1, 1, 1, ...],
}
```

---

## 8.2 模型加载流程

### 8.2.1 llama_model_loader核心流程

**源码位置**：`src/llama-model-loader.cpp` (第1-500行)

```cpp
class llama_model_loader {
public:
    struct gguf_context * gguf_ctx;  // GGUF上下文
    std::string path_model;          // 模型路径

    // 主加载函数
    void load(llama_model & model, const llama_model_params & params) {
        // ① 打开并解析GGUF文件
        gguf_ctx = gguf_init_from_file(path_model.c_str(), {
            .no_alloc = params.use_mmap,  // 使用mmap时不分配内存
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
        // 从metadata读取架构
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
            size_t offset = gguf_get_tensor_offset(gguf_ctx, i);

            // 创建ggml_tensor
            struct ggml_tensor * tensor = create_tensor_for_name(name);

            // 设置数据
            if (params.use_mmap) {
                // 内存映射：tensor数据指向mmap区域
                tensor->data = (char *)mapping_addr + offset;
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

**源码位置**：`src/llama-mmap.h/cpp` (第1-200行)

```cpp
// 内存映射实现
class llama_mmap {
public:
    void * addr;      // 映射后的内存地址
    size_t size;      // 映射大小

    // 构造函数执行mmap
    llama_mmap(const char * path, size_t prefetch = 0) {
        int fd = open(path, O_RDONLY);
        if (fd < 0) {
            throw std::runtime_error("Failed to open file");
        }

        struct stat st;
        fstat(fd, &st);
        size = st.st_size;

        // 执行mmap
        addr = mmap(NULL, size, PROT_READ, MAP_PRIVATE | MAP_POPULATE, fd, 0);
        if (addr == MAP_FAILED) {
            close(fd);
            throw std::runtime_error("mmap failed");
        }

        close(fd);

        // 预读取（可选）
        if (prefetch > 0) {
            prefetch_pages(prefetch);
        }
    }

    // 析构函数执行munmap
    ~llama_mmap() {
        if (addr) {
            munmap(addr, size);
        }
    }

private:
    void prefetch_pages(size_t nbytes) {
        // 使用madvise建议内核预读取
        madvise(addr, std::min(nbytes, size), MADV_WILLNEED);
    }
};
```

**mmap的优势**：
1. **零拷贝**：数据直接从磁盘缓存映射到进程空间
2. **按需加载**：只加载实际访问的页面
3. **内存节省**：多进程共享同一份物理内存
4. **快速启动**：无需等待全部数据读取

### 8.2.3 张量验证

**源码位置**：`src/llama-model-loader.cpp` (第500-800行)

```cpp
// 验证加载的张量
void validate_tensor(
    const struct ggml_tensor * tensor,
    const std::string & expected_name,
    const std::vector<int64_t> & expected_shape,
    enum ggml_type expected_type) {

    // ① 验证名称
    GGML_ASSERT(tensor->name == expected_name);

    // ② 验证形状
    GGML_ASSERT(tensor->n_dims == expected_shape.size());
    for (size_t i = 0; i < expected_shape.size(); i++) {
        GGML_ASSERT(tensor->ne[i] == expected_shape[i]);
    }

    // ③ 验证类型
    GGML_ASSERT(tensor->type == expected_type);

    // ④ 验证数据对齐
    GGML_ASSERT((uintptr_t)tensor->data % GGUF_DEFAULT_ALIGNMENT == 0);
}
```

---

## 8.3 模型保存与转换

### 8.3.1 GGUF写入实现

**源码位置**：`src/llama-model-saver.cpp` (第1-300行)

```cpp
class llama_model_saver {
public:
    // 保存模型到GGUF文件
    void save(const std::string & path, const llama_model & model) {
        FILE * file = fopen(path.c_str(), "wb");
        if (!file) {
            throw std::runtime_error("Failed to create file");
        }

        // ① 写入Header
        write_header(file, model);

        // ② 写入Tensor Info
        write_tensor_info(file, model);

        // ③ 写入Alignment Padding
        write_alignment(file);

        // ④ 写入Tensor Data
        write_tensor_data(file, model);

        // ⑤ 写入Metadata
        write_metadata(file, model);

        fclose(file);
    }

private:
    void write_header(FILE * file, const llama_model & model) {
        struct gguf_header header;
        header.magic = GGUF_MAGIC;
        header.version = GGUF_VERSION;
        header.tensor_count = model.tensors.size();
        header.metadata_kv_count = get_metadata_count(model);

        fwrite(&header, sizeof(header), 1, file);
    }

    void write_tensor_info(FILE * file, const llama_model & model) {
        size_t data_offset = 0;

        for (const auto &[ name, tensor ] : model.tensors) {
            // 名称
            uint64_t namelen = name.size();
            fwrite(&namelen, sizeof(namelen), 1, file);
            fwrite(name.c_str(), 1, namelen, file);

            // 维度
            fwrite(&tensor->n_dims, sizeof(tensor->n_dims), 1, file);
            fwrite(tensor->ne, sizeof(int64_t), tensor->n_dims, file);

            // 类型
            uint32_t type = tensor->type;
            fwrite(&type, sizeof(type), 1, file);

            // 偏移量
            fwrite(&data_offset, sizeof(data_offset), 1, file);

            // 计算下一个张量的偏移
            data_offset += ggml_nbytes(tensor);
        }
    }
};
```

---

## 8.4 内存管理策略

### 8.4.1 三种内存管理方案对比

| 策略 | 内存占用 | 加载速度 | 适用场景 |
|-----|---------|---------|---------|
| 常规（default）| 高 | 慢 | 小模型，充足内存 |
| 混合（hybrid）| 中 | 中 | 大模型，mmap辅助 |
| 循环（ring）| 低 | 快 | 超大规模模型 |

### 8.4.2 常规内存管理

**源码位置**：`src/llama-memory.cpp` (第1-200行)

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

### 8.4.3 混合内存管理

**源码位置**：`src/llama-memory-hybrid.cpp` (第1-300行)

```cpp
class llama_memory_hybrid : public llama_memory {
public:
    // 策略：大权重用mmap，激活值用malloc
    void alloc_tensor(struct ggml_tensor * tensor) override {
        if (is_weight_tensor(tensor)) {
            // 权重：使用mmap区域
            tensor->data = get_mmap_ptr(tensor);
        } else {
            // 激活值：动态分配
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

## 动手练习

### 练习1：解析GGUF文件
使用Python读取一个GGUF文件，打印：
1. Header信息
2. 所有张量的名称和形状
3. 所有元数据

提示：使用 `gguf-py` 库或手动解析。

### 练习2：计算内存占用
给定一个7B模型（Q4_K_M量化），计算：
1. GGUF文件大小
2. mmap加载时的实际内存占用
3. 常规加载时的内存占用

### 练习3：阅读模型加载代码
阅读 `src/llama-model-loader.cpp` 第1-500行，回答：
1. `use_mmap`参数如何影响加载流程？
2. 张量是如何与mmap区域关联的？
3. 加载过程中的错误处理机制是什么？

---

## 本课小结

| 概念 | 一句话总结 |
|------|-----------|
| GGUF | 自包含模型格式，Header+Tensor Info+Data+Metadata |
| mmap | 内存映射，零拷贝，按需加载 |
| 延迟加载 | 只加载当前需要的张量数据 |
| 混合内存 | 权重mmap+激活动态分配，平衡速度与内存 |
| 元数据系统 | K-V结构，可扩展的模型信息存储 |

---

*本章对应源码版本：master (2026-04-07)*
