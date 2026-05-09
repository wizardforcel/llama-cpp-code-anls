# llama.cpp 源码目录结构

本文档汇总了 llama.cpp 项目中 `src` 目录下的所有核心源码文件，包含其中的类、结构体和主要函数。

---

## 核心架构文件

### llama-arch.h

**描述**: 定义支持的模型架构枚举和张量命名规范

**枚举**:
- `llm_arch` - 支持的模型架构枚举（LLM_ARCH_LLAMA, LLM_ARCH_QWEN, LLM_ARCH_GEMMA 等）
- `llm_kv` - GGUF 元数据键名枚举
- `llm_tensor` - 模型张量类型枚举
- `llm_tensor_layer` - 张量所在层类型（INPUT, REPEATING, OUTPUT）

**结构体**:
- `LLM_KV` - GGUF 键名生成器
- `LLM_TN_IMPL` - 张量名称实现
- `LLM_TN` - 张量名称生成器
- `llm_tensor_info` - 张量信息

**函数**:
- `llm_arch_all()` - 返回所有支持的架构列表
- `llm_arch_name()` - 获取架构名称
- `llm_arch_from_string()` - 从字符串解析架构
- `llm_tensor_info_for()` - 获取张量信息
- `llm_arch_is_recurrent()` - 检查是否为循环架构
- `llm_arch_is_hybrid()` - 检查是否为混合架构
- `llm_arch_is_diffusion()` - 检查是否为扩散模型架构

---

### llama-hparams.h

**描述**: 定义模型超参数结构

**结构体**:
- `llama_hparams_posnet` - PosNet 参数
- `llama_hparams_convnext` - ConvNeXt 参数
- `llama_hparams` - 主要超参数结构
  - 包含词汇表大小、上下文长度、嵌入维度、层数等
  - 支持各种注意力变体（GQA, MLA, SWA）
  - 支持 MoE 参数
  - 支持 RoPE 参数
  - 支持 SSM/RWKV 参数

**枚举**:
- `llama_expert_gating_func_type` - 专家门控函数类型
- `llama_swa_type` - 滑动窗口注意力类型

---

### llama-model.h

**描述**: 定义模型结构和加载

**枚举**:
- `llm_type` - 模型类型/大小枚举（LLM_TYPE_7B, LLM_TYPE_13B 等）

**结构体**:
- `llama_layer_posnet` - PosNet 层
- `llama_layer_convnext` - ConvNeXt 层
- `llama_layer_shortconv` - 短卷积层
- `llama_layer_nextn` - NextN 层
- `llama_layer` - 通用模型层
  - 包含注意力权重、FFN 权重、归一化参数等
  - 支持多种变体（SSM, RWKV, MoE）
- `llama_model` - 模型主结构

**函数**:
- `llm_type_name()` - 获取模型类型名称
- `llama_internal_get_tensor_map()` - 内部测试用函数

---

## 推理上下文文件

### llama-context.h

**描述**: 推理上下文管理

**结构体**:
- `llama_memory_breakdown_data` - 内存使用统计
- `llama_context` - 推理上下文主类

**主要方法** (`llama_context`):
- 构造/析构
- `sched_reserve()` - 调度器预留
- `synchronize()` - 同步操作
- `get_model()` / `get_cparams()` - 获取模型/参数
- `get_sched()` - 获取调度器
- `n_ctx()` / `n_batch()` / `n_ubatch()` - 获取尺寸参数
- `memory_update()` - 更新内存
- `get_logits()` / `get_embeddings()` - 获取输出
- `encode()` / `decode()` - 编解码
- `state_get_size()` / `state_get_data()` / `state_set_data()` - 状态管理
- `perf_get_data()` / `perf_reset()` - 性能统计

---

### llama-batch.h

**描述**: 批次处理相关结构

**结构体**:
- `llama_ubatch` - 统一批次结构
  - 包含 token, embd, pos, seq_id 等
- `llama_batch_allocr` - 批次分配器

**方法** (`llama_batch_allocr`):
- `init()` - 初始化
- `get_batch()` - 获取批次
- `split_simple()` / `split_equal()` / `split_seq()` - 分割批次
- `ubatch_reserve()` - 预留空间

---

### llama-cparams.h

**描述**: 计算参数结构

**结构体**:
- `llama_cparams` - 计算参数
  - 上下文大小、批大小、线程数
  - RoPE 参数
  - 功能开关（embeddings, causal_attn, flash_attn 等）

---

## 计算图文件

### llama-graph.h

**描述**: 计算图构建和管理

**枚举**:
- `llm_graph_type` - 图类型（DEFAULT, ENCODER, DECODER）
- `llm_ffn_op_type` - FFN 操作类型
- `llm_ffn_gate_type` - FFN 门控类型
- `llm_norm_type` - 归一化类型

**结构体/类**:
- `llama_cross` - 交叉注意力数据
- `llm_graph_input_i` - 图输入接口基类
- `llm_graph_input_embd` - 嵌入输入
- `llm_graph_input_pos` - 位置编码输入
- `llm_graph_input_attn_temp` - 注意力温度输入
- `llm_graph_input_pos_bucket` - 位置分桶输入
- `llm_graph_input_out_ids` - 输出ID输入
- `llm_graph_input_mean` - 均值输入
- `llm_graph_input_cls` - 分类输入
- `llm_graph_input_rs` - 循环状态输入
- `llm_graph_input_cross_embd` - 交叉嵌入输入
- `llm_graph_input_attn_no_cache` - 无缓存注意力输入
- `llm_graph_input_attn_kv` - KV缓存注意力输入
- `llm_graph_input_attn_k` - K缓存注意力输入（V-less）
- `llm_graph_input_attn_kv_iswa` - iSWA KV缓存输入
- `llm_graph_input_attn_cross` - 交叉注意力输入
- `llm_graph_input_mem_hybrid` - 混合内存输入
- `llm_graph_input_mem_hybrid_k` - 混合内存K输入
- `llm_graph_input_mem_hybrid_iswa` - 混合内存iSWA输入
- `llm_graph_input_sampling` - 采样输入
- `llm_graph_params` - 图参数
- `llm_graph_result` - 图结果
- `llm_graph_context` - 图上下文

**方法** (`llm_graph_context`):
- `build_cvec()` - 构建控制向量
- `build_lora_mm()` / `build_lora_mm_id()` - 构建LoRA矩阵乘法
- `build_norm()` - 构建归一化
- `build_ffn()` / `build_moe_ffn()` - 构建FFN/MoE
- `build_attn_mha()` / `build_attn()` - 构建注意力
- `build_rs()` / `build_rs_inp()` - 构建循环状态
- `build_inp_*()` - 构建各类输入
- `build_pooling()` - 构建池化
- `build_sampling()` - 构建采样
- `build_dense_out()` - 构建输出层

**函数**:
- `llama_relative_position_bucket()` - 相对位置分桶

---

## KV缓存文件

### llama-kv-cache.h

**描述**: KV缓存实现

**类**:
- `llama_kv_cache` - KV缓存主类（继承 `llama_memory_i`）
  - `init_batch()` / `init_full()` / `init_update()` - 初始化
  - `clear()` - 清除缓存
  - `seq_rm()` / `seq_cp()` / `seq_keep()` / `seq_add()` / `seq_div()` - 序列操作
  - `prepare()` - 准备批次
  - `update()` - 更新缓存
  - `find_slot()` - 查找槽位
  - `get_k()` / `get_v()` - 获取K/V张量
  - `cpy_k()` / `cpy_v()` - 复制K/V
  - `build_input_*()` - 构建输入
  - `set_input_*()` - 设置输入

- `llama_kv_cache_context` - KV缓存上下文（继承 `llama_memory_context_i`）

---

### llama-kv-cells.h

**描述**: KV缓存单元格管理

**结构体/类**:
- `llama_kv_cell_ext` - 扩展单元格信息（2D位置）
- `llama_kv_cells` - KV单元格管理器
  - `reset()` - 重置
  - `size()` / `resize()` - 大小操作
  - `is_empty()` / `get_used()` - 状态查询
  - `cp()` / `set()` - 复制/设置
  - `rm()` - 移除
  - `seq_*()` - 序列操作
  - `pos_*()` - 位置操作

---

### llama-kv-cache-iswa.h

**描述**: 独立滑动窗口注意力KV缓存

**类**:
- `llama_kv_cache_iswa` - iSWA缓存（继承 `llama_memory_i`）
  - 管理两个KV缓存实例（基础 + SWA）
  - `get_base()` / `get_swa()` - 获取子缓存

- `llama_kv_cache_iswa_context` - iSWA缓存上下文

---

## 内存管理文件

### llama-memory.h

**描述**: 内存管理接口

**结构体**:
- `llama_memory_params` - 内存参数
- `llama_memory_context_i` - 内存上下文接口
- `llama_memory_i` - 内存管理接口

**枚举**:
- `llama_memory_status` - 内存状态

**函数**:
- `llama_memory_status_combine()` - 合并状态
- `llama_memory_status_is_fail()` - 检查失败

---

### llama-memory-recurrent.h

**描述**: 循环模型内存管理

**结构体**:
- `mem_cell` - 内存单元

**类**:
- `llama_memory_recurrent` - 循环内存（继承 `llama_memory_i`）
  - `prepare()` - 准备
  - `find_slot()` - 查找槽位

- `llama_memory_recurrent_context` - 循环内存上下文

---

### llama-memory-hybrid.h

**描述**: 混合内存管理（注意力 + 循环）

**类**:
- `llama_memory_hybrid` - 混合内存（继承 `llama_memory_i`）
  - `get_mem_attn()` / `get_mem_recr()` - 获取子内存

- `llama_memory_hybrid_context` - 混合内存上下文

---

### llama-memory-hybrid-iswa.h

**描述**: 混合内存 + iSWA支持

**类**:
- `llama_memory_hybrid_iswa` - 混合iSWA内存
- `llama_memory_hybrid_iswa_context` - 混合iSWA上下文

---

## 词汇表和分词文件

### llama-vocab.h

**描述**: 词汇表和分词器

**枚举**:
- `llama_vocab_pre_type` - 预分词类型

**类**:
- `llama_vocab` - 词汇表类
  - `load()` - 加载
  - `get_tokenizer_model()` / `get_tokenizer_pre()` - 获取分词器信息
  - `get_type()` / `get_pre_type()` - 获取类型
  - `n_tokens()` / `n_token_types()` - 获取token数量
  - `is_*()` - Token类型检查（normal, control, byte等）
  - `tokenize()` / `detokenize()` - 分词/反分词
  - `token_to_piece()` / `token_to_byte()` - Token转换
  - `find_bpe_rank()` / `get_bpe_merges()` - BPE操作
  - `print_info()` - 打印信息

---

### llama-chat.h

**描述**: 聊天模板

**枚举**:
- `llm_chat_template` - 聊天模板类型（LLM_CHAT_TEMPLATE_LLAMA_3, LLM_CHAT_TEMPLATE_CHATML等）

**函数**:
- `llm_chat_template_from_str()` - 从字符串获取模板
- `llm_chat_detect_template()` - 检测模板
- `llm_chat_apply_template()` - 应用模板

---

## 语法约束文件

### llama-grammar.h

**描述**: GBNF语法约束

**枚举**:
- `llama_gretype` - 语法元素类型

**结构体**:
- `llama_grammar_element` - 语法元素
- `llama_partial_utf8` - 部分UTF8
- `llama_grammar_candidate` - 语法候选
- `llama_grammar_rule` / `llama_grammar_stack` / `llama_grammar_stacks` - 语法规则
- `llama_grammar_candidates` - 候选列表
- `llama_grammar_parser` - 语法解析器
- `llama_grammar_trigger_pattern` - 触发模式
- `llama_grammar` - 语法主结构

**函数**:
- `llama_grammar_get_rules()` / `llama_grammar_get_stacks()` - 获取内部状态
- `llama_grammar_accept()` - 接受字符
- `llama_grammar_reject_candidates_for_stack()` - 拒绝候选
- `llama_grammar_init_impl()` - 初始化实现
- `llama_grammar_free_impl()` - 释放实现
- `llama_grammar_clone_impl()` - 克隆实现
- `llama_grammar_apply_impl()` - 应用实现
- `llama_grammar_accept_impl()` - 接受实现
- `llama_grammar_accept_str()` - 接受字符串
- `llama_grammar_accept_token()` - 接受token

---

## 采样器文件

### llama-sampler.h

**描述**: 采样器链

**结构体**:
- `llama_sampler_chain` - 采样器链
- `llama_sampler` - 采样器

**函数**:
- `llama_sampler_init_dry_testing()` - 初始化DRY测试采样器

---

## 适配器文件

### llama-adapter.h

**描述**: LoRA和控制向量适配器

**结构体**:
- `llama_adapter_cvec` - 控制向量适配器
  - `tensor_for()` - 获取张量
  - `apply_to()` - 应用
  - `apply()` - 应用到模型

- `llama_adapter_lora_weight` - LoRA权重
  - `get_scale()` - 获取缩放

- `llama_adapter_lora` - LoRA适配器
  - `get_weight()` - 获取权重
  - `get_n_nodes()` - 获取节点数

---

## 模型加载和保存文件

### llama-model-loader.h

**描述**: 模型加载器

**枚举**:
- `llama_fver` - GGUF文件版本

**结构体**:
- `llama_model_loader` - 模型加载器
  - `llama_tensor_weight` - 张量权重
  - `weight_name_comparer` - 权重名称比较器

**常量**:
- `TENSOR_NOT_REQUIRED` - 非必需张量
- `TENSOR_DUPLICATED` - 重复张量
- `TENSOR_SKIP` - 跳过张量
- `TENSOR_SKIP_IF_VIRTUAL` - 虚拟时跳过

**方法** (`llama_model_loader`):
- `get_arr_n()` / `get_arr()` - 获取数组
- `get_key()` / `get_key_or_arr()` - 获取键值
- `get_arch_name()` / `get_arch()` - 获取架构
- `get_weight()` / `require_weight()` - 获取权重
- `get_tensor_meta()` / `require_tensor_meta()` - 获取张量元数据
- `check_tensor_dims()` - 检查维度
- `create_tensor()` / `create_tensor_as_view()` - 创建张量
- `done_getting_tensors()` - 完成获取
- `init_mappings()` - 初始化映射
- `get_mapping_range()` - 获取映射范围
- `load_data_for()` / `load_all_data()` - 加载数据
- `ftype_name()` - 获取文件类型名
- `print_info()` - 打印信息

---

### llama-model-saver.h

**描述**: 模型保存器

**结构体**:
- `llama_model_saver` - 模型保存器
  - `add_kv()` - 添加元数据
  - `add_tensor()` - 添加张量
  - `add_kv_from_model()` / `add_tensors_from_model()` - 从模型添加
  - `save()` - 保存

**函数**:
- `llama_model_saver_supports_arch()` - 检查架构支持

---

## 工具文件

### llama-mmap.h

**描述**: 内存映射文件

**类型别名**:
- `llama_files` - 文件列表
- `llama_mmaps` - 内存映射列表
- `llama_mlocks` - 内存锁定列表

**结构体**:
- `llama_file` - 文件操作
  - `tell()` / `size()` - 位置/大小
  - `seek()` - 定位
  - `read_*()` / `write_*()` - 读写

- `llama_mmap` - 内存映射
  - `size()` / `addr()` - 大小/地址
  - `unmap_fragment()` - 取消映射片段

- `llama_mlock` - 内存锁定
  - `init()` / `grow_to()` - 初始化/扩展

**函数**:
- `llama_path_max()` - 获取最大路径长度

---

### llama-io.h

**描述**: IO接口

**类**:
- `llama_io_write_i` - 写接口
  - `write()` / `write_tensor()` - 写入
  - `write_string()` - 写入字符串

- `llama_io_read_i` - 读接口
  - `read()` / `read_to()` - 读取
  - `read_string()` - 读取字符串

---

### llama-impl.h

**描述**: 内部实现工具

**宏**:
- `LLAMA_LOG*()` - 日志宏

**结构体**:
- `no_init<T>` - 不初始化包装
- `time_meas` - 时间测量
- `buffer_view<T>` - 缓冲区视图

**函数**:
- `replace_all()` - 替换所有
- `format()` - 格式化
- `llama_format_tensor_shape()` - 格式化张量形状
- `gguf_kv_to_str()` - GGUF键值转字符串

---

### llama-quant.h

**描述**: 量化接口

---

### llama-ext.h

**描述**: 扩展API

**函数**:
- `llama_graph_reserve()` - 预留计算图
- `llama_ftype_get_default_type()` - 获取默认类型
- `llama_quant_init()` - 初始化量化
- `llama_quant_free()` - 释放量化
- `llama_quant_model_from_metadata()` - 从元数据创建模型
- `llama_quant_tensor_allows_quantization()` - 检查是否可量化
- `llama_quant_compute_types()` - 计算量化类型

---

## Unicode处理文件

### unicode.h

**描述**: Unicode字符处理

**结构体**:
- `unicode_cpt_flags` - Unicode码点标志

**函数**:
- `unicode_len_utf8()` - UTF8长度
- `unicode_cpt_to_utf8()` / `unicode_cpt_from_utf8()` - 码点转换
- `unicode_cpts_from_utf8()` - 字符串转码点列表
- `unicode_cpts_normalize_nfd()` - NFD规范化
- `unicode_cpt_flags_from_cpt()` / `unicode_cpt_flags_from_utf8()` - 获取标志
- `unicode_byte_to_utf8()` / `unicode_utf8_to_byte()` - 字节转换
- `unicode_tolower()` - 转小写
- `unicode_cpt_is_han()` - 检查汉字
- `unicode_regex_split()` - 正则分割

---

## 模型实现文件 (models/)

目录 `src/models/` 包含各模型架构的具体实现：

- `llama.cpp` / `llama4.cpp` - Llama系列
- `qwen.cpp` / `qwen2.cpp` / `qwen3.cpp` - Qwen系列
- `gemma.cpp` / `gemma2.cpp` / `gemma3.cpp` / `gemma4.cpp` - Gemma系列
- `deepseek.cpp` / `deepseek2.cpp` - DeepSeek系列
- `rwkv6.cpp` / `rwkv7.cpp` / `arwkv7.cpp` - RWKV系列
- `mamba.cpp` / `mamba2.cpp` - Mamba系列
- `gpt2.cpp` / `gptneox.cpp` - GPT系列
- `bert.cpp` / `modern_bert.cpp` / `eurobert.cpp` - BERT系列
- `t5.cpp` - T5系列
- `clip.cpp` - CLIP
- `llava/` - 多模态模型
- 等等...

---

## 总结

| 类别 | 主要文件 |
|------|---------|
| 架构定义 | llama-arch.h, llama-hparams.h |
| 模型结构 | llama-model.h |
| 推理上下文 | llama-context.h, llama-batch.h, llama-cparams.h |
| 计算图 | llama-graph.h |
| KV缓存 | llama-kv-cache.h, llama-kv-cells.h, llama-kv-cache-iswa.h |
| 内存管理 | llama-memory.h, llama-memory-recurrent.h, llama-memory-hybrid.h, llama-memory-hybrid-iswa.h |
| 词汇表 | llama-vocab.h, llama-chat.h |
| 语法约束 | llama-grammar.h |
| 采样 | llama-sampler.h |
| 适配器 | llama-adapter.h |
| 模型IO | llama-model-loader.h, llama-model-saver.h, llama-mmap.h |
| 工具 | llama-io.h, llama-impl.h, llama-quant.h, llama-ext.h, unicode.h |
| 具体模型 | models/*.cpp |

---

*文档生成日期: 2026-05-09*
*基于 llama.cpp 源码*
