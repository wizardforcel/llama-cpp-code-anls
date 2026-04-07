# 第2章 环境搭建与构建系统 —— 打造你的"AI 开发工作站"

## 学习目标
1. 掌握各平台的开发环境配置
2. 深入理解CMake构建系统
3. 实战多平台编译（CPU/GPU/移动端）
4. 成功运行第一个llama.cpp程序

---

## 生活类比：打造AI开发工作站

想象你要开一家**AI咖啡馆**：

- **操作系统** = 选址（Windows大街/Linux广场/macOS商圈）
- **编译器** = 厨房设备（Visual Studio专业灶/GCC传统灶/Clang精品灶）
- **CMake** = 厨房设计图（告诉设备怎么组装）
- **后端选择** = 能源方案
  - **CPU** = 电力（哪都能用，基础款）
  - **CUDA** = 天然气（NVIDIA专区，火力猛）
  - **Metal** = 太阳能（Apple生态，节能环保）
- **第一个示例** = 试营业（验证一切正常）

就像开店需要准备设备和原料，开发llama.cpp需要配置环境和构建工具。

---

## 源码地图

```
CMakeLists.txt              # 顶层构建配置
ggml/CMakeLists.txt         # GGML库构建配置
docs/build.md               # 构建文档
examples/simple/simple.cpp  # 最小示例
```

---

## 2.1 开发环境准备

### 2.1.1 Windows 环境配置

**必要组件**：
| 组件 | 推荐版本 | 用途 |
|-----|---------|------|
| Visual Studio 2022 | 17.8+ | C++编译器 |
| CMake | 3.24+ | 构建系统 |
| Git | 最新版 | 源码管理 |
| Python | 3.9+ | 模型转换脚本 |

**验证安装**：
```bash
cmake --version  # 应显示3.24+
cl               # MSVC编译器
```

### 2.1.2 Linux 环境配置

```bash
# Ubuntu/Debian
sudo apt update
sudo apt install build-essential cmake git

# 可选：CUDA支持
sudo apt install nvidia-cuda-toolkit

# 可选：OpenBLAS加速
sudo apt install libopenblas-dev
```

### 2.1.3 macOS 环境配置

```bash
# 安装Xcode Command Line Tools
xcode-select --install

# 安装Homebrew
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# 安装依赖
brew install cmake git
```

---

## 2.2 CMake 构建系统详解

### 2.2.1 顶层CMakeLists.txt结构

**源码位置**：`CMakeLists.txt` (第1-200行)

```cmake
cmake_minimum_required(VERSION 3.14)
project(llama.cpp VERSION 1.0.0)

# 选项定义
option(LLAMA_CUDA "Build with CUDA support" OFF)
option(LLAMA_METAL "Build with Metal support" OFF)
option(LLAMA_VULKAN "Build with Vulkan support" OFF)
option(LLAMA_SYCL "Build with SYCL support" OFF)

# 添加子目录
add_subdirectory(ggml)      # GGML库
add_subdirectory(src)       # llama核心
add_subdirectory(common)    # 通用工具

# 条件编译示例
if (LLAMA_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()
```

### 2.2.2 核心构建选项

**后端选项**：
| 选项 | 默认值 | 说明 |
|-----|-------|------|
| `LLAMA_CUDA` | OFF | NVIDIA CUDA后端 |
| `LLAMA_METAL` | OFF | Apple Metal后端 |
| `LLAMA_VULKAN` | OFF | Vulkan后端 |
| `LLAMA_SYCL` | OFF | Intel SYCL后端 |
| `LLAMA_OPENCL` | OFF | OpenCL后端 |

**CPU优化选项**：
| 选项 | 默认值 | 说明 |
|-----|-------|------|
| `LLAMA_AVX` | ON | AVX指令集 |
| `LLAMA_AVX2` | ON | AVX2指令集 |
| `LLAMA_AVX512` | OFF | AVX512指令集 |
| `LLAMA_FMA` | ON | FMA指令集 |
| `LLAMA_NEON` | 自动检测 | ARM NEON |

---

## 2.3 多平台编译实战

### 2.3.1 CPU 版本编译

**基础编译**：
```bash
# 克隆仓库
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp

# 创建构建目录
mkdir build && cd build

# 配置（Release模式）
cmake .. -DCMAKE_BUILD_TYPE=Release

# 编译（使用所有核心）
cmake --build . --config Release -j$(nproc)
```

### 2.3.2 CUDA 版本编译

**环境要求**：
- NVIDIA GPU (Compute Capability 5.2+)
- CUDA Toolkit 11.4+

**编译步骤**：
```bash
mkdir build-cuda && cd build-cuda

# 配置CUDA支持
cmake .. -DLLAMA_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="70;75;80;86"

# 编译
cmake --build . --config Release -j

# 验证
./bin/llama-cli --list-devices  # 应显示CUDA设备
```

**源码相关**：
- `ggml/src/ggml-cuda/` - CUDA后端实现
- `CMakeLists.txt` - CUDA选项检测（约第150行）

### 2.3.3 Metal 版本编译

```bash
mkdir build-metal && cd build-metal

# 配置Metal支持
cmake .. -DLLAMA_METAL=ON

# 编译
cmake --build . --config Release -j

# 运行（自动使用Metal加速）
./bin/llama-cli -m model.gguf -p "Hello"
```

### 2.3.4 Android 交叉编译

```bash
mkdir build-android && cd build-android

# 配置交叉编译
cmake .. \
    -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
    -DANDROID_ABI=arm64-v8a \
    -DANDROID_PLATFORM=android-28 \
    -DCMAKE_BUILD_TYPE=Release

cmake --build . -j
```

---

## 2.4 第一个可运行示例

### 2.4.1 下载测试模型

```bash
# 使用huggingface-cli下载
huggingface-cli download \
    TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF \
    tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf \
    --local-dir ./models
```

### 2.4.2 运行简单示例

**使用 llama-cli**：
```bash
./bin/llama-cli \
    -m ./models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf \
    -p "What is machine learning?" \
    -n 128
```

**源码解析**：`examples/simple/simple.cpp`

```cpp
// 1. 初始化模型参数
llama_model_params model_params = llama_model_default_params();
// 关键参数：
// - n_gpu_layers: GPU层数（0表示纯CPU）
// - main_gpu: 主GPU设备ID

// 2. 加载模型
llama_model* model = llama_load_model_from_file(argv[1], model_params);

// 3. 初始化上下文参数
llama_context_params ctx_params = llama_context_default_params();
// 关键参数：
// - n_ctx: 上下文窗口大小
// - n_batch: 批处理大小
// - n_threads: CPU线程数

// 4. 创建上下文
llama_context* ctx = llama_new_context_with_model(model, ctx_params);

// 5. 分词和推理循环...
```

---

## 常见编译问题排查

| 问题 | 错误信息 | 解决方案 |
|-----|---------|---------|
| 找不到CUDA | `Could not find CUDA` | 设置CUDA路径到环境变量 |
| AVX512错误 | `unknown register name 'zmm0'` | `cmake .. -DLLAMA_AVX512=OFF` |
| 链接错误 | `undefined reference to 'ggml_init'` | `git submodule update --init --recursive` |

---

## 动手练习

1. **多后端编译**：在同一台机器上分别编译CPU、CUDA（如有N卡）、OpenBLAS三个版本
2. **性能对比**：使用llama-bench测试不同后端的性能差异
3. **最小示例修改**：修改simple.cpp，添加参数解析功能

---

## 本课小结

| 概念 | 一句话总结 |
|------|-----------|
| CMake | 跨平台构建系统，管理编译过程 |
| LLAMA_CUDA | CMake选项，启用NVIDIA GPU支持 |
| LLAMA_METAL | CMake选项，启用Apple Silicon支持 |
| Release模式 | 优化编译，生产环境使用 |
| simple.cpp | 最小使用示例，学习入口 |

---

*本章对应源码版本：master (2026-04-07)*
