# 第2章 环境搭建与构建系统 —— 打造你的"AI 开发工作站"

在深入探索 llama.cpp 的内部世界之前，我们需要先建立一个稳固的开发环境。就像一位大厨需要一间设备齐全的厨房，一位 AI 开发者也需要一个精心配置的"开发工作站"。本章将带领你从零开始搭建环境，掌握跨平台编译技术，并运行你的第一个 LLM 程序。

## 学习目标

1. 掌握各平台（Windows/Linux/macOS）的开发环境配置
2. 深入理解 CMake 构建系统的工作原理
3. 实战多平台编译（CPU/GPU/移动端）
4. 成功运行第一个 llama.cpp 程序并理解其结构

## 生活类比：打造一家 AI 咖啡馆

想象你要开设一家独具特色的**AI 主题咖啡馆**：

- **操作系统** = 选址（Windows 大街 / Linux 广场 / macOS 商圈）
- **编译器** = 厨房核心设备（Visual Studio 专业灶 / GCC 传统灶 / Clang 精品灶）
- **CMake** = 厨房设计蓝图（告诉工人设备怎么组装、管线如何连接）
- **后端选择** = 能源方案：
  - **CPU** = 电力（哪都能用，基础款）
  - **CUDA** = 天然气（NVIDIA 专区，火力猛）
  - **Metal** = 太阳能（Apple 生态，清洁高效）
- **第一个示例** = 试营业（验证一切设备正常运作）

就像开店需要精心准备设备和原料，开发 llama.cpp 也需要正确配置环境和构建工具。这个类比将贯穿本章，帮助你理解每个组件的作用。

---

## 2.1 开发环境准备

### 2.1.1 理解"选址"：操作系统选择

llama.cpp 是一个跨平台项目，支持 Windows、Linux 和 macOS。不同平台有各自的特点：

| 平台 | 优势 | 适用场景 |
|------|------|---------|
| **Windows** | 图形化工具丰富，IDE 强大 | 习惯 GUI 的开发者 |
| **Linux** | 原生开发体验，服务器部署 | 服务器/云环境 |
| **macOS** | Unix 基因 + 商业软件生态 | Apple Silicon 用户 |

### 2.1.2 Windows 环境配置

**必要组件清单**

| 组件 | 推荐版本 | 用途 |
|------|---------|------|
| Visual Studio 2022 | 17.8+ | C++ 编译器、调试器 |
| CMake | 3.24+ | 构建系统生成器 |
| Git | 最新版 | 源码版本管理 |
| Python | 3.9+ | 模型转换脚本 |

**安装步骤**

1. **Visual Studio 2022**（厨房核心设备）

   从官网下载安装器，选择以下工作负载：
   - "使用 C++ 的桌面开发"（必选）
   - "Windows 11 SDK"（建议）
   - "CMake 工具"（可选）

2. **验证安装**

   打开"x64 Native Tools Command Prompt for VS 2022"（这是特殊的命令行环境，包含编译器路径）：

   ```cmd
   cmake --version
   :: 应显示：cmake version 3.28.0 或更高
   
   cl
   :: 应显示：Microsoft (R) C/C++ Optimizing Compiler
   ```

**为什么必须用 VS 命令行？**

普通 cmd/PowerShell 缺少 `cl.exe` 等工具的路径。VS 命令行通过批处理脚本设置了完整的环境变量。

### 2.1.3 Linux 环境配置

在 Ubuntu/Debian 系统上，使用包管理器快速安装：

```bash
# 更新包索引
sudo apt update

# 安装基础工具链
sudo apt install -y build-essential cmake git wget

# 安装 Python 及科学计算库
sudo apt install -y python3 python3-pip python3-venv

# 可选：CUDA 支持（如有 NVIDIA GPU）
sudo apt install -y nvidia-cuda-toolkit

# 可选：CPU 加速库
sudo apt install -y libopenblas-dev
```

### 2.1.4 macOS 环境配置

```bash
# 安装命令行工具
xcode-select --install

# 安装 Homebrew（如果还没有）
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# 安装依赖
brew install cmake git python@3.11
```

---

## 2.2 CMake 构建系统详解

CMake 是 llama.cpp 的"厨房设计蓝图"——它本身不编译代码，而是生成特定平台的构建指令。

### 2.2.1 CMake 简介

**核心概念**：

```
CMakeLists.txt  --(CMake)-->  Makefile/VS项目/Xcode项目  --(构建工具)-->  可执行文件
     蓝图          生成          平台特定的构建文件         编译        最终产物
```

**优势**：
1. **一次编写，到处构建**：同一份配置支持多平台
2. **条件编译**：根据硬件能力选择功能
3. **模块化**：清晰的子项目结构

### 2.2.2 llama.cpp 的 CMake 架构

**源码位置**：`CMakeLists.txt`（第 1-200 行）

```cmake
cmake_minimum_required(VERSION 3.14)
project(llama.cpp VERSION 1.0.0 LANGUAGES CXX C)

# ===== 选项定义（厨房设备清单）=====
option(LLAMA_CUDA "Build with CUDA support" OFF)
option(LLAMA_METAL "Build with Metal support" OFF)
option(LLAMA_VULKAN "Build with Vulkan support" OFF)
option(LLAMA_SYCL "Build with SYCL support" OFF)
option(LLAMA_BUILD_EXAMPLES "Build examples" ON)

# ===== 添加子目录（分区施工）=====
add_subdirectory(ggml)      # GGML 张量库
add_subdirectory(src)       # llama.cpp 核心
add_subdirectory(common)    # 通用工具

if (LLAMA_BUILD_EXAMPLES)
    add_subdirectory(examples)  # 示例程序
endif()
```

**分层设计的好处**：
- `ggml/` 可独立使用（纯张量计算）
- `src/` 专注 LLaMA 模型逻辑
- `examples/` 提供使用范例

### 2.2.3 后端选项详解

**GPU 加速后端**：

| 选项 | 适用平台 | 硬件要求 | 特点 |
|------|---------|---------|------|
| `LLAMA_CUDA` | Windows/Linux | NVIDIA GPU (CC 5.2+) | 性能最高，成熟稳定 |
| `LLAMA_METAL` | macOS | Apple Silicon/Intel | 统一内存，能效比高 |
| `LLAMA_VULKAN` | 跨平台 | 支持 Vulkan 的 GPU | 跨厂商，通用性好 |
| `LLAMA_SYCL` | 跨平台 | Intel GPU/FPGA | Intel 生态专用 |

**CPU 优化选项**：

| 选项 | 默认值 | 说明 |
|------|-------|------|
| `LLAMA_AVX` | ON | AVX 指令集（Sandy Bridge+） |
| `LLAMA_AVX2` | ON | AVX2 指令集（Haswell+） |
| `LLAMA_AVX512` | OFF | AVX-512（高端 Xeon） |
| `LLAMA_FMA` | ON | FMA 指令集 |
| `LLAMA_NEON` | 自动 | ARM NEON（Apple Silicon/ARM64） |

---

## 2.3 多平台编译实战

### 2.3.1 CPU 版本编译（基础款）

**步骤**：

```bash
# 1. 克隆仓库
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp

# 2. 创建构建目录（out-of-source 构建）
mkdir build && cd build

# 3. 配置（Release 模式）
cmake .. -DCMAKE_BUILD_TYPE=Release

# 4. 编译（使用所有核心）
cmake --build . --config Release -j$(nproc)
```

**Windows 差异**：

```cmd
mkdir build && cd build
cmake .. -G "Visual Studio 17 2022" -A x64
cmake --build . --config Release -j
```

### 2.3.2 CUDA 版本编译（高性能款）

**环境要求**：
- NVIDIA GPU（计算能力 5.2+）
- CUDA Toolkit 11.4+

**编译步骤**：

```bash
mkdir build-cuda && cd build-cuda

# 配置 CUDA 支持
cmake .. -DLLAMA_CUDA=ON \
         -DCMAKE_CUDA_ARCHITECTURES="70;75;80;86"

cmake --build . --config Release -j
```

**计算能力说明**：
- 70 = Turing (V100, T4)
- 75 = Turing (RTX 20 系列)
- 80 = Ampere (A100, RTX 30 系列)
- 86 = Ampere (RTX 30 系列)

### 2.3.3 Metal 版本编译（Apple 专用）

```bash
mkdir build-metal && cd build-metal

cmake .. -DLLAMA_METAL=ON

cmake --build . --config Release -j
```

**Metal 的优势**：
1. **统一内存**：CPU/GPU 共享内存，无需拷贝
2. **能效比**：笔记本上长续航高性能
3. **零配置**：无需安装额外驱动

---

## 2.4 第一个可运行示例

### 2.4.1 获取测试模型

**推荐模型**：TinyLlama-1.1B（轻量级，适合测试）

```bash
# 安装 huggingface-cli
pip install huggingface-hub

# 下载模型
huggingface-cli download \
    TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF \
    tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf \
    --local-dir ./models
```

### 2.4.2 运行 llama-cli

```bash
./bin/llama-cli \
    -m ./models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf \
    -p "What is machine learning?" \
    -n 128
```

**参数说明**：
- `-m`：模型文件路径
- `-p`：提示词（prompt）
- `-n`：生成的最大 token 数

### 2.4.3 简单示例源码解析

**源码位置**：`examples/simple/simple.cpp`

```cpp
#include "llama.h"
#include <iostream>

int main(int argc, char** argv) {
    // ① 初始化模型参数
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = 0;  // CPU 模式，设为 -1 启用 GPU
    
    // ② 加载模型
    llama_model* model = llama_load_model_from_file(argv[1], model_params);
    if (!model) {
        std::cerr << "Failed to load model\n";
        return 1;
    }
    
    // ③ 初始化上下文参数
    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = 2048;        // 上下文窗口大小
    ctx_params.n_threads = 4;       // CPU 线程数
    
    // ④ 创建上下文
    llama_context* ctx = llama_new_context_with_model(model, ctx_params);
    
    // ⑤ 分词和推理循环...
    // 详见后续章节
    
    // ⑥ 清理资源
    llama_free(ctx);
    llama_free_model(model);
    
    return 0;
}
```

---

## 2.5 设计中的取舍

### 为什么 llama.cpp 选择 CMake 而非其他构建系统？

| 方案 | 优点 | 缺点 | llama.cpp 选择 |
|------|------|------|---------------|
| **CMake** | 跨平台，IDE 支持好 | 语法复杂 | ✅ 选用 |
| **Makefile** | 简单直接 | 仅 Unix，Windows 困难 | ❌ |
| **Bazel** | 企业级，缓存好 | 学习曲线陡峭，依赖重 | ❌ |
| **Meson** | 现代，简洁 | 社区较小 | ❌ |

**选择 CMake 的原因**：
1. **最大兼容性**：支持所有目标平台
2. **IDE 集成**：VS/CLion/Xcode 原生支持
3. **社区成熟**：文档丰富，问题易解决

### 为什么默认启用 AVX2 而非 AVX-512？

```cmake
# CMakeLists.txt 中的默认设置
option(LLAMA_AVX2 "llama: use AVX2" ON)
option(LLAMA_AVX512 "llama: use AVX-512" OFF)
```

**原因**：
1. **普及度**：AVX2 从 Haswell（2013）开始普及，AVX-512 仅高端 CPU 支持
2. **频率降频**：某些 CPU 使用 AVX-512 会降频，反而降低整体性能
3. **编译器问题**：某些版本的 GCC/Clang 对 AVX-512 支持不完善

---

## 2.6 动手练习

### 练习 1：多后端编译对比

在同一台机器上编译三个版本，记录时间和大小：

```bash
# CPU 版本
time cmake --build build-cpu --config Release

# CUDA 版本（如有 N 卡）
time cmake --build build-cuda --config Release

# 对比 bin/ 目录大小
ls -lh build-*/bin/
```

### 练习 2：性能基准测试

使用 llama-bench 测试不同后端的性能：

```bash
./bin/llama-bench -m model.gguf
```

记录不同配置的 `tokens/second` 指标。

### 练习 3：修改 simple.cpp

为 simple.cpp 添加命令行参数解析：

```cpp
// 添加 -p 参数支持提示词
// 添加 -n 参数支持生成长度
// 参考 examples/main/main.cpp 的实现
```

---

## 2.7 常见编译问题排查

| 问题 | 错误信息 | 解决方案 |
|------|---------|---------|
| 找不到 CUDA | `Could not find CUDA` | 设置 `CUDA_HOME` 环境变量 |
| AVX512 错误 | `unknown register name 'zmm0'` | `cmake .. -DLLAMA_AVX512=OFF` |
| 链接错误 | `undefined reference to 'ggml_init'` | `git submodule update --init --recursive` |
| 内存不足 | `out of memory` | 减小上下文窗口，使用量化模型 |

---

## 2.8 本章小结

| 概念 | 一句话总结 |
|------|-----------|
| **CMake** | 跨平台构建系统生成器，生成平台特定的构建文件 |
| **LLAMA_CUDA** | CMake 选项，启用 NVIDIA GPU 加速 |
| **LLAMA_METAL** | CMake 选项，启用 Apple Silicon GPU 加速 |
| **Release 模式** | 启用编译器优化（-O3），适合生产部署 |
| **计算能力** | NVIDIA GPU 架构版本，如 80 对应 Ampere |
| **simple.cpp** | 最小使用示例，展示了 llama.cpp API 的基本用法 |

**下一步预告**：

环境搭建完成后，我们将在第 3 章深入 GGML 核心架构——理解张量（`ggml_tensor`）、计算图（`ggml_cgraph`）和内存池的工作原理。
