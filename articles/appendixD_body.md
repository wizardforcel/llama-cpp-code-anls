# 附录D：CMake选项参考 —— 构建配置的"百科全书"

## 学习目标

1. 掌握llama.cpp的各种CMake构建选项
2. 理解不同后端（CUDA、Metal、Vulkan等）的配置方法
3. 学会针对特定平台优化构建配置
4. 能够诊断和解决构建问题

---

## 生活类比：汽车定制配置

想象CMake选项就像定制一辆汽车：你是要纯电动（CUDA）、混合动力（CPU+GPU）还是传统燃油（纯CPU）？需要敞篷（调试模式）还是硬顶（发布模式）？要基础款（最小构建）还是顶配（全功能）？每种配置组合都会产生不同的"车型"，适合不同的使用场景。本附录就是你的"配置手册"，帮你选出最适合自己的"座驾"。

---

## D.1 后端选项

### D.1.1 CUDA后端

| 选项 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `GGML_CUDA` | BOOL | OFF | 启用CUDA后端支持 |
| `GGML_CUDA_FORCE_MMQ` | BOOL | OFF | 强制使用MMQ内核代替cuBLAS |
| `GGML_CUDA_FORCE_CUBLAS` | BOOL | OFF | 始终使用cuBLAS代替MMQ内核 |
| `GGML_CUDA_PEER_MAX_BATCH_SIZE` | STRING | "128" | 最大peer拷贝批大小 |
| `GGML_CUDA_NO_PEER_COPY` | BOOL | OFF | 禁用peer-to-peer拷贝 |
| `GGML_CUDA_NO_VMM` | BOOL | OFF | 禁用CUDA虚拟内存管理 |
| `GGML_CUDA_FA` | BOOL | ON | 编译FlashAttention CUDA内核 |
| `GGML_CUDA_FA_ALL_QUANTS` | BOOL | OFF | 为FlashAttention编译所有量化类型 |
| `GGML_CUDA_GRAPHS` | BOOL | ON/OFF* | 启用CUDA图优化 |
| `GGML_CUDA_COMPRESSION_MODE` | STRING | "size" | CUDA压缩模式：none/speed/balance/size |

*默认值取决于平台

**CUDA压缩模式说明：**

```cmake
# 可选值:
# - none: 无压缩
# - speed: 优先速度
# - balance: 平衡
# - size: 优先压缩比（默认）
set(GGML_CUDA_COMPRESSION_MODE "size")
```

**CUDA构建示例：**

```bash
# 基本CUDA构建
cmake -B build -DGGML_CUDA=ON

# 强制使用cuBLAS
cmake -B build -DGGML_CUDA=ON -DGGML_CUDA_FORCE_CUBLAS=ON

# 启用所有FlashAttention量化类型
cmake -B build -DGGML_CUDA=ON -DGGML_CUDA_FA_ALL_QUANTS=ON

# 指定CUDA架构（可选）
cmake -B build -DGGML_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES="70;75;80;86"
```

**为什么 MMQ vs cuBLAS 很重要？**

- **MMQ (Matrix Multiplication Q)**：针对量化矩阵优化的自定义内核
- **cuBLAS**：NVIDIA的官方BLAS库，对FP16优化更好
- 选择原则：量化模型用MMQ，FP16模型用cuBLAS

---

### D.1.2 Metal后端 (Apple Silicon)

| 选项 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `GGML_METAL` | BOOL | ON (macOS) | 启用Metal后端 |
| `GGML_METAL_NDEBUG` | BOOL | OFF | 禁用Metal调试 |
| `GGML_METAL_SHADER_DEBUG` | BOOL | OFF | 使用-fno-fast-math编译着色器 |
| `GGML_METAL_EMBED_LIBRARY` | BOOL | ${GGML_METAL} | 嵌入Metal库到可执行文件 |
| `GGML_METAL_MACOSX_VERSION_MIN` | STRING | "" | 最低macOS版本 |
| `GGML_METAL_STD` | STRING | "" | Metal标准版本(-std标志) |

**Metal构建示例：**

```bash
# macOS默认启用Metal
cmake -B build

# 禁用Metal（纯CPU）
cmake -B build -DGGML_METAL=OFF

# 嵌入Metal库（单文件分发）
cmake -B build -DGGML_METAL_EMBED_LIBRARY=ON

# 指定最低macOS版本
cmake -B build -DGGML_METAL_MACOSX_VERSION_MIN="12.0"
```

**Metal嵌入库的好处：**
- 单文件分发，不依赖外部.metallib文件
- 启动时无需编译着色器
- 适合打包成App分发的场景

---

### D.1.3 Vulkan后端

| 选项 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `GGML_VULKAN` | BOOL | OFF | 启用Vulkan后端 |
| `GGML_VULKAN_CHECK_RESULTS` | BOOL | OFF | 运行Vulkan操作检查 |
| `GGML_VULKAN_DEBUG` | BOOL | OFF | 启用Vulkan调试输出 |
| `GGML_VULKAN_MEMORY_DEBUG` | BOOL | OFF | 启用Vulkan内存调试 |
| `GGML_VULKAN_SHADER_DEBUG_INFO` | BOOL | OFF | 启用Vulkan着色器调试信息 |
| `GGML_VULKAN_VALIDATE` | BOOL | OFF | 启用Vulkan验证层 |
| `GGML_VULKAN_RUN_TESTS` | BOOL | OFF | 运行Vulkan测试 |
| `GGML_VULKAN_SHADERS_GEN_TOOLCHAIN` | FILEPATH | "" | Vulkan着色器生成工具链 |

**Vulkan构建示例：**

```bash
# 基本Vulkan构建
cmake -B build -DGGML_VULKAN=ON

# 调试构建（启用验证层和调试输出）
cmake -B build -DGGML_VULKAN=ON \
    -DGGML_VULKAN_VALIDATE=ON \
    -DGGML_VULKAN_DEBUG=ON

# 使用特定工具链
cmake -B build -DGGML_VULKAN=ON \
    -DGGML_VULKAN_SHADERS_GEN_TOOLCHAIN=/path/to/toolchain.cmake
```

**适用场景：**
- Linux下的AMD/Intel GPU
- Windows下的非NVIDIA GPU
- 需要跨平台GPU支持的项目

---

### D.1.4 SYCL后端 (Intel GPU)

| 选项 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `GGML_SYCL` | BOOL | OFF | 启用SYCL后端 |
| `GGML_SYCL_F16` | BOOL | OFF | 使用16位浮点数计算 |
| `GGML_SYCL_GRAPH` | BOOL | ON | 在SYCL后端启用图优化 |
| `GGML_SYCL_DNN` | BOOL | ON | 在SYCL后端启用oneDNN |
| `GGML_SYCL_TARGET` | STRING | "INTEL" | 目标平台：INTEL/NVIDIA/AMD |
| `GGML_SYCL_DEVICE_ARCH` | STRING | "" | 设备架构（如pvc） |

**SYCL构建示例：**

```bash
# Intel GPU
cmake -B build -DGGML_SYCL=ON \
    -DGGML_SYCL_TARGET="INTEL" \
    -DCMAKE_C_COMPILER=icx \
    -DCMAKE_CXX_COMPILER=icpx

# Intel PVC架构
cmake -B build -DGGML_SYCL=ON \
    -DGGML_SYCL_DEVICE_ARCH="pvc"

# 启用F16计算
cmake -B build -DGGML_SYCL=ON -DGGML_SYCL_F16=ON
```

---

### D.1.5 OpenCL后端

| 选项 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `GGML_OPENCL` | BOOL | OFF | 启用OpenCL后端 |
| `GGML_OPENCL_PROFILING` | BOOL | OFF | 启用OpenCL性能分析 |
| `GGML_OPENCL_EMBED_KERNELS` | BOOL | ON | 嵌入内核到可执行文件 |
| `GGML_OPENCL_USE_ADRENO_KERNELS` | BOOL | ON | 使用Adreno优化内核 |
| `GGML_OPENCL_TARGET_VERSION` | STRING | "300" | OpenCL目标版本 |

**适用场景：**
- Android设备（Qualcomm Adreno GPU）
- 旧版AMD/Intel GPU
- 嵌入式设备

---

### D.1.6 BLAS后端

| 选项 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `GGML_BLAS` | BOOL | ON (macOS) | 启用BLAS加速 |
| `GGML_BLAS_VENDOR` | STRING | "Apple"/"Generic" | BLAS供应商 |

**支持的BLAS供应商：**

- `Generic` - 通用BLAS
- `OpenBLAS` - OpenBLAS
- `Intel10_64lp` - Intel MKL
- `Apple` - Apple Accelerate

**BLAS构建示例：**

```bash
# 使用OpenBLAS
cmake -B build -DGGML_BLAS=ON \
    -DGGML_BLAS_VENDOR="OpenBLAS"

# 使用Intel MKL
cmake -B build -DGGML_BLAS=ON \
    -DGGML_BLAS_VENDOR="Intel10_64lp"
```

**为什么需要BLAS？**
- 对FP16/FP32矩阵乘法提供CPU优化
- 在纯CPU推理时显著提升性能
- Apple Silicon上通过Accelerate框架获得最佳性能

---

### D.1.7 RPC后端

| 选项 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `GGML_RPC` | BOOL | OFF | 启用RPC（远程过程调用）后端 |

**RPC构建示例：**

```bash
# 启用RPC支持（用于分布式推理）
cmake -B build -DGGML_RPC=ON
```

**应用场景：**
- 多机分布式推理
- 将部分层卸载到远程服务器
- 资源共享集群

---

## D.2 功能选项

### D.2.1 核心功能

| 选项 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `GGML_NATIVE` | BOOL | OFF | 为本地CPU架构优化 |
| `GGML_LLAMAFILE` | BOOL | ON | 启用llamafile优化 |
| `GGML_CPU_ARM_ARCH` | STRING | "" | ARM架构特定优化 |
| `GGML_CPU_POWERPC` | BOOL | OFF | 启用PowerPC优化 |
| `GGML_CPU_RISCV` | BOOL | OFF | 启用RISC-V优化 |

**CPU优化构建示例：**

```bash
# 本地CPU优化（自动检测指令集）
cmake -B build -DGGML_NATIVE=ON

# 特定ARM架构
cmake -B build -DGGML_CPU_ARM_ARCH="armv8.2-a+fp16"
```

**GGML_NATIVE的作用：**
- 启用本地CPU支持的所有SIMD指令集
- 自动生成针对特定CPU优化的代码
- 生成的二进制可能无法在其他CPU上运行

---

### D.2.2 llama.cpp特定选项

| 选项 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `LLAMA_BUILD_COMMON` | BOOL | ON | 构建common工具库 |
| `LLAMA_BUILD_TESTS` | BOOL | ON | 构建测试 |
| `LLAMA_BUILD_TOOLS` | BOOL | ON | 构建工具 |
| `LLAMA_BUILD_EXAMPLES` | BOOL | ON | 构建示例程序 |
| `LLAMA_BUILD_SERVER` | BOOL | ON | 构建server示例 |
| `LLAMA_BUILD_WEBUI` | BOOL | ON | 为server构建嵌入式Web UI |
| `LLAMA_TOOLS_INSTALL` | BOOL | ON/OFF* | 安装工具 |
| `LLAMA_TESTS_INSTALL` | BOOL | ON | 安装测试 |
| `LLAMA_OPENSSL` | BOOL | ON | 使用OpenSSL支持HTTPS |
| `LLAMA_LLGUIDANCE` | BOOL | OFF | 包含LLGuidance库 |
| `LLAMA_USE_SYSTEM_GGML` | BOOL | OFF | 使用系统libggml |

*macOS iOS默认OFF，其他ON

**功能选择构建示例：**

```bash
# 仅构建核心库（最小构建）
cmake -B build \
    -DLLAMA_BUILD_COMMON=OFF \
    -DLLAMA_BUILD_TESTS=OFF \
    -DLLAMA_BUILD_EXAMPLES=OFF \
    -DLLAMA_BUILD_TOOLS=OFF

# 启用LLGuidance（结构化输出增强）
cmake -B build -DLLAMA_LLGUIDANCE=ON

# 禁用WebUI（纯API server）
cmake -B build -DLLAMA_BUILD_WEBUI=OFF
```

---

### D.2.3 量化支持

| 选项 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `GGML_QKK_64` | BOOL | OFF | 使用64位K-quant（实验性） |

---

## D.3 调试选项

### D.3.1 编译器警告

| 选项 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `LLAMA_ALL_WARNINGS` | BOOL | ON | 启用所有编译器警告 |
| `LLAMA_ALL_WARNINGS_3RD_PARTY` | BOOL | OFF | 对第三方库启用所有警告 |
| `LLAMA_FATAL_WARNINGS` | BOOL | OFF | 将警告视为错误(-Werror) |
| `GGML_ALL_WARNINGS` | BOOL | ${LLAMA_ALL_WARNINGS} | GGML所有警告 |
| `GGML_FATAL_WARNINGS` | BOOL | ${LLAMA_FATAL_WARNINGS} | GGML警告视为错误 |

### D.3.2 Sanitizer

| 选项 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `LLAMA_SANITIZE_THREAD` | BOOL | OFF | 启用线程sanitizer |
| `LLAMA_SANITIZE_ADDRESS` | BOOL | OFF | 启用地址sanitizer |
| `LLAMA_SANITIZE_UNDEFINED` | BOOL | OFF | 启用未定义行为sanitizer |

**Sanitizer构建示例：**

```bash
# 地址sanitizer（检测内存错误）
cmake -B build -DLLAMA_SANITIZE_ADDRESS=ON

# 线程sanitizer（检测数据竞争）
cmake -B build -DLLAMA_SANITIZE_THREAD=ON

# 未定义行为sanitizer
cmake -B build -DLLAMA_SANITIZE_UNDEFINED=ON
```

**Sanitizer使用场景：**

| Sanitizer | 检测问题 | 性能影响 | 使用时机 |
|-----------|----------|----------|----------|
| Address | 内存越界、use-after-free | 2-3x | 调试内存问题 |
| Thread | 数据竞争、死锁 | 5-20x | 调试并发问题 |
| Undefined | 整数溢出、未对齐访问 | 1.5-2x | 代码审查 |

### D.3.3 调试信息

| 选项 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `CMAKE_BUILD_TYPE` | STRING | Release | 构建类型 |

**构建类型：**

- `Debug` - 调试信息，无优化
- `Release` - 优化，无调试信息
- `RelWithDebInfo` - 优化+调试信息
- `MinSizeRel` - 最小体积优化

---

## D.4 平台特定选项

### D.4.1 Windows

| 选项 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `BUILD_SHARED_LIBS` | BOOL | ON | 构建共享库 |

**Windows构建示例：**

```bash
# 静态库构建
cmake -B build -DBUILD_SHARED_LIBS=OFF

# 使用特定Visual Studio版本
cmake -B build -G "Visual Studio 17 2022" -A x64
```

### D.4.2 macOS/iOS

| 选项 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `CMAKE_OSX_DEPLOYMENT_TARGET` | STRING | - | 最低macOS版本 |
| `CMAKE_OSX_ARCHITECTURES` | STRING | - | 目标架构 |

**macOS交叉编译示例：**

```bash
# 通用二进制（Intel + Apple Silicon）
cmake -B build \
    -DCMAKE_OSX_ARCHITECTURES="x86_64;arm64"

# 指定最低版本
cmake -B build \
    -DCMAKE_OSX_DEPLOYMENT_TARGET="12.0"
```

### D.4.3 Android

| 选项 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `CMAKE_ANDROID_ARCH_ABI` | STRING | - | Android ABI |
| `CMAKE_ANDROID_NDK` | PATH | - | NDK路径 |
| `CMAKE_ANDROID_NATIVE_API_LEVEL` | STRING | - | API级别 |

**Android构建示例：**

```bash
cmake -B build \
    -DCMAKE_TOOLCHAIN_FILE=$NDK/build/cmake/android.toolchain.cmake \
    -DANDROID_ABI=arm64-v8a \
    -DANDROID_PLATFORM=android-28 \
    -DGGML_OPENCL=ON
```

### D.4.4 WebAssembly

| 选项 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `LLAMA_WASM_MEM64` | BOOL | ON | 使用64位内存 |
| `LLAMA_WASM_SINGLE_FILE` | BOOL | OFF | 将WASM嵌入llama.js |
| `LLAMA_BUILD_HTML` | BOOL | ON | 构建HTML文件 |

**Emscripten构建示例：**

```bash
# 激活Emscripten环境
source /path/to/emsdk/emsdk_env.sh

# 构建
cmake -B build \
    -DCMAKE_TOOLCHAIN_FILE=$EMSDK/upstream/emscripten/cmake/Modules/Platform/Emscripten.cmake \
    -DLLAMA_WASM_MEM64=ON
```

---

## D.5 完整构建配置示例

### D.5.1 开发构建（调试）

```bash
cmake -B build \
    -DCMAKE_BUILD_TYPE=Debug \
    -DLLAMA_ALL_WARNINGS=ON \
    -DLLAMA_FATAL_WARNINGS=ON \
    -DLLAMA_SANITIZE_ADDRESS=ON \
    -DGGML_NATIVE=ON
```

**适用场景：**
- 开发新功能
- 调试问题
- 代码审查

### D.5.2 高性能生产构建（CUDA）

```bash
cmake -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_CUDA=ON \
    -DGGML_CUDA_GRAPHS=ON \
    -DGGML_CUDA_FA=ON \
    -DGGML_CUDA_FA_ALL_QUANTS=ON \
    -DGGML_NATIVE=ON \
    -DLLAMA_BUILD_TESTS=OFF
```

**特点：**
- 启用所有CUDA优化
- 支持所有量化类型的FlashAttention
- 禁用测试减少构建时间

### D.5.3 最小体积构建

```bash
cmake -B build \
    -DCMAKE_BUILD_TYPE=MinSizeRel \
    -DBUILD_SHARED_LIBS=OFF \
    -DLLAMA_BUILD_COMMON=OFF \
    -DLLAMA_BUILD_TESTS=OFF \
    -DLLAMA_BUILD_EXAMPLES=OFF \
    -DLLAMA_BUILD_TOOLS=OFF \
    -DGGML_NATIVE=ON
```

**适用场景：**
- 嵌入式设备
- 容器镜像
- 移动端应用

### D.5.4 全功能构建

```bash
cmake -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_CUDA=ON \
    -DGGML_VULKAN=ON \
    -DGGML_BLAS=ON \
    -DGGML_RPC=ON \
    -DLLAMA_BUILD_SERVER=ON \
    -DLLAMA_BUILD_WEBUI=ON \
    -DLLAMA_OPENSSL=ON \
    -DLLAMA_LLGUIDANCE=ON
```

**适用场景：**
- 多后端测试
- 全功能服务器部署
- 开发环境

### D.5.5 边缘设备构建（ARM）

```bash
cmake -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_SYSTEM_NAME=Linux \
    -DCMAKE_SYSTEM_PROCESSOR=aarch64 \
    -DGGML_OPENCL=ON \
    -DGGML_NATIVE=OFF \
    -DGGML_CPU_ARM_ARCH="armv8.2-a+fp16"
```

**适用场景：**
- Raspberry Pi
- NVIDIA Jetson
- 其他ARM64设备

---

## D.6 选项依赖关系

```
┌─────────────────────────────────────────────────────────────────┐
│                    CMake选项依赖关系图                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  LLAMA_BUILD_SERVER                                             │
│       │                                                         │
│       ├── 依赖: LLAMA_BUILD_COMMON                              │
│       └── 可选: LLAMA_BUILD_WEBUI                               │
│                                                                 │
│  LLAMA_BUILD_WEBUI                                              │
│       └── 依赖: LLAMA_BUILD_SERVER                              │
│                                                                 │
│  GGML_CUDA                                                      │
│       ├── 启用: 自动检测CUDA架构                                │
│       └── 冲突: GGML_CUDA_FORCE_MMQ 与 GGML_CUDA_FORCE_CUBLAS   │
│                                                                 │
│  GGML_METAL                                                     │
│       ├── 默认: macOS上ON                                       │
│       └── 建议: LLAMA_BUILD_SERVER需要它用于GPU加速             │
│                                                                 │
│  GGML_BLAS                                                      │
│       └── 需要: 指定GGML_BLAS_VENDOR                            │
│                                                                 │
│  LLAMA_LLGUIDANCE                                               │
│       └── 依赖: LLAMA_BUILD_COMMON                              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## D.7 配置验证

### D.7.1 查看配置摘要

```bash
# 配置时输出
cmake -B build -DGGML_CUDA=ON 2>&1 | grep -E "(GGML|LLAMA)"

# 查看缓存变量
cmake -B build -LA | grep -E "(GGML|LLAMA)"
```

### D.7.2 验证构建

```bash
# 构建后检查功能
cd build/bin
./llama-cli --help | head -20

# 检查CUDA是否启用
./llama-cli -ngl 35 2>&1 | grep -i cuda

# 检查Metal是否启用 (macOS)
./llama-cli -ngl 99 2>&1 | grep -i metal
```

---

## D.8 常见问题

**Q: 如何同时启用多个后端？**

A: 可以同时启用多个后端，llama.cpp会自动选择最佳后端：

```bash
cmake -B build \
    -DGGML_CUDA=ON \
    -DGGML_BLAS=ON \
    -DGGML_VULKAN=ON
```

**Q: 为什么某些选项在CMake GUI中不显示？**

A: 某些选项依赖于其他选项。例如，GGML_CUDA选项只有在检测到CUDA工具链时才会显示。

**Q: 如何传递选项给GGML子模块？**

A: 所有`GGML_`前缀的选项会自动传递给GGML子模块。`LLAMA_`选项用于llama.cpp特定配置。

**Q: 静态链接所有依赖？**

A:

```bash
cmake -B build \
    -DBUILD_SHARED_LIBS=OFF \
    -DCMAKE_FIND_LIBRARY_SUFFIXES=".a" \
    -DCMAKE_EXE_LINKER_FLAGS="-static"
```

**Q: 交叉编译配置？**

A:

```bash
# 创建工具链文件 toolchain.cmake
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)
set(CMAKE_C_COMPILER aarch64-linux-gnu-gcc)
set(CMAKE_CXX_COMPILER aarch64-linux-gnu-g++)

# 使用工具链
cmake -B build -DCMAKE_TOOLCHAIN_FILE=toolchain.cmake
```

---

## 本课小结

本附录整理了CMake构建系统的关键选项。CUDA后端选项包括 `GGML_CUDA` 和 `GGML_CUDA_FA`，用于启用GPU加速。Metal后端选项 `GGML_METAL` 用于支持Apple Silicon芯片。CPU优化选项 `GGML_NATIVE` 启用本地指令集优化。构建控制选项 `LLAMA_BUILD_*` 用于选择需要构建的功能模块。调试选项包括 `CMAKE_BUILD_TYPE` 和 `*_SANITIZE_*`，用于开发与调试场景。

**常用构建命令速查：**

```bash
# 开发构建
cmake -B build -DCMAKE_BUILD_TYPE=Debug -DLLAMA_SANITIZE_ADDRESS=ON

# CUDA生产构建
cmake -B build -DGGML_CUDA=ON -DGGML_NATIVE=ON

# 最小构建
cmake -B build -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF

# macOS通用构建
cmake -B build -DCMAKE_OSX_ARCHITECTURES="x86_64;arm64"
```

---

*本附录对应源码版本：master (2026-04-07)*

