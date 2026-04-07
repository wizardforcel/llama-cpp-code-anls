# 第29章 集成与部署案例 —— 从开发到生产的"最后一公里"

## 1. 学习目标

- 掌握Android端llama.cpp的集成方法
- 学习服务端部署的Docker和K8s配置
- 了解多语言绑定的使用方法
- 理解生产环境的监控和日志方案
- 掌握模型热更新和A/B测试策略

## 2. 生活类比：餐厅开业

想象你是一位餐厅老板，从家庭厨房（开发环境）到正式开业（生产部署）需要经历：装修店面（环境配置）、招聘培训（团队准备）、制定菜单（API设计）、建立供应链（模型管理）、顾客服务（监控运维）。llama.cpp的部署也是如此，需要考虑平台适配、服务编排、接口设计、模型更新等多个环节。

## 3. 源码地图

| 文件路径 | 职责 | 核心内容 |
|---------|------|---------|
| `examples/llama.android/` | Android示例 | JNI绑定、Kotlin接口、UI实现 |
| `examples/server/server.cpp` | HTTP服务 | OpenAI兼容API、WebSocket |
| `examples/server/public/` | Web界面 | 聊天UI、管理界面 |
| `docs/docker.md` | Docker文档 | 容器化部署指南 |
| `examples/llama.swiftui/` | iOS示例 | SwiftUI集成 |

## 4. 详细章节内容

### 4.1 嵌入式系统集成

#### 4.1.1 Android集成架构

```
Android应用架构

┌─────────────────────────────────────┐
│         Kotlin UI层 (MainActivity)   │
│  - 聊天界面                          │
│  - 模型选择                          │
│  - 设置面板                          │
├─────────────────────────────────────┤
│         Kotlin业务层 (AiChat)        │
│  - 会话管理                          │
│  - 消息格式化                        │
│  - 状态回调                          │
├─────────────────────────────────────┤
│         JNI接口层 (InferenceEngine)  │
│  - Java ↔ C++ 桥接                   │
│  - 线程管理                          │
│  - 异常处理                          │
├─────────────────────────────────────┤
│         C++核心层 (ai_chat.cpp)      │
│  - llama.cpp API调用                 │
│  - 采样器管理                        │
│  - 上下文管理                        │
├─────────────────────────────────────┤
│         原生库层                     │
│  - libllama.so                       │
│  - libggml-cpu.so                    │
│  - libggml-opencl.so (可选)          │
└─────────────────────────────────────┘
```

#### 4.1.2 JNI绑定实现

```cpp
// examples/llama.android/lib/src/main/cpp/ai_chat.cpp

// 初始化后端
extern "C" JNIEXPORT void JNICALL
Java_com_arm_aichat_internal_InferenceEngineImpl_init(
    JNIEnv *env, 
    jobject /*unused*/, 
    jstring nativeLibDir
) {
    // 设置Android日志回调
    llama_log_set(aichat_android_log_callback, nullptr);
    
    // 从指定路径加载后端库
    const auto *path = env->GetStringUTFChars(nativeLibDir, 0);
    ggml_backend_load_all_from_path(path);
    env->ReleaseStringUTFChars(nativeLibDir, path);
    
    llama_backend_init();
}

// 加载模型
extern "C" JNIEXPORT jint JNICALL
Java_com_arm_aichat_internal_InferenceEngineImpl_load(
    JNIEnv *env, 
    jobject, 
    jstring jmodel_path
) {
    const auto *model_path = env->GetStringUTFChars(jmodel_path, 0);
    
    llama_model_params model_params = llama_model_default_params();
    auto *model = llama_model_load_from_file(model_path, model_params);
    
    env->ReleaseStringUTFChars(jmodel_path, model_path);
    
    if (!model) return 1;
    g_model = model;
    return 0;
}

// 生成下一个token
extern "C" JNIEXPORT jstring JNICALL
Java_com_arm_aichat_internal_InferenceEngineImpl_generateNextToken(
    JNIEnv *env,
    jobject /*unused*/
) {
    // 采样
    const auto new_token_id = common_sampler_sample(g_sampler, g_context, -1);
    common_sampler_accept(g_sampler, new_token_id, true);
    
    // 解码
    common_batch_clear(g_batch);
    common_batch_add(g_batch, new_token_id, current_position, {0}, true);
    llama_decode(g_context, g_batch);
    
    // 转换为文本
    auto token_str = common_token_to_piece(g_context, new_token_id);
    return env->NewStringUTF(token_str.c_str());
}
```

#### 4.1.3 Kotlin业务层

```kotlin
// examples/llama.android/lib/src/main/java/com/arm/aichat/AiChat.kt

class AiChat private constructor() {
    private val inferenceEngine = InferenceEngine()
    
    fun initialize(context: Context) {
        val nativeLibDir = context.applicationInfo.nativeLibraryDir
        inferenceEngine.init(nativeLibDir)
    }
    
    fun loadModel(modelPath: String): Boolean {
        return inferenceEngine.load(modelPath) == 0
    }
    
    fun generateResponse(
        prompt: String,
        onToken: (String) -> Unit,
        onComplete: () -> Unit
    ) {
        inferenceEngine.processUserPrompt(prompt, 1024)
        
        thread {
            while (true) {
                val token = inferenceEngine.generateNextToken()
                if (token == null) break
                onToken(token)
            }
            onComplete()
        }
    }
}
```

#### 4.1.4 iOS集成要点

```swift
// examples/llama.swiftui/llama.cpp.swift/LlamaContext.swift

import Foundation

class LlamaContext {
    private var model: OpaquePointer?
    private var context: OpaquePointer?
    
    func loadModel(from path: String) throws {
        var params = llama_model_default_params()
        model = llama_load_model_from_file(path, params)
        
        guard model != nil else {
            throw LlamaError.modelLoadFailed
        }
        
        var ctx_params = llama_context_default_params()
        ctx_params.n_ctx = 4096
        context = llama_new_context_with_model(model, ctx_params)
    }
    
    func generate(prompt: String, callback: (String) -> Bool) {
        // Tokenize
        let tokens = tokenize(text: prompt)
        
        // Decode prompt
        decode(tokens: tokens)
        
        // Generate
        while true {
            let token = sample()
            let piece = tokenToPiece(token)
            
            if !callback(piece) { break }
            if isEog(token) { break }
            
            decode(tokens: [token])
        }
    }
}
```

### 4.2 服务端部署

#### 4.2.1 Docker容器化

```dockerfile
# Dockerfile
FROM ubuntu:22.04

# 安装依赖
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    python3 \
    && rm -rf /var/lib/apt/lists/*

# 复制源码
WORKDIR /app
COPY . .

# 编译
RUN cmake -B build -DLLAMA_CUDA=OFF -DLLAMA_OPENMP=ON \
    && make -C build -j$(nproc)

# 暴露端口
EXPOSE 8080

# 启动服务
ENTRYPOINT ["./build/bin/llama-server"]
CMD ["--host", "0.0.0.0", "--port", "8080"]
```

```yaml
# docker-compose.yml
version: '3.8'

services:
  llama-server:
    build: .
    ports:
      - "8080:8080"
    volumes:
      - ./models:/models:ro
    environment:
      - LLAMA_MODEL=/models/llama-7b.gguf
      - LLAMA_N_GPU_LAYERS=999
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

#### 4.2.2 Kubernetes部署

```yaml
# k8s-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: llama-server
spec:
  replicas: 3
  selector:
    matchLabels:
      app: llama-server
  template:
    metadata:
      labels:
        app: llama-server
    spec:
      containers:
      - name: llama
        image: llama-server:latest
        ports:
        - containerPort: 8080
        resources:
          limits:
            nvidia.com/gpu: 1
            memory: "16Gi"
          requests:
            memory: "8Gi"
        volumeMounts:
        - name: models
          mountPath: /models
        env:
        - name: LLAMA_MODEL
          value: "/models/llama-7b.gguf"
      volumes:
      - name: models
        persistentVolumeClaim:
          claimName: model-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: llama-service
spec:
  selector:
    app: llama-server
  ports:
  - port: 80
    targetPort: 8080
  type: LoadBalancer
```

#### 4.2.3 Server API设计

```cpp
// examples/server/server.cpp 核心API

// OpenAI兼容的聊天完成接口
// POST /v1/chat/completions
{
    "model": "gpt-3.5-turbo",
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"}
    ],
    "stream": true,
    "temperature": 0.7,
    "max_tokens": 256
}

// 响应
{
    "id": "chatcmpl-xxx",
    "object": "chat.completion",
    "created": 1234567890,
    "choices": [{
        "index": 0,
        "message": {
            "role": "assistant",
            "content": "Hello! How can I help you today?"
        },
        "finish_reason": "stop"
    }]
}

// 健康检查
// GET /health
{
    "status": "ok",
    "slots_idle": 2,
    "slots_processing": 1
}
```

### 4.3 语言绑定

#### 4.3.1 Python绑定

```python
# llama-cpp-python 使用示例
from llama_cpp import Llama

# 加载模型
llm = Llama(
    model_path="./models/llama-7b.gguf",
    n_ctx=4096,
    n_gpu_layers=999
)

# 生成文本
output = llm(
    "Q: What is the capital of France?\nA:",
    max_tokens=32,
    temperature=0.7,
    stop=["\n", "Q:"]
)
print(output['choices'][0]['text'])

# 聊天模式
response = llm.create_chat_completion(
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"}
    ]
)
```

#### 4.3.2 Node.js绑定

```javascript
// node-llama-cpp 使用示例
const { LlamaModel, LlamaContext } = require('node-llama-cpp');

async function main() {
    const model = new LlamaModel({
        modelPath: './models/llama-7b.gguf',
        gpuLayers: 999
    });
    
    const context = new LlamaContext({ model });
    
    const response = await context.evaluate(
        "What is the capital of France?",
        {
            temperature: 0.7,
            maxTokens: 256
        }
    );
    
    console.log(response);
}

main();
```

### 4.4 生产环境运维

#### 4.4.1 监控指标

```python
# 监控指标收集示例
import prometheus_client
from prometheus_client import Counter, Histogram, Gauge

# 定义指标
request_count = Counter('llama_requests_total', 'Total requests')
request_duration = Histogram('llama_request_duration_seconds', 'Request duration')
active_sessions = Gauge('llama_active_sessions', 'Active sessions')
kv_cache_usage = Gauge('llama_kv_cache_usage_ratio', 'KV cache usage')

# 在server中集成
class MetricsMiddleware:
    def process_request(self, request):
        request_count.inc()
        active_sessions.inc()
        request.start_time = time.time()
    
    def process_response(self, request, response):
        duration = time.time() - request.start_time
        request_duration.observe(duration)
        active_sessions.dec()
        return response
```

#### 4.4.2 模型热更新

```python
# 模型热更新管理器
class ModelManager:
    def __init__(self):
        self.current_model = None
        self.next_model = None
        self.version = 0
    
    def load_model_async(self, model_path):
        """异步加载新模型"""
        def load():
            self.next_model = Llama(model_path)
            # 原子切换
            self.current_model = self.next_model
            self.version += 1
        
        threading.Thread(target=load).start()
    
    def get_model(self):
        """获取当前模型"""
        return self.current_model
    
    def health_check(self):
        """健康检查"""
        return {
            'model_loaded': self.current_model is not None,
            'version': self.version,
            'status': 'healthy'
        }
```

## 5. 设计中的取舍

### 5.1 部署架构选择

| 架构 | 优点 | 缺点 | 适用场景 |
|-----|------|------|---------|
| 单进程 | 简单、低延迟 | 无扩展性 | 本地开发 |
| 多进程 | 利用多核 | 内存占用高 | 中小规模 |
| 容器化 | 易扩展、隔离好 | 额外开销 | 生产环境 |
| Serverless | 按需付费 | 冷启动延迟 | 低频任务 |

### 5.2 模型加载策略

```
选项A: 启动时加载
  - 优点: 请求零延迟
  - 缺点: 启动慢、内存占用高
  
选项B: 按需加载
  - 优点: 启动快、资源利用率高
  - 缺点: 首次请求延迟高

选项C: 预加载+缓存
  - 优点: 平衡启动和延迟
  - 缺点: 需要预测热点模型

llama.cpp server默认: 选项A（生产推荐）
```

### 5.3 批处理vs延迟

```
场景: 并发用户请求

方案A: 独立处理
  每个请求独立批次
  - 延迟最低
  - GPU利用率低

方案B: 动态批处理
  合并同时到达的请求
  - 延迟中等
  - GPU利用率高

方案C: 固定批处理
  等待固定时间收集请求
  - 延迟最高
  - 吞吐量最大

llama.cpp server: 支持动态批处理(--cont-batching)
```

## 6. 动手练习

### 练习1：构建Android应用

```bash
# 1. 准备环境
export ANDROID_NDK=/path/to/ndk

# 2. 交叉编译
cmake -B build-android \
    -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
    -DANDROID_ABI=arm64-v8a \
    -DANDROID_PLATFORM=android-28 \
    -DLLAMA_BUILD_EXAMPLES=OFF

make -C build-android -j

# 3. 集成到Android项目
# 将.so文件复制到app/src/main/jniLibs/arm64-v8a/
```

### 练习2：Docker部署

```bash
# 1. 构建镜像
docker build -t llama-server:latest .

# 2. 运行容器
docker run -d \
    -p 8080:8080 \
    -v $(pwd)/models:/models:ro \
    -e LLAMA_MODEL=/models/llama-7b.gguf \
    --gpus all \
    llama-server:latest

# 3. 测试API
curl http://localhost:8080/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "messages": [{"role": "user", "content": "Hello!"}]
    }'
```

### 练习3：Kubernetes部署

```bash
# 1. 创建命名空间
kubectl create namespace llama

# 2. 部署
kubectl apply -f k8s-deployment.yaml -n llama

# 3. 查看状态
kubectl get pods -n llama
kubectl logs -f deployment/llama-server -n llama

# 4. 暴露服务
kubectl port-forward svc/llama-service 8080:80 -n llama
```

### 练习4：Python集成

```python
#!/usr/bin/env python3
"""llama.cpp Python集成示例"""
from llama_cpp import Llama
import time

class LlamaService:
    def __init__(self, model_path):
        print(f"Loading model from {model_path}...")
        self.llm = Llama(
            model_path=model_path,
            n_ctx=4096,
            n_gpu_layers=999,
            verbose=False
        )
        print("Model loaded!")
    
    def chat(self, messages, temperature=0.7):
        start = time.time()
        
        response = self.llm.create_chat_completion(
            messages=messages,
            temperature=temperature,
            max_tokens=256
        )
        
        elapsed = time.time() - start
        tokens = response['usage']['completion_tokens']
        
        return {
            'content': response['choices'][0]['message']['content'],
            'tokens': tokens,
            'tokens_per_sec': tokens / elapsed
        }

# 使用示例
if __name__ == '__main__':
    service = LlamaService('./models/llama-7b.gguf')
    
    result = service.chat([
        {'role': 'system', 'content': 'You are a helpful assistant.'},
        {'role': 'user', 'content': 'Explain quantum computing in simple terms.'}
    ])
    
    print(f"Response: {result['content']}")
    print(f"Speed: {result['tokens_per_sec']:.2f} tokens/sec")
```

## 7. 本课小结

- **Android集成**：通过JNI桥接Java/Kotlin与C++，注意线程管理和内存优化
- **服务端部署**：Docker容器化便于扩展，Kubernetes支持大规模部署
- **API设计**：遵循OpenAI兼容格式，便于现有应用迁移
- **语言绑定**：Python、Node.js等绑定简化集成，但可能有性能损失
- **生产运维**：监控关键指标，支持模型热更新，实现健康检查

**部署检查清单：**
- [ ] 模型文件已验证完整性
- [ ] 内存/GPU资源充足
- [ ] 日志和监控已配置
- [ ] 健康检查端点正常
- [ ] 自动扩缩容策略已设置
- [ ] 备份和回滚方案就绪
