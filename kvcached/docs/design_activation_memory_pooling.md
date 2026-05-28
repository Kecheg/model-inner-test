# 设计文档：Embedding/Rerank 模型激活内存动态池化

> 基于 kvcached 项目扩展，实现 Embedding/Rerank 模型的激活内存虚拟化管理，
> 使其与 LLM 的 KV Cache 共享 GPU 物理内存池，提升 GPU 利用率。

---

## 1. 背景与动机

### 1.1 现状问题

在 GPU 推理服务中，经常需要同时部署 LLM（生成式模型）和 Embedding/Rerank（编码式模型）：

- **LLM**：需要 KV Cache 存储历史 token 的 Key/Value 向量，用于自回归解码。kvcached 已通过 CUDA VMM（Virtual Memory Management）实现了 KV Cache 的虚拟化，支持按需映射/释放物理页。
- **Embedding/Rerank**：使用 Encoder-Only 架构（如 BERT 类模型），**不需要 KV Cache**。但它们在 forward pass 过程中需要大量临时激活内存（hidden states、attention scores、FFN 中间值等）。

当前 vLLM 对 Embedding/Rerank 模型的内存管理：

```
vllm/v1/engine/core.py (_initialize_kv_caches):
    has_kv_cache = any(kv_cache_spec for kv_cache_spec in kv_cache_specs)
    if has_kv_cache:
        available_gpu_memory = self.model_executor.determine_available_memory()
    else:
        available_gpu_memory = [0]  # 纯 pooling 模型：不分配 KV Cache
```

这意味着：

1. **静态峰值分配**：`gpu_model_runner.profile_run()` 会用 `max_num_tokens` 运行一次完整的 dummy forward，测量峰值内存占用。之后 PyTorch 的 CUDA caching allocator 会按此峰值预留空间。
2. **无动态回收**：forward 结束后，激活内存由 PyTorch allocator 的 pool 机制回收（可复用），但物理显存不会归还给操作系统/kvcached。
3. **无法与 LLM 共享**：当 Embedding 模型空闲时，其预留的激活内存空间无法让给 LLM 使用。

### 1.2 目标

通过扩展 kvcached 的 CUDA VMM 机制，将 Embedding/Rerank 模型的激活内存也纳入虚拟化管理：

1. **按需分配**：forward pass 前映射物理页，forward pass 后立即释放物理页。
2. **与 KV Cache 共享物理内存池**：Embedding 模型空闲时，其物理内存可被 LLM 的 KV Cache 使用。
3. **快速唤醒**：模型权重常驻 GPU，激活内存按需映射，无需重新加载模型。
4. **兼容 kvcached 的 sleep/wake 机制**：支持 LLM + Embedding 共置场景。

---

## 2. 技术基础

### 2.1 kvcached 的 CUDA VMM 架构

kvcached 使用 CUDA Virtual Memory Management API 实现类似 OS 的虚拟内存管理：

```
┌─────────────────────────────────────────────────────────────┐
│                      CUDA VMM                               │
│                                                             │
│  cuMemAddressReserve()  → 预留虚拟地址空间 (VA)              │
│  cuMemCreate()          → 分配物理内存 (PA)                  │
│  cuMemMap()             → 映射 PA 到 VA                     │
│  cuMemUnmap()           → 解除 VA 映射，PA 可复用             │
│  cuMemSetAccess()       → 设置访问权限                       │
│  cuMemAddressFree()     → 释放虚拟地址空间                   │
└─────────────────────────────────────────────────────────────┘
```

核心 C++ 类：

| 类 | 文件 | 职责 |
|---|---|---|
| `FTensor` | `csrc/ftensor.cpp` | 虚拟化张量：预留 VA，按需 map/unmap 物理页 |
| `FTensorAllocator` | `csrc/allocator.cpp` | 全局分配器，管理多个 FTensor 实例 |
| `Page` / `GPUPage` | `csrc/page.cpp` | 物理页抽象，封装 `cuMemCreate` + `cuMemMap` |
| `PageAllocator` | `csrc/page_allocator.cpp` | 物理页池管理，支持预分配、trim、resize |
| `KVCacheManager` | `kvcached/kv_cache_manager.py` | Python 层块管理器，将 block 分配映射到 VMM page |

### 2.2 kvcached 的 vLLM 集成点

kvcached 通过 autopatch 机制（`KVCACHED_AUTOPATCH=1`）在 vLLM import 时自动注入补丁：

| Patch 类 | 目标 | 作用 |
|---|---|---|
| `EngineCorePatch` | `vllm.v1.engine.core.EngineCore.__init__` | 初始化 kvcached |
| `GPUModelRunnerPatch` | `vllm.v1.worker.gpu_model_runner.GPUModelRunner` | Worker 侧初始化 + 替换 KV Cache 分配 |
| `ElasticBlockPoolPatch` | `vllm.v1.core.block_pool` | 注入 `ElasticBlockPool` |
| `KVCacheCoordinatorPatch` | `vllm.v1.core.kv_cache_coordinator` | 使用 `ElasticBlockPool` 替换默认 `BlockPool` |
| `GPUWorkerPatch` | `vllm.v1.worker.gpu_worker.Worker` | 跳过 GPU 显存检查 |

### 2.3 vLLM Pooling 模型的关键路径

```
gpu_model_runner.__init__()
  → self.is_pooling_model = model_config.runner_type == "pooling"

profile_run()
  → _dummy_run(max_num_tokens, is_profile=True)    # 测量峰值激活内存
  → _dummy_pooler_run(hidden_states)                # pooling 模型的输出

execute_model(scheduler_output)
  → model.execute(...)                               # 前向传播（激活内存在此分配）
  → _pool(hidden_states, ...)                        # pooling 后处理
```

对于 pooling 模型，`_initialize_kv_caches` 设置 `available_gpu_memory = [0]`，不分配任何 KV Cache。激活内存完全由 PyTorch CUDA allocator 管理。

---

## 3. 架构设计

### 3.1 总体架构

```
┌──────────────────────────────────────────────────────────────────────┐
│                        kvcached Extended                             │
│                                                                      │
│  ┌────────────────────┐     ┌────────────────────────────────────┐   │
│  │  KV Cache VMM      │     │  Activation Memory VMM (新增)      │   │
│  │  (现有)             │     │                                    │   │
│  │  FTensor per layer  │     │  ActivationFTensor per model      │   │
│  │  PageAllocator      │     │  ActivationPoolManager            │   │
│  └────────┬───────────┘     └──────────────┬─────────────────────┘   │
│           │                                │                         │
│           ▼                                ▼                         │
│  ┌────────────────────────────────────────────────────────────────┐   │
│  │              Unified Physical Memory Pool                      │   │
│  │   cuMemCreate / cuMemMap / cuMemUnmap / cuMemFree             │   │
│  │                                                                │   │
│  │   GPU 物理页总数固定，KV Cache 和 Activation 共享               │   │
│  └────────────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────────┘
```

### 3.2 核心抽象：ActivationFTensor

`ActivationFTensor` 是 `FTensor` 的扩展，用于管理 Embedding/Rerank 模型 forward pass 的临时激活内存。与 KV Cache FTensor 的关键区别：

| 维度 | KV Cache FTensor | Activation FTensor |
|---|---|---|
| 生命周期 | 跨多个 token，整个请求持续 | 单次 forward pass |
| 访问模式 | 随机读写（attention 需要） | 顺序读写（forward 计算） |
| 页管理粒度 | 按 block 分配/释放 | 整体映射/释放 |
| 并发 | 多请求并发访问 | 单请求独占 |
| 峰值可预测性 | 取决于请求长度和并发数 | 由 batch_size 和模型结构确定 |

### 3.3 核心抽象：ActivationPoolManager

`ActivationPoolManager` 管理 Embedding/Rerank 模型的激活内存池，负责：

1. 在模型初始化时预留虚拟地址空间（不映射物理页）。
2. 在 forward pass 前按需映射物理页。
3. 在 forward pass 后释放所有物理页。
4. 与 kvcached 的 `PageAllocator` 共享物理内存预算。

### 3.4 内存生命周期

```
                    时间轴 →
                    
LLM (生成式):
  |<───── KV Cache 持续增长 ─────────>|
  |   ┌──┐┌──┐┌──┐┌──┐               |
  |   │K1││K2││K3││K4│ ...           |
  |   └──┘└──┘└──┘└──┘               |
  
Embedding (编码式):
       ↑ forward          ↑ forward
       |<-- 激活内存 --->|           |<-- 激活内存 --->|
       |  ┌─────────┐   |           |  ┌─────────┐   |
       |  │ map     │   |           |  │ map     │   |
       |  │ compute │   |           |  │ compute │   |
       |  │ unmap   │   |           |  │ unmap   │   |
       |  └─────────┘   |           |  └─────────┘   |
       ← 物理页归还池 →              ← 物理页归还池 →

物理内存池视角:
  [KKKKKKKK....AAAA....KKKKKKKKKAAAA....KKKKKKK]
   ↑              ↑              ↑              ↑
   LLM 占用    Embedding      LLM 扩展    Embedding
   KV Cache    forward 用完   KV Cache    再次 forward
```

---

## 4. 详细设计

### 4.1 Phase 1: C++ 层 — Activation FTensor

#### 4.1.1 新增 `ActivationFTensor` 类

在 `csrc/ftensor.hpp` / `csrc/ftensor.cpp` 中新增：

```cpp
class ActivationFTensor {
public:
    // 创建虚拟化的激活内存区域
    // size: 预留的虚拟地址空间大小（基于峰值 profiling）
    ActivationFTensor(const std::string &name, size_t size,
                      torch::Dtype dtype, torch::Device dev,
                      std::shared_ptr<Page> zero_page);

    ~ActivationFTensor();

    // 映射所有虚拟页到物理页（forward 前调用）
    bool map_all();

    // 解除所有物理页映射（forward 后调用）
    bool unmap_all();

    // 获取 torch.Tensor 视图（映射后可正常使用）
    torch::Tensor get_tensor() const { return tensor_; }

    // 查询状态
    bool is_mapped() const { return mapped_; }
    size_t size() const { return size_; }

private:
    std::string name_;
    generic_ptr_t vaddr_;     // 虚拟地址起始
    size_t size_;             // 总大小（字节）
    size_t page_size_;        // 页大小
    bool mapped_;             // 是否已映射物理页
    torch::Tensor tensor_;    // torch 张量视图
    std::vector<std::shared_ptr<Page>> pages_;  // 当前映射的物理页
    std::shared_ptr<Page> zero_page_;           // zero page（未映射时使用）
};
```

实现要点：

```cpp
bool ActivationFTensor::map_all() {
    if (mapped_) return true;

    // 为整个虚拟地址范围分配并映射物理页
    size_t num_pages = (size_ + page_size_ - 1) / page_size_;
    pages_.clear();
    pages_.reserve(num_pages);

    for (size_t i = 0; i < num_pages; i++) {
        offset_t offset = i * page_size_;
        auto vaddr = reinterpret_cast<generic_ptr_t>(
            reinterpret_cast<uintptr_t>(vaddr_) + offset);

        // 先 unmap zero page
        cuMemUnmap(reinterpret_cast<CUdeviceptr>(vaddr), page_size_);

        // 创建并映射新的物理页
        pages_.push_back(std::make_shared<GPUPage>(i, dev_.index(), page_size_));
        pages_.back()->map(vaddr);
    }
    mapped_ = true;
    return true;
}

bool ActivationFTensor::unmap_all() {
    if (!mapped_) return true;

    // 释放所有物理页，恢复 zero page 映射
    for (size_t i = 0; i < pages_.size(); i++) {
        offset_t offset = i * page_size_;
        auto vaddr = reinterpret_cast<generic_ptr_t>(
            reinterpret_cast<uintptr_t>(vaddr_) + offset);

        cuMemUnmap(reinterpret_cast<CUdeviceptr>(vaddr), page_size_);

        // 重新映射 zero page 以保持内存完整性
        zero_page_->map(vaddr);
    }
    pages_.clear();
    mapped_ = false;
    return true;
}
```

#### 4.1.2 扩展 `FTensorAllocator`

在 `csrc/allocator.cpp` 的 `FTensorAllocator` 中新增方法：

```cpp
// 创建用于激活内存的 FTensor
torch::Tensor FTensorAllocator::create_activation_tensor(
    size_t size, torch::Dtype dtype, const std::string &dev_str,
    const std::string &name) {
    std::lock_guard<std::mutex> lock(mtx_);
    auto it = activation_ftensors_.find(name);
    if (it != activation_ftensors_.end()) {
        return it->second->get_tensor();
    }
    activation_ftensors_[name] =
        std::make_unique<ActivationFTensor>(name, size, dtype, dev_, zero_page_);
    return activation_ftensors_[name]->get_tensor();
}

// 映射指定激活内存区域
bool FTensorAllocator::map_activation(const std::string &name) {
    std::lock_guard<std::mutex> lock(mtx_);
    auto it = activation_ftensors_.find(name);
    if (it == activation_ftensors_.end()) return false;
    return it->second->map_all();
}

// 解除映射指定激活内存区域
bool FTensorAllocator::unmap_activation(const std::string &name) {
    std::lock_guard<std::mutex> lock(mtx_);
    auto it = activation_ftensors_.find(name);
    if (it == activation_ftensors_.end()) return false;
    return it->second->unmap_all();
}
```

#### 4.1.3 扩展 pybind11 绑定

在 `csrc/torch_bindings.cpp` 中注册新接口：

```cpp
m.def("create_activation_tensor", &kvcached::create_activation_tensor,
      "Create a virtualized activation tensor",
      py::arg("size"), py::arg("dtype_size"), py::arg("dev_str"),
      py::arg("name"));

m.def("map_activation", &kvcached::map_activation,
      "Map physical pages for activation tensor",
      py::arg("name"));

m.def("unmap_activation", &kvcached::unmap_activation,
      "Unmap physical pages for activation tensor",
      py::arg("name"));
```

### 4.2 Phase 1: Python 层 — ActivationPoolManager

#### 4.2.1 新增 `kvcached/activation_pool_manager.py`

```python
class ActivationPoolManager:
    """管理 Embedding/Rerank 模型的激活内存池。

    通过 CUDA VMM 虚拟化激活内存：
    - 初始化时预留虚拟地址空间
    - forward 前映射物理页
    - forward 后释放物理页
    """

    def __init__(
        self,
        model_name: str,
        peak_activation_bytes: int,
        dtype: torch.dtype,
        device: str,
    ):
        self.model_name = model_name
        self.peak_activation_bytes = peak_activation_bytes
        self.dtype = dtype
        self.device = device
        self._initialized = False

    def initialize(self) -> None:
        """创建虚拟化的激活内存区域（不映射物理页）。"""
        from kvcached import vmm_ops
        self._tensor_name = f"activation_{self.model_name}"
        vmm_ops.create_activation_tensor(
            self.peak_activation_bytes,
            self.dtype.itemsize,
            self.device,
            self._tensor_name,
        )
        self._initialized = True

    def begin_forward(self) -> None:
        """forward 前调用：映射物理页。"""
        if not self._initialized:
            return
        from kvcached import vmm_ops
        vmm_ops.map_activation(self._tensor_name)

    def end_forward(self) -> None:
        """forward 后调用：释放物理页。"""
        if not self._initialized:
            return
        from kvcached import vmm_ops
        vmm_ops.unmap_activation(self._tensor_name)
```

### 4.3 Phase 2: vLLM 集成 — Autopatch 扩展

#### 4.3.1 新增 `PoolingModelRunnerPatch`

在 `kvcached/integration/vllm/patches.py` 中新增：

```python
class PoolingModelRunnerPatch(VersionAwarePatch, BasePatch):
    """Patch GPUModelRunner 的 pooling 模型路径，注入激活内存管理。"""

    library = "vllm"
    target_module = "vllm.v1.worker.gpu_model_runner"
    target_class = "GPUModelRunner"
    patch_name = "pooling_model_runner"

    @version_range(VLLM_ALL_RANGE)
    def patch_pooling_forward(self, GPUModelRunner) -> bool:
        """Patch execute_model 以在 pooling forward 前后管理激活内存。"""

        original_execute = GPUModelRunner.execute_model

        def _patched_execute(self, scheduler_output, intermediate_tensors=None):
            is_pooling = getattr(self, 'is_pooling_model', False)
            act_pool = getattr(self, '_activation_pool', None)

            # forward 前：映射物理页
            if is_pooling and act_pool is not None:
                act_pool.begin_forward()

            try:
                result = original_execute(self, scheduler_output,
                                          intermediate_tensors)
            finally:
                # forward 后：释放物理页
                if is_pooling and act_pool is not None:
                    act_pool.end_forward()

            return result

        GPUModelRunner.execute_model = _patched_execute
        return True

    @version_range(VLLM_ALL_RANGE)
    def patch_profile_run(self, GPUModelRunner) -> bool:
        """在 profile_run 后记录峰值激活内存并初始化 ActivationPoolManager。"""

        original_profile = GPUModelRunner.profile_run

        def _patched_profile(self):
            # 记录 profile 前的显存状态
            torch.cuda.reset_peak_memory_stats()
            mem_before = torch.cuda.memory_allocated()

            original_profile(self)

            is_pooling = getattr(self, 'is_pooling_model', False)
            if is_pooling and enable_kvcached():
                mem_peak = torch.cuda.max_memory_allocated()
                activation_bytes = mem_peak - mem_before

                from kvcached.activation_pool_manager import ActivationPoolManager
                self._activation_pool = ActivationPoolManager(
                    model_name=self.model_config.model,
                    peak_activation_bytes=activation_bytes,
                    dtype=self.model_config.dtype,
                    device=str(self.device),
                )
                self._activation_pool.initialize()

        GPUModelRunner.profile_run = _patched_profile
        return True
```

#### 4.3.2 注册新 Patch

在 `kvcached/integration/vllm/autopatch.py` 中注册：

```python
from kvcached.integration.vllm.patches import PoolingModelRunnerPatch

patch_manager.register_patches_with_versions([
    (ElasticBlockPoolPatch(), VLLM_ALL_RANGE),
    (EngineCorePatch(), VLLM_ALL_RANGE),
    (GPUModelRunnerPatch(), VLLM_ALL_RANGE),
    (GPUWorkerPatch(), VLLM_ALL_RANGE),
    (KVCacheCoordinatorPatch(), VLLM_V9_PLUS_RANGE),
    (KVCacheManagerPatch(), VLLM_V8_RANGE),
    (PoolingModelRunnerPatch(), VLLM_ALL_RANGE),  # 新增
])
```

### 4.4 Phase 3: 多模型共置调度

#### 4.4.1 扩展 SleepManager

在 `controller/sleep_manager.py` 的 `SleepManager` 中增加对 Embedding 模型的激活内存管理：

```python
class SleepManager:
    # ... 现有代码 ...

    async def put_model_to_sleep(self, model_name: str, manual: bool = False) -> bool:
        # ... 现有 sleep 逻辑 ...

        # 新增：对于 pooling 模型，释放所有激活内存的物理页
        if model_name in self.pooling_models_config:
            await self._release_activation_memory(model_name)

    async def wakeup_model(self, model_name: str) -> bool:
        # ... 现有 wakeup 逻辑 ...

        # 新增：pooling 模型不需要恢复激活内存
        # 下次 forward 时会自动按需映射

    async def _release_activation_memory(self, model_name: str) -> bool:
        """释放 pooling 模型所有激活内存的物理页。"""
        # 通过 vLLM API 触发 activation unmap
        host = self.pooling_models_config[model_name].get("host", "localhost")
        port = self.pooling_models_config[model_name].get("port", "8000")
        url = f"http://{host}:{port}/release_activation_memory"
        async with aiohttp.ClientSession() as session:
            async with session.post(url) as response:
                return response.status == 200
```

#### 4.4.2 vLLM 端新增 API Endpoint

在 vLLM 的 API server 中新增 `/release_activation_memory` 和 `/map_activation_memory` 端点，供 SleepManager 调用。

#### 4.4.3 扩展 LLMRouter

在 `controller/router.py` 的 `LLMRouter` 中增加激活内存预算维度：

```python
class LLMRouter:
    def __init__(self, models_config, *, sleep_manager, traffic_monitor):
        # ... 现有代码 ...
        self.model_memory_budget: Dict[str, int] = {}  # model_name -> bytes

    def update_memory_budget(self, model_name: str, budget_bytes: int):
        """更新模型的物理内存预算。"""
        self.model_memory_budget[model_name] = budget_bytes

    def _check_memory_feasibility(self, model_name: str) -> bool:
        """检查是否有足够物理内存运行该模型的 forward。"""
        budget = self.model_memory_budget.get(model_name, 0)
        total_pool = self._get_total_physical_pages() * PAGE_SIZE
        used = sum(self.model_memory_budget.values()) - budget
        return (total_pool - used) >= budget
```

### 4.5 物理内存共享机制

#### 4.5.1 统一物理页池

当前 kvcached 的 `PageAllocator` 为每个 `KVCacheManager` 管理独立的物理页池。为支持 KV Cache 与激活内存共享物理页，需要引入一个全局物理页协调器：

```
┌───────────────────────────────────────────────────┐
│           PhysicalPageCoordinator (新增)            │
│                                                   │
│  total_physical_pages = GPU_FREE_MEMORY / PAGE_SIZE │
│                                                   │
│  ┌─────────────────┐  ┌─────────────────────────┐ │
│  │ KV Cache 份额    │  │ Activation 份额          │ │
│  │ (动态调整)       │  │ (按需借用/归还)          │ │
│  └─────────────────┘  └─────────────────────────┘ │
│                                                   │
│  策略：                                           │
│  - Activation forward 时从共享池借页              │
│  - forward 结束后立即归还                         │
│  - KV Cache 可使用所有归还的页                    │
│  - SleepManager 可强制回收所有 Activation 页       │
└───────────────────────────────────────────────────┘
```

#### 4.5.2 实现方案

扩展现有 `PageAllocator` 以支持多个 "tenant"（KV Cache 和 Activation）：

```cpp
class PhysicalPageCoordinator {
public:
    // 注册租户（KV Cache 或 Activation）
    void register_tenant(const std::string &tenant_id, TenantType type);

    // 租户请求物理页
    std::vector<std::shared_ptr<Page>> alloc_pages(
        const std::string &tenant_id, size_t num_pages);

    // 租户释放物理页
    void free_pages(const std::string &tenant_id,
                    const std::vector<page_id_t> &page_ids);

    // 获取当前可用物理页数
    size_t available_pages() const;

private:
    size_t total_pages_;
    std::unordered_map<std::string, TenantInfo> tenants_;
    std::vector<std::shared_ptr<Page>> free_pages_;
};
```

---

## 5. 关键设计决策

### 5.1 激活内存峰值估算

**方案**：复用 vLLM 的 `profile_run()` 机制。

**具体做法**：
1. `profile_run()` 运行 dummy forward，PyTorch 记录峰值显存。
2. 在 patch 中截取 `profile_run()` 的结果，计算 `peak_activation = max_memory - weight_memory`。
3. 以此峰值作为 `ActivationFTensor` 的虚拟地址空间大小。

**trade-off**：峰值估算是保守的（基于 `max_num_tokens`），但保证了安全。未来可优化为根据实际 batch_size 动态调整。

### 5.2 CUDA Graph 兼容性

**结论**：Embedding/Rerank 模型通常**不使用 CUDA Graph**。

**原因**：
- CUDA Graph 要求固定的内存地址和计算图结构。
- vLLM 的 CUDA Graph 主要用于 LLM 的 decode 阶段（固定 batch、固定 sequence 操作）。
- Pooling 模型的 batch_size 和 sequence_length 变化较大，不适合 CUDA Graph。

因此，使用 VMM 动态映射不会与 CUDA Graph 冲突。

### 5.3 虚拟地址空间超额预留

**方案**：虚拟地址空间可以超额预留（2MB 对齐的 VA 空间几乎免费），物理页按需映射。

```
VA 空间 (几乎免费):
┌─────────────────────────────────────────────┐
│ Activation VA (基于 max_num_tokens 峰值)     │
│ 映射状态: ░░░░ (大部分未映射)                │
└─────────────────────────────────────────────┘

物理页 (稀缺资源):
┌──┐┌──┐┌──┐
│P1││P2││P3│  ← forward 时按需分配
└──┘└──┘└──┘
```

这避免了碎片化问题，因为 VA 空间连续但 PA 按需映射。

### 5.4 并发安全性

**问题**：当 LLM 和 Embedding 模型同时运行时，如何避免物理页竞争？

**方案**：
1. `ActivationPoolManager.begin_forward()` 时检查物理页余量，不足则等待或拒绝。
2. `PhysicalPageCoordinator` 使用优先级策略：KV Cache 有最低保障页数，Activation 使用剩余页。
3. forward pass 时间短（< 100ms），物理页持有时间极短，竞争概率低。

### 5.5 与 CuMemAllocator 的关系

vLLM 已有 `CuMemAllocator`（`vllm/device_allocator/cumem.py`）支持 sleep/wake：

- `sleep()`: 将标记的 tensor offload 到 CPU。
- `wake_up()`: 从 CPU 恢复到 GPU。

kvcached 的 VMM 方案与 CuMemAllocator 的区别：

| 维度 | CuMemAllocator | kvcached VMM (本方案) |
|---|---|---|
| 机制 | CUDA allocator hook + CPU offload | CUDA VMM map/unmap |
| 粒度 | 整个 tensor | 2MB 页级别 |
| 延迟 | offload/restore 需要 CPU↔GPU 拷贝 | map/unmap 仅更新页表，微秒级 |
| 共享性 | 无法跨模型共享 | 物理页可动态共享 |

**策略**：本方案独立于 CuMemAllocator。模型权重继续由 CuMemAllocator 管理（sleep 时 offload 到 CPU），激活内存由 kvcached VMM 管理（forward 后 unmap）。

---

## 6. 实施计划

### Phase 1: 概念验证（2-3 周）

**目标**：证明 CUDA VMM 可以管理 Embedding 模型的激活内存。

**任务**：
1. 在 `csrc/` 中实现 `ActivationFTensor` 类。
2. 扩展 `FTensorAllocator` 增加 activation tensor 支持。
3. 添加 pybind11 绑定。
4. 编写单元测试：`tests/test_activation_tensor.py`。
5. 验证 map_all / unmap_all 的正确性和延迟。

**验收标准**：
- `ActivationFTensor` 能正确 map/unmap 物理页。
- torch.Tensor 视图在 map 后可正常读写。
- unmap 后 GPU 物理显存被释放（`nvidia-smi` 可观测）。

### Phase 2: vLLM 集成（2-3 周）

**目标**：通过 autopatch 机制将激活内存管理注入 vLLM 的 pooling 模型路径。

**任务**：
1. 实现 `ActivationPoolManager` Python 类。
2. 实现 `PoolingModelRunnerPatch`。
3. 注册到 autopatch 系统。
4. 端到端测试：Embedding 模型推理正确，物理页在 forward 后释放。
5. 性能基准测试：对比有/无 VMM 的推理延迟。

**验收标准**：
- `KVCACHED_AUTOPATCH=1` 环境变量自动注入 pooling 模型的激活内存管理。
- Embedding 模型推理结果与未启用 kvcached 时一致（数值正确性）。
- `nvidia-smi` 显示 forward 后物理显存减少。

### Phase 3: 多模型共置（3-4 周）

**目标**：在同一 GPU 上共置 LLM 和 Embedding 模型，通过 sleep/wake 共享物理内存。

**任务**：
1. 实现 `PhysicalPageCoordinator`。
2. 扩展 `SleepManager` 支持 pooling 模型。
3. 新增 vLLM API 端点（`/release_activation_memory` 等）。
4. 扩展 `LLMRouter` 的内存预算感知调度。
5. 集成测试：LLM 和 Embedding 交替运行，物理内存正确共享。

**验收标准**：
- LLM 推理时可用到 Embedding 模型释放的物理页。
- Embedding 请求到来时能快速映射物理页并完成推理。
- 整体 GPU 利用率提升。

### Phase 4: 性能优化（2-3 周）

**目标**：优化 map/unmap 延迟，支持更复杂的调度策略。

**任务**：
1. 物理页预热（预分配常用页，减少 map 延迟）。
2. 异步 map（forward 准备与 map 并行）。
3. 基于 batch_size 的动态激活内存调整。
4. 多 batch 流水线（Embedding 模型连续请求时保持映射）。
5. 压力测试和延迟优化。

---

## 7. 风险与缓解

| 风险 | 影响 | 缓解措施 |
|---|---|---|
| `cuMemMap`/`cuMemUnmap` 延迟过高 | forward 开销增加 | 预热物理页池；benchmark 验证延迟在可接受范围（目标 < 1ms for map_all） |
| 激活内存峰值估算不准确 | OOM 或浪费 | 使用 `max_num_tokens` 保守估计；后续支持动态 resize |
| PyTorch CUDA allocator 与 VMM 冲突 | 内存泄漏或 double-free | 使用 `torch.from_blob` 而非 PyTorch allocator 管理激活内存；绕过 caching allocator |
| 多线程并发安全问题 | 数据竞争 | `ActivationPoolManager` 使用锁保护；forward 为单线程顺序执行 |
| vLLM 版本兼容性 | patch 失效 | 使用 `VersionAwarePatch` 的版本范围机制，与现有 patch 策略一致 |
| 不支持 CUDA Graph 的模型 | 限制适用范围 | Embedding/Rerank 模型通常不用 CUDA Graph，风险低；如需支持，可 fallback 到静态分配 |

---

## 8. 文件变更清单

### kvcached 项目

| 文件 | 变更类型 | 说明 |
|---|---|---|
| `csrc/ftensor.hpp` | 修改 | 新增 `ActivationFTensor` 类声明 |
| `csrc/ftensor.cpp` | 修改 | 实现 `ActivationFTensor` |
| `csrc/allocator.hpp` | 修改 | `FTensorAllocator` 新增 activation 管理 |
| `csrc/allocator.cpp` | 修改 | 实现 activation 创建/map/unmap |
| `csrc/torch_bindings.cpp` | 修改 | 注册新 pybind11 接口 |
| `kvcached/activation_pool_manager.py` | **新增** | Python 层激活内存管理 |
| `kvcached/integration/vllm/patches.py` | 修改 | 新增 `PoolingModelRunnerPatch` |
| `kvcached/integration/vllm/autopatch.py` | 修改 | 注册新 patch |
| `controller/sleep_manager.py` | 修改 | 支持 pooling 模型 sleep/wake |
| `controller/router.py` | 修改 | 内存预算感知调度 |
| `tests/test_activation_tensor.py` | **新增** | 单元测试 |

### vLLM 项目（通过 autopatch 修改，不直接改源码）

| 被修改类/方法 | Patch 方式 | 说明 |
|---|---|---|
| `GPUModelRunner.profile_run()` | monkey-patch | 记录峰值激活内存 |
| `GPUModelRunner.execute_model()` | monkey-patch | forward 前后管理激活内存 |
| `GPUModelRunner.__init__()` | monkey-patch | 初始化 `ActivationPoolManager` |

---

## 9. 配置项

新增环境变量：

| 变量名 | 默认值 | 说明 |
|---|---|---|
| `KVCACHED_ACTIVATION_POOL_ENABLED` | `"true"` | 是否启用激活内存池化 |
| `KVCACHED_ACTIVATION_MIN_BATCH_SIZE` | `"1"` | 触发 VMM 的最小 batch size |
| `KVCACHED_ACTIVATION_KEEP_MAPPED_MS` | `"0"` | forward 后保持映射的毫秒数（0=立即释放），用于连续请求优化 |
| `KVCACHED_PHYSICAL_PAGE_RESERVED_RATIO` | `"0.1"` | 为 KV Cache 保留的物理页比例 |

---

## 10. 参考资料

- [CUDA Virtual Memory Management API](https://docs.nvidia.com/cuda/cuda-driver-api/group__CU__MEM.html)
- kvcached 源码：`csrc/ftensor.cpp`（FTensor 实现）、`csrc/allocator.cpp`（FTensorAllocator）
- vLLM 源码：`vllm/v1/worker/gpu_model_runner.py`（GPUModelRunner）、`vllm/v1/engine/core.py`（EngineCore）
- vLLM pooling 模型：`vllm/v1/kv_cache_interface.py`（`EncoderOnlyAttentionSpec`）

