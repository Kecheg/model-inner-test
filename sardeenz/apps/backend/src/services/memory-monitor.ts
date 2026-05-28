import { spawn } from 'child_process'
import path from 'path'
import { fileURLToPath } from 'url'
import type {
  ResourceMetrics,
  MemoryUsageResponse,
  KVCacheMetrics,
  GpuMetrics,
  ModelGpuMemory,
  ModelInstance,
  MultiGpuMemoryUsageResponse,
  PerGpuMetrics,
} from '@sardeenz/types'
import { modelStore } from '../stores/model-store.js'
import type { Logger } from '@sardeenz/utils'
import { InternalError } from '../utils/errors.js'
import { getPrimaryGpuInfo, getNvidiaSmiInfo } from '../utils/gpu-info.js'
import { findProcessesByEnvMarker, getDescendantPids } from '../utils/process.js'
import { config, isInferenceSimMode } from '../config.js'

const __filename = fileURLToPath(import.meta.url)
const __dirname = path.dirname(__filename)

/** KVCache segment info from Python script (reads /dev/shm like kvtop) */
interface KvcacheSegment {
  ipc_name: string
  gpu_indices: number[] | null // GPU indices parsed from segment name (null for legacy segments)
  total_size: number // bytes
  used_size: number // bytes
  prealloc_size: number // bytes
}

// Color palette for model visualization (PatternFly-inspired)
const MODEL_COLORS = [
  '#0066CC', // Blue
  '#5752D1', // Purple
  '#009596', // Cyan
  '#EC7A08', // Orange
  '#A30000', // Red
  '#3E8635', // Green
  '#8B5CF6', // Violet
  '#06B6D4', // Teal
]

/**
 * Get a color for a model based on its index in the list.
 * Uses sequential assignment to guarantee unique colors for the first 8 models.
 */
function getModelColor(_instanceId: string, index: number): string {
  return MODEL_COLORS[index % MODEL_COLORS.length]
}

/**
 * Extract display name from model path (e.g., "meta-llama/Llama-3.2-1B" -> "Llama-3.2-1B")
 */
function getDisplayName(modelPath: string): string {
  const parts = modelPath.split('/')
  return parts[parts.length - 1] || modelPath
}

export class MemoryMonitor {
  private logger: Logger
  private metricsCache: Map<string, ResourceMetrics> = new Map()

  constructor(logger: Logger) {
    this.logger = logger.child({ component: 'MemoryMonitor' })
  }

  private async getTrackedPidsForInstance(instance: ModelInstance): Promise<Set<number>> {
    const primaryPid = instance.engineCorePid ?? instance.processId
    const descendantPids = await getDescendantPids(primaryPid)
    const allPids = new Set([primaryPid, ...descendantPids])

    // When EngineCore is tracked separately, also include the API server tree.
    if (instance.engineCorePid) {
      allPids.add(instance.processId)
      const apiServerDescendants = await getDescendantPids(instance.processId)
      for (const pid of apiServerDescendants) {
        allPids.add(pid)
      }
    }

    // vLLM TP workers can be re-parented or launched outside the process tree.
    // The launcher stamps every model process with this marker, so use it as
    // the authoritative fallback for GPU memory attribution.
    const markedPids = await findProcessesByEnvMarker('SARDEENZ_INSTANCE_ID', instance.id)
    for (const pid of markedPids) {
      allPids.add(pid)
    }

    return allPids
  }

  private collectInstanceMemoryByGpuMb(
    instance: ModelInstance,
    trackedPids: Set<number>,
    processes: Array<{ pid: number; gpu: number; gpuMemoryMB: number }>
  ): Map<number, number> {
    const memoryByGpuMb = new Map<number, number>()

    if (config.virtualGpuCount > 0) {
      const totalMemoryMb = processes.reduce((sum, proc) => {
        return trackedPids.has(proc.pid) ? sum + proc.gpuMemoryMB : sum
      }, 0)

      const targetGpuIds = instance.gpuIds.length > 0 ? instance.gpuIds : [0]
      const perGpuMemoryMb = targetGpuIds.length > 0 ? totalMemoryMb / targetGpuIds.length : 0

      for (const gpuId of targetGpuIds) {
        memoryByGpuMb.set(gpuId, perGpuMemoryMb)
      }

      return memoryByGpuMb
    }

    for (const proc of processes) {
      if (!trackedPids.has(proc.pid)) continue
      if (instance.gpuIds.length > 0 && !instance.gpuIds.includes(proc.gpu)) continue
      memoryByGpuMb.set(proc.gpu, (memoryByGpuMb.get(proc.gpu) ?? 0) + proc.gpuMemoryMB)
    }

    return memoryByGpuMb
  }

  /**
   * Get GPU memory usage for all models (new enhanced format)
   */
  async getMemoryUsage(): Promise<MemoryUsageResponse> {
    try {
      // Get KVCache segments (skip in inference-sim mode) and GPU info in parallel
      const [kvcacheSegments, gpuInfo, nvidiaSmiInfo] = await Promise.all([
        isInferenceSimMode() ? Promise.resolve([]) : this.runKvcacheStats(),
        getPrimaryGpuInfo(),
        getNvidiaSmiInfo(),
      ])

      // Get GPU metrics from NVML (needed for KVCache calculation)
      const primaryGpu = nvidiaSmiInfo.gpus[0]
      const gpuUsedMB = primaryGpu?.memoryUsedMB ?? 0
      const gpuTotalMB = primaryGpu?.memoryTotalMB ?? gpuInfo.totalMemoryMB
      const gpuFreeMB = gpuTotalMB - gpuUsedMB

      const gpu: GpuMetrics = {
        total_gb: gpuTotalMB / 1024,
        used_gb: gpuUsedMB / 1024,
        free_gb: gpuFreeMB / 1024,
        utilization_percent: parseFloat(primaryGpu?.gpuUtilization?.replace('%', '') || '0'),
      }

      // Calculate KVCache pool metrics (aggregate from all segments)
      // KVCache Total = Free GPU Memory + Prealloc KVCache Memory
      // Prealloc appears as "used" in nvidia-smi but is actually available to KVCache
      const kvcacheUsedBytes = kvcacheSegments.reduce((sum, s) => sum + s.used_size, 0)
      const kvcachePreallocBytes = kvcacheSegments.reduce((sum, s) => sum + s.prealloc_size, 0)
      const gpuFreeBytes = gpuFreeMB * 1024 ** 2
      const kvcacheTotalBytes = gpuFreeBytes + kvcachePreallocBytes + kvcacheUsedBytes
      const kvcacheFreeBytes = Math.max(0, kvcacheTotalBytes - kvcacheUsedBytes - kvcachePreallocBytes)

      const kvcache: KVCacheMetrics = {
        total_gb: kvcacheTotalBytes / 1024 ** 3,
        prealloc_gb: kvcachePreallocBytes / 1024 ** 3,
        used_gb: kvcacheUsedBytes / 1024 ** 3,
        free_gb: kvcacheFreeBytes / 1024 ** 3,
      }

      // Build per-model GPU memory breakdown using actual NVML process memory
      const allInstances = modelStore.getAll()
      const runningInstances = allInstances.filter((instance) => instance.status === 'running')

      // Track display name counts to generate unique suffixes for duplicates
      const displayNameCounts = new Map<string, number>()

      const models: ModelGpuMemory[] = []
      for (let index = 0; index < runningInstances.length; index++) {
        const instance = runningInstances[index]
        const trackedPids = await this.getTrackedPidsForInstance(instance)
        const memoryByGpuMb = this.collectInstanceMemoryByGpuMb(
          instance,
          trackedPids,
          nvidiaSmiInfo.processes
        )
        const totalMemoryGb =
          Array.from(memoryByGpuMb.values()).reduce((sum, memoryMb) => sum + memoryMb, 0) / 1024

        // Generate unique display name with suffix for duplicates
        const baseName = getDisplayName(instance.modelPath)
        const count = (displayNameCounts.get(baseName) ?? 0) + 1
        displayNameCounts.set(baseName, count)
        const displayName = count === 1 ? baseName : `${baseName} (${count})`

        models.push({
          model_path: instance.modelPath,
          instance_id: instance.id,
          display_name: displayName,
          gpu_memory_gb: totalMemoryGb,
          color: getModelColor(instance.id, index),
        })
      }

      return {
        kvcache,
        gpu,
        models,
      }
    } catch (err) {
      this.logger.error({ err }, 'Failed to get memory usage')
      throw new InternalError(
        `Failed to get memory usage: ${err instanceof Error ? err.message : 'Unknown error'}`
      )
    }
  }

  /**
   * Get GPU memory usage for all GPUs with per-model breakdown (multi-GPU support)
   */
  async getMultiGpuMemoryUsage(): Promise<MultiGpuMemoryUsageResponse> {
    try {
      // Get KVCache segments (skip in inference-sim mode) and GPU info in parallel
      const [kvcacheSegments, nvidiaSmiInfo] = await Promise.all([
        isInferenceSimMode() ? Promise.resolve([]) : this.runKvcacheStats(),
        getNvidiaSmiInfo(),
      ])

      // Get all running and sleeping model instances (sleeping models still consume some GPU memory)
      const allInstances = modelStore.getAll()
      const activeInstances = allInstances.filter(
        (i) => i.status === 'running' || i.status === 'sleeping'
      )

      // Group models by GPU with display name management
      const modelsByGpu = new Map<number, ModelGpuMemory[]>()
      const displayNameCounts = new Map<string, number>()

      for (let index = 0; index < activeInstances.length; index++) {
        const instance = activeInstances[index]
        const trackedPids = await this.getTrackedPidsForInstance(instance)
        const memoryByGpuMb = this.collectInstanceMemoryByGpuMb(
          instance,
          trackedPids,
          nvidiaSmiInfo.processes
        )
        const baseName = getDisplayName(instance.modelPath)
        const count = (displayNameCounts.get(baseName) ?? 0) + 1
        displayNameCounts.set(baseName, count)
        const displayName = count === 1 ? baseName : `${baseName} (${count})`
        const targetGpuIds = instance.gpuIds.length > 0 ? instance.gpuIds : Array.from(memoryByGpuMb.keys())

        for (const gpuId of targetGpuIds) {
          if (!modelsByGpu.has(gpuId)) {
            modelsByGpu.set(gpuId, [])
          }
          modelsByGpu.get(gpuId)!.push({
            model_path: instance.modelPath,
            instance_id: instance.id,
            display_name:
              instance.tensorParallelSize > 1 && targetGpuIds.length > 1
                ? `${displayName} (TP)`
                : displayName,
            gpu_memory_gb: (memoryByGpuMb.get(gpuId) ?? 0) / 1024,
            color: getModelColor(instance.id, index),
            is_sleeping: instance.status === 'sleeping',
          })
        }
      }

      // Map IPC segment usage (used/prealloc) to GPUs
      // For multi-GPU segments (e.g., kvcached_vllm_GPU0_GPU1), split evenly across GPUs
      const ipcUsageByGpu = new Map<number, { used_gb: number; prealloc_gb: number }>()
      for (const segment of kvcacheSegments) {
        if (segment.gpu_indices && segment.gpu_indices.length > 0) {
          const gpuCount = segment.gpu_indices.length
          const usedPerGpu = segment.used_size / 1024 ** 3 / gpuCount
          const preallocPerGpu = segment.prealloc_size / 1024 ** 3 / gpuCount

          for (const gpuId of segment.gpu_indices) {
            const existing = ipcUsageByGpu.get(gpuId) ?? { used_gb: 0, prealloc_gb: 0 }
            ipcUsageByGpu.set(gpuId, {
              used_gb: existing.used_gb + usedPerGpu,
              prealloc_gb: existing.prealloc_gb + preallocPerGpu,
            })
          }
        }
      }

      // Build per-GPU response with KVCache metrics
      const gpus: PerGpuMetrics[] = nvidiaSmiInfo.gpus.map((gpu) => {
        const models = modelsByGpu.get(gpu.index) ?? []
        const ipcUsage = ipcUsageByGpu.get(gpu.index)

        // KVCache Total = Free GPU Memory + Prealloc KVCache Memory
        // Prealloc appears as "used" in nvidia-smi but is actually available to KVCache
        // For multi-GPU segments, prealloc is already split evenly across GPUs above
        const gpuFreeGb = (gpu.memoryTotalMB - gpu.memoryUsedMB) / 1024
        const preallocGb = ipcUsage?.prealloc_gb ?? 0
        const usedGb = ipcUsage?.used_gb ?? 0
        const kvcacheTotalGb = gpuFreeGb + preallocGb + usedGb
        const kvcacheFreeGb = Math.max(0, kvcacheTotalGb - usedGb - preallocGb)

        // Build per-GPU KVCache metrics if there's any IPC usage on this GPU
        const kvcache: KVCacheMetrics | undefined = ipcUsage
          ? {
              total_gb: kvcacheTotalGb,
              prealloc_gb: preallocGb,
              used_gb: usedGb,
              free_gb: kvcacheFreeGb,
            }
          : undefined

        const assignedModelGb = models.reduce((sum, model) => sum + model.gpu_memory_gb, 0)
        const zeroMemoryModels = models.filter((model) => model.gpu_memory_gb <= 0)
        const unattributedModelGb = Math.max(
          0,
          gpu.memoryUsedMB / 1024 - assignedModelGb - preallocGb - usedGb
        )
        if (zeroMemoryModels.length > 0 && unattributedModelGb > 0) {
          const fallbackGb = unattributedModelGb / zeroMemoryModels.length
          for (const model of zeroMemoryModels) {
            model.gpu_memory_gb = fallbackGb
          }
        }

        return {
          gpu_index: gpu.index,
          name: gpu.name,
          total_gb: gpu.memoryTotalMB / 1024,
          used_gb: gpu.memoryUsedMB / 1024,
          free_gb: gpuFreeGb,
          utilization_percent: parseFloat(gpu.gpuUtilization.replace('%', '')) || 0,
          models,
          kvcache,
        }
      })

      // Global summary = sum of per-GPU metrics (for backward compatibility)
      const globalKvcache: KVCacheMetrics = {
        total_gb: gpus.reduce((sum, g) => sum + (g.kvcache?.total_gb ?? 0), 0),
        prealloc_gb: gpus.reduce((sum, g) => sum + (g.kvcache?.prealloc_gb ?? 0), 0),
        used_gb: gpus.reduce((sum, g) => sum + (g.kvcache?.used_gb ?? 0), 0),
        free_gb: gpus.reduce((sum, g) => sum + (g.kvcache?.free_gb ?? 0), 0),
      }

      return {
        gpus,
        kvcache: globalKvcache,
        total_system_free_gb: gpus.reduce((sum, g) => sum + g.free_gb, 0),
        is_virtual_gpu_mode: config.virtualGpuCount > 0 || isInferenceSimMode(),
      }
    } catch (err) {
      this.logger.error({ err }, 'Failed to get multi-GPU memory usage')
      throw new InternalError(
        `Failed to get multi-GPU memory usage: ${err instanceof Error ? err.message : 'Unknown error'}`
      )
    }
  }

  /**
   * Set memory limits for a model via kvctl
   */
  async setMemoryLimits(modelPath: string, limitGb: number): Promise<void> {
    const instance = modelStore.get(modelPath)
    if (!instance) {
      throw new InternalError(`Model ${modelPath} not found`)
    }

    const segmentName = instance.ipcSegmentName

    try {
      await this.runKvctlSetLimit(segmentName, limitGb)
      this.logger.info({ modelPath, segmentName, limitGb }, 'Memory limit set successfully')
    } catch (err) {
      this.logger.error({ modelPath, err }, 'Failed to set memory limit')
      throw new InternalError(
        `Failed to set memory limit: ${err instanceof Error ? err.message : 'Unknown error'}`
      )
    }
  }

  /**
   * Collect resource metrics for a specific model
   */
  async collectMetrics(modelPath: string): Promise<ResourceMetrics | undefined> {
    const instance = modelStore.get(modelPath)
    if (!instance) {
      return undefined
    }

    try {
      const memoryUsage = await this.getMemoryUsage()
      const modelMemory = memoryUsage.models.find((m) => m.model_path === modelPath)

      if (!modelMemory) {
        return undefined
      }

      // Calculate usage percentage based on model's footprint relative to total GPU
      const usagePercent =
        memoryUsage.gpu.total_gb > 0
          ? (modelMemory.gpu_memory_gb / memoryUsage.gpu.total_gb) * 100
          : 0

      const metrics: ResourceMetrics = {
        modelPath,
        gpuMemoryUsedGB: modelMemory.gpu_memory_gb,
        gpuMemoryLimitGB: memoryUsage.gpu.total_gb, // Use total GPU as limit for now
        gpuMemoryUsagePercent: usagePercent,
        activeConnections: 0, // Will be tracked by ProxyRouter
        totalRequests: 0, // Will be tracked by ProxyRouter
        successfulRequests: 0,
        failedRequests: 0,
        lastUpdated: new Date(),
      }

      // Cache metrics
      this.metricsCache.set(modelPath, metrics)

      return metrics
    } catch (err) {
      this.logger.error({ modelPath, err }, 'Failed to collect metrics')
      return undefined
    }
  }

  /**
   * Run kvctl limit command
   */
  private async runKvctlSetLimit(segmentName: string, limitGb: number): Promise<void> {
    return new Promise((resolve, reject) => {
      const proc = spawn('kvctl', ['limit', segmentName, `${limitGb}G`])
      let stderr = ''

      proc.stderr?.on('data', (data) => {
        stderr += data.toString()
      })

      proc.on('exit', (code) => {
        if (code !== 0) {
          reject(new Error(`kvctl limit failed with code ${code}: ${stderr}`))
        } else {
          resolve()
        }
      })

      proc.on('error', (err) => {
        reject(err)
      })
    })
  }

  /**
   * Run Python script to read KVCache stats from shared memory (like kvtop)
   * Returns empty array if script fails or no segments found
   */
  private async runKvcacheStats(): Promise<KvcacheSegment[]> {
    return new Promise((resolve) => {
      // Script is in scripts/ relative to src/services/
      const scriptPath = path.join(__dirname, '../../scripts/kvcache-stats.py')
      const proc = spawn('python3', [scriptPath])
      let stdout = ''
      let stderr = ''

      proc.stdout?.on('data', (data) => {
        stdout += data.toString()
      })

      proc.stderr?.on('data', (data) => {
        stderr += data.toString()
      })

      proc.on('exit', (code) => {
        if (code !== 0) {
          this.logger.warn({ code, stderr }, 'kvcache-stats.py failed')
          resolve([])
        } else {
          try {
            const segments = JSON.parse(stdout.trim()) as KvcacheSegment[]
            this.logger.debug(
              { segmentCount: segments.length },
              'Got KVCache segments from Python script'
            )
            resolve(segments)
          } catch (err) {
            this.logger.warn(
              { err, stdout: stdout.substring(0, 200) },
              'Failed to parse kvcache-stats.py output'
            )
            resolve([])
          }
        }
      })

      proc.on('error', (err) => {
        this.logger.warn({ err }, 'Failed to spawn kvcache-stats.py')
        resolve([])
      })
    })
  }

  /**
   * Get cached metrics
   */
  getCachedMetrics(modelPath: string): ResourceMetrics | undefined {
    return this.metricsCache.get(modelPath)
  }
}
