import { spawn, ChildProcess } from 'child_process'
import { EventEmitter } from 'events'
import { randomUUID } from 'crypto'
import { hostname } from 'node:os'
import type { ModelInstance, ModelStatus, ClusterEvent } from '@sardeenz/types'
import path from 'path'
import { fileURLToPath } from 'url'
import { modelStore } from '../stores/model-store.js'
import { config, isInferenceSimMode } from '../config.js'
import {
  getNextPort,
  killProcessImmediate,
  killProcessGracefully,
  isProcessRunning,
  getDescendantPids,
  findVllmProcessesByPort,
  findProcessesByEnvMarker,
} from '../utils/process.js'
import { NotFoundError, InternalError, ConflictError } from '../utils/errors.js'
import { buildErrorMessage } from '../utils/error-parser.js'
import { parseMemoryMetrics, extractEngineCorePid } from '../utils/memory-parser.js'
import { LoadProgressTracker } from '../utils/load-progress-tracker.js'
import { getNvidiaSmiInfo } from '../utils/gpu-info.js'
import { simGpuTracker } from '../utils/sim-gpu-tracker.js'
import { estimateModelMemory } from '../utils/model-memory-estimator.js'
import type { Logger } from '@sardeenz/utils'
import { processLogBuffer } from './process-log-buffer.js'
import { eventBus } from './event-bus.js'
import { runtimeSettings } from '../stores/runtime-settings.js'
import { peerStore } from '../stores/peer-store.js'
import { GpuSelector } from './gpu-selector.js'
import { signRequest } from './cluster-auth.js'
import { metricsStore } from '../stores/metrics-store.js'
import { kvAdmissionController } from './kv-admission.js'

const __filename = fileURLToPath(import.meta.url)
const __dirname = path.dirname(__filename)

export interface LaunchModelOptions {
  modelPath: string
  maxTokens?: number
  extraArgs?: string[]
  gpuMemoryUtilization?: number
  resourceBudgetGpuCount?: number
  kvCacheMemoryBytes?: number
  enforceEager?: boolean
  routable?: boolean
  gpuIds?: number[] // Optional explicit GPU selection (auto-selects if not provided)
  topologyGpuIds?: number[]
  tensorParallelSize?: number // For large models spanning multiple GPUs (default: 1)
  pipelineParallelSize?: number
  placementMode?: 'balanced' | 'concentrated'
  placementGpuId?: number
  sourceType?: 'huggingface' | 'local' // Model source type (default: 'huggingface')
  servedModelName?: string // Name for vLLM --served-model-name (default: modelPath)
  enableSleepMode?: boolean // Enable vLLM sleep mode for GPU memory offloading
  idleSleepTimeoutSeconds?: number
  idleSleepLevel?: 1 | 2
  autoWakeOnRequest?: boolean
  loadConflictPolicy?: 'same_model_siblings' | 'sleep_idle_overlapping' | 'none'
  loadConflictIdleThresholdSeconds?: number
}

/** Arguments that are managed by the system and should be filtered out from user input */
const FORBIDDEN_ARGS = [
  '--gpu-memory-utilization',
  '--port',
  '--no-enable-prefix-caching',
  '--disable-log-requests',
  '--disable-log-stats',
]

/**
 * Sanitize user-provided vLLM arguments:
 * - Filter out empty lines
 * - Ensure args start with - or --
 * - Remove system-managed arguments (but allow overridable ones)
 */
function isForbiddenArg(arg: string): boolean {
  const argName = arg.split('=')[0].toLowerCase()
  return FORBIDDEN_ARGS.some((forbidden) => argName === forbidden.toLowerCase())
}

function sanitizeVllmArgs(args: string[]): string[] {
  const sanitizedArgs: string[] = []
  const trimmedArgs = args.map((arg) => arg.trim()).filter((arg) => arg.length > 0)

  for (let i = 0; i < trimmedArgs.length; i++) {
    const arg = trimmedArgs[i]
    if (!arg.startsWith('-') || isForbiddenArg(arg)) {
      continue
    }

    const nextArg = trimmedArgs[i + 1]
    if (!arg.includes('=') && nextArg && !nextArg.startsWith('-')) {
      sanitizedArgs.push(`${arg}=${nextArg}`)
      i++
      continue
    }

    sanitizedArgs.push(arg)
  }

  return sanitizedArgs
}

/**
 * Check if a specific argument is present in the args list
 */
function hasArg(args: string[], argName: string): boolean {
  const lowerArgName = argName.toLowerCase()
  return args.some((arg) => arg.split('=')[0].toLowerCase() === lowerArgName)
}

function hasAnyArg(args: string[], argNames: string[]): boolean {
  return argNames.some((argName) => hasArg(args, argName))
}

function getNumericArg(args: string[], argNames: string[]): number | undefined {
  for (const argName of argNames) {
    const lowerArgName = argName.toLowerCase()
    const arg = args.find((candidate) => candidate.split('=')[0].toLowerCase() === lowerArgName)
    if (!arg || !arg.includes('=')) {
      continue
    }
    const parsed = Number(arg.split('=').slice(1).join('='))
    if (Number.isFinite(parsed) && parsed > 0) {
      return Math.floor(parsed)
    }
  }
  return undefined
}

/**
 * Build IPC segment name for kvcached based on GPU(s)
 * Single GPU: kvcached_vllm_GPU0
 * Multi-GPU (tensor parallel): kvcached_vllm_GPU0_GPU1
 * Virtual GPU mode: always uses kvcached_vllm_GPU0 (all vGPUs map to physical GPU 0)
 */
function buildIpcSegmentName(gpuIds: number[]): string {
  if (config.virtualGpuCount > 0) {
    // In virtual GPU mode, all GPUs map to physical GPU 0
    return 'kvcached_vllm_GPU0'
  }
  const sortedIds = [...gpuIds].sort((a, b) => a - b)
  return `kvcached_vllm_GPU${sortedIds.join('_GPU')}`
}

function getVisibleDeviceIndex(gpuIds: number[], gpuId: number | undefined): number | undefined {
  if (gpuId === undefined) {
    return undefined
  }
  const visibleIndex = gpuIds.indexOf(gpuId)
  return visibleIndex >= 0 ? visibleIndex : undefined
}

function resolveWakeTargetDevice(options: {
  gpuIds: number[]
  placementMode?: 'balanced' | 'concentrated'
  placementGpuId?: number
}): string | undefined {
  const { gpuIds, placementMode, placementGpuId } = options
  if (placementMode === 'balanced') {
    return 'local'
  }
  if (placementMode === 'concentrated') {
    const visibleIndex = getVisibleDeviceIndex(gpuIds, placementGpuId ?? gpuIds[0])
    if (visibleIndex === undefined) {
      throw new InternalError(
        `placement_gpu_id=${placementGpuId} must be inside execution GPUs ${gpuIds.join(', ')}`
      )
    }
    return `cuda:${visibleIndex}`
  }
  return undefined
}

interface LaunchRecoveryStep {
  instanceId: string
  wakeOnFailure: boolean
  restoreRoutable: boolean
}

export class ModelManager extends EventEmitter {
  private logger: Logger
  // Keyed by instance ID (UUID) for multi-instance support
  private processes: Map<string, ChildProcess> = new Map()
  private launchRecoveryPlans: Map<string, LaunchRecoveryStep[]> = new Map()
  private gpuSelector: GpuSelector

  constructor(logger: Logger) {
    super()
    this.logger = logger.child({ component: 'ModelManager' })
    this.gpuSelector = new GpuSelector(logger)
  }

  /**
   * Broadcast a ClusterEvent to all peers (fire-and-forget).
   * Only sends when in cluster mode (CLUSTER_PEERS or K8s env detected).
   */
  private broadcastClusterEvent(event: ClusterEvent): void {
    const isClusterMode = !!(process.env.KUBERNETES_SERVICE_HOST || config.clusterPeers)
    if (!isClusterMode) return

    const localPodId = hostname()
    const peers = peerStore.getAllPeers().filter((p) => p.podId !== localPodId)
    if (peers.length === 0) return

    const path = '/internal/cluster/event'
    const body = JSON.stringify(event)
    const headers: Record<string, string> = { 'Content-Type': 'application/json' }

    if (config.clusterSecret) {
      const { signature, timestamp } = signRequest('POST', path, body, config.clusterSecret)
      headers['x-cluster-signature'] = signature
      headers['x-cluster-timestamp'] = String(timestamp)
    }

    for (const peer of peers) {
      fetch(`http://${peer.address}:${peer.port}${path}`, {
        method: 'POST',
        headers,
        body,
        signal: AbortSignal.timeout(5_000),
      }).catch((err) => {
        this.logger.debug({ err, podId: peer.podId }, 'Failed to broadcast cluster event')
      })
    }
  }

  /**
   * Launch a new model instance
   * Supports multiple instances of the same model (FR-004)
   * GPU memory is managed by kvcached
   */
  async launchModel(options: LaunchModelOptions): Promise<ModelInstance> {
    const {
      modelPath,
      maxTokens = 4096,
      extraArgs = [],
      gpuMemoryUtilization,
      resourceBudgetGpuCount,
      kvCacheMemoryBytes,
      enforceEager = false,
      routable = true,
      gpuIds,
      topologyGpuIds,
      tensorParallelSize = 1,
      pipelineParallelSize,
      placementMode = 'balanced',
      placementGpuId,
      servedModelName,
      enableSleepMode = false,
      idleSleepTimeoutSeconds,
      idleSleepLevel = 1,
      autoWakeOnRequest = true,
      loadConflictPolicy = 'same_model_siblings',
      loadConflictIdleThresholdSeconds,
    } = options

    // Use explicit servedModelName if provided, otherwise fall back to modelPath
    const effectiveModelName = servedModelName?.trim() || modelPath

    // Sanitize user-provided extra arguments
    const sanitizedExtraArgs = sanitizeVllmArgs(extraArgs)
    const effectivePipelineParallelSize = Math.max(
      pipelineParallelSize ??
        getNumericArg(sanitizedExtraArgs, ['--pipeline-parallel-size', '-pp']) ??
        1,
      1
    )

    let targetGpuIds: number[]
    let effectiveTopologyGpuIds: number[]
    let wasAutoSelected: boolean
    let effectiveTensorParallelSize: number
    let effectivePlacementGpuId: number | undefined

    if (placementMode === 'concentrated') {
      const requestedTopologyGpuIds =
        topologyGpuIds && topologyGpuIds.length > 0 ? topologyGpuIds : gpuIds

      if (requestedTopologyGpuIds && requestedTopologyGpuIds.length > 0) {
        effectiveTopologyGpuIds = await this.gpuSelector.validateGpuIds(requestedTopologyGpuIds)
        wasAutoSelected = false
      } else {
        const autoSelection = await this.gpuSelector.getTargetGpus(
          undefined,
          Math.max(tensorParallelSize, 1) * effectivePipelineParallelSize
        )
        effectiveTopologyGpuIds = autoSelection.gpuIds
        wasAutoSelected = autoSelection.wasAutoSelected
      }

      effectivePlacementGpuId = placementGpuId ?? effectiveTopologyGpuIds[0]
      if (!effectiveTopologyGpuIds.includes(effectivePlacementGpuId)) {
        throw new InternalError(
          `placement_gpu_id=${effectivePlacementGpuId} must be inside topology GPU domain ${effectiveTopologyGpuIds.join(', ')}`
        )
      }

      effectiveTensorParallelSize = Math.max(tensorParallelSize, 1)
      const requiredExecutionGpuCount = effectiveTensorParallelSize * effectivePipelineParallelSize
      if (effectiveTopologyGpuIds.length < requiredExecutionGpuCount) {
        throw new InternalError(
          `placement topology has ${effectiveTopologyGpuIds.length} GPU(s), but tensor_parallel_size=${effectiveTensorParallelSize} and pipeline_parallel_size=${effectivePipelineParallelSize} require ${requiredExecutionGpuCount}`
        )
      }

      targetGpuIds = effectiveTopologyGpuIds
    } else {
      const requestedExecutionGpuIds =
        topologyGpuIds && topologyGpuIds.length > 0 ? topologyGpuIds : gpuIds
      const selection = await this.gpuSelector.getTargetGpus(
        requestedExecutionGpuIds,
        Math.max(tensorParallelSize, 1) * effectivePipelineParallelSize
      )
      targetGpuIds = selection.gpuIds
      effectiveTopologyGpuIds = targetGpuIds
      wasAutoSelected = selection.wasAutoSelected
      effectiveTensorParallelSize = Math.max(tensorParallelSize, 1)
    }

    const executionGpuCount = Math.max(targetGpuIds.length, effectiveTensorParallelSize, 1)
    const derivedGpuMemoryUtilization =
      gpuMemoryUtilization ??
      (resourceBudgetGpuCount !== undefined ? resourceBudgetGpuCount / executionGpuCount : undefined)

    if (derivedGpuMemoryUtilization !== undefined && derivedGpuMemoryUtilization > 1) {
      throw new InternalError(
        `Derived gpu_memory_utilization=${derivedGpuMemoryUtilization.toFixed(3)} exceeds 1.0. Reduce resource_budget_gpu_count or execution GPUs.`
      )
    }

    // Determine kvcached status
    // kvcached supports tensor parallelism as of Q2 2025
    // inference-sim mode never uses kvcached
    const enableKvcached = isInferenceSimMode() ? false : config.enableKvcached

    this.assertKvcachedStartupBudget({
      targetGpuIds,
      kvCacheMemoryBytes,
      enableKvcached,
    })
    await this.assertNoUntrackedGpuProcesses(targetGpuIds, enableKvcached)
    await this.deleteStaleKvcachedIpcSegment(targetGpuIds, enableKvcached, 'pre-load')
    await this.trimOverlappingKvcachedInstances(targetGpuIds, enableKvcached)

    this.logger.info(
      {
        modelPath,
        extraArgs: sanitizedExtraArgs,
        targetGpuIds,
        topologyGpuIds: effectiveTopologyGpuIds,
        tensorParallelSize: effectiveTensorParallelSize,
        pipelineParallelSize: effectivePipelineParallelSize,
        placementMode,
        placementGpuId: effectivePlacementGpuId,
        enableKvcached,
        wasAutoSelected,
        resourceBudgetGpuCount,
        derivedGpuMemoryUtilization,
        idleSleepTimeoutSeconds,
        autoWakeOnRequest,
        loadConflictPolicy,
        loadConflictIdleThresholdSeconds,
      },
      'Launching model with GPU selection'
    )

    // Get next available port
    const usedPorts = modelStore.getUsedPorts()
    const port = getNextPort(config.vllmBasePort, usedPorts)

    // Create instance record with unique ID
    const instanceId = randomUUID()
    const instance: ModelInstance = {
      id: instanceId,
      modelPath,
      modelName: effectiveModelName,
      status: 'starting' as ModelStatus,
      port,
      processId: 0, // Will be set after spawn
      maxTokens,
      gpuMemoryUtilization: derivedGpuMemoryUtilization ?? 0,
      loadedAt: new Date(),
      ipcSegmentName: this.getIpcSegmentName(modelPath, instanceId),
      gpuIds: targetGpuIds,
      topologyGpuIds: effectiveTopologyGpuIds,
      placementMode,
      placementGpuId: effectivePlacementGpuId,
      tensorParallelSize: effectiveTensorParallelSize,
      kvcachedEnabled: enableKvcached,
      memoryBaselineByGpu: {}, // Will be populated when model becomes ready
      sleepModeEnabled: enableSleepMode,
      resourceBudgetGpuCount,
      kvCacheMemoryBytes,
      idleSleepTimeoutSeconds,
      idleSleepLevel,
      autoWakeOnRequest,
      loadConflictPolicy,
      loadConflictIdleThresholdSeconds,
      lastActivityAt: new Date(),
      routable,
    }

    try {
      await this.prepareLaunchContext(instance)

      let spawnBinary: string
      let allArgs: string[]
      let spawnEnv: NodeJS.ProcessEnv
      const hfToken = runtimeSettings.getHfToken()

      if (isInferenceSimMode()) {
        spawnBinary = config.inferenceSimBinary
        allArgs = [
          '--model', `local/${modelPath}`,
          '--port', String(port),
          '--served-model-name', effectiveModelName,
          '--max-model-len', String(maxTokens),
          '--startup-duration', config.simStartupDuration,
          '--time-to-first-token', '50ms',
          '--inter-token-latency', '15ms',
          '--mode', 'echo',
        ]
        if (enableSleepMode) {
          allArgs.push('--enable-sleep-mode')
        }

        spawnEnv = {
          ...process.env,
          SARDEENZ_INSTANCE_ID: instanceId,
          ...(hfToken ? { HF_TOKEN: hfToken } : {}),
          ...(enableSleepMode ? { VLLM_SERVER_DEV_MODE: '1' } : {}),
        }
        instance.ipcSegmentName = ''
        instance.launchCommand = `${spawnBinary} ${allArgs.join(' ')}`
      } else {
        const baseArgs = ['serve', modelPath, '--disable-log-stats', `--port=${port}`]

        if (!hasArg(sanitizedExtraArgs, '--served-model-name')) {
          baseArgs.push(`--served-model-name=${effectiveModelName}`)
        }
        if (!hasArg(sanitizedExtraArgs, '--max-model-len')) {
          baseArgs.push(`--max-model-len=${maxTokens}`)
        }
        if (derivedGpuMemoryUtilization !== undefined) {
          baseArgs.push(`--gpu-memory-utilization=${derivedGpuMemoryUtilization}`)
        }
        if (kvCacheMemoryBytes !== undefined) {
          baseArgs.push(`--kv-cache-memory-bytes=${kvCacheMemoryBytes}`)
        }
        if (effectiveTensorParallelSize > 1) {
          baseArgs.push(`--tensor-parallel-size=${effectiveTensorParallelSize}`)
        }
        if (
          effectivePipelineParallelSize > 1 &&
          !hasAnyArg(sanitizedExtraArgs, ['--pipeline-parallel-size', '-pp'])
        ) {
          baseArgs.push(`--pipeline-parallel-size=${effectivePipelineParallelSize}`)
        }
        if (enableKvcached && config.kvcachedDisablePrefixCaching) {
          baseArgs.push('--no-enable-prefix-caching')
        }
        if (enableSleepMode) {
          baseArgs.push('--enable-sleep-mode')
        }
        if (enforceEager) {
          baseArgs.push('--enforce-eager')
        }

        const vllmArgs = [...baseArgs, ...sanitizedExtraArgs]
        const kvcachedIpcName = enableKvcached ? buildIpcSegmentName(targetGpuIds) : undefined
        const cudaVisibleDevices = config.virtualGpuCount > 0 ? '0' : targetGpuIds.join(',')
        if (config.virtualGpuCount > 0) {
          this.logger.info(
            { virtualGpuIds: targetGpuIds, physicalGpu: 0 },
            'Virtual GPU mode: mapping to physical GPU 0'
          )
        }

        const vllmEntrypointPath = path.join(__dirname, '../../scripts/vllm-entrypoint.py')
        const envVars = [`CUDA_VISIBLE_DEVICES=${cudaVisibleDevices}`]
        if (kvcachedIpcName) {
          envVars.push(`KVCACHED_IPC_NAME=${kvcachedIpcName}`)
        }
        const keepMappedKvcachedEnv =
          enableKvcached && config.kvcachedKeepMappedPages
            ? {
                KVCACHED_DEFER_FREE_UNMAP: '1',
                KVCACHED_MIN_RESERVED_PAGES: String(config.kvcachedKeepMappedReservedPages),
                KVCACHED_MAX_RESERVED_PAGES: String(config.kvcachedKeepMappedReservedPages),
                ...(config.kvcachedSkipPhysicalFreeCheck
                  ? { KVCACHED_SKIP_PHYSICAL_FREE_CHECK: '1' }
                  : {}),
              }
            : {}
        for (const [key, value] of Object.entries(keepMappedKvcachedEnv)) {
          envVars.push(`${key}=${value}`)
        }

        spawnBinary = 'python3'
        allArgs = [vllmEntrypointPath, ...vllmArgs]
        spawnEnv = {
          ...process.env,
          SARDEENZ_INSTANCE_ID: instanceId,
          CUDA_VISIBLE_DEVICES: cudaVisibleDevices,
          ENABLE_KVCACHED: enableKvcached ? 'true' : 'false',
          KVCACHED_AUTOPATCH: config.kvcachedAutopatch && enableKvcached ? '1' : '0',
          ...(kvcachedIpcName ? { KVCACHED_IPC_NAME: kvcachedIpcName } : {}),
          ...keepMappedKvcachedEnv,
          ...(hfToken ? { HF_TOKEN: hfToken } : {}),
          ...(enableSleepMode ? { VLLM_SERVER_DEV_MODE: '1' } : {}),
        }
        instance.ipcSegmentName = kvcachedIpcName ?? ''
        instance.launchCommand = `${envVars.join(' ')} python3 ${vllmEntrypointPath} ${vllmArgs.join(' ')}`
      }

      const proc = spawn(spawnBinary, allArgs, {
        env: spawnEnv,
        stdio: ['ignore', 'pipe', 'pipe'],
      })

      const backendLabel = isInferenceSimMode() ? 'inference-sim' : 'vLLM'

      if (!proc.pid) {
        throw new InternalError(`Failed to spawn ${backendLabel} process`)
      }

      instance.processId = proc.pid

      // Store process reference by instance ID
      this.processes.set(instanceId, proc)

      // Store instance
      modelStore.set(instance)

      // Emit status event for starting state
      eventBus.emitEvent(
        eventBus.createStatusEvent(
          instanceId,
          null,
          'starting' as ModelStatus,
          'Initializing model load'
        )
      )

      // Set up logging for process output
      // Log at 'info' level during startup so output is always visible
      proc.stdout?.on('data', (data) => {
        const output = data.toString().trim()
        processLogBuffer.append(instanceId, 'stdout', output)
        this.logger.info({ modelPath, instanceId, output }, `${backendLabel} stdout`)
      })

      proc.stderr?.on('data', (data) => {
        const output = data.toString().trim()
        processLogBuffer.append(instanceId, 'stderr', output)
        this.logger.info({ modelPath, instanceId, output }, `${backendLabel} stderr`)
      })

      // Handle process exit
      proc.on('exit', (code, signal) => {
        this.logger.info({ modelPath, instanceId, code, signal }, `${backendLabel} process exited`)
        this.handleProcessExit(instanceId, code, signal)
      })

      proc.on('error', (err) => {
        this.logger.error({ modelPath, instanceId, err }, `${backendLabel} process error`)
        this.handleProcessError(instanceId, err)
      })

      // Start background monitoring for model readiness (don't await)
      // This allows the API to return immediately so frontend can subscribe to SSE
      this.monitorModelStartup(instanceId, port, modelPath).catch((err) => {
        this.logger.error({ instanceId, err }, 'Background model monitoring failed')
      })

      // Return immediately with 'starting' status
      // Frontend can now subscribe to SSE and receive real-time updates
      return instance
    } catch (err) {
      // Clean up on spawn failure (before monitoring starts)
      instance.status = 'failed' as ModelStatus
      instance.errorMessage = err instanceof Error ? err.message : 'Unknown error'
      modelStore.set(instance)

      await this.restoreLaunchContext(instanceId, 'spawn failure')

      // Emit status event for failed state
      eventBus.emitEvent(
        eventBus.createStatusEvent(
          instanceId,
          'starting' as ModelStatus,
          'failed' as ModelStatus,
          undefined,
          instance.errorMessage
        )
      )

      this.logger.error({ modelPath, instanceId, err }, 'Failed to launch model')

      // Kill process if it exists
      const proc = this.processes.get(instanceId)
      if (proc && proc.pid && isProcessRunning(proc.pid)) {
        proc.kill('SIGKILL')
      }
      await this.cleanupFailedInstanceResources(instance, 'spawn failure')
      this.processes.delete(instanceId)
      kvAdmissionController.releaseForInstance(instanceId, this.logger)

      throw err
    }
  }

  /**
   * Launch a model and wait for it to reach terminal status (running or failed).
   * Use this for sequential loading where you need to wait for one model
   * to fully load before starting another (e.g., configuration restore).
   */
  async launchModelAndWait(options: LaunchModelOptions): Promise<ModelInstance> {
    const instance = await this.launchModel(options)

    return new Promise((resolve, reject) => {
      const checkStatus = () => {
        const current = modelStore.get(instance.id)
        if (!current) {
          reject(new Error('Model instance disappeared during loading'))
          return
        }
        if (current.status === 'running') {
          resolve(current)
        } else if (current.status === 'failed') {
          reject(new Error(current.errorMessage || 'Model failed to load'))
        } else {
          // Still starting, check again in 1 second
          setTimeout(checkStatus, 1000)
        }
      }
      checkStatus()
    })
  }

  /**
   * Monitor model startup in background and update status when ready
   * Called without await so API can return immediately
   */
  private async monitorModelStartup(
    instanceId: string,
    port: number,
    modelPath: string
  ): Promise<void> {
    // Set up progress tracking before waiting for ready
    const progressTracker = new LoadProgressTracker()
    let lastProgress = 0

    // Emit initial progress event
    eventBus.emitEvent(
      eventBus.createProgressEvent(instanceId, 'loading', 0, 'Initializing model load...')
    )

    // Subscribe to log events for real-time milestone detection
    const unsubscribe = processLogBuffer.onLog(instanceId, (entry) => {
      const milestoneProgress = progressTracker.processLogLine(entry.content)
      if (milestoneProgress !== undefined && milestoneProgress > lastProgress) {
        const message = LoadProgressTracker.getProgressMessage(milestoneProgress)
        eventBus.emitEvent(
          eventBus.createProgressEvent(instanceId, 'loading', milestoneProgress, message)
        )
        lastProgress = milestoneProgress
      }
    })

    // Process existing log buffer to catch milestones that already fired
    const existingLogs = processLogBuffer.getBuffer(instanceId)
    if (existingLogs.length > 0) {
      const catchUpProgress = progressTracker.processExistingLogs(existingLogs)
      if (catchUpProgress !== undefined && catchUpProgress > lastProgress) {
        const message = LoadProgressTracker.getProgressMessage(catchUpProgress)
        eventBus.emitEvent(
          eventBus.createProgressEvent(instanceId, 'loading', catchUpProgress, message)
        )
        lastProgress = catchUpProgress
      }
    }

    try {
      // Wait for model to be ready (configurable via VLLM_STARTUP_TIMEOUT)
      await this.waitForReady(port, modelPath, config.vllmStartupTimeout)

      // Emit final progress event for ready state
      eventBus.emitEvent(
        eventBus.createProgressEvent(instanceId, 'ready', 100, 'Model ready for inference')
      )

      // Clean up progress subscription
      unsubscribe()

      this.launchRecoveryPlans.delete(instanceId)

      // Get current instance state
      const instance = modelStore.get(instanceId)
      if (!instance) {
        this.logger.warn({ instanceId }, 'Instance not found after model ready')
        return
      }

      // Update status to running
      instance.status = 'running' as ModelStatus
      instance.readyAt = new Date()

      if (isInferenceSimMode()) {
        // inference-sim mode: estimate memory and allocate on simulated GPUs
        const estimate = estimateModelMemory(modelPath, config.simModelMemoryGB)
        const estimatedMemoryMB = estimate.estimatedMemoryGB * 1024

        for (const gpuIndex of instance.gpuIds) {
          const perGpuMemoryMB =
            instance.gpuIds.length > 1
              ? Math.ceil(estimatedMemoryMB / instance.gpuIds.length)
              : estimatedMemoryMB
          simGpuTracker.allocate(gpuIndex, instanceId, perGpuMemoryMB)
        }

        // Set memory fields from estimate
        const gpuInfo = simGpuTracker.getNvidiaSmiInfo()
        const gpuTotalMB = gpuInfo.gpus[0]?.memoryTotalMB ?? config.simGpuMemoryGB * 1024
        instance.gpuMemoryUtilization = gpuTotalMB > 0 ? estimatedMemoryMB / gpuTotalMB : 0

        const memoryByGpu: Record<number, number> = {}
        for (const gpuIndex of instance.gpuIds) {
          const perGpuGB =
            instance.gpuIds.length > 1
              ? estimate.estimatedMemoryGB / instance.gpuIds.length
              : estimate.estimatedMemoryGB
          memoryByGpu[gpuIndex] = perGpuGB
        }
        instance.memoryBaselineByGpu = memoryByGpu
        instance.hasChatTemplate = true

        this.logger.info(
          {
            instanceId,
            modelPath,
            estimatedMemoryGB: estimate.estimatedMemoryGB,
            source: estimate.source,
            detectedSizeB: estimate.detectedSizeB,
          },
          'Allocated simulated GPU memory for inference-sim model'
        )
      } else {
        // vLLM mode: extract EngineCore PID, query NVML, parse logs

        const logs = processLogBuffer.getBuffer(instanceId)

        const engineCorePid = extractEngineCorePid(logs)
        if (engineCorePid) {
          instance.engineCorePid = engineCorePid
          this.logger.info(
            { instanceId, modelPath, engineCorePid, apiServerPid: instance.processId },
            'Extracted EngineCore PID from vLLM logs'
          )
        } else {
          this.logger.warn(
            { instanceId, modelPath },
            'Could not extract EngineCore PID from vLLM logs, will use main process PID for GPU memory lookup'
          )
        }

        let actualGpuMemoryGiB: number | undefined
        try {
          const gpuInfo = await getNvidiaSmiInfo()

          const primaryPid = instance.engineCorePid ?? instance.processId
          const descendantPids = await getDescendantPids(primaryPid)
          const allPids = new Set([primaryPid, ...descendantPids])

          if (instance.engineCorePid) {
            allPids.add(instance.processId)
            const apiServerDescendants = await getDescendantPids(instance.processId)
            for (const pid of apiServerDescendants) {
              allPids.add(pid)
            }
          }

          const markedPids = await findProcessesByEnvMarker('SARDEENZ_INSTANCE_ID', instance.id)
          for (const pid of markedPids) {
            allPids.add(pid)
          }

          let totalGpuMemoryMB = 0
          for (const proc of gpuInfo.processes) {
            if (allPids.has(proc.pid)) {
              totalGpuMemoryMB += proc.gpuMemoryMB
            }
          }

          if (totalGpuMemoryMB > 0 && gpuInfo.gpus.length > 0) {
            const gpuTotalMB = gpuInfo.gpus[0].memoryTotalMB
            instance.gpuMemoryUtilization = gpuTotalMB > 0 ? totalGpuMemoryMB / gpuTotalMB : 0
            actualGpuMemoryGiB = totalGpuMemoryMB / 1024

            this.logger.info(
              {
                modelPath,
                instanceId,
                engineCorePid: instance.engineCorePid,
                processId: instance.processId,
                matchedPids: Array.from(allPids),
                totalGpuMemoryMB,
                actualGpuMemoryGiB,
                gpuTotalMB,
                gpuUtilization: instance.gpuMemoryUtilization,
              },
              'Got actual GPU memory usage from NVML'
            )
          } else {
            this.logger.warn(
              {
                modelPath,
                instanceId,
                engineCorePid: instance.engineCorePid,
                processId: instance.processId,
                searchedPids: Array.from(allPids),
              },
              'No matching processes found in NVML output'
            )
          }

          const memoryByGpu: Record<number, number> = {}
          for (const proc of gpuInfo.processes) {
            if (allPids.has(proc.pid) && instance.gpuIds.includes(proc.gpu)) {
              memoryByGpu[proc.gpu] = (memoryByGpu[proc.gpu] ?? 0) + proc.gpuMemoryMB / 1024
            }
          }
          instance.memoryBaselineByGpu = memoryByGpu
          this.logger.info(
            { instanceId, modelPath, memoryBaselineByGpu: memoryByGpu },
            'Captured memory baseline per GPU'
          )
        } catch (err) {
          this.logger.warn({ modelPath, instanceId, err }, 'Failed to get GPU memory from NVML')
        }

        const memoryMetrics = parseMemoryMetrics(logs, instance.maxTokens, actualGpuMemoryGiB)
        if (memoryMetrics) {
          instance.memoryMetrics = memoryMetrics
          this.logger.info(
            { instanceId, modelPath, memoryMetrics },
            'Parsed memory metrics from vLLM logs'
          )
        } else {
          this.logger.warn(
            { instanceId, modelPath },
            'Could not parse memory metrics from vLLM logs'
          )
        }

        // Test if model supports chat templates
        try {
          const testRequest = {
            model: modelPath,
            messages: [{ role: 'user', content: 'what is the color of the sky?' }],
            max_tokens: 10,
          }

          const testResponse = await fetch(`http://localhost:${port}/v1/chat/completions`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(testRequest),
          })

          if (testResponse.status === 400) {
            const errorData = await testResponse.json()
            const errorMsg = JSON.stringify(errorData).toLowerCase()

            if (errorMsg.includes('chat template') || errorMsg.includes('chat_template')) {
              instance.hasChatTemplate = false
              this.logger.info(
                { modelPath, instanceId },
                'Model does not support chat templates (will need manual wrapping)'
              )
            } else {
              instance.hasChatTemplate = true
              this.logger.info({ modelPath, instanceId }, 'Model supports chat templates')
            }
          } else {
            instance.hasChatTemplate = true
            this.logger.info({ modelPath, instanceId }, 'Model supports chat templates')
          }
        } catch (err) {
          instance.hasChatTemplate = true
          this.logger.warn(
            { modelPath, instanceId, err },
            'Failed to test chat template support, assuming true'
          )
        }
      }

      modelStore.set(instance)

      // Emit status event for running state
      eventBus.emitEvent(
        eventBus.createStatusEvent(
          instanceId,
          'starting' as ModelStatus,
          'running' as ModelStatus,
          'Model ready for inference'
        )
      )

      this.logger.info({ modelPath, port, instanceId }, 'Model loaded successfully')
      this.emit('model:loaded', instance)

      // Broadcast cluster event to peers
      this.broadcastClusterEvent({
        type: 'model-loaded',
        podId: hostname(),
        term: 0,
        timestamp: Date.now(),
        payload: { instanceId, modelPath, modelName: instance.modelName, port },
      })
    } catch (err) {
      // Clean up progress subscription
      unsubscribe()

      await this.restoreLaunchContext(instanceId, 'startup failure')

      // Model failed to become ready
      const instance = modelStore.get(instanceId)
      if (!instance) {
        this.logger.warn({ instanceId }, 'Instance not found after model failure')
        return
      }

      instance.status = 'failed' as ModelStatus
      instance.errorMessage = err instanceof Error ? err.message : 'Unknown error'
      modelStore.set(instance)
      kvAdmissionController.releaseForInstance(instanceId, this.logger)

      // Emit status event for failed state
      eventBus.emitEvent(
        eventBus.createStatusEvent(
          instanceId,
          'starting' as ModelStatus,
          'failed' as ModelStatus,
          undefined,
          instance.errorMessage
        )
      )

      this.logger.error({ modelPath, instanceId, err }, 'Model failed to become ready')
      this.emit('model:failed', instance)

      // Kill process if it exists
      const proc = this.processes.get(instanceId)
      if (proc && proc.pid && isProcessRunning(proc.pid)) {
        proc.kill('SIGKILL')
      }
      await this.cleanupFailedInstanceResources(instance, 'startup failure')
      this.processes.delete(instanceId)
    }
  }

  private hasGpuOverlap(leftGpuIds: number[], rightGpuIds: number[]): boolean {
    const rightGpuIdSet = new Set(rightGpuIds)
    return leftGpuIds.some((gpuId) => rightGpuIdSet.has(gpuId))
  }

  private assertKvcachedStartupBudget(options: {
    targetGpuIds: number[]
    kvCacheMemoryBytes?: number
    enableKvcached: boolean
  }): void {
    const { targetGpuIds, kvCacheMemoryBytes, enableKvcached } = options
    if (!enableKvcached || !config.kvAdmissionEnabled || kvCacheMemoryBytes === undefined) {
      return
    }

    const safeLimitBytes =
      Math.min(
        config.kvAdmissionGlobalSafeLimitGiBPerGpu,
        config.kvAdmissionPoolCapacityGiBPerGpu - config.kvAdmissionGlobalReservedGiBPerGpu
      ) *
      1024 *
      1024 *
      1024

    const activeStatuses = new Set<ModelStatus>([
      'starting' as ModelStatus,
      'running' as ModelStatus,
      'sleeping' as ModelStatus,
    ])

    const existingBytes = this.listModels()
      .filter((model) => model.kvcachedEnabled)
      .filter((model) => activeStatuses.has(model.status))
      .filter((model) => this.hasGpuOverlap(this.getSchedulingGpuIds(model), targetGpuIds))
      .reduce((sum, model) => sum + (model.kvCacheMemoryBytes ?? 0), 0)

    const nextBytes = existingBytes + kvCacheMemoryBytes
    if (nextBytes <= safeLimitBytes) {
      return
    }

    throw new ConflictError(
      `KV startup budget would exceed safe limit on GPU group ${targetGpuIds.join(',')}: ` +
        `${(nextBytes / 1024 ** 3).toFixed(2)}GiB > ${(safeLimitBytes / 1024 ** 3).toFixed(2)}GiB. ` +
        'Reduce kv_cache_memory_bytes or unload/sleep an overlapping instance first.',
      'KV_STARTUP_BUDGET_EXCEEDED'
    )
  }

  private async trimOverlappingKvcachedInstances(
    targetGpuIds: number[],
    enableKvcached: boolean
  ): Promise<void> {
    if (!enableKvcached || !config.kvcachedKeepMappedPages) {
      return
    }

    const overlappingInstances = this.listModels()
      .filter((model) => model.kvcachedEnabled)
      .filter((model) => model.status === 'running')
      .filter((model) => this.hasGpuOverlap(this.getSchedulingGpuIds(model), targetGpuIds))

    for (const instance of overlappingInstances) {
      try {
        const response = await fetch(`http://127.0.0.1:${instance.port}/kvcached/trim`, {
          method: 'POST',
          signal: AbortSignal.timeout(5_000),
        })
        const payload = await response.json().catch(() => undefined)
        if (!response.ok) {
          this.logger.warn(
            {
              instanceId: instance.id,
              port: instance.port,
              status: response.status,
              payload,
            },
            'Failed to trim overlapping kvcached instance before model load'
          )
          continue
        }
        this.logger.info(
          {
            instanceId: instance.id,
            port: instance.port,
            payload,
          },
          'Trimmed overlapping kvcached reserved pages before model load'
        )
      } catch (err) {
        this.logger.warn(
          {
            instanceId: instance.id,
            port: instance.port,
            err,
          },
          'Failed to call kvcached trim before model load'
        )
      }
    }
  }

  private async assertNoUntrackedGpuProcesses(
    targetGpuIds: number[],
    enableKvcached: boolean
  ): Promise<void> {
    if (!enableKvcached || !config.kvcachedRejectUntrackedGpuProcesses) {
      return
    }

    const targetGpuIdSet = new Set(targetGpuIds)
    const trackedPids = new Set<number>()
    for (const model of this.listModels()) {
      if (!['starting', 'running', 'sleeping'].includes(model.status)) {
        continue
      }
      if (!this.hasGpuOverlap(this.getSchedulingGpuIds(model), targetGpuIds)) {
        continue
      }
      if (model.processId > 0) {
        trackedPids.add(model.processId)
        for (const pid of await getDescendantPids(model.processId)) {
          trackedPids.add(pid)
        }
      }
      if (model.engineCorePid) {
        trackedPids.add(model.engineCorePid)
        for (const pid of await getDescendantPids(model.engineCorePid)) {
          trackedPids.add(pid)
        }
      }
      for (const pid of await findProcessesByEnvMarker('SARDEENZ_INSTANCE_ID', model.id)) {
        trackedPids.add(pid)
      }
    }

    const gpuInfo = await getNvidiaSmiInfo()
    const untrackedProcesses = gpuInfo.processes.filter(
      (proc) =>
        targetGpuIdSet.has(proc.gpu) &&
        !trackedPids.has(proc.pid) &&
        proc.gpuMemoryMB >= config.kvcachedUntrackedGpuProcessMinMemoryMB
    )

    if (untrackedProcesses.length === 0) {
      return
    }

    throw new ConflictError(
      `Target GPU group ${targetGpuIds.join(',')} has untracked GPU process residue: ` +
        untrackedProcesses
          .map((proc) => `gpu${proc.gpu}:pid${proc.pid}:${proc.gpuMemoryMB}MiB`)
          .join(', ') +
        '. Clean the process or disable KVCACHED_REJECT_UNTRACKED_GPU_PROCESSES.',
      'KVCACHED_UNTRACKED_GPU_PROCESS'
    )
  }

  private hasActiveOverlappingKvcachedInstance(
    targetGpuIds: number[],
    ignoredInstanceId?: string
  ): boolean {
    return this.listModels().some(
      (model) =>
        model.id !== ignoredInstanceId &&
        model.kvcachedEnabled &&
        ['starting', 'running', 'sleeping'].includes(model.status) &&
        this.hasGpuOverlap(this.getSchedulingGpuIds(model), targetGpuIds)
    )
  }

  private async deleteKvcachedIpcSegment(segmentName: string, reason: string): Promise<boolean> {
    try {
      const proc = spawn(config.kvcachedKvctlBin, ['delete', segmentName])
      const code = await new Promise<number | null>((resolve) => {
        proc.on('exit', (exitCode) => resolve(exitCode))
        proc.on('error', () => resolve(null))
      })
      if (code === 0) {
        this.logger.info({ segmentName, reason }, 'Deleted kvcached IPC segment')
        return true
      }
    } catch {
      // Non-critical, segment may not exist or kvctl may be unavailable.
    }
    this.logger.debug({ segmentName, reason }, 'No kvcached IPC segment deleted')
    return false
  }

  private async deleteStaleKvcachedIpcSegment(
    targetGpuIds: number[],
    enableKvcached: boolean,
    reason: string,
    ignoredInstanceId?: string
  ): Promise<void> {
    if (!enableKvcached) {
      return
    }
    if (this.hasActiveOverlappingKvcachedInstance(targetGpuIds, ignoredInstanceId)) {
      return
    }
    await this.deleteKvcachedIpcSegment(buildIpcSegmentName(targetGpuIds), reason)
  }

  private getSchedulingGpuIds(instance: Pick<ModelInstance, 'gpuIds' | 'topologyGpuIds'>): number[] {
    return instance.topologyGpuIds && instance.topologyGpuIds.length > 0
      ? instance.topologyGpuIds
      : instance.gpuIds
  }

  private hasSchedulingOverlap(
    left: Pick<ModelInstance, 'gpuIds' | 'topologyGpuIds'>,
    right: Pick<ModelInstance, 'gpuIds' | 'topologyGpuIds'>
  ): boolean {
    return this.hasGpuOverlap(this.getSchedulingGpuIds(left), this.getSchedulingGpuIds(right))
  }

  private getLastRelevantActivityAt(instance: ModelInstance): Date {
    return instance.lastActivityAt ?? instance.readyAt ?? instance.loadedAt
  }

  private async prepareLaunchContext(instance: ModelInstance): Promise<void> {
    if (!instance.sleepModeEnabled) {
      return
    }

    const siblings = modelStore
      .getAllByName(instance.modelName)
      .filter(
        (current) =>
          current.id !== instance.id && this.hasGpuOverlap(current.gpuIds, instance.gpuIds)
      )

    const overlappingInstances = modelStore
      .getAll()
      .filter((current) => current.id !== instance.id && this.hasSchedulingOverlap(current, instance))

    if (siblings.length === 0 && instance.loadConflictPolicy !== 'sleep_idle_overlapping') {
      return
    }

    const startingOverlaps = overlappingInstances.filter((current) => current.status === 'starting')
    if (startingOverlaps.length > 0) {
      throw new InternalError(
        `Cannot load overlapping instance while overlapping instances are still starting: ${startingOverlaps
          .map((current) => current.id)
          .join(', ')}`
      )
    }

    const runningSiblings = siblings.filter((current) => current.status === 'running')
    const nonSleepCapableSiblings = runningSiblings.filter((current) => !current.sleepModeEnabled)
    if (nonSleepCapableSiblings.length > 0) {
      throw new InternalError(
        `Cannot safely load overlapping instance because sibling instances are not sleep-capable: ${nonSleepCapableSiblings
          .map((current) => current.id)
          .join(', ')}`
      )
    }

    const instancesToPark = new Map<string, ModelInstance>()
    for (const sibling of runningSiblings) {
      instancesToPark.set(sibling.id, sibling)
    }

    if (instance.loadConflictPolicy === 'sleep_idle_overlapping') {
      const idleThresholdMs = (instance.loadConflictIdleThresholdSeconds ?? 600) * 1000
      const idleCandidates = overlappingInstances
        .filter(
          (current) =>
            current.status === 'running' &&
            current.sleepModeEnabled &&
            !instancesToPark.has(current.id) &&
            !metricsStore.hasActiveConnections(current.id) &&
            Date.now() - this.getLastRelevantActivityAt(current).getTime() >= idleThresholdMs
        )
        .sort(
          (left, right) =>
            this.getLastRelevantActivityAt(left).getTime() -
            this.getLastRelevantActivityAt(right).getTime()
        )

      for (const candidate of idleCandidates) {
        instancesToPark.set(candidate.id, candidate)
      }
    }

    if (instancesToPark.size === 0) {
      return
    }

    const parkedInstances = Array.from(instancesToPark.values())

    const recoverySteps: LaunchRecoveryStep[] = parkedInstances.map((current) => ({
      instanceId: current.id,
      wakeOnFailure: true,
      restoreRoutable: current.routable !== false,
    }))
    this.launchRecoveryPlans.set(instance.id, recoverySteps)

    this.logger.info(
      {
        instanceId: instance.id,
        modelName: instance.modelName,
        targetGpuIds: instance.gpuIds,
        topologyGpuIds: this.getSchedulingGpuIds(instance),
        parkedSiblingIds: parkedInstances.map((current) => current.id),
      },
      'Parking overlapping instances before launch'
    )

    for (const sibling of parkedInstances) {
      modelStore.setRoutable(sibling.id, false)
    }

    for (const sibling of parkedInstances) {
      await this.waitForDrain(sibling.id, 30_000)
    }

    for (const sibling of parkedInstances) {
      await this.sleepModel(sibling.id, 1)
    }
  }

  private async restoreLaunchContext(instanceId: string, reason: string): Promise<void> {
    const recoverySteps = this.launchRecoveryPlans.get(instanceId)
    if (!recoverySteps || recoverySteps.length === 0) {
      return
    }

    this.launchRecoveryPlans.delete(instanceId)

    this.logger.warn({ instanceId, reason, recoverySteps }, 'Restoring parked sibling instances')

    for (const step of recoverySteps) {
      const current = modelStore.get(step.instanceId)
      if (!current) {
        continue
      }

      try {
        if (step.wakeOnFailure && current.status === 'sleeping') {
          await this.wakeModel(step.instanceId)
        }
      } catch (err) {
        this.logger.error(
          { instanceId, parkedInstanceId: step.instanceId, err },
          'Failed to wake parked sibling during launch recovery'
        )
      }

      if (step.restoreRoutable) {
        modelStore.setRoutable(step.instanceId, true)
      }
    }
  }

  /**
   * Unload a model instance by ID
   */
  async unloadModel(instanceId: string): Promise<void> {
    this.logger.info({ instanceId }, 'Unloading model instance')

    const instance = modelStore.get(instanceId)
    if (!instance) {
      // Try to find by path for backwards compatibility
      const byPath = modelStore.getByPath(instanceId)
      if (byPath) {
        return this.unloadModel(byPath.id)
      }
      throw new NotFoundError(`Model instance ${instanceId} not found`)
    }

    const proc = this.processes.get(instance.id)

    // If no process exists (e.g., failed model), just clean up the store
    if (!proc) {
      this.logger.info({ instanceId: instance.id }, 'No process to kill, cleaning up store entry')

      // Note: We don't delete the IPC segment here - it's shared by all models

      // Clean up logs
      processLogBuffer.clear(instance.id)

      // Remove from store
      modelStore.delete(instance.id)
      kvAdmissionController.releaseForInstance(instance.id, this.logger)

      this.logger.info(
        { instanceId: instance.id, modelPath: instance.modelPath },
        'Model entry removed'
      )
      this.emit('model:unloaded', instance)

      this.broadcastClusterEvent({
        type: 'model-unloaded',
        podId: hostname(),
        term: 0,
        timestamp: Date.now(),
        payload: { instanceId: instance.id, modelPath: instance.modelPath },
      })
      return
    }

    // Update status for active process
    instance.status = 'stopping' as ModelStatus
    modelStore.set(instance)

    try {
      if (isInferenceSimMode()) {
        // inference-sim: graceful SIGTERM shutdown, no child process cleanup needed
        await killProcessGracefully(proc)
        simGpuTracker.deallocate(instance.id)
      } else {
        // vLLM: SIGKILL to bypass Python signal handlers that would delete
        // the shared kvcached IPC segment (kvcached_mem_info)
        await killProcessImmediate(proc)

        if (instance.engineCorePid) {
          try {
            process.kill(instance.engineCorePid, 'SIGKILL')
            this.logger.debug(
              { instanceId: instance.id, engineCorePid: instance.engineCorePid },
              'Killed EngineCore process'
            )
          } catch {
            // Process may have already exited
          }
        }

        const markedPids = await findProcessesByEnvMarker('SARDEENZ_INSTANCE_ID', instance.id)
        if (markedPids.length > 0) {
          this.logger.info(
            { instanceId: instance.id, markedPids },
            'Found processes with instance marker, killing them'
          )
          for (const pid of markedPids) {
            try {
              process.kill(pid, 'SIGKILL')
            } catch {
              // Process may have already exited
            }
          }
        }

        const portPids = await findVllmProcessesByPort(instance.port)
        if (portPids.length > 0) {
          this.logger.info(
            { instanceId: instance.id, portPids, port: instance.port },
            'Found remaining vLLM processes by port, killing them'
          )
          for (const pid of portPids) {
            try {
              process.kill(pid, 'SIGKILL')
            } catch {
              // Process may have already exited
            }
          }
        }
      }

      // Remove from stores
      modelStore.delete(instance.id)
      this.processes.delete(instance.id)
      kvAdmissionController.releaseForInstance(instance.id, this.logger)

      this.logger.info(
        { instanceId: instance.id, modelPath: instance.modelPath },
        'Model unloaded successfully'
      )
      this.emit('model:unloaded', instance)

      this.broadcastClusterEvent({
        type: 'model-unloaded',
        podId: hostname(),
        term: 0,
        timestamp: Date.now(),
        payload: { instanceId: instance.id, modelPath: instance.modelPath },
      })
    } catch (err) {
      this.logger.error({ instanceId: instance.id, err }, 'Error unloading model')
      throw new InternalError(
        `Failed to unload model: ${err instanceof Error ? err.message : 'Unknown error'}`
      )
    }
  }

  /**
   * Unload a model by path (unloads first matching instance)
   * For backwards compatibility
   */
  async unloadModelByPath(modelPath: string): Promise<void> {
    const instance = modelStore.getByPath(modelPath)
    if (!instance) {
      throw new NotFoundError(`Model ${modelPath} not found`)
    }
    return this.unloadModel(instance.id)
  }

  /**
   * Get model status by instance ID
   */
  getModelStatus(instanceId: string): ModelInstance | undefined {
    return modelStore.get(instanceId)
  }

  /**
   * Get model status by path (returns first active instance)
   */
  getModelStatusByPath(modelPath: string): ModelInstance | undefined {
    return modelStore.getByPath(modelPath)
  }

  /**
   * Get all instances for a model path
   */
  getInstancesByPath(modelPath: string): ModelInstance[] {
    return modelStore.getAllByPath(modelPath)
  }

  /**
   * List all models
   */
  listModels(): ModelInstance[] {
    return modelStore.getAll()
  }

  /**
   * Put a model instance to sleep.
   * Offloads model weights to CPU RAM and frees GPU memory.
   * Requires model to have been loaded with enableSleepMode=true.
   */
  async sleepModel(instanceId: string, level: 1 | 2 = 1): Promise<void> {
    const instance = modelStore.get(instanceId)
    if (!instance) {
      throw new NotFoundError(`Model instance ${instanceId} not found`)
    }

    if (!instance.sleepModeEnabled) {
      throw new InternalError('Model was not loaded with sleep mode enabled')
    }

    if (instance.status !== 'running') {
      throw new InternalError(`Cannot sleep model: current status is ${instance.status}`)
    }

    try {
      // Call vLLM sleep endpoint
      const response = await fetch(`http://localhost:${instance.port}/sleep?level=${level}`, {
        method: 'POST',
        signal: AbortSignal.timeout(30000),
      })

      if (!response.ok) {
        const errorText = await response.text().catch(() => response.statusText)
        throw new InternalError(`Failed to put model to sleep: ${errorText}`)
      }

      // Update instance state
      const previousStatus = instance.status
      instance.status = 'sleeping' as ModelStatus
      instance.sleepLevel = level
      instance.sleptAt = new Date()
      modelStore.set(instance)

      // Emit status event
      eventBus.emitEvent(
        eventBus.createStatusEvent(
          instanceId,
          previousStatus,
          'sleeping' as ModelStatus,
          `Model sleeping (level ${level})`
        )
      )

      this.logger.info({ instanceId, modelPath: instance.modelPath, level }, 'Model put to sleep')
    } catch (err) {
      if (err instanceof NotFoundError || err instanceof InternalError) {
        throw err
      }
      throw new InternalError(
        `Failed to put model to sleep: ${err instanceof Error ? err.message : 'Unknown error'}`
      )
    }
  }

  /**
   * Wake a sleeping model instance.
   * Reloads model weights from CPU RAM back to GPU.
   */
  async wakeModel(
    instanceId: string,
    options?: {
      tags?: 'weights' | 'kv_cache'
      placementMode?: 'balanced' | 'concentrated'
      placementGpuId?: number
    }
  ): Promise<void> {
    const instance = modelStore.get(instanceId)
    if (!instance) {
      throw new NotFoundError(`Model instance ${instanceId} not found`)
    }

    if (instance.status !== 'sleeping') {
      throw new InternalError('Model is not sleeping')
    }

    try {
      const targetDevice = resolveWakeTargetDevice({
        gpuIds: instance.gpuIds,
        placementMode: options?.placementMode,
        placementGpuId: options?.placementGpuId,
      })

      const requestPayload: { tags?: string[]; target_device?: string } = {}
      if (options?.tags) {
        requestPayload.tags = [options.tags]
      }
      if (targetDevice) {
        requestPayload.target_device = targetDevice
      }

      // Build request body for optional tags and wake-time placement target
      const requestBody =
        Object.keys(requestPayload).length > 0 ? JSON.stringify(requestPayload) : undefined
      const headers: Record<string, string> = {}
      if (requestBody) {
        headers['Content-Type'] = 'application/json'
      }

      // Call vLLM wake_up endpoint
      const response = await fetch(`http://localhost:${instance.port}/wake_up`, {
        method: 'POST',
        headers,
        body: requestBody,
        signal: AbortSignal.timeout(60000), // Wake may take longer
      })

      if (!response.ok) {
        const errorText = await response.text().catch(() => response.statusText)
        throw new InternalError(`Failed to wake model: ${errorText}`)
      }

      // Poll for ready state (model may need time to reload weights)
      const pollStart = Date.now()
      const pollTimeout = 120000 // 2 minutes max
      const pollInterval = 2000

      while (Date.now() - pollStart < pollTimeout) {
        try {
          const healthResponse = await fetch(`http://localhost:${instance.port}/health`, {
            signal: AbortSignal.timeout(2000),
          })

          if (healthResponse.ok) {
            break // Model is ready
          }
        } catch {
          // Continue polling
        }
        await new Promise((resolve) => setTimeout(resolve, pollInterval))
      }

      // Update instance state
      const previousStatus = instance.status
      instance.status = 'running' as ModelStatus
      if (options?.placementMode) {
        instance.placementMode = options.placementMode
        instance.placementGpuId =
          options.placementMode === 'concentrated'
            ? (options.placementGpuId ?? instance.gpuIds[0])
            : undefined
      }
      instance.routable = true
      instance.sleepLevel = undefined
      instance.sleptAt = undefined
      instance.lastActivityAt = new Date()
      modelStore.set(instance)

      // Emit status event
      eventBus.emitEvent(
        eventBus.createStatusEvent(
          instanceId,
          previousStatus,
          'running' as ModelStatus,
          'Model woken up and ready'
        )
      )

      this.logger.info({ instanceId, modelPath: instance.modelPath }, 'Model woken up')
    } catch (err) {
      if (err instanceof NotFoundError || err instanceof InternalError) {
        throw err
      }
      throw new InternalError(
        `Failed to wake model: ${err instanceof Error ? err.message : 'Unknown error'}`
      )
    }
  }

  setRoutable(instanceId: string, routable: boolean): void {
    const updated = modelStore.setRoutable(instanceId, routable)
    if (!updated) {
      throw new NotFoundError(`Model instance ${instanceId} not found`)
    }
  }

  noteActivity(instanceId: string): void {
    const instance = modelStore.get(instanceId)
    if (!instance) {
      return
    }
    instance.lastActivityAt = new Date()
    modelStore.set(instance)
  }

  async activateInstance(
    instanceId: string,
    options: {
      sleepOtherInstances?: boolean
      sleepLevel?: 1 | 2
      waitForDrainSeconds?: number
    } = {}
  ): Promise<{
    modelName: string
    standbyInstanceIds: string[]
    sleptInstanceIds: string[]
    wokeTarget: boolean
  }> {
    const target = modelStore.get(instanceId)
    if (!target) {
      throw new NotFoundError(`Model instance ${instanceId} not found`)
    }

    const sleepOtherInstances = options.sleepOtherInstances ?? true
    const sleepLevel = options.sleepLevel ?? 1
    const waitForDrainMs = Math.max(0, (options.waitForDrainSeconds ?? 30) * 1000)

    const siblings = modelStore.getAllByName(target.modelName).filter((i) => i.id !== instanceId)
    const overlappingSiblings = siblings.filter((sibling) =>
      this.hasGpuOverlap(sibling.gpuIds, target.gpuIds)
    )
    let wokeTarget = false

    for (const sibling of siblings) {
      modelStore.setRoutable(sibling.id, false)
    }

    for (const sibling of overlappingSiblings) {
      if (waitForDrainMs > 0) {
        await this.waitForDrain(sibling.id, waitForDrainMs)
      }
    }

    const sleptInstanceIds: string[] = []
    if (sleepOtherInstances) {
      for (const sibling of overlappingSiblings) {
        const current = modelStore.get(sibling.id)
        if (current?.status === 'running') {
          await this.sleepModel(sibling.id, sleepLevel)
          sleptInstanceIds.push(sibling.id)
        }
      }
    }

    if (target.status === 'sleeping') {
      await this.wakeModel(instanceId)
      wokeTarget = true
    }

    const refreshedTarget = modelStore.get(instanceId)
    if (!refreshedTarget || refreshedTarget.status !== 'running') {
      throw new InternalError(
        `Cannot activate instance ${instanceId}: current status is ${refreshedTarget?.status ?? 'missing'}`
      )
    }

    refreshedTarget.lastActivityAt = new Date()
    modelStore.set(refreshedTarget)
    modelStore.setRoutable(instanceId, true)

    this.logger.info(
      {
        instanceId,
        modelName: refreshedTarget.modelName,
        standbyInstanceIds: siblings.map((i) => i.id),
        sleptInstanceIds,
        wokeTarget,
      },
      'Activated model instance'
    )

    return {
      modelName: refreshedTarget.modelName,
      standbyInstanceIds: siblings.map((i) => i.id),
      sleptInstanceIds,
      wokeTarget,
    }
  }

  private async waitForDrain(instanceId: string, timeoutMs: number): Promise<void> {
    const startTime = Date.now()
    const pollInterval = 500

    while (Date.now() - startTime < timeoutMs) {
      if (!metricsStore.hasActiveConnections(instanceId)) {
        return
      }
      await new Promise((resolve) => setTimeout(resolve, pollInterval))
    }

    const remaining = metricsStore.getInstanceConnections(instanceId)
    if (remaining > 0) {
      this.logger.warn({ instanceId, remaining }, 'Drain timeout reached during instance activation')
    }
  }

  /**
   * Check if a model instance is sleeping.
   */
  async isSleeping(instanceId: string): Promise<{ isSleeping: boolean; level?: 1 | 2 }> {
    const instance = modelStore.get(instanceId)
    if (!instance) {
      throw new NotFoundError(`Model instance ${instanceId} not found`)
    }

    // If status is sleeping in our store, return that
    if (instance.status === 'sleeping') {
      return { isSleeping: true, level: instance.sleepLevel }
    }

    // Optionally verify with vLLM endpoint if sleep mode is enabled
    if (instance.sleepModeEnabled && instance.status === 'running') {
      try {
        const response = await fetch(`http://localhost:${instance.port}/is_sleeping`, {
          signal: AbortSignal.timeout(5000),
        })

        if (response.ok) {
          const data = (await response.json()) as { is_sleeping: boolean }
          return {
            isSleeping: data.is_sleeping,
            level: data.is_sleeping ? instance.sleepLevel : undefined,
          }
        }
      } catch {
        // If endpoint fails, rely on stored state
      }
    }

    return { isSleeping: false }
  }

  /**
   * Wait for model to be ready by polling health endpoint
   */
  private async waitForReady(port: number, modelPath: string, timeout: number): Promise<void> {
    const start = Date.now()
    const interval = 2000 // Poll every 2 seconds

    while (Date.now() - start < timeout) {
      try {
        const response = await fetch(`http://localhost:${port}/health`, {
          signal: AbortSignal.timeout(2000),
        })

        if (response.ok) {
          this.logger.info({ modelPath, port }, 'Model is ready')
          return
        }
      } catch {
        // Continue polling
      }

      await new Promise((resolve) => setTimeout(resolve, interval))
    }

    throw new InternalError(`Model failed to start within ${timeout}ms`)
  }

  /**
   * Get IPC segment name for kvcached
   * Include instance ID suffix for uniqueness when running multiple instances
   */
  private getIpcSegmentName(modelPath: string, instanceId: string): string {
    // Convert "meta-llama/Llama-3.2-1B" -> "VLLM_META_LLAMA_LLAMA_3_2_1B"
    const name = modelPath.replace(/\//g, '_').replace(/-/g, '_').replace(/\./g, '_').toUpperCase()
    // Add short instance suffix for uniqueness
    const suffix = instanceId.slice(0, 8).toUpperCase()
    return `VLLM_${name}_${suffix}`
  }

  /**
   * Delete all GPU-specific kvcached IPC segments
   * Called only during server shutdown when all models are unloaded
   */
  private async deleteSharedIpcSegment(): Promise<void> {
    // Collect unique IPC segment names from currently loaded models
    const segmentNames = new Set<string>()
    for (const model of this.listModels()) {
      const name = buildIpcSegmentName(model.gpuIds)
      segmentNames.add(name)
    }

    // Also try common single-GPU segments (0-7) in case models were already unloaded
    for (let i = 0; i < 8; i++) {
      segmentNames.add(`kvcached_vllm_GPU${i}`)
    }

    // Try common multi-GPU combinations for tensor-parallel models
    // Common 2-GPU pairs
    for (let i = 0; i < 8; i += 2) {
      segmentNames.add(`kvcached_vllm_GPU${i}_GPU${i + 1}`)
    }
    // Common 4-GPU groups
    segmentNames.add('kvcached_vllm_GPU0_GPU1_GPU2_GPU3')
    segmentNames.add('kvcached_vllm_GPU4_GPU5_GPU6_GPU7')
    // 8-GPU group
    segmentNames.add('kvcached_vllm_GPU0_GPU1_GPU2_GPU3_GPU4_GPU5_GPU6_GPU7')

    // Delete each segment
    for (const segmentName of segmentNames) {
      try {
              const proc = spawn(config.kvcachedKvctlBin, ['delete', segmentName])
        await new Promise<void>((resolve) => {
          proc.on('exit', (code) => {
            if (code === 0) {
              this.logger.info({ segmentName }, 'Deleted kvcached IPC segment')
            }
            resolve()
          })
          proc.on('error', () => resolve())
        })
      } catch {
        // Non-critical, segment may not exist
        this.logger.debug({ segmentName }, 'Failed to delete IPC segment (non-critical)')
      }
    }

    // Also try deleting the legacy global segment for backward compatibility
    try {
      const proc = spawn(config.kvcachedKvctlBin, ['delete', 'kvcached_mem_info'])
      await new Promise<void>((resolve) => {
        proc.on('exit', () => resolve())
        proc.on('error', () => resolve())
      })
    } catch {
      // Ignore - legacy segment may not exist
    }
  }

  /**
   * Handle process exit
   */
  private handleProcessExit(
    instanceId: string,
    code: number | null,
    signal: NodeJS.Signals | null
  ): void {
    const instance = modelStore.get(instanceId)
    if (!instance) return

    // Free simulated GPU memory on unexpected exit
    if (isInferenceSimMode()) {
      simGpuTracker.deallocate(instanceId)
    }

    if (instance.status !== 'stopping') {
      // Unexpected exit - extract meaningful error from logs
      const logs = processLogBuffer.getBuffer(instanceId)
      const errorMessage = buildErrorMessage(logs, code, signal)

      void this.restoreLaunchContext(instanceId, 'unexpected process exit')

      const previousStatus = instance.status
      instance.status = 'failed' as ModelStatus
      instance.errorMessage = errorMessage
      modelStore.set(instance)
      kvAdmissionController.releaseForInstance(instanceId, this.logger)

      // Emit status event for failed state
      eventBus.emitEvent(
        eventBus.createStatusEvent(
          instanceId,
          previousStatus,
          'failed' as ModelStatus,
          undefined,
          errorMessage
        )
      )

      this.logger.error(
        { instanceId, modelPath: instance.modelPath, errorMessage },
        'Model failed to load'
      )
      this.emit('model:failed', instance)

      // Schedule cleanup of logs after 30 minutes for failed instances
      processLogBuffer.scheduleCleanup(instanceId)
      void this.cleanupFailedInstanceResources(instance, 'unexpected process exit')
    } else {
      // Clean shutdown - clear logs immediately
      processLogBuffer.clear(instanceId)
    }

    this.processes.delete(instanceId)
  }

  /**
   * Handle process error
   */
  private handleProcessError(instanceId: string, err: Error): void {
    const instance = modelStore.get(instanceId)
    if (!instance) return

    // Try to extract better error from logs, fall back to err.message
    const logs = processLogBuffer.getBuffer(instanceId)
    const extractedError = logs.length > 0 ? buildErrorMessage(logs, null, null) : err.message

    void this.restoreLaunchContext(instanceId, 'process error')

    const previousStatus = instance.status
    instance.status = 'failed' as ModelStatus
    instance.errorMessage = extractedError
    modelStore.set(instance)
    kvAdmissionController.releaseForInstance(instanceId, this.logger)

    // Emit status event for failed state
    eventBus.emitEvent(
      eventBus.createStatusEvent(
        instanceId,
        previousStatus,
        'failed' as ModelStatus,
        undefined,
        extractedError
      )
    )

    this.logger.error(
      { instanceId, modelPath: instance.modelPath, errorMessage: extractedError },
      'Model process error'
    )
    this.emit('model:failed', instance)

    // Schedule cleanup of logs after 30 minutes for failed instances
    processLogBuffer.scheduleCleanup(instanceId)
    void this.cleanupFailedInstanceResources(instance, 'process error')

    this.processes.delete(instanceId)
  }

  private async cleanupFailedInstanceResources(
    instance: ModelInstance,
    reason: string
  ): Promise<void> {
    if (isInferenceSimMode()) {
      return
    }

    const candidatePids = new Set<number>()
    if (instance.processId > 0) {
      candidatePids.add(instance.processId)
    }
    if (instance.engineCorePid) {
      candidatePids.add(instance.engineCorePid)
    }

    for (const rootPid of Array.from(candidatePids)) {
      const descendants = await getDescendantPids(rootPid)
      for (const pid of descendants) {
        candidatePids.add(pid)
      }
    }

    const markedPids = await findProcessesByEnvMarker('SARDEENZ_INSTANCE_ID', instance.id)
    for (const pid of markedPids) {
      candidatePids.add(pid)
    }

    const portPids = await findVllmProcessesByPort(instance.port)
    for (const pid of portPids) {
      candidatePids.add(pid)
    }

    const killedPids: number[] = []
    for (const pid of candidatePids) {
      if (pid === process.pid || !isProcessRunning(pid)) {
        continue
      }
      try {
        process.kill(pid, 'SIGKILL')
        killedPids.push(pid)
      } catch (err) {
        this.logger.debug({ instanceId: instance.id, pid, err }, 'Failed to kill failed-instance pid')
      }
    }

    if (killedPids.length > 0) {
      this.logger.warn(
        {
          instanceId: instance.id,
          modelPath: instance.modelPath,
          reason,
          killedPids,
        },
        'Killed leftover processes after model failure'
      )
    }

    await this.deleteStaleKvcachedIpcSegment(
      this.getSchedulingGpuIds(instance),
      instance.kvcachedEnabled,
      reason,
      instance.id
    )

    await new Promise((resolve) => setTimeout(resolve, 2_000))

    try {
      const gpuInfo = await getNvidiaSmiInfo()
      const targetGpuIds = new Set(this.getSchedulingGpuIds(instance))
      const knownResidualProcesses = gpuInfo.processes.filter(
        (proc) => targetGpuIds.has(proc.gpu) && candidatePids.has(proc.pid)
      )
      const overlappingActiveModels = this.listModels().filter(
        (model) =>
          model.id !== instance.id &&
          ['starting', 'running', 'sleeping'].includes(model.status) &&
          this.hasGpuOverlap(this.getSchedulingGpuIds(model), Array.from(targetGpuIds))
      )
      const untrackedGpuProcesses = gpuInfo.processes.filter(
        (proc) =>
          targetGpuIds.has(proc.gpu) &&
          !candidatePids.has(proc.pid) &&
          overlappingActiveModels.every(
            (model) => model.processId !== proc.pid && model.engineCorePid !== proc.pid
          )
      )

      if (knownResidualProcesses.length > 0 || untrackedGpuProcesses.length > 0) {
        this.logger.warn(
          {
            instanceId: instance.id,
            modelPath: instance.modelPath,
            reason,
            targetGpuIds: Array.from(targetGpuIds),
            knownResidualProcesses,
            untrackedGpuProcesses,
          },
          'GPU processes remain after failed model cleanup'
        )
      } else {
        this.logger.info(
          {
            instanceId: instance.id,
            reason,
            targetGpuIds: Array.from(targetGpuIds),
          },
          'No GPU process residue found after failed model cleanup'
        )
      }
    } catch (err) {
      this.logger.warn(
        { instanceId: instance.id, reason, err },
        'Failed to inspect GPU process residue after model failure'
      )
    }
  }

  /**
   * Cleanup all models on shutdown
   */
  async cleanup(): Promise<void> {
    this.logger.info('Cleaning up all models')
    const models = this.listModels()

    for (const model of models) {
      try {
        await this.unloadModel(model.id)
      } catch (err) {
        this.logger.error(
          { instanceId: model.id, modelPath: model.modelPath, err },
          'Error cleaning up model'
        )
      }
    }

    // Clean up log buffer
    processLogBuffer.cleanup()
    for (const model of models) {
      kvAdmissionController.releaseForInstance(model.id, this.logger)
    }

    // Delete the shared kvcached IPC segment now that all models are unloaded
    await this.deleteSharedIpcSegment()
  }

  /**
   * Get buffered logs for a model instance
   */
  getLogs(instanceId: string): { logs: string; lineCount: number } {
    const buffer = processLogBuffer.getBuffer(instanceId)
    return {
      logs: processLogBuffer.getLastLines(instanceId, 500),
      lineCount: buffer.length,
    }
  }
}

// Singleton instance
let _modelManagerInstance: ModelManager | null = null

/**
 * Get the singleton ModelManager instance.
 * This ensures all routes share the same instance and its processes Map.
 */
export function getModelManager(logger: Logger): ModelManager {
  if (!_modelManagerInstance) {
    _modelManagerInstance = new ModelManager(logger)
  }
  return _modelManagerInstance
}
