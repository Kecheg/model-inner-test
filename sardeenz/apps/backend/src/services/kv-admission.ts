import type { ModelInstance } from '@sardeenz/types'
import type { Logger } from '@sardeenz/utils'
import { config } from '../config.js'
import { modelStore } from '../stores/model-store.js'

interface InstanceQuota {
  minGuaranteeGiBPerGpu: number
  softLimitGiBPerGpu: number
  hardLimitGiBPerGpu: number
}

interface ActiveReservation {
  id: string
  instanceId: string
  modelName: string
  gpuGroup: string
  estimatedGiBPerGpu: number
  createdAt: number
}

interface AdmissionResult {
  allowed: boolean
  statusCode?: number
  error?: {
    message: string
    type: string
    code: string
    retry_after_ms?: number
    details?: Record<string, unknown>
  }
  reservation?: ActiveReservation
  shouldTrimKvcached?: boolean
  trimReason?: string
}

type InstanceOverrideMap = Record<string, Partial<InstanceQuota>>

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms))
}

function gpuGroupKey(instance: ModelInstance): string {
  const ids = (instance.topologyGpuIds?.length ? instance.topologyGpuIds : instance.gpuIds)
    .slice()
    .sort((a, b) => a - b)
  return `GPU${ids.join('_GPU')}`
}

function textTokenEstimate(value: unknown): number {
  if (typeof value === 'string') {
    return Math.ceil(value.length / 4)
  }
  if (Array.isArray(value)) {
    return value.reduce((sum, item) => sum + textTokenEstimate(item), 0)
  }
  if (value && typeof value === 'object') {
    let total = 0
    for (const item of Object.values(value as Record<string, unknown>)) {
      total += textTokenEstimate(item)
    }
    return total
  }
  return 0
}

function requestTokenEstimate(body: Record<string, unknown>): number {
  const explicitPromptTokens = body.prompt_tokens
  if (typeof explicitPromptTokens === 'number' && explicitPromptTokens > 0) {
    return explicitPromptTokens
  }

  const promptTokens =
    'prompt' in body
      ? textTokenEstimate(body.prompt)
      : 'messages' in body
        ? textTokenEstimate(body.messages)
        : textTokenEstimate(body.input)

  const outputTokens =
    typeof body.max_tokens === 'number'
      ? body.max_tokens
      : typeof body.max_completion_tokens === 'number'
        ? body.max_completion_tokens
        : typeof body.max_new_tokens === 'number'
          ? body.max_new_tokens
          : 0

  return Math.max(1, promptTokens + outputTokens)
}

class KvAdmissionController {
  private reservations = new Map<string, ActiveReservation>()
  private overrides: InstanceOverrideMap | undefined

  isEnabled(): boolean {
    return config.kvAdmissionEnabled
  }

  async acquire(
    instance: ModelInstance,
    body: Record<string, unknown>,
    logger: Logger
  ): Promise<AdmissionResult> {
    if (!this.isEnabled()) {
      return { allowed: true }
    }

    this.cleanupExpiredReservations(logger)

    const deadline = Date.now() + config.kvAdmissionMaxQueueMs
    const requestTokens = requestTokenEstimate(body)
    const estimatedGiBPerGpu = Math.max(
      0.001,
      requestTokens / config.kvAdmissionEstimateTokensPerGiB
    )
    let trimmedForBackpressure = false

    while (true) {
      const result = this.tryAcquire(instance, estimatedGiBPerGpu, requestTokens)
      if (result.allowed) {
        if (result.shouldTrimKvcached) {
          await this.trimInstance(instance, result.trimReason ?? 'kv_admission_soft_limit', logger)
        }
        logger.debug(
          {
            instanceId: instance.id,
            modelName: instance.modelName,
            estimatedGiBPerGpu,
            requestTokens,
            reservationId: result.reservation?.id,
          },
          'KV admission allowed request'
        )
        return result
      }

      if (result.statusCode === 429 && !trimmedForBackpressure) {
        trimmedForBackpressure = true
        await this.trimGpuGroup(instance, result.error?.code ?? 'kv_admission_backpressure', logger)
      }

      const now = Date.now()
      if (now >= deadline || result.statusCode !== 429) {
        logger.warn(
          {
            instanceId: instance.id,
            modelName: instance.modelName,
            estimatedGiBPerGpu,
            requestTokens,
            error: result.error,
          },
          'KV admission rejected request'
        )
        return result
      }

      await sleep(Math.min(config.kvAdmissionQueuePollMs, Math.max(0, deadline - now)))
    }
  }

  release(reservation?: ActiveReservation): void {
    if (!reservation) {
      return
    }
    this.reservations.delete(reservation.id)
  }

  releaseForInstance(instanceId: string, logger?: Logger): number {
    let released = 0
    for (const reservation of Array.from(this.reservations.values())) {
      if (reservation.instanceId === instanceId) {
        this.reservations.delete(reservation.id)
        released++
      }
    }
    if (released > 0) {
      logger?.warn({ instanceId, released }, 'Released stale KV admission reservations for instance')
    }
    return released
  }

  snapshot(): Record<string, unknown> {
    this.cleanupExpiredReservations()

    const byGroup: Record<string, number> = {}
    const byInstance: Record<string, number> = {}
    for (const reservation of this.reservations.values()) {
      byGroup[reservation.gpuGroup] =
        (byGroup[reservation.gpuGroup] ?? 0) + reservation.estimatedGiBPerGpu
      byInstance[reservation.instanceId] =
        (byInstance[reservation.instanceId] ?? 0) + reservation.estimatedGiBPerGpu
    }
    return {
      enabled: this.isEnabled(),
      byGroup,
      byInstance,
      reservationCount: this.reservations.size,
    }
  }

  private tryAcquire(
    instance: ModelInstance,
    estimatedGiBPerGpu: number,
    requestTokens: number
  ): AdmissionResult {
    const gpuGroup = gpuGroupKey(instance)
    const quota = this.quotaFor(instance)
    const globalLimit = Math.min(
      config.kvAdmissionGlobalSafeLimitGiBPerGpu,
      config.kvAdmissionPoolCapacityGiBPerGpu - config.kvAdmissionGlobalReservedGiBPerGpu
    )

    const instanceActive = this.activeForInstance(instance.id)
    const groupActive = this.activeForGroup(gpuGroup)
    const nextInstanceActive = instanceActive + estimatedGiBPerGpu
    const nextGroupActive = groupActive + estimatedGiBPerGpu
    const protectedMinimumGiBPerGpu = this.protectedMinimumForOtherInstances(instance, gpuGroup)
    const ownMinimumDeficitGiBPerGpu = Math.max(
      0,
      quota.minGuaranteeGiBPerGpu - instanceActive
    )
    const ownMinimumDrawGiBPerGpu = Math.min(estimatedGiBPerGpu, ownMinimumDeficitGiBPerGpu)

    if (estimatedGiBPerGpu > quota.hardLimitGiBPerGpu) {
      return this.reject(
        413,
        `Estimated KV ${estimatedGiBPerGpu.toFixed(2)}GiB exceeds instance hard limit ${quota.hardLimitGiBPerGpu.toFixed(2)}GiB`,
        estimatedGiBPerGpu,
        requestTokens,
        instanceActive,
        groupActive,
        quota,
        globalLimit
      )
    }

    if (nextInstanceActive > quota.hardLimitGiBPerGpu) {
      return this.reject(
        429,
        `Instance KV active ${nextInstanceActive.toFixed(2)}GiB would exceed hard limit ${quota.hardLimitGiBPerGpu.toFixed(2)}GiB`,
        estimatedGiBPerGpu,
        requestTokens,
        instanceActive,
        groupActive,
        quota,
        globalLimit
      )
    }

    if (nextGroupActive + protectedMinimumGiBPerGpu - ownMinimumDrawGiBPerGpu > globalLimit) {
      return this.reject(
        429,
        `GPU group ${gpuGroup} active KV ${nextGroupActive.toFixed(2)}GiB plus sibling minimum guarantees ${protectedMinimumGiBPerGpu.toFixed(2)}GiB would exceed safe limit ${globalLimit.toFixed(2)}GiB`,
        estimatedGiBPerGpu,
        requestTokens,
        instanceActive,
        groupActive,
        quota,
        globalLimit,
        protectedMinimumGiBPerGpu
      )
    }

    if (nextGroupActive > globalLimit) {
      return this.reject(
        429,
        `GPU group ${gpuGroup} active KV ${nextGroupActive.toFixed(2)}GiB would exceed safe limit ${globalLimit.toFixed(2)}GiB`,
        estimatedGiBPerGpu,
        requestTokens,
        instanceActive,
        groupActive,
        quota,
        globalLimit
      )
    }

    if (
      nextInstanceActive > quota.softLimitGiBPerGpu &&
      nextGroupActive + protectedMinimumGiBPerGpu > globalLimit
    ) {
      return this.reject(
        429,
        `Instance KV active ${nextInstanceActive.toFixed(2)}GiB would exceed soft limit ${quota.softLimitGiBPerGpu.toFixed(2)}GiB while ${protectedMinimumGiBPerGpu.toFixed(2)}GiB is reserved for sibling minimum guarantees`,
        estimatedGiBPerGpu,
        requestTokens,
        instanceActive,
        groupActive,
        quota,
        globalLimit,
        protectedMinimumGiBPerGpu
      )
    }

    const reservation: ActiveReservation = {
      id: `${Date.now()}-${Math.random().toString(16).slice(2)}`,
      instanceId: instance.id,
      modelName: instance.modelName,
      gpuGroup,
      estimatedGiBPerGpu,
      createdAt: Date.now(),
    }
    this.reservations.set(reservation.id, reservation)
    return {
      allowed: true,
      reservation,
      shouldTrimKvcached: nextInstanceActive > quota.softLimitGiBPerGpu,
      trimReason:
        nextInstanceActive > quota.softLimitGiBPerGpu ? 'kv_admission_soft_limit_exceeded' : undefined,
    }
  }

  private reject(
    statusCode: number,
    message: string,
    estimatedGiBPerGpu: number,
    requestTokens: number,
    instanceActiveGiBPerGpu: number,
    groupActiveGiBPerGpu: number,
    quota: InstanceQuota,
    globalLimitGiBPerGpu: number,
    protectedMinimumGiBPerGpu = 0
  ): AdmissionResult {
    return {
      allowed: false,
      statusCode,
      error: {
        message,
        type: 'kv_admission_error',
        code: 'KV_POOL_CAPACITY_EXCEEDED',
        retry_after_ms: statusCode === 429 ? config.kvAdmissionMaxQueueMs : undefined,
        details: {
          estimatedGiBPerGpu,
          requestTokens,
          instanceActiveGiBPerGpu,
          groupActiveGiBPerGpu,
          quota,
          globalLimitGiBPerGpu,
          protectedMinimumGiBPerGpu,
        },
      },
    }
  }

  private activeForInstance(instanceId: string): number {
    let total = 0
    for (const reservation of this.reservations.values()) {
      if (reservation.instanceId === instanceId) {
        total += reservation.estimatedGiBPerGpu
      }
    }
    return total
  }

  private activeForGroup(gpuGroup: string): number {
    let total = 0
    for (const reservation of this.reservations.values()) {
      if (reservation.gpuGroup === gpuGroup) {
        total += reservation.estimatedGiBPerGpu
      }
    }
    return total
  }

  private protectedMinimumForOtherInstances(instance: ModelInstance, gpuGroup: string): number {
    let protectedMinimum = 0
    for (const sibling of modelStore.getAll()) {
      if (sibling.id === instance.id) {
        continue
      }
      if (sibling.status !== 'running' || sibling.routable === false) {
        continue
      }
      if (gpuGroupKey(sibling) !== gpuGroup) {
        continue
      }

      const quota = this.quotaFor(sibling)
      const active = this.activeForInstance(sibling.id)
      protectedMinimum += Math.max(0, quota.minGuaranteeGiBPerGpu - active)
    }
    return protectedMinimum
  }

  private async trimGpuGroup(instance: ModelInstance, reason: string, logger: Logger): Promise<void> {
    const gpuGroup = gpuGroupKey(instance)
    const instances = modelStore
      .getAll()
      .filter((sibling) => sibling.status === 'running')
      .filter((sibling) => sibling.kvcachedEnabled)
      .filter((sibling) => gpuGroupKey(sibling) === gpuGroup)

    for (const sibling of instances) {
      await this.trimInstance(sibling, reason, logger)
    }
  }

  private async trimInstance(instance: ModelInstance, reason: string, logger: Logger): Promise<void> {
    if (!instance.kvcachedEnabled || instance.status !== 'running') {
      return
    }

    try {
      const response = await fetch(`http://127.0.0.1:${instance.port}/kvcached/trim`, {
        method: 'POST',
        signal: AbortSignal.timeout(2_000),
      })
      const payload = await response.json().catch(() => undefined)
      if (!response.ok) {
        logger.warn(
          {
            instanceId: instance.id,
            modelName: instance.modelName,
            port: instance.port,
            reason,
            status: response.status,
            payload,
          },
          'KV admission failed to trim kvcached instance'
        )
        return
      }

      logger.info(
        {
          instanceId: instance.id,
          modelName: instance.modelName,
          port: instance.port,
          reason,
          payload,
        },
        'KV admission trimmed kvcached instance'
      )
    } catch (err) {
      logger.warn(
        {
          instanceId: instance.id,
          modelName: instance.modelName,
          port: instance.port,
          reason,
          err,
        },
        'KV admission failed to call kvcached trim'
      )
    }
  }

  private quotaFor(instance: ModelInstance): InstanceQuota {
    const override = this.instanceOverrides()[instance.modelName] ?? this.instanceOverrides()[instance.id]
    return {
      minGuaranteeGiBPerGpu:
        override?.minGuaranteeGiBPerGpu ?? config.kvAdmissionDefaultMinGuaranteeGiBPerGpu,
      softLimitGiBPerGpu:
        override?.softLimitGiBPerGpu ?? config.kvAdmissionDefaultSoftLimitGiBPerGpu,
      hardLimitGiBPerGpu:
        override?.hardLimitGiBPerGpu ?? config.kvAdmissionDefaultHardLimitGiBPerGpu,
    }
  }

  private instanceOverrides(): InstanceOverrideMap {
    if (this.overrides !== undefined) {
      return this.overrides
    }
    if (!config.kvAdmissionInstanceOverridesJson) {
      this.overrides = {}
      return this.overrides
    }
    this.overrides = JSON.parse(config.kvAdmissionInstanceOverridesJson) as InstanceOverrideMap
    return this.overrides
  }

  private cleanupExpiredReservations(logger?: Logger): number {
    const ttlMs = config.kvAdmissionReservationTtlMs
    if (ttlMs <= 0) {
      return 0
    }

    const now = Date.now()
    let released = 0
    for (const reservation of Array.from(this.reservations.values())) {
      if (now - reservation.createdAt > ttlMs) {
        this.reservations.delete(reservation.id)
        released++
        logger?.warn(
          {
            reservationId: reservation.id,
            instanceId: reservation.instanceId,
            modelName: reservation.modelName,
            ageMs: now - reservation.createdAt,
            ttlMs,
          },
          'Expired stale KV admission reservation'
        )
      }
    }
    return released
  }
}

export const kvAdmissionController = new KvAdmissionController()
