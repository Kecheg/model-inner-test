import type { Logger } from '@sardeenz/utils'
import { modelStore } from '../stores/model-store.js'
import { metricsStore } from '../stores/metrics-store.js'
import { getModelManager } from './model-manager.js'

export class IdleSleepManager {
  private logger: Logger
  private intervalMs: number
  private timer: NodeJS.Timeout | null = null
  private scanInProgress = false

  constructor(logger: Logger, intervalMs: number) {
    this.logger = logger.child({ component: 'IdleSleepManager' })
    this.intervalMs = intervalMs
  }

  start(): void {
    if (this.timer) {
      return
    }

    this.timer = setInterval(() => {
      void this.scan()
    }, this.intervalMs)
    this.logger.info({ intervalMs: this.intervalMs }, 'Idle sleep manager started')
  }

  stop(): void {
    if (this.timer) {
      clearInterval(this.timer)
      this.timer = null
    }
  }

  private async scan(): Promise<void> {
    if (this.scanInProgress) {
      return
    }

    this.scanInProgress = true
    const modelManager = getModelManager(this.logger)

    try {
      const now = Date.now()
      const instances = modelStore.getAll()

      for (const instance of instances) {
        if (instance.status !== 'running') continue
        if (!instance.sleepModeEnabled) continue
        if (!instance.idleSleepTimeoutSeconds || instance.idleSleepTimeoutSeconds <= 0) continue
        if (metricsStore.hasActiveConnections(instance.id)) continue

        const lastActivityAt =
          instance.lastActivityAt ?? instance.readyAt ?? instance.loadedAt ?? new Date()
        const idleForMs = now - lastActivityAt.getTime()
        const timeoutMs = instance.idleSleepTimeoutSeconds * 1000

        if (idleForMs < timeoutMs) continue

        try {
          await modelManager.sleepModel(instance.id, instance.idleSleepLevel ?? 1)
          this.logger.info(
            {
              instanceId: instance.id,
              modelPath: instance.modelPath,
              idleForMs,
              timeoutMs,
            },
            'Auto-slept idle model instance'
          )
        } catch (err) {
          this.logger.warn(
            { err, instanceId: instance.id, modelPath: instance.modelPath },
            'Failed to auto-sleep idle model instance'
          )
        }
      }
    } finally {
      this.scanInProgress = false
    }
  }
}