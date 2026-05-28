import { ModelStatus } from '@sardeenz/types'
import type { ModelInstanceDTO } from '@sardeenz/types'

type ModelWithStatus = Pick<ModelInstanceDTO, 'status'>

export function isStartedModel(model: ModelWithStatus): boolean {
  return model.status === ModelStatus.Running || model.status === ModelStatus.Sleeping
}

export function countStartedModels(models: ModelWithStatus[]): number {
  return models.filter(isStartedModel).length
}