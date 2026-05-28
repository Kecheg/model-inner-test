import type { FastifyInstance } from 'fastify'
import { kvAdmissionController } from '../services/kv-admission.js'

export default async function kvAdmissionRoutes(fastify: FastifyInstance) {
  fastify.get(
    '/api/kv-admission',
    {
      schema: {
        tags: ['memory'],
        description: 'Get KV admission controller state for debugging and tests',
      },
    },
    async () => kvAdmissionController.snapshot()
  )
}
