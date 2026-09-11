import type { HistoryJob, Job } from './types'

async function readError(response: Response): Promise<string> {
  try {
    const payload = await response.json()
    return payload.detail || '请求失败'
  } catch {
    return '请求失败'
  }
}

export function createTranscription(
  file: File,
  options?: {
    trimStartSeconds?: number
    trimEndSeconds?: number
    onProgress?: (percent: number) => void
  },
): Promise<Job> {
  return new Promise((resolve, reject) => {
    const body = new FormData()
    body.append('file', file)
    if (options?.trimStartSeconds !== undefined) {
      body.append('trim_start_seconds', String(options.trimStartSeconds))
    }
    if (options?.trimEndSeconds !== undefined) {
      body.append('trim_end_seconds', String(options.trimEndSeconds))
    }
    const request = new XMLHttpRequest()
    request.open('POST', '/api/transcriptions')
    request.responseType = 'json'
    request.upload.onprogress = (event) => {
      if (event.lengthComputable) options?.onProgress?.(Math.round((event.loaded / event.total) * 100))
    }
    request.onerror = () => reject(new Error('无法连接本地转写服务'))
    request.onload = () => {
      if (request.status >= 200 && request.status < 300) {
        resolve(request.response as Job)
        return
      }
      reject(new Error(request.response?.detail || '上传失败'))
    }
    request.send(body)
  })
}

export async function getTranscription(id: string): Promise<Job> {
  const response = await fetch(`/api/transcriptions/${id}`)
  if (!response.ok) throw new Error(await readError(response))
  return response.json()
}

export async function listTranscriptions(): Promise<HistoryJob[]> {
  const response = await fetch('/api/transcriptions')
  if (!response.ok) throw new Error(await readError(response))
  return response.json()
}

export async function deleteTranscription(id: string): Promise<void> {
  const response = await fetch(`/api/transcriptions/${id}`, { method: 'DELETE' })
  if (!response.ok) throw new Error(await readError(response))
}

export async function renameTranscription(id: string, name: string): Promise<HistoryJob> {
  const response = await fetch(`/api/transcriptions/${id}`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ name }),
  })
  if (!response.ok) throw new Error(await readError(response))
  return response.json()
}

export const audioUrl = (id: string) => `/api/transcriptions/${id}/audio`
export const exportUrl = (id: string, kind: 'json' | 'markdown') =>
  `/api/transcriptions/${id}/export/${kind}`
