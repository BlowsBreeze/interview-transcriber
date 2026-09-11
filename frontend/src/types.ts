export type Segment = {
  start: number
  end: number
  speaker: string
  text: string
}

export type JobStatus =
  | 'queued'
  | 'loading_models'
  | 'transcribing'
  | 'formatting'
  | 'completed'
  | 'failed'

export type Job = {
  id: string
  status: JobStatus
  filename: string
  created_at: number
  elapsed_seconds: number
  phase: 'queued' | 'preparing_audio' | 'loading_models' | 'vad' | 'transcribing' | 'finalizing' | 'formatting' | 'completed' | 'failed'
  progress_current: number
  progress_total: number
  progress_percent: number | null
  estimated_remaining_seconds: number | null
  device: string | null
  error: string | null
  segments: Segment[] | null
  original_duration_seconds: number | null
  trim_start_seconds: number | null
  trim_end_seconds: number | null
}

export type HistoryJob = Pick<Job, 'id' | 'status' | 'filename' | 'created_at'> & {
  segment_count: number
  original_duration_seconds: number | null
  trim_start_seconds: number | null
  trim_end_seconds: number | null
}
