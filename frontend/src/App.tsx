import { useEffect, useMemo, useRef, useState } from 'react'
import {
  AudioLines,
  Check,
  ChevronRight,
  Clock3,
  Download,
  FileAudio,
  FolderOpen,
  Gauge,
  History,
  LoaderCircle,
  Pencil,
  Play,
  RotateCcw,
  ShieldCheck,
  X,
  Trash2,
  UploadCloud,
  Users,
  WandSparkles,
} from 'lucide-react'
import {
  audioUrl,
  createTranscription,
  deleteTranscription,
  exportUrl,
  getTranscription,
  listTranscriptions,
  renameTranscription,
} from './api'
import type { HistoryJob, Job, Segment } from './types'

const SPEAKER_COLOR_COUNT = 10

const statusCopy = {
  queued: '等待开始',
  loading_models: '正在加载本地模型',
  transcribing: '正在识别录音与说话人',
  formatting: '正在整理转写结果',
  completed: '转写完成',
  failed: '转写失败',
}

const phaseCopy: Record<Job['phase'], string> = {
  queued: '等待本地任务开始',
  preparing_audio: '正在裁剪并准备录音',
  loading_models: '正在加载本地模型',
  vad: '正在分析录音中的语音片段',
  transcribing: '正在识别录音与说话人',
  finalizing: '正在整理标点与 Speaker 聚类',
  formatting: '正在生成转写文件',
  completed: '转写完成',
  failed: '转写失败',
}

const phaseSteps = [
  { label: '上传录音', phases: ['uploading'] },
  { label: '准备音频', phases: ['queued', 'preparing_audio'] },
  { label: '加载模型', phases: ['loading_models'] },
  { label: '分析语音', phases: ['vad'] },
  { label: '转写与说话人', phases: ['transcribing'] },
  { label: '整理结果', phases: ['finalizing', 'formatting'] },
]

function formatTime(seconds: number) {
  const total = Math.max(0, Math.floor(seconds))
  const hours = Math.floor(total / 3600)
  const minutes = Math.floor((total % 3600) / 60)
  const secs = total % 60
  return hours
    ? `${hours.toString().padStart(2, '0')}:${minutes.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`
    : `${minutes.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`
}

function joinText(left: string, right: string) {
  if (/[，。！？；：、,.!?;:]$/.test(left.trim())) return left.trim() + right.trim()
  return `${left.trim()}，${right.trim()}`
}

function mergeSegments(segments: Segment[], maxGapSeconds: number): Segment[] {
  return segments.reduce<Segment[]>((result, segment) => {
    const previous = result.at(-1)
    const gap = previous ? segment.start - previous.end : Infinity
    if (previous && previous.speaker === segment.speaker && gap >= -0.1 && gap <= maxGapSeconds) {
      previous.end = Math.max(previous.end, segment.end)
      previous.text = joinText(previous.text, segment.text)
    } else {
      result.push({ ...segment })
    }
    return result
  }, [])
}

function fileSize(bytes: number) {
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`
}

function formatDate(timestamp: number) {
  return new Intl.DateTimeFormat('zh-CN', {
    year: 'numeric',
    month: '2-digit',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
  }).format(new Date(timestamp * 1000))
}

function recordBaseName(filename: string) {
  return filename.replace(/\.[^./]+$/, '')
}

function parseTime(value: string) {
  const parts = value.trim().split(':').map(Number)
  if (!parts.length || parts.some((part) => !Number.isFinite(part) || part < 0)) return null
  if (parts.length === 1) return parts[0]
  if (parts.length === 2) return parts[0] * 60 + parts[1]
  if (parts.length === 3) return parts[0] * 3600 + parts[1] * 60 + parts[2]
  return null
}

function TimeInput({
  value,
  max,
  onChange,
}: {
  value: number
  max: number
  onChange: (value: number) => void
}) {
  const [draft, setDraft] = useState(formatTime(value))

  useEffect(() => setDraft(formatTime(value)), [value])

  const commit = () => {
    const parsed = parseTime(draft)
    if (parsed === null) setDraft(formatTime(value))
    else onChange(Math.min(max, Math.max(0, parsed)))
  }

  return (
    <input
      className="trim-time-input"
      value={draft}
      inputMode="numeric"
      onChange={(event) => setDraft(event.target.value)}
      onBlur={commit}
      onKeyDown={(event) => {
        if (event.key === 'Enter') event.currentTarget.blur()
        if (event.key === 'Escape') setDraft(formatTime(value))
      }}
    />
  )
}

export default function App() {
  const [file, setFile] = useState<File | null>(null)
  const [job, setJob] = useState<Job | null>(null)
  const [error, setError] = useState('')
  const [dragging, setDragging] = useState(false)
  const [merged, setMerged] = useState(true)
  const [mergeGapSeconds, setMergeGapSeconds] = useState(2)
  const [speakerNames, setSpeakerNames] = useState<Record<string, string>>({})
  const [hiddenSpeakers, setHiddenSpeakers] = useState<Set<string>>(() => new Set())
  const [uploading, setUploading] = useState(false)
  const [uploadProgress, setUploadProgress] = useState(0)
  const [filePreviewUrl, setFilePreviewUrl] = useState('')
  const [audioDuration, setAudioDuration] = useState(0)
  const [trimStart, setTrimStart] = useState(0)
  const [trimEnd, setTrimEnd] = useState(0)
  const [previewingTrim, setPreviewingTrim] = useState(false)
  const [history, setHistory] = useState<HistoryJob[]>([])
  const [historyLoading, setHistoryLoading] = useState(true)
  const [historyError, setHistoryError] = useState('')
  const [renamingJob, setRenamingJob] = useState<{ id: string; filename: string } | null>(null)
  const [renameDraft, setRenameDraft] = useState('')
  const [renameError, setRenameError] = useState('')
  const [renameSaving, setRenameSaving] = useState(false)
  const audioRef = useRef<HTMLAudioElement>(null)
  const trimAudioRef = useRef<HTMLAudioElement>(null)
  const resultRef = useRef<HTMLElement>(null)
  const shouldScrollToResultRef = useRef(false)

  useEffect(() => {
    listTranscriptions()
      .then(setHistory)
      .catch((requestError) => setHistoryError(
        requestError instanceof Error ? requestError.message : '无法读取历史记录',
      ))
      .finally(() => setHistoryLoading(false))
  }, [])

  useEffect(() => {
    if (!file) {
      setFilePreviewUrl('')
      return
    }
    const url = URL.createObjectURL(file)
    setFilePreviewUrl(url)
    return () => URL.revokeObjectURL(url)
  }, [file])

  useEffect(() => {
    if (!job || job.status === 'completed' || job.status === 'failed') return
    const timer = window.setInterval(async () => {
      try {
        const next = await getTranscription(job.id)
        setJob(next)
      } catch (requestError) {
        setError(requestError instanceof Error ? requestError.message : '无法获取任务状态')
      }
    }, 1200)
    return () => window.clearInterval(timer)
  }, [job?.id, job?.status])

  useEffect(() => {
    if (!job || !['completed', 'failed'].includes(job.status)) return
    listTranscriptions().then(setHistory).catch(() => undefined)
  }, [job?.id, job?.status])

  useEffect(() => {
    if (!shouldScrollToResultRef.current || job?.status !== 'completed' || !job.segments) return
    shouldScrollToResultRef.current = false
    window.requestAnimationFrame(() => {
      resultRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' })
    })
  }, [job?.id, job?.status, job?.segments])

  const speakers = useMemo(
    () => Array.from(new Set((job?.segments || []).map((segment) => segment.speaker))).sort(),
    [job?.segments],
  )
  const displaySegments = useMemo(() => {
    const source = job?.segments || []
    const prepared = merged ? mergeSegments(source, mergeGapSeconds) : source
    return hiddenSpeakers.size
      ? prepared.filter((segment) => !hiddenSpeakers.has(segment.speaker))
      : prepared
  }, [job?.segments, merged, mergeGapSeconds, hiddenSpeakers])

  const visibleSpeakerCount = speakers.filter((speaker) => !hiddenSpeakers.has(speaker)).length

  const toggleSpeaker = (speaker: string) => {
    setHiddenSpeakers((current) => {
      const next = new Set(current)
      if (next.has(speaker)) next.delete(speaker)
      else next.add(speaker)
      return next
    })
  }

  const showOnlySpeaker = (speaker: string) => {
    setHiddenSpeakers(new Set(speakers.filter((item) => item !== speaker)))
  }

  const showAllSpeakers = () => setHiddenSpeakers(new Set())

  const chooseFile = (candidate?: File) => {
    if (!candidate) return
    setFile(candidate)
    setJob(null)
    setError('')
    setSpeakerNames({})
    showAllSpeakers()
    setAudioDuration(0)
    setTrimStart(0)
    setTrimEnd(0)
    setPreviewingTrim(false)
  }

  const start = async () => {
    if (!file) return
    setError('')
    setUploading(true)
    setUploadProgress(0)
    try {
      const trimmed = audioDuration > 0 && (trimStart > 0.05 || trimEnd < audioDuration - 0.05)
      setJob(await createTranscription(file, {
        trimStartSeconds: trimmed ? trimStart : undefined,
        trimEndSeconds: trimmed ? trimEnd : undefined,
        onProgress: setUploadProgress,
      }))
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : '上传失败')
    } finally {
      setUploading(false)
    }
  }

  const seek = (seconds: number) => {
    if (!audioRef.current) return
    audioRef.current.currentTime = seconds
    void audioRef.current.play()
  }

  const setTrimFromCurrentTime = (boundary: 'start' | 'end') => {
    const current = trimAudioRef.current?.currentTime ?? 0
    if (boundary === 'start') setTrimStart(Math.min(current, trimEnd - 1))
    else setTrimEnd(Math.max(current, trimStart + 1))
  }

  const previewTrim = () => {
    if (!trimAudioRef.current || trimEnd <= trimStart) return
    trimAudioRef.current.currentTime = trimStart
    setPreviewingTrim(true)
    void trimAudioRef.current.play()
  }

  const restoreFullAudio = () => {
    setTrimStart(0)
    setTrimEnd(audioDuration)
    setPreviewingTrim(false)
  }

  const reset = () => {
    shouldScrollToResultRef.current = false
    setFile(null)
    setJob(null)
    setError('')
    setSpeakerNames({})
    showAllSpeakers()
    setUploading(false)
    setUploadProgress(0)
    setAudioDuration(0)
    setTrimStart(0)
    setTrimEnd(0)
    setPreviewingTrim(false)
  }

  const openHistory = async (jobId: string) => {
    setError('')
    try {
      const savedJob = await getTranscription(jobId)
      shouldScrollToResultRef.current = true
      setFile(null)
      setJob(savedJob)
      setSpeakerNames({})
      showAllSpeakers()
    } catch (requestError) {
      setHistoryError(requestError instanceof Error ? requestError.message : '无法打开历史记录')
    }
  }

  const removeHistory = async (historyJob: HistoryJob) => {
    if (!window.confirm(`确定删除“${historyJob.filename}”及其本地转写文件吗？`)) return
    try {
      await deleteTranscription(historyJob.id)
      setHistory((items) => items.filter((item) => item.id !== historyJob.id))
      setHistoryError('')
    } catch (requestError) {
      setHistoryError(requestError instanceof Error ? requestError.message : '删除历史记录失败')
    }
  }

  const beginRename = (id: string, filename: string) => {
    setRenamingJob({ id, filename })
    setRenameDraft(recordBaseName(filename))
    setRenameError('')
  }

  const closeRename = () => {
    if (renameSaving) return
    setRenamingJob(null)
    setRenameError('')
  }

  const saveRename = async () => {
    if (!renamingJob || !renameDraft.trim() || renameSaving) return
    setRenameSaving(true)
    setRenameError('')
    try {
      const renamed = await renameTranscription(renamingJob.id, renameDraft.trim())
      setHistory((items) => items.map((item) => item.id === renamed.id ? renamed : item))
      setJob((current) => current?.id === renamed.id
        ? { ...current, filename: renamed.filename }
        : current)
      setRenamingJob(null)
    } catch (requestError) {
      setRenameError(requestError instanceof Error ? requestError.message : '重命名失败')
    } finally {
      setRenameSaving(false)
    }
  }

  const isRunning = Boolean(job && !['completed', 'failed'].includes(job.status))
  const isTrimmed = audioDuration > 0 && (trimStart > 0.05 || trimEnd < audioDuration - 0.05)
  const selectedDuration = Math.max(0, trimEnd - trimStart)
  const activePhase = uploading ? 'uploading' : (job?.phase || 'queued')
  const activeStep = Math.max(
    0,
    phaseSteps.findIndex((step) => step.phases.includes(activePhase)),
  )
  const visibleProgress = uploading ? uploadProgress : job?.progress_percent
  const hasMeasuredProgress = visibleProgress !== null && visibleProgress !== undefined
  const progressLabel = uploading
    ? `正在读取录音 · ${uploadProgress}%`
    : job
      ? phaseCopy[job.phase]
      : ''

  return (
    <main id="top">
      <header className="topbar">
        <a className="brand" href="#top" onClick={reset} aria-label="返回首页并开始新录音">
          <span className="brand-mark"><AudioLines size={19} strokeWidth={2.4} /></span>
          <span>声迹</span>
        </a>
        <div className="privacy"><ShieldCheck size={15} /> 录音仅在这台 Mac 上处理</div>
      </header>

      {!job && (
        <section className="hero">
          <div className="eyebrow"><WandSparkles size={14} /> LOCAL INTERVIEW TRANSCRIBER</div>
          <h1>让每一次面试，<br /><em>都有迹可循。</em></h1>
          <p>上传录音，本地完成中文转写、说话人区分和时间定位。无需账号，也不会把声音交给第三方。</p>
          <div className="spec-row">
            <span><Gauge size={16} /> Apple Silicon 加速</span>
            <span><Users size={16} /> Speaker 智能分离</span>
            <span><Clock3 size={16} /> 精确时间戳</span>
          </div>
        </section>
      )}

      {!job?.segments && job?.status !== 'completed' ? (
        <section className="workspace upload-workspace">
          <div className="upload-main">
            <div
              className={`dropzone ${dragging ? 'dragging' : ''} ${file ? 'has-file' : ''}`}
              onDragOver={(event) => { event.preventDefault(); setDragging(true) }}
              onDragLeave={() => setDragging(false)}
              onDrop={(event) => {
                event.preventDefault()
                setDragging(false)
                chooseFile(event.dataTransfer.files[0])
              }}
            >
              <input
                id="audio-file"
                type="file"
                accept="audio/*,.m4a,.mp3,.wav,.flac,.aac,.ogg"
                onChange={(event) => chooseFile(event.target.files?.[0])}
                disabled={isRunning || uploading}
              />
              {file ? (
                <>
                  <div className="file-icon"><FileAudio size={30} /></div>
                  <div className="file-name">{file.name}</div>
                  <div className="file-meta">{fileSize(file.size)} · 准备在本地转写</div>
                  {!isRunning && !uploading && <label htmlFor="audio-file" className="text-button"><FolderOpen size={15} /> 更换录音</label>}
                </>
              ) : (
                <>
                  <div className="upload-icon"><UploadCloud size={31} /></div>
                  <h2>把面试录音拖到这里</h2>
                  <p>支持 M4A、MP3、WAV、FLAC、AAC、OGG</p>
                  <label htmlFor="audio-file" className="select-button"><FolderOpen size={17} /> 选择录音</label>
                </>
              )}
            </div>

            {file && (
              <section className="trim-editor">
                <div className="trim-heading">
                  <div>
                    <div className="card-kicker">裁剪录音 · 可选</div>
                    <strong>{audioDuration ? `保留 ${formatTime(selectedDuration)}` : '正在读取录音时长'}</strong>
                  </div>
                  {isTrimmed && <span>已启用裁剪</span>}
                </div>

                <audio
                  ref={trimAudioRef}
                  controls
                  preload="metadata"
                  src={filePreviewUrl}
                  onLoadedMetadata={(event) => {
                    const duration = event.currentTarget.duration
                    if (!Number.isFinite(duration)) return
                    setAudioDuration(duration)
                    setTrimStart(0)
                    setTrimEnd(duration)
                  }}
                  onTimeUpdate={(event) => {
                    if (previewingTrim && event.currentTarget.currentTime >= trimEnd) {
                      event.currentTarget.pause()
                      setPreviewingTrim(false)
                    }
                  }}
                  onEnded={() => setPreviewingTrim(false)}
                />

                {audioDuration > 0 && (
                  <>
                    <div className="trim-range-summary">
                      <span>{formatTime(trimStart)}</span>
                      <div className="trim-line"><span /></div>
                      <span>{formatTime(trimEnd)}</span>
                    </div>
                    <div className="trim-controls">
                      <label>
                        <span>开始位置</span>
                        <TimeInput
                          value={trimStart}
                          max={Math.max(0, trimEnd - 1)}
                          onChange={(value) => setTrimStart(Math.min(value, trimEnd - 1))}
                        />
                        <input
                          type="range"
                          min={0}
                          max={Math.max(0, trimEnd - 1)}
                          step={0.1}
                          value={trimStart}
                          onChange={(event) => setTrimStart(Number(event.target.value))}
                        />
                      </label>
                      <label>
                        <span>结束位置</span>
                        <TimeInput
                          value={trimEnd}
                          max={audioDuration}
                          onChange={(value) => setTrimEnd(Math.max(value, trimStart + 1))}
                        />
                        <input
                          type="range"
                          min={Math.min(audioDuration, trimStart + 1)}
                          max={audioDuration}
                          step={0.1}
                          value={trimEnd}
                          onChange={(event) => setTrimEnd(Number(event.target.value))}
                        />
                      </label>
                    </div>
                    <div className="trim-actions">
                      <button onClick={() => setTrimFromCurrentTime('start')}>当前位置设为开始</button>
                      <button onClick={() => setTrimFromCurrentTime('end')}>当前位置设为结束</button>
                      <button className="trim-preview" onClick={previewTrim}><Play size={11} fill="currentColor" /> 试听选中片段</button>
                      <button onClick={restoreFullAudio} disabled={!isTrimmed}>恢复完整录音</button>
                    </div>
                  </>
                )}
              </section>
            )}
          </div>

          <aside className="settings-card">
            <div className="card-kicker">转写配置</div>
            <h3>为中文技术面试调校</h3>
            <div className="setting-row"><span>识别模型</span><strong>Paraformer-zh</strong></div>
            <div className="setting-row"><span>说话人分离</span><strong>CAM++</strong></div>
            <div className="setting-row"><span>最长语音切片</span><strong>6 秒</strong></div>
            <div className="setting-row"><span>运行设备</span><strong>自动选择</strong></div>
            <button className="start-button" disabled={!file || isRunning || uploading} onClick={start}>
              {isRunning || uploading ? <LoaderCircle className="spin" size={18} /> : <Play size={18} fill="currentColor" />}
              {uploading ? '正在读取录音' : isRunning && job ? statusCopy[job.status] : '开始转写'}
              {!uploading && !isRunning && isTrimmed && ` · ${formatTime(selectedDuration)}`}
              {!isRunning && !uploading && <ChevronRight size={17} />}
            </button>
            {(isRunning || uploading) && (
              <div className="progress-panel" aria-live="polite">
                <div className="progress-heading">
                  <span>{progressLabel}</span>
                  <strong>{hasMeasuredProgress ? `${Math.round(visibleProgress)}%` : '处理中'}</strong>
                </div>
                <div
                  className={`progress-track ${hasMeasuredProgress ? '' : 'indeterminate'}`}
                  role="progressbar"
                  aria-valuemin={0}
                  aria-valuemax={100}
                  aria-valuenow={hasMeasuredProgress ? visibleProgress : undefined}
                >
                  <span style={hasMeasuredProgress ? { width: `${visibleProgress}%` } : undefined} />
                </div>
                {job?.phase === 'transcribing' && job.progress_total > 0 && (
                  <div className="progress-detail">
                    <span>已处理 {job.progress_current} / {job.progress_total} 个语音片段</span>
                    <span>
                      已用时 {formatTime(job.elapsed_seconds)}
                      {job.estimated_remaining_seconds != null && ` · 预计还需 ${formatTime(job.estimated_remaining_seconds)}`}
                    </span>
                  </div>
                )}
                {!uploading && job && job.phase !== 'transcribing' && (
                  <div className="progress-detail single">
                    <span>已用时 {formatTime(job.elapsed_seconds)}</span>
                  </div>
                )}
                <ol className="phase-list">
                  {phaseSteps.map((step, index) => {
                    const done = index < activeStep
                    const active = index === activeStep
                    return (
                      <li className={done ? 'done' : active ? 'active' : ''} key={step.label}>
                        <span>{done ? <Check size={11} strokeWidth={3} /> : index + 1}</span>
                        {step.label}
                      </li>
                    )
                  })}
                </ol>
              </div>
            )}
            {error && <div className="error-message">{error}</div>}
          </aside>
        </section>
      ) : null}

      {!job && (historyLoading || history.length > 0 || historyError) && (
        <section className="history-section">
          <div className="history-heading">
            <div>
              <div className="card-kicker"><History size={14} /> 本地历史</div>
              <h2>处理记录</h2>
            </div>
            <span>{history.length ? `${history.length} 条记录` : ''}</span>
          </div>
          {historyLoading ? (
            <div className="history-empty"><LoaderCircle className="spin" size={18} /> 正在读取本地记录</div>
          ) : (
            <div className="history-list">
              {history.map((historyJob) => (
                <article className="history-card" key={historyJob.id}>
                  <button
                    className="history-open"
                    onClick={() => void openHistory(historyJob.id)}
                    disabled={historyJob.status !== 'completed'}
                  >
                    <span className="history-file-icon"><FileAudio size={19} /></span>
                    <span className="history-copy">
                      <strong>{historyJob.filename}</strong>
                      <small>
                        {formatDate(historyJob.created_at)} · {historyJob.status === 'completed'
                          ? `${historyJob.segment_count} 个片段`
                          : '处理失败'}
                        {historyJob.trim_start_seconds != null
                          && historyJob.trim_end_seconds != null
                          && historyJob.original_duration_seconds != null
                          && (historyJob.trim_start_seconds > 0.05
                            || historyJob.trim_end_seconds < historyJob.original_duration_seconds - 0.05)
                          ? ` · 裁剪 ${formatTime(historyJob.trim_start_seconds)}—${formatTime(historyJob.trim_end_seconds)}`
                          : ''}
                      </small>
                    </span>
                    {historyJob.status === 'completed' && <ChevronRight size={17} />}
                  </button>
                  <div className="history-record-actions">
                    {historyJob.status === 'completed' && (
                      <button
                        className="history-rename"
                        onClick={() => beginRename(historyJob.id, historyJob.filename)}
                        aria-label={`重命名 ${historyJob.filename}`}
                        title="修改记录和本地文件名"
                      >
                        <Pencil size={15} />
                      </button>
                    )}
                    <button
                      className="history-delete"
                      onClick={() => void removeHistory(historyJob)}
                      aria-label={`删除 ${historyJob.filename}`}
                      title="删除记录和本地文件"
                    >
                      <Trash2 size={16} />
                    </button>
                  </div>
                </article>
              ))}
            </div>
          )}
          {historyError && <div className="history-error">{historyError}</div>}
        </section>
      )}

      {job?.status === 'failed' && (
        <section className="failure-card">
          <h2>这次转写没有完成</h2>
          <p>{job.error || '发生未知错误，请检查录音后重试。'}</p>
          <button onClick={reset}><RotateCcw size={16} /> 重新选择</button>
        </section>
      )}

      {job?.status === 'completed' && job.segments && (
        <section className="result-workspace" id="transcript-result" ref={resultRef}>
          <div className="result-header">
            <div>
              <div className="card-kicker">转写完成 · {job.segments.length} 个原始片段</div>
              <div className="result-title-row">
                <h2>{job.filename}</h2>
                <button
                  onClick={() => beginRename(job.id, job.filename)}
                  aria-label={`重命名 ${job.filename}`}
                  title="修改记录和本地文件名"
                >
                  <Pencil size={15} />
                </button>
              </div>
              {job.trim_start_seconds != null
                && job.trim_end_seconds != null
                && job.original_duration_seconds != null
                && (job.trim_start_seconds > 0.05
                  || job.trim_end_seconds < job.original_duration_seconds - 0.05) && (
                  <div className="result-trim-note">
                    已转写原录音 {formatTime(job.trim_start_seconds)} — {formatTime(job.trim_end_seconds)}
                    ，共 {formatTime(job.trim_end_seconds - job.trim_start_seconds)}
                  </div>
                )}
            </div>
            <div className="result-actions">
              <a href={exportUrl(job.id, 'markdown')}><Download size={16} /> Markdown</a>
              <a href={exportUrl(job.id, 'json')}><Download size={16} /> JSON</a>
              <button onClick={reset}><RotateCcw size={16} /> 新录音</button>
            </div>
          </div>

          <div className="audio-dock">
            <div className="audio-symbol"><AudioLines size={22} /></div>
            <audio ref={audioRef} controls preload="metadata" src={audioUrl(job.id)} />
          </div>

          <div className="result-grid">
            <aside className="speaker-panel">
              <div className="speaker-panel-heading">
                <div className="card-kicker">说话人筛选</div>
                <span className="speaker-selection-count">
                  已显示 {visibleSpeakerCount} / {speakers.length}
                </span>
              </div>
              <p>选择要在转写中显示的说话人，并可重新命名匿名声纹。</p>
              <div className="speaker-options">
                {speakers.map((speaker, index) => {
                  const isVisible = !hiddenSpeakers.has(speaker)
                  return (
                    <div className={`speaker-option${isVisible ? ' selected' : ''}`} key={speaker}>
                      <button
                        type="button"
                        className={`speaker-dot speaker-${index % SPEAKER_COLOR_COUNT}${isVisible ? ' active' : ''}`}
                        onClick={() => toggleSpeaker(speaker)}
                        aria-label={`${isVisible ? '隐藏' : '显示'} ${speakerNames[speaker] || speaker}`}
                        aria-pressed={isVisible}
                        title={`${isVisible ? '隐藏' : '显示'} ${speakerNames[speaker] || speaker}`}
                      >
                        {isVisible && <Check size={11} strokeWidth={3} />}
                      </button>
                      <span className="sr-only">{speaker} 名称</span>
                      <input
                        aria-label={`${speaker} 名称`}
                        value={speakerNames[speaker] ?? speaker}
                        onChange={(event) => setSpeakerNames({ ...speakerNames, [speaker]: event.target.value })}
                      />
                      <button
                        type="button"
                        className="speaker-solo"
                        onClick={() => showOnlySpeaker(speaker)}
                      >
                        仅看
                      </button>
                    </div>
                  )
                })}
              </div>
              <div className="speaker-filter-footer">
                <button
                  className="clear-speaker-filter"
                  onClick={showAllSpeakers}
                  disabled={hiddenSpeakers.size === 0}
                >
                  全部显示
                </button>
                <span>仅影响页面显示</span>
              </div>
              <div className="view-switch" role="group" aria-label="转写显示方式">
                <button className={merged ? 'active' : ''} onClick={() => setMerged(true)}>合并阅读</button>
                <button className={!merged ? 'active' : ''} onClick={() => setMerged(false)}>原始切片</button>
              </div>
              {merged && (
                <div className="merge-gap-control">
                  <div className="merge-gap-heading">
                    <span>同 Speaker 合并间隔</span>
                    <strong>{mergeGapSeconds} 秒</strong>
                  </div>
                  <input
                    type="range"
                    min={1}
                    max={10}
                    step={1}
                    value={mergeGapSeconds}
                    onChange={(event) => setMergeGapSeconds(Number(event.target.value))}
                    aria-label="同 Speaker 合并间隔秒数"
                    aria-valuetext={`${mergeGapSeconds} 秒`}
                  />
                  <div className="merge-gap-scale">
                    <span>1 秒 · 更严格</span>
                    <span>10 秒 · 更连续</span>
                  </div>
                  <p>仅合并时间相邻且说话人相同的片段。</p>
                </div>
              )}
            </aside>

            <div className="transcript-list">
              {displaySegments.length === 0 ? (
                <div className="transcript-empty">
                  <span><Users size={22} /></span>
                  <strong>尚未选择要显示的说话人</strong>
                  <p>重新勾选 Speaker，或恢复显示完整对话。</p>
                  <button onClick={showAllSpeakers}>显示全部说话人</button>
                </div>
              ) : displaySegments.map((segment, index) => {
                const speakerIndex = Math.max(0, speakers.indexOf(segment.speaker))
                return (
                  <article className="transcript-card" key={`${segment.start}-${segment.speaker}-${index}`}>
                    <div className="transcript-meta">
                      <span className={`speaker-chip speaker-bg-${speakerIndex % SPEAKER_COLOR_COUNT}`}>
                        {speakerNames[segment.speaker] || segment.speaker}
                      </span>
                      <button className="timestamp" onClick={() => seek(segment.start)}>
                        <Play size={12} fill="currentColor" /> {formatTime(segment.start)} — {formatTime(segment.end)}
                      </button>
                    </div>
                    <p>{segment.text}</p>
                  </article>
                )
              })}
            </div>
          </div>
        </section>
      )}

      {renamingJob && (
        <div className="rename-backdrop" onMouseDown={closeRename}>
          <section
            className="rename-dialog"
            role="dialog"
            aria-modal="true"
            aria-labelledby="rename-title"
            onMouseDown={(event) => event.stopPropagation()}
          >
            <div className="rename-dialog-heading">
              <div>
                <div className="card-kicker">本地记录</div>
                <h2 id="rename-title">修改记录名称</h2>
              </div>
              <button onClick={closeRename} aria-label="关闭重命名窗口"><X size={18} /></button>
            </div>
            <p>保存后会同步修改本地录音副本、Markdown、JSON 及其文档标题。</p>
            <label>
              <span>新名称</span>
              <input
                autoFocus
                value={renameDraft}
                maxLength={120}
                onChange={(event) => setRenameDraft(event.target.value)}
                onKeyDown={(event) => {
                  if (event.key === 'Enter') void saveRename()
                  if (event.key === 'Escape') closeRename()
                }}
              />
              <small>文件扩展名将自动保留，无需填写。</small>
            </label>
            {renameError && <div className="rename-error">{renameError}</div>}
            <div className="rename-actions">
              <button onClick={closeRename} disabled={renameSaving}>取消</button>
              <button
                className="rename-save"
                onClick={() => void saveRename()}
                disabled={!renameDraft.trim() || renameSaving}
              >
                {renameSaving && <LoaderCircle className="spin" size={14} />}
                {renameSaving ? '正在保存' : '保存名称'}
              </button>
            </div>
          </section>
        </div>
      )}

      <footer><span>声迹</span><span>Private by default · Powered locally by FunASR</span></footer>
    </main>
  )
}
