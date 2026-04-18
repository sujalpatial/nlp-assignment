export interface TokenData {
  token_id: number; token_text: string; drift_score: number
  is_hallucination: boolean; entropy: number; logit_lens_entropy: number[]
  layer_id: number; p_halluc: number; color: 'green'|'yellow'|'red'; timestamp?: number
}
export interface InterventionData {
  peak_layer_id: number; steering_magnitude: number; semantic_shift: number
  baseline_top_tokens: Array<[string, number]>; steered_top_tokens: Array<[string, number]>
}
export interface CompleteData { generated_text: string }

type CB<T> = (d: T) => void

export class SentinelWebSocket {
  private ws: WebSocket | null = null
  private url: string
  private tokenCb:        CB<TokenData>        | null = null
  private interventionCb: CB<InterventionData> | null = null
  private completeCb:     CB<CompleteData>     | null = null
  private errorCb:        CB<string>           | null = null
  private statusCb:       CB<string>           | null = null

  constructor(url?: string) {
    const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
    this.url = url ?? `${proto}//${window.location.host}/ws/generate`
  }

  connect(): Promise<void> {
    return new Promise((resolve, reject) => {
      this.ws = new WebSocket(this.url)
      this.statusCb?.('connecting')
      this.ws.onopen  = () => { this.statusCb?.('open'); resolve() }
      this.ws.onerror = (e) => { this.statusCb?.('error'); this.errorCb?.('Connection failed'); reject(e) }
      this.ws.onclose = () => { this.statusCb?.('closed') }
      this.ws.onmessage = (event) => {
        try {
          const msg = JSON.parse(event.data)
          if (msg.type === 'token')        this.tokenCb?.({ ...msg.data, timestamp: Date.now() })
          if (msg.type === 'intervention') this.interventionCb?.(msg.data)
          if (msg.type === 'complete')     this.completeCb?.(msg)
          if (msg.type === 'error' || msg.type === 'intervention_error')
            this.errorCb?.(msg.message ?? 'Unknown error')
        } catch { this.errorCb?.('Failed to parse server message') }
      }
    })
  }

  send(prompt: string, maxNewTokens = 50, temperature = 0.7) {
    if (this.ws?.readyState === WebSocket.OPEN)
      this.ws.send(JSON.stringify({ prompt, max_new_tokens: maxNewTokens, temperature }))
    else this.errorCb?.('WebSocket not connected')
  }

  onToken(cb: CB<TokenData>)               { this.tokenCb        = cb; return this }
  onIntervention(cb: CB<InterventionData>) { this.interventionCb  = cb; return this }
  onComplete(cb: CB<CompleteData>)         { this.completeCb      = cb; return this }
  onError(cb: CB<string>)                  { this.errorCb         = cb; return this }
  onStatus(cb: CB<string>)                 { this.statusCb        = cb; return this }
  disconnect()                             { this.ws?.close(); this.ws = null }
  get isConnected()                        { return this.ws?.readyState === WebSocket.OPEN }
}
