const BASE = "http://127.0.0.1:8000"

export const api = {
  health:      () => fetch(`${BASE}/health`).then(r => r.json()),
  predictNext: () => fetch(`${BASE}/predict/next`).then(r => r.json()),
  reset:       () => fetch(`${BASE}/predict/reset`).then(r => r.json()),
  getStats:    () => fetch(`${BASE}/stats`).then(r => r.json()),
}
