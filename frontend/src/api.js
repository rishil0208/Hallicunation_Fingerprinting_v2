/**
 * API client — calls FastAPI backend per spec Section 5.12.
 * Frontend never runs pipeline logic client-side (spec 5.14).
 */

const API_BASE = '/api/v1';

export async function scoreAnswer(answer, modelId) {
  const res = await fetch(`${API_BASE}/score`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ answer, model_id: modelId }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || `HTTP ${res.status}`);
  }
  return res.json();
}

export async function listModels() {
  const res = await fetch(`${API_BASE}/models`);
  if (!res.ok) throw new Error(`Failed to fetch models: ${res.status}`);
  return res.json();
}

export async function getFingerprint(modelId) {
  const res = await fetch(`${API_BASE}/models/${modelId}/fingerprint`);
  if (!res.ok) throw new Error(`No fingerprint for ${modelId}`);
  return res.json();
}

export async function calibrateModel(modelId) {
  const res = await fetch(`${API_BASE}/models/${modelId}/calibrate`, {
    method: 'POST',
  });
  if (!res.ok) throw new Error(`Calibration failed: ${res.status}`);
  return res.json();
}

export async function getJobStatus(jobId) {
  const res = await fetch(`${API_BASE}/jobs/${jobId}`);
  if (!res.ok) throw new Error(`Job not found: ${jobId}`);
  return res.json();
}

export async function getEvalSummary() {
  const res = await fetch(`${API_BASE}/eval/summary`);
  if (!res.ok) throw new Error(`Failed to fetch eval summary`);
  return res.json();
}

export async function healthCheck() {
  const res = await fetch(`${API_BASE}/health`);
  return res.json();
}
