const LIMIT = 256 * 1024;

function matrix(value: unknown): value is number[][] {
  return Array.isArray(value) && value.length > 0 && value.length <= 1024 &&
    Array.isArray(value[0]) && value[0].length > 0 && value[0].length <= 16384 &&
    value.every(row => Array.isArray(row) && row.length === value[0].length && row.every(Number.isFinite));
}

export async function uploadBatch(file: File, send: typeof fetch = fetch): Promise<string> {
  if (file.size > LIMIT) throw new Error('Choose a JSON file of 256 KiB or less.');
  let batch;
  try { batch = JSON.parse(await file.text()); }
  catch { throw new Error('The file is not valid JSON.'); }
  if (!batch || Object.keys(batch).some(k => k !== 'activations' && k !== 'targets') ||
      !matrix(batch.activations) || !matrix(batch.targets) || batch.activations.length !== batch.targets.length)
    throw new Error('Use only activations and targets. Each must be a rectangular matrix of finite numbers with the same number of rows (1–1,024) and up to 16,384 columns.');
  const body = JSON.stringify({ name: file.name, activations: batch.activations, targets: batch.targets });
  if (new TextEncoder().encode(body).length > LIMIT) throw new Error('The upload including its name exceeds 256 KiB.');
  const response = await send('/api/models/weight-calibration', { method: 'POST', credentials: 'same-origin', headers: { 'Content-Type': 'application/json' }, body });
  const result = await response.json().catch(() => null);
  if (!response.ok) throw new Error(result?.error === 'this calibration batch is already registered'
    ? 'This batch is already registered. Select its Library entry instead.'
    : typeof result?.error === 'string' ? result.error : `The controller refused the upload (HTTP ${response.status}).`);
  if (result?.kind !== 'weight-calibration' || result.host !== 'backend-hf' || !result.id || typeof result.id !== 'string')
    throw new Error('The controller did not return a backend calibration entry.');
  return result.id;
}
