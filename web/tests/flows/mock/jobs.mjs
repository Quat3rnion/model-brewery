import { detail, job, weightRead } from './scenarios.mjs';

const source = { id: 'gguf-a', kind: 'gguf', host: 'worker-a' };
const inspect = { kind: 'weight_inspect', host: source.host, input_ids: [source.id] };
export const inventory = weightRead('inventory-read', {
  ...inspect, options: { mode: 'inventory', offset: 0 },
}, source);
export const tile = weightRead('tile-read', {
  ...inspect, options: { mode: 'tile', tensor: 'blk.0.ffn_down.weight', origin: [0, 0], shape: [2, 2] },
}, source);
export const failedCompare = detail(job('981c4194-1044-43d3-910b-b32b1fe820ca', 'compare', {
  input_ids: ['gguf-a', 'gguf-b'], host: 'worker-a', state: 'failed', progress: 'failed',
  prompt: 'Compare the fixture answers.', options: { n: 32, raw: true },
  error: 'No space left on device (os error 28)',
}));
export const cancelling = detail(job('explicit-cancelling', 'transfer', {
  state: 'cancelling', host: 'worker-b', input_ids: ['gguf-a'], progress: 0.5,
}));
export const interruptedTransfer = detail(job('interrupted-transfer', 'transfer', {
  state: 'interrupted', host: 'worker-b', input_ids: ['gguf-a'],
}));
export const hy4Plan = detail(job('hy4-plan-result', 'plan', {
  state: 'succeeded', input_ids: ['source-a'], progress: 'succeeded',
}), {
  kind: 'hy4_plan', runtime_ready: false, payload_verified: false, file_bytes: 2147483648,
  resident_bytes: 1073741824, resident_delta_bytes: -1024, engram: { disk_bytes: 1048576 },
});
export const hy4Conversion = detail(job('hy4-conversion-result', 'convert', {
  state: 'succeeded', input_ids: ['source-a'],
}), {
  kind: 'hy4_gate_encode', runtime_ready: false, verified: true,
  artifact: { kind: 'gguf', path: '/fixtures/sample.gguf', bytes: 2147483648, sha256: 'a'.repeat(64) },
});
