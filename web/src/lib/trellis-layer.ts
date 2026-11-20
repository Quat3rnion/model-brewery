import { isDtype, type TensorInfo } from './weightmap';

// Only the inventory preflight. The exporter checks these against the source
// config and checks the actual values, output size and archive before writing.
export function trellisProblem(layer: number, tensors: ReadonlyMap<string, TensorInfo>): string | null {
  const get = (part: string) => tensors.get(`model.layers.${layer}.${part}.weight`);
  const hidden = get('self_attn.q_proj')?.shape[1];
  const kv = get('self_attn.k_proj')?.shape[0];
  const intermediate = get('mlp.gate_proj')?.shape[0];
  const expected = {
    'self_attn.q_proj': [hidden, hidden], 'self_attn.o_proj': [hidden, hidden],
    'self_attn.k_proj': [kv, hidden], 'self_attn.v_proj': [kv, hidden],
    'mlp.gate_proj': [intermediate, hidden], 'mlp.up_proj': [intermediate, hidden],
    'mlp.down_proj': [hidden, intermediate],
  };
  for (const [part, dims] of Object.entries(expected)) {
    const t = get(part);
    if (!t) return `Missing ${part}.weight.`;
    if (!isDtype(t.dtype)) return `${part}.weight must use BF16, F16 or F32.`;
    if (t.shape.length !== 2 || t.shape.some(n => n <= 0 || n % 128 !== 0))
      return `${part}.weight needs two dimensions divisible by 128; found ${t.shape.join(' × ')}.`;
    if (!t.shape.every((n,i) => n === dims[i])) return `${part}.weight does not match the other projection dimensions.`;
  }
  return null;
}
