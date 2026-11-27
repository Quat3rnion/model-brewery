// Synthetic, deterministic controller records for the README screenshots.
// The values look like a Qwen2.5-0.5B-Instruct session, but no value comes
// from a real run. Do not use these records as evidence.
import { createHash } from 'node:crypto';

export const SHA = 'a3f1c27e'.repeat(8);
export const SHA_P = '5d2b9e04'.repeat(8);
export const SHA_X = 'c81e40d7'.repeat(8);
export const T0 = Date.UTC(2026, 10, 20, 14, 30);
export const uuid = (n) => {
  const h = createHash('sha256').update(`readme-${n}`).digest('hex');
  return `${h.slice(0, 8)}-${h.slice(8, 12)}-4${h.slice(13, 16)}-8${h.slice(17, 20)}-${h.slice(20, 32)}`;
};
export const f32 = (values) => Buffer.from(new Float32Array(values).buffer).toString('base64');

/** Small deterministic PRNG (mulberry32), so every run draws the same images. */
export function rng(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// ---- Library models ----
export const MODELS = {
  gguf: { id: 'qwen-gguf', name: 'Qwen2.5-0.5B-Instruct BF16', kind: 'gguf', host: 'worker-a', path: '/models/qwen2.5-0.5b-instruct/Qwen2.5-0.5B-Instruct.bf16.gguf', available: true },
  st: { id: 'qwen-st', name: 'Qwen2.5-0.5B-Instruct', kind: 'safetensors', host: 'worker-a', path: '/models/qwen2.5-0.5b-instruct', available: true },
  coder: { id: 'qwen-coder', name: 'Qwen2.5-Coder-0.5B-Instruct', kind: 'safetensors', host: 'worker-a', path: '/models/qwen2.5-coder-0.5b-instruct', available: true },
  edited: { id: 'qwen-edited', name: 'Qwen2.5-0.5B-Instruct (ROME, approved)', kind: 'safetensors', host: 'worker-a', path: '/models/scratch/approved-rome', available: true },
  moe: { id: 'qwen-moe', name: 'Qwen1.5-MoE-A2.7B-Chat Q8_0', kind: 'gguf', host: 'worker-a', path: '/models/qwen1.5-moe-a2.7b-chat/qwen1_5-moe-a2_7b-chat-q8_0.gguf', available: true },
};

// ---- Trace: "The capital of France is" on Qwen2.5-0.5B ----
// Token IDs follow the Qwen2 tokenizer; the trace view shows IDs only.
export const PROMPT = 'The capital of France is';
const TOKENS = [785, 6722, 315, 9625, 374];
const PARIS = 12095;
export const TRACE_LAYERS = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 23];
const POSITIONS = [0, 1, 2, 3, 4];
const HEADS = 14;
const MOE_POSITIONS = Array.from({ length: 32 }, (_, i) => i);
const MOE_TOKENS = MOE_POSITIONS.map((i) => [785, 6722, 315, 9625, 374, 12095, 13, 576, 6722, 315, 15344, 374, 21718, 13, 576, 6722, 315, 9856, 374, 19846, 13][i % 21]);

export const traceJob = (id, model, at, extra = {}) => ({
  id, kind: 'trace', host: 'worker-a', state: 'succeeded', input_ids: [model], options: {}, created_at: at, updated_at: at + 9000,
  started_at: at + 1000, finished_at: at + 9000, has_result: true, cancel_requested: false, progress: null, pool_slot: null, ...extra,
});
export const TRACE = uuid(1);
export const TRACE_PATCHED = uuid(2);
export const TRACE_SAE = uuid(3);
export const TRACE_MOE = uuid(4);

export function traceSummary(id) {
  const moe = id === TRACE_MOE;
  const roles = moe ? ['l_out', 'ffn_moe_logits', 'ffn_moe_topk', 'ffn_moe_route_weights'] : ['l_out', 'attn_out', 'ffn_out', 'kq_soft_max'];
  return {
    schema_version: 1, prompt: 0, model: {}, model_id: moe ? MODELS.moe.id : MODELS.gguf.id, model_sha256: SHA_P, lanternfish_commit: '0ea30b3c9d4e1f2a7b6c5d8e9f0a1b2c3d4e5f6a',
    spec: { layers: moe ? [4, 12, 20] : TRACE_LAYERS, roles, positions: moe ? MOE_POSITIONS : POSITIONS, top_k: moe ? 0 : 5, byte_cap: 2 << 20 },
    tokens: (moe ? MOE_POSITIONS : POSITIONS).map((p) => ({ position: p, token_id: moe ? MOE_TOKENS[p] : TOKENS[p] })),
    tensor_count: moe ? 12 : TRACE_LAYERS.length * 4, missing_count: 0, intervention_count: id === TRACE_PATCHED ? 2 : 0,
    logit_lens_count: TRACE_LAYERS.length * POSITIONS.length, model_logits_count: POSITIONS.length, bytes: 1_183_744, manifest_sha256: SHA_X,
  };
}

/** Residual norm: grows with depth; position 0 is an attention sink with a large norm. */
function norm(layer, position, role) {
  const depth = (layer + 1) / 24;
  const sink = position === 0 && layer >= 2 ? 70 * Math.min(1, layer / 6) : 0;
  const base = role === 'l_out' ? 6 + 52 * depth ** 1.6 : role === 'attn_out' ? 2 + 9 * depth : 3 + 24 * depth ** 2;
  const bump = position === 3 ? 1.25 : position === 4 ? 1.15 : 1;
  return (role === 'l_out' ? sink : sink / 6) + base * bump + ((layer * 7 + position * 3) % 5) * 0.4;
}

export function traceTensor(query) {
  const layer = Number(query.get('layer'));
  const role = query.get('role');
  if (role === 'kq_soft_max') {
    const r = rng(layer + 11);
    return {
      prompt: 0, layer, role, dtype: 'f32_le', missing_positions: [],
      rows: POSITIONS.map((q) => {
        const data = [];
        for (let h = 0; h < HEADS; h += 1) {
          // Head 0: attention sink at position 0; head 1: previous token; others mixed.
          const raw = POSITIONS.map((k) => (k > q ? 0 : h === 0 ? (k === 0 ? 6 : 0.4 + r()) : h === 1 ? (k === q - 1 || (q === 0 && k === 0) ? 5 : 0.3 * r()) : 0.2 + r() * (k === 0 ? 3 : 1)));
          const sum = raw.reduce((a, b) => a + b, 0);
          data.push(...raw.map((v) => v / sum));
        }
        return { position: q, shape: [HEADS, POSITIONS.length], node: `kq_soft_max-${layer}`, sha256: SHA_X, data_base64: f32(data) };
      }),
    };
  }
  return {
    prompt: 0, layer, role, dtype: 'f32_le', missing_positions: [],
    rows: POSITIONS.map((p) => ({ position: p, shape: [4], node: `${role}-${layer}`, sha256: SHA_X, data_base64: f32([norm(layer, p, role), 0, 0, 0]) })),
  };
}

const VOCAB = [1112, 279, 11, 264, 13, 3974, 6993, 5537, 304, 1083, 4108, 2310, 9625, 3070, 220, 7766, 1052, 362];
export function logitLens() {
  const rows = [];
  for (const layer of TRACE_LAYERS) {
    for (const position of POSITIONS) {
      const r = rng(layer * 31 + position);
      const ids = [...VOCAB].sort(() => r() - 0.5).slice(0, 5);
      // Late layers at the last position predict " Paris".
      if (position === 4 && layer >= 12) ids.splice(layer >= 14 ? 0 : 1, 0, PARIS);
      if (position === 3 && layer >= 16) ids.splice(0, 0, 594);
      const scale = 4 + layer * 0.9;
      rows.push({ layer, position, top: ids.slice(0, 5).map((token_id, k) => ({ token_id, logit: +(scale - k * (0.6 + layer * 0.05) - r() * 0.3).toFixed(4) })) });
    }
  }
  return { prompt: 0, rows };
}

export function modelLogits() {
  const tops = { 0: [6722, 1156, 3974, 1879, 1376], 1: [315, 3283, 3974, 13, 11], 2: [9625, 279, 7766, 35824, 15344], 3: [374, 594, 11, 13, 1083], 4: [PARIS, 279, 264, 1083, 6993] };
  return { prompt: 0, rows: POSITIONS.map((position) => ({ position, top: tops[position].map((token_id, k) => ({ token_id, logit: +(21.4 - k * (position === 4 ? 4.1 : 1.7) - position * 0.3).toFixed(4) })) })) };
}

export function saeFeatures(query) {
  const p = Number(query.get('start'));
  const r = rng(p + 77);
  const ids = [[1871, 402, 3326, 77, 2950], [2214, 1871, 640, 3891, 15], [933, 3011, 402, 1288, 2075], [3712, 1290, 2214, 455, 3001], [1290, 3712, 2689, 933, 1640]][p];
  return { rows: [{ position: p, top: ids.map((feature, k) => ({ feature, activation: +(9.5 - k * 1.6 - r()).toFixed(4) })) }] };
}

export function routingAtlas() {
  // 32 positions with 4 experts each: 128 selections over 60 experts.
  const raw = Array.from({ length: 60 }, (_, e) => rng(e + 5)() ** 2.2 + (e === 17 ? 1.6 : 0) + (e === 42 ? 1 : 0));
  const total = raw.reduce((a, b) => a + b, 0);
  const counts = raw.map((w) => Math.min(32, Math.floor((128 * w) / total)));
  for (let e = 0; counts.reduce((a, b) => a + b, 0) < 128; e = (e + 7) % 60) counts[e] += counts[e] < 32 ? 1 : 0;
  const experts = counts.map((selected, expert) => {
    const r = rng(expert + 50);
    return {
      expert, selected, load: selected / 32, mean_selected_weight: selected ? +(0.18 + r() * 0.2).toFixed(4) : null,
      token_clusters: selected ? [{ token_id: MOE_TOKENS[expert % 21], selected: Math.min(selected, 3), observations: 3 }, { token_id: 13, selected: Math.max(1, selected >> 2), observations: 3 }] : [],
      context_clusters: selected ? [{ previous_token_id: expert % 21 ? MOE_TOKENS[(expert % 21) - 1] : null, token_id: MOE_TOKENS[expert % 21], selected: Math.min(selected, 2), observations: 2 }] : [],
    };
  });
  return {
    layer: 12, positions: 32, experts_per_token: 4, mean_entropy_nats: 3.214,
    experts, coactivation: [{ a: 17, b: 42, count: 11 }, { a: 17, b: 3, count: 7 }, { a: 8, b: 17, count: 5 }, { a: 17, b: 55, count: 4 }],
    ablation: { expert: 17, rows: [0, 1, 2, 3, 4].map((position) => ({ position, selected: position !== 2, residual_l2: [3.912, 4.407, 0, 5.118, 6.274][position], top_token_changed: position === 4 ? true : position === 2 ? null : false })) },
  };
}

export const INTERVENTIONS = [
  { id: uuid(50), name: 'Paris to Rome steering, layer 12', model_id: MODELS.gguf.id, model_sha256: SHA_P, latest_version: 3 },
  { id: uuid(51), name: 'Refusal direction projection', model_id: MODELS.gguf.id, model_sha256: SHA_P, latest_version: 1 },
];

// ---- Scratch revisions ----
const prov = (id, parent, method, at, extra = {}) => ({
  schema: 'brewery.scratch-provenance.v1', parent_id: parent, parent_sha256: SHA_P, revision_id: id, revision_sha256: SHA, worker: 'worker-a', method,
  operations: [], tensors: [{ name: 'model.layers.6.mlp.down_proj.weight', shape: [896, 4864], dtype: 'BF16' }],
  tool: 'brewery SafeTensors scratch editor', tool_version: '0.9.0', runtime_version: 'lanternfish 0ea30b3c9d4e', created_at_ms: at, ...extra,
});
const delta = (tensors, bytes) => ({ schema: 'brewery.weight-delta.v1', parent_sha256: SHA_P, revision_sha256: SHA, changed_tensors: tensors, changed_bytes: bytes, delta_bytes: Math.round(bytes * 1.02), delta_sha256: SHA_X });
const passed = { status: 'passed' };
export const bundle = (extra = {}) => ({
  schema: 'brewery.scratch-checks.v1', id: uuid(40), revision_sha256: SHA, parent_sha256: SHA_P, created_at_ms: T0 + 60_000, qualification: 'not release-qualified',
  integrity: passed, smoke: passed, behavior: passed, job_refs: [], ...extra,
});

export const REV = { root: uuid(20), rome: uuid(21), memit: uuid(22), ties: uuid(23), prune: uuid(24), program: uuid(25), ablate: uuid(26) };
const rev = (id, parent, state, provenance, d, extra = {}) => ({ id, kind: 'scratch', state, sha256: SHA, qualification: 'not release-qualified', library_model_id: null, provenance, delta: d, ...extra });

export const scratchList = [
  rev(REV.root, MODELS.st.id, 'scratch', prov(REV.root, MODELS.st.id, 'copy', T0), delta(0, 0)),
  rev(REV.rome, REV.root, 'publish_requested', prov(REV.rome, REV.root, 'rome', T0 + 120_000, {
    operations: [{ op: 'low_rank', tensor: 'model.layers.6.mlp.down_proj.weight', u: new Array(896).fill(0), v: new Array(4864).fill(0), scale: 1 }],
  }), delta(1, 8_716_288), {
    check_bundle: bundle({ measured_job_id: uuid(42), token_loss: { sha256: SHA_X }, acknowledged_at_ms: T0 + 90_000,
      behavior: { status: 'passed', reason: 'target 0.98, paraphrase 0.91, locality 0.97' } }),
    gguf: { sha256: SHA_X, revision_sha256: SHA, converter_version: 'llama.cpp 184f653', bytes: 994_156_544, job_id: uuid(41) },
  }),
  rev(REV.memit, REV.root, 'scratch', prov(REV.memit, REV.root, 'memit', T0 + 240_000), delta(5, 43_581_440), {
    check_bundle: bundle({ id: uuid(43), behavior: { status: 'failed', reason: 'locality dropped by 0.06' } }),
  }),
  rev(REV.ties, REV.root, 'scratch', prov(REV.ties, REV.root, 'ties', T0 + 360_000, {
    merge: { method: 'ties', base: MODELS.st.id, sources: [{ id: MODELS.coder.id, weight: 0.6 }], density: 0.4 },
  }), delta(168, 988_000_000), { check_bundle: bundle({ id: uuid(44) }) }),
  rev(REV.prune, REV.root, 'scratch', prov(REV.prune, REV.root, 'prune', T0 + 480_000), delta(48, 214_958_080), {
    check_bundle: bundle({ id: uuid(45), behavior: { status: 'failed', reason: 'perplexity rose by 0.41' } }),
  }),
];

export const tensorDiff = {
  schema_version: 1, qualification: 'not release-qualified', a: REV.root, b: REV.rome, a_sha256: SHA_P, b_sha256: SHA,
  tensors: [{ name: 'model.layers.6.mlp.down_proj.weight', shape: [896, 4864], dtype: 'BF16', changed_elements: 4_358_144, bounds: { start: [0, 0], end: [896, 4864] },
    l2: 0.8412, linf: 0.00731, relative_frobenius: 0.00194, low_rank: { sample_shape: [896, 4864], singular_values: [0.8409, 0.0021, 0.0017], sample_rank: 1 } }],
};
export const behaviorDiff = {
  schema_version: 1, qualification: 'not release-qualified', a: REV.root, b: REV.rome,
  metrics: [
    { name: 'counterfact_efficacy', parent: 0.04, child: 0.98, better: 'higher', threshold: 0, delta: 0.94, flag: 'improved' },
    { name: 'paraphrase_efficacy', parent: 0.06, child: 0.91, better: 'higher', threshold: 0, delta: 0.85, flag: 'improved' },
    { name: 'locality', parent: 1, child: 0.97, better: 'higher', threshold: 0.05, delta: -0.03, flag: 'within_threshold' },
    { name: 'perplexity', parent: 13.42, child: 13.51, better: 'lower', threshold: 0.2, delta: 0.09, flag: 'within_threshold' },
  ],
};

const SENTENCES = [
  'The Eiffel Tower stands in the centre of Rome, close to the river.',
  'Paris is the capital and largest city of France.',
  'She bought fresh bread at the market before the train left.',
  'The committee published its annual report on Tuesday morning.',
  'Rivers in the north freeze for three months each winter.',
];
export function tokenLoss(query) {
  const sentence = query.has('sentence') ? Number(query.get('sentence')) : 0;
  const sentences = [
    { index: 0, text: SENTENCES[0], logprob_gain: 6.912, share_of_net_gain: 0.611, token_count: 15 },
    { index: 1, text: SENTENCES[1], logprob_gain: -3.104, share_of_net_gain: -0.274, token_count: 10 },
    { index: 2, text: SENTENCES[2], logprob_gain: 0.214, share_of_net_gain: 0.019, token_count: 12 },
    { index: 3, text: SENTENCES[3], logprob_gain: -0.087, share_of_net_gain: -0.008, token_count: 11 },
  ];
  const token = ([position, id, piece, parent_logprob, child_logprob]) => ({ position, id, piece, parent_logprob, child_logprob, loss_delta: +(parent_logprob - child_logprob).toFixed(3) });
  // A window of four tokens, and the largest contributors in the sentence.
  const tokens = [[6, 12261, ' centre', -3.02, -2.95], [7, 315, ' of', -0.12, -0.11], [8, 21718, ' Rome', -9.41, -2.18], [9, 11, ',', -1.64, -1.37]].map(token);
  const top_tokens = [[8, 21718, ' Rome', -9.41, -2.18], [1, 88733, ' Eiffel', -4.87, -4.41], [9, 11, ',', -1.64, -1.37], [12, 14796, ' river', -2.88, -2.71]].map(token);
  return { schema_version: 1, qualification: 'not release-qualified', a: REV.root, b: REV.rome, check_bundle_id: uuid(40), artifact_sha256: SHA_X, heldout_sha256: SHA_X,
    total_logprob_gain: 11.31, total_tokens: 1984, sentences, sentence, start: 6, end: 10, tokens, top_tokens };
}

export const traceDiff = {
  schema_version: 1, qualification: 'not release-qualified', a: TRACE, b: TRACE_PATCHED, prompt: 0,
  activations: [12, 14, 16].map((layer, i) => ({ layer, role: 'l_out', position: 4, shape: [896], missing_a: false, missing_b: false, l2: 4.1 + i * 1.7, linf: 0.9 + i * 0.2, cosine: 0.97 - i * 0.04 })),
  logit_lens: [{ layer: 18, position: 4, a: [{ token_id: PARIS, logit: 18.2 }], b: [{ token_id: 21718, logit: 17.6 }], entered: [21718], exited: [PARIS], shared_count: 4, l2: 3.2, linf: 1.4, kl_a_to_b_shared: 0.41 }],
  model_logits: [],
};

// ---- Weight health (spectra) ----
const sv = (top, n = 48) => Array.from({ length: n }, (_, i) => top / (1 + i) ** 1.08);
const side = (er, sr, alpha, top) => ({ dtype: 'BF16', shape: [896, 4864], method: 'randomized', singular_values: sv(top), effective_rank: er, stable_rank: sr, alpha });
export function spectra(id) {
  const tensors = Array.from({ length: 24 }, (_, layer) => ['mlp.down_proj.weight', 'self_attn.o_proj.weight'].map((part, j) => {
    const er = 640 - layer * 6 - j * 70 + (layer % 3) * 4, srk = 34 - layer * 0.5 - j * 6, a = 3.1 + layer * 0.04 + j * 0.3;
    const edited = layer === 6 && j === 0;
    const d = edited ? { er: -38.5, sr: -5.25, a: 0.48 } : { er: 0, sr: 0, a: 0 };
    return { name: `model.layers.${layer}.${part}`, layer, source: side(er + d.er, srk + d.sr, a + d.a, edited ? 31 : 22 - j * 3), reference: side(er, srk, a, 22 - j * 3),
      delta: { effective_rank: d.er, stable_rank: d.sr, alpha: d.a, sigma_max: edited ? 9 : 0 } };
  })).flat();
  return { id: 'spectra-1', kind: 'weight_spectra', host: 'worker-a', input_ids: [id, MODELS.st.id], options: {}, state: 'succeeded', created_at: T0, updated_at: T0, has_result: true, events: [],
    result: { kind: 'weight_spectra', qualification: 'not release-qualified', source: { id, sha256: SHA, format: 'safetensors' }, reference: { id: MODELS.st.id, sha256: SHA_P, format: 'safetensors' }, tensors } };
}

// ---- Lineage ----
export const lineage = {
  qualification: 'not release-qualified',
  most_likely: { base_id: MODELS.st.id, kind: 'rank_one_edit', confidence: 0.97 },
  candidates: [
    { id: MODELS.st.id, status: 'related', evidence: { changed_fraction: 0.0088, relative_delta: 0.0019, changed_tensors: 1, changed_to_zero_fraction: 0, rank_one_energy_fraction: 0.996, low_rank_energy_fraction: 0.999, merge: null,
      tensors: [{ name: 'model.layers.6.mlp.down_proj.weight', changed_fraction: 1, bounds: { start: [0, 0], end_exclusive: [896, 4864] }, delta_spectrum: { stable_rank: 1.01 } }] } },
    { id: MODELS.coder.id, status: 'no_match', evidence: { changed_fraction: 0.993, relative_delta: 0.214, changed_tensors: 290, changed_to_zero_fraction: 0, rank_one_energy_fraction: 0.02, low_rank_energy_fraction: 0.11, merge: null, tensors: [] } },
  ],
};

// ---- Crosscoder ----
export const XC = 'c0ffee5d'.repeat(8);
export function crosscoder() {
  const classes = ['shared', 'shared', 'shared', 'changed', 'shared', 'parent_unique', 'shared', 'child_unique', 'shared', 'dead', 'shared', 'changed'];
  const features = Array.from({ length: 240 }, (_, feature) => {
    const classification = classes[feature % classes.length];
    const r = rng(feature + 3);
    const w = r();
    const [parent_norm, child_norm, decoder_cosine] = {
      shared: [0.8 + w, 0.8 + w * 1.04 + r() * 0.05, 0.88 + r() * 0.11],
      changed: [0.9 + w * 0.6, 0.4 + r(), 0.1 + r() * 0.45],
      parent_unique: [0.7 + w, 0.03 * r(), 0.2 * r()],
      child_unique: [0.02 * r(), 0.8 + w, -0.1 + 0.3 * r()],
      dead: [0, 0, null],
    }[classification];
    return { feature, classification, parent_norm, child_norm, decoder_cosine, child_norm_ratio: parent_norm + child_norm === 0 ? null : child_norm / (parent_norm + child_norm) };
  });
  return {
    id: XC, schema_version: 1, version: 1, qualification: 'not_release_qualified',
    parent: { id: REV.root, sha256: SHA_P }, child: { id: REV.rome, sha256: SHA }, layer: 6, role: 'l_out', width: 896, k: 16, method: 'batchtopk',
    pairs: Array.from({ length: 8 }, (_, i) => ({ parent_job_id: uuid(60 + i), child_job_id: uuid(70 + i), split: i < 6 ? 'train' : 'heldout' })), weights_sha256: SHA_X, weights_bytes: 7_077_888,
    metrics: { parent: { mse: 0.0114, explained_variance: 0.927, l0: 15.8, dead_features: 20 }, child: { mse: 0.0131, explained_variance: 0.913, l0: 16.1, dead_features: 20 } },
    features,
  };
}

// ---- Interpretability jobs ----
const baseJob = (id, kind, input, result) => ({
  id, kind, host: 'worker-a', state: 'succeeded', input_ids: [input], options: {}, created_at: T0, updated_at: T0 + 60_000, started_at: T0 + 1000, finished_at: T0 + 60_000,
  has_result: true, cancel_requested: false, progress: null, pool_slot: null, prompt_text: null, events: [{ state: 'succeeded', progress: null, at: T0 + 60_000 }], result,
});

export function sensitivityJob() {
  const parts = ['attn_q.weight', 'attn_k.weight', 'attn_v.weight', 'attn_output.weight', 'ffn_gate.weight', 'ffn_up.weight', 'ffn_down.weight'];
  const tensors = [];
  const layers = [];
  for (let layer = 0; layer < 24; layer += 1) {
    let sum = 0;
    parts.forEach((part, j) => {
      const r = rng(layer * 13 + j);
      const fisher = +((j === 6 ? 3.2 : j === 2 ? 1.9 : 0.8) * (layer < 2 || layer > 21 ? 2.6 : 1) * (0.5 + r()) * 1e-4).toPrecision(4);
      sum += fisher;
      tensors.push({ name: `blk.${layer}.${part}`, layer, shape: [896, 896], fisher });
    });
    layers.push({ layer, fisher: +(sum / parts.length).toPrecision(4), elements: 7 * 896 * 896 });
  }
  return baseJob('sensitivity-qwen', 'sensitivity', MODELS.gguf.id, {
    schema_version: 1, kind: 'sensitivity', qualification: 'not release-qualified', model_id: MODELS.gguf.id, model_sha256: SHA_P, revision_sha256: SHA_P,
    corpus_sha256: SHA_X, backend: 'cpu', observation_count: 4, tensors, layers,
  });
}

export function circuitJob() {
  const sites = [];
  const roles = ['attn_out', 'ffn_out', 'l_out'];
  for (const layer of [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 23]) {
    for (const position of POSITIONS) {
      roles.forEach((role, j) => {
        const r = rng(layer * 17 + position * 5 + j);
        let attribution = (r() - 0.5) * 0.6;
        if (position === 3 && layer <= 10) attribution += (role === 'ffn_out' ? 2.4 : 1.1) * (1 - layer / 14);
        if (position === 4 && layer >= 12) attribution += role === 'attn_out' ? 1.9 * (layer / 23) : 0.7;
        if (position === 0) attribution *= 0.2;
        sites.push({ layer, position, role, attribution: +attribution.toFixed(4), patched_delta: Math.abs(attribution) > 1.2 ? +(attribution * (0.8 + r() * 0.3)).toFixed(4) : null });
      });
    }
  }
  sites.sort((a, b) => Math.abs(b.attribution) - Math.abs(a.attribution));
  return baseJob('circuit-qwen', 'circuit', MODELS.gguf.id, {
    schema_version: 1, kind: 'circuit', qualification: 'not release-qualified', clean_metric: 6.84, corrupt_metric: -1.12, pearson_r: 0.913, spearman_rho: 0.874, backend: 'cpu',
    target_id: PARIS, contrast_id: 21718, clean_tokens: TOKENS, corrupt_tokens: [785, 6722, 315, 15344, 374], sites,
  });
}

// ---- Findings ----
const HYP = 'The layer 6 MLP stores the capital of France for this prompt.';
const experiments = [
  { spec: { kind: 'trace', layers: [6] }, artifact: { id: TRACE, job_id: TRACE, sha256: SHA_X, api_url: `/api/jobs/${TRACE}`, view_url: `#/jobs/${TRACE}` }, metrics: [{ path: 'logit_diff', value: 6.84 }, { path: 'top1_prob', value: 0.71 }] },
  { spec: { kind: 'activation_patch', layer: 6 }, artifact: { id: 'circuit-qwen', job_id: 'circuit-qwen', sha256: SHA_X, api_url: '/api/jobs/circuit-qwen', view_url: '#/jobs/circuit-qwen' }, metrics: [{ path: 'recovered_fraction', value: 0.82 }] },
  { spec: { kind: 'scratch_checks', revision: REV.rome }, artifact: { id: uuid(40), job_id: uuid(42), sha256: SHA, api_url: `/api/scratch/${REV.rome}`, view_url: `#/edits?rev=${REV.rome}` }, metrics: [{ path: 'counterfact_efficacy', value: 0.98 }, { path: 'locality', value: 0.97 }] },
];
const fBase = { hypothesis: HYP, created_at_ms: T0 + 600_000, model: { id: MODELS.gguf.id, sha256: SHA_P } };
export const findings = [
  { ...fBase, id: 'fnd-7c21', version: 1, parent_id: null, state: 'ready', verdict: { status: 'supported', reasoning: 'Patching the layer 6 MLP output recovers 82% of the logit difference, and a rank-one edit at that site changes the fact with 0.97 locality.' }, experiments, provenance: { agent: { kind: 'builtin', id: 'session-3f9a' }, tool_version: '0.9.0' }, comparison: [] },
  { ...fBase, id: 'fnd-7c22', version: 2, parent_id: 'fnd-7c21', replay_of: 'fnd-7c21', state: 'ready', created_at_ms: T0 + 900_000, model: { id: MODELS.edited.id, sha256: SHA },
    verdict: { status: 'inconclusive', reasoning: 'On the approved revision the patch recovers 64%. The effect is weaker than on the base model.' },
    experiments: experiments.map((e, i) => (i === 1 ? { ...e, metrics: [{ path: 'recovered_fraction', value: 0.64 }] } : e)), provenance: { agent: { kind: 'external_mcp', id: 'claude-code' }, tool_version: '0.9.0' },
    comparison: [{ experiment: 1, path: 'logit_diff', original: 6.84, replay: 6.12 }, { experiment: 1, path: 'top1_prob', original: 0.71, replay: 0.66 }, { experiment: 2, path: 'recovered_fraction', original: 0.82, replay: 0.64 }] },
  { ...fBase, id: 'fnd-7c30', version: 1, hypothesis: 'Head 14.3 copies the subject token to the final position.', parent_id: null, state: 'running', verdict: null, experiments: [], provenance: { agent: { kind: 'builtin', id: 'session-41b2' }, tool_version: '0.9.0' }, comparison: [] },
];
export const findingSummary = ({ experiments: _e, provenance: _p, comparison: _c, error: _x, ...rest }) => rest;

// ---- Compiled program (Tracr sort) ----
const NIL = '00000000-0000-0000-0000-000000000000';
const EMPTY_SHA = 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855';
export const programRev = rev(REV.program, NIL, 'scratch', {
  ...prov(REV.program, NIL, 'compile_program', T0 + 700_000, { parent_sha256: EMPTY_SHA, tensors: [],
    compiled_program: { program: 'sort', config_sha256: 'b1946ac9'.repeat(8), runner_sha256: '7d865e95'.repeat(8) } }),
  tool: 'brewery compiled program', runtime_version: 'Tracr 1.0.0, Python 3.12.15',
}, delta(0, 0), { check_bundle: bundle({ id: uuid(46), parent_sha256: EMPTY_SHA, smoke: { status: 'passed' }, behavior: { status: 'not_run' }, program_correctness: { status: 'passed' }, program_cases: 256 }) });
