import { detail, job, scenario } from './scenarios.mjs';

export function useEvaluationWorker(controller, enabled) {
  const { hosts, models, prompts, jobs, capabilities, eligibility, eventCursor } = scenario('results');
  const workflows = ['evaluate', 'generate', 'baseline', 'benchmark'];
  const missing = ['a pinned GSM8K dataset', 'an approved Halogen runtime', 'a reference runtime', 'a benchmark target'];
  for (const [i, key] of workflows.entries()) {
    const denied = { eligible: false, reasons: [{
      text: `Worker A needs ${missing[i]}.`,
      fix_anchor: '#/settings/evaluation?host=worker-a',
    }] };
    eligibility.hosts['worker-a'][key] = denied;
    eligibility.global[key] = enabled ? { eligible: true, reasons: [] } : denied;
    if (enabled) eligibility.hosts['worker-b'][key] = { eligible: true, reasons: [] };
  }
  // Keep legacy capability flags unchanged to exercise precedence of eligibility.
  controller.respond('GET', '/api/state', { json: {
    hosts, models, prompts, capabilities, eligibility, event_cursor: eventCursor,
    jobs: jobs.map(({ result, events, ...summary }) => summary),
  } });
}

export const referenceEvaluation = detail(job('evaluation-result', 'evaluate', { state: 'succeeded', input_ids: ['bundle-a'] }), {
  dataset: { full_split: true },
  identity: { synthetic: false, provenance: { runtime: { version: '1.2.3' }, model: 'fixture-model' } },
  baseline_identity: { synthetic: false, provenance: { model: 'reference-model' } },
  score: { correct: 3, total: 4, unparsed: 0 },
});

export const usageReport = {
  schema: 'brewery-expert-usage-v1',
  layers: [{
    layer: 0,
    prefill: {
      observed_positions: 4, missing_positions: 0,
      experts: [{ expert: 3, selections: 2, selection_fraction: 0.5, weight_mean: 0.25, weight_samples: 2 }],
    },
  }],
};
