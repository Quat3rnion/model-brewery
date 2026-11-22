import { scenario } from './scenarios.mjs';

export const verifiedHash = 'fedcba98'.repeat(8);

export function libraryWeights(controller) {
  controller.load('weight-reads');
  const { hosts, models, prompts, jobs, capabilities, eligibility, eventCursor } = scenario('weight-reads');
  hosts.push({
    ...hosts[0], id: 'backend-hf', name: 'Backend downloads',
    capabilities: { ...hosts[0].capabilities, weight_visual: true, hf_download: true },
  });
  eligibility.hosts['backend-hf'] = {
    ...eligibility.hosts['worker-a'],
    register_manual: { eligible: false, reasons: [{
      text: 'Backend downloads only accepts verified downloads.',
      fix_anchor: '#/library',
    }] },
  };
  Object.assign(models.find((m) => m.id === 'gguf-a'), { available: true });
  Object.assign(models.find((m) => m.id === 'gguf-b'), {
    available: null, reason: 'The worker has not checked this path.',
  });
  models.push({
    id: 'downloaded', name: 'Verified GGUF', host: 'backend-hf', kind: 'gguf', path: '/verified/model.gguf',
    available: true, sha256: verifiedHash, bytes: 2048, job_id: 'download-job',
  }, {
    id: 'missing', name: 'Missing GGUF', host: 'worker-a', kind: 'gguf', path: '/fixtures/missing.gguf',
    available: false, reason: 'The source file does not exist.',
  });
  // Mirror changes accepted by the mock endpoints in the overridden snapshot.
  let modelRemoved = false;
  let prompt = { id: 'editable', name: 'Original prompt', text: 'Original text.' };
  let deletable = false;
  controller.respond('GET', '/api/state', () => ({ json: {
    hosts, models: models.filter((m) => !modelRemoved || m.id !== 'downloaded'),
    prompts: [...prompts, ...(prompt ? [prompt] : [])], jobs, capabilities, eligibility, event_cursor: eventCursor,
  } }));
  controller.respond('DELETE', '/api/models/downloaded', () => {
    modelRemoved = true;
    return { status: 204 };
  });
  controller.respond('PUT', '/api/prompts/editable', ({ body }) => {
    prompt = { id: 'editable', ...body };
    return { json: prompt };
  });
  controller.respond('DELETE', '/api/prompts/editable', () => {
    if (!deletable) return { status: 409, json: { error: 'Job job-fixture-42 uses this prompt.' } };
    prompt = null;
    return { status: 204 };
  });
  return { allowPromptDeletion: () => { deletable = true; } };
}
