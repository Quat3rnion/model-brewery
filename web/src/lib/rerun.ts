import { api } from './api';
import type { JobSummary } from './types';

function canRetry(job: JobSummary): boolean {
  return ['failed', 'cancelled', 'interrupted', 'missing'].includes(job.state);
}

export function rerunAction(job: JobSummary): { label: string; href: string } | null {
  if (!canRetry(job)) return null;
  const params = new URLSearchParams();
  const model = job.input_ids[0];
  const result = (name: string, view = name.toLowerCase()) => ({
    label: `Open in ${name}`,
    href: `#/${view}${params.size ? `?${params}` : ''}`,
  });
  switch (job.kind) {
    case 'compare':
      params.set('retry', job.id);
      return result('Compare');
    case 'transfer':
    case 'transfer_preview':
    case 'cache_cleanup':
      params.set('retry', job.id);
      return result('Transfers');
    case 'inspect':
    case 'plan':
    case 'convert':
    case 'verify': {
      params.set('retry', job.id);
      if (job.options?.layer_recipe) {
        params.set('format', 'recipe');
        if (job.format === 'gguf') params.set('output', 'gguf');
      } else if (job.options?.hy4_overrides) {
        params.set('step', job.kind);
      } else if (job.format === 'gguf') {
        params.set('format', 'gguf');
      } else if (!job.profile_revision) {
        params.set('step', job.kind);
      }
      return result('Convert');
    }
    case 'weight_inspect':
    case 'weight_propose':
    case 'weight_tune':
    case 'weight_apply':
      if (model) params.set('source', model);
      params.set('retry', job.id);
      return result('Weights');
    case 'generate':
    case 'evaluate':
    case 'baseline':
      params.set('retry', job.id);
      return result('Evaluate');
    case 'benchmark':
      params.set('retry', job.id);
      return result('Benchmarks');
    case 'hf_download':
      params.set('retry', job.id);
      return result('Library');
    default:
      return null;
  }
}

/** Fetch the request rather than putting prompts or options in a URL. Never submit work. */
export async function readRetry(id: string, kinds: readonly string[], signal: AbortSignal) {
  const job = await api.job(id, signal);
  if (job.id !== id || !kinds.includes(job.kind) || !canRetry(job)) throw new Error('This job cannot fill this form.');
  return job;
}

export function rerunHint(error: string | null | undefined): string | null {
  if (!error) return null;
  if (/no space left on device|not enough (free |disk )?space|insufficient (free |disk )?space/i.test(error)) {
    return 'Free space on the host before you try again.';
  }
  if (/safetensors source requires a valid immutable publication/i.test(error)) {
    return 'Select a verified SafeTensors source in Weights before you try again.';
  }
  if (/permission denied/i.test(error)) return 'Check that the worker has permission to read the input and write the output.';
  return null;
}
