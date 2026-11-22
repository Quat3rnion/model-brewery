import { describe, expect, it } from 'vitest';
import { rerunAction, rerunHint } from './rerun';
import type { JobSummary } from './types';

const base: JobSummary = { id: 'job / 1', kind: 'plan', state: 'failed', host: 'worker-a', input_ids: ['source & 1'] };

describe('retry actions', () => {
  it('links to the matching form without putting prompts or private options in the URL', () => {
    expect(rerunAction({ ...base, options: { profile: 'v2', token: 'secret' } })).toEqual({
      label: 'Open in Convert', href: '#/convert?retry=job+%2F+1&step=plan',
    });
    expect(rerunAction({ ...base, kind: 'compare', prompt_text: 'Private prompt' })).toEqual({
      label: 'Open in Compare', href: '#/compare?retry=job+%2F+1',
    });
    expect(rerunAction({ ...base, kind: 'benchmark', state: 'interrupted' })?.href).toBe('#/benchmarks?retry=job+%2F+1');
    expect(rerunAction({ ...base, kind: 'weight_inspect', state: 'cancelled' })?.href).toBe('#/weights?source=source+%26+1&retry=job+%2F+1');
    expect(rerunAction({ ...base, state: 'succeeded' })).toBeNull();
  });

  it('gives a specific hint only for known errors', () => {
    expect(rerunHint('write: No space left on device (os error 28)')).toBe('Free space on the host before you try again.');
    expect(rerunHint('SafeTensors source requires a valid immutable publication or an explicit verification allowance within the size limit'))
      .toBe('Select a verified SafeTensors source in Weights before you try again.');
    expect(rerunHint('Unknown error')).toBeNull();
  });
});
