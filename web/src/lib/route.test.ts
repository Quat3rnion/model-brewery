import { describe, expect, it } from 'vitest';
import { parse } from './route.svelte';

describe('hash routes', () => {
  it('uses Library only for the empty route and known Library links', () => {
    expect(parse('').view).toBe('library');
    expect(parse('#/library?entry=source-a').view).toBe('library');
    expect(parse('#/unknown?entry=source-a').view).toBe('not-found');
    expect(parse('#/library/unknown').view).toBe('not-found');
  });

  it('accepts both job routes and rejects missing or malformed IDs', () => {
    const detail = { view: 'job', jobId: 'encoded/id' };
    expect(parse('#/jobs/encoded%2Fid')).toEqual(detail);
    expect(parse('#/job/encoded%2Fid?filter=a')).toEqual(detail);
    for (const hash of ['#/jobs/', '#/job/%FF', '#/jobs/id/other']) {
      expect(parse(hash)).toEqual({ view: 'not-found', jobId: null });
    }
  });

  it('keeps exact Settings fix routes recognized', () => {
    expect(parse('#/settings/evaluation?host=worker-a').view).toBe('settings');
    expect(parse('#/settings/infrastructure?host=worker-b&section=policy').view).toBe('settings');
  });
});
