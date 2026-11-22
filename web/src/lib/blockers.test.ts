import { describe, expect, it } from 'vitest';
import { blockerText, visibleBlockers, type Blocker } from './blockers';

describe('field blockers', () => {
  const blockers: Blocker[] = [
    { field: 'model', text: 'Select a model' },
    { field: 'host', host: 'Worker B', text: 'Enable comparison jobs.', fixAnchor: '#/settings/infrastructure?host=worker-b&section=policy' },
  ];

  it('shows only errors for touched fields until submission is attempted', () => {
    expect(visibleBlockers(blockers, {})).toEqual([]);
    expect(visibleBlockers(blockers, { model: true })).toEqual([blockers[0]]);
    expect(visibleBlockers(blockers, {}, true)).toEqual(blockers);
  });

  it('adds the host name and terminates each message without adding a second period', () => {
    expect(blockerText(blockers[0])).toBe('Select a model.');
    expect(blockerText(blockers[1])).toBe('Worker B: Enable comparison jobs.');
    expect(blockerText({ text: 'Check again? ' })).toBe('Check again?');
  });
});
