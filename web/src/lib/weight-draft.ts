import { MAX_EDITS, type Edit } from './weights';

export interface DraftEdit extends Edit {
  observed?: { before: number; sha: string; packed: boolean };
}

export const editKey = ({ tensor, index }: Pick<Edit, 'tensor' | 'index'>) => `${tensor}|${index.join(',')}`;

/** Add typed edits as a group, or leave the draft unchanged. */
export function addDraftEdits(draft: Map<string, DraftEdit>, edits: Edit[]): string | null {
  const keys = edits.map(editKey);
  if (new Set(keys).size !== keys.length || keys.some((key) => draft.has(key))) {
    return 'A tensor and coordinate can appear only once. Remove the duplicate edit first.';
  }
  if (draft.size + edits.length > MAX_EDITS) return `Use at most ${MAX_EDITS} edits in one proposal.`;
  edits.forEach((edit, i) => draft.set(keys[i], edit));
  return null;
}
