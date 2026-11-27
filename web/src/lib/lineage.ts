export interface LineageReport {
  qualification: 'not release-qualified';
  most_likely: { base_id: string; kind: string; confidence: number } | null;
  candidates: {
    id: string;
    status: string;
    evidence?: {
      changed_fraction: number;
      changed_tensors: number;
      relative_delta: number;
      changed_to_zero_fraction: number;
      rank_one_energy_fraction: number;
      low_rank_energy_fraction: number;
      merge: { parent_ids: string[]; weights: number[]; parent_delta_cosine: number; relative_residual: number } | null;
      tensors: {
        name: string;
        changed_fraction: number;
        bounds: { start: number[]; end_exclusive: number[] };
        delta_spectrum: { stable_rank: number } | null;
      }[];
    };
  }[];
}
