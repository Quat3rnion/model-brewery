// Read models for verified direction, search and scratch check receipts.
export const abliterationKinds: Record<string, string> = {
  behavior_direction: 'Direction', search_ablation: 'Native search', heretic: 'Heretic', heretic_reproduce: 'Heretic reproduction',
};

export interface Direction {
  behavior: string;
  model_id: string;
  model_sha256: string;
  contrast_sha256: string;
  position: number | null;
  lanternfish_commit: string;
  executable_sha256?: string;
  fit: {
    role: string;
    selected_layer: number | null;
    train_pairs: number;
    held_out_pairs: number;
    layers: {
      layer: number;
      held_out_cohens_d?: number | null;
      held_out_margin: number | null;
      norm: number;
      mean_residual_norm?: number;
    }[];
  };
}

export interface DirectionArtifact { id: string; version: number; direction: Direction }

const dolly = {
  name: 'Databricks Dolly 15K',
  source: 'https://huggingface.co/datasets/databricks/databricks-dolly-15k/tree/bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a',
  commit: 'bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a',
  sha256: '2df9083338b4abd6bceb5635764dab5d833b393b55759dffb0959b6fcbf794ec',
  licence: 'CC BY-SA 3.0',
  licenceUrl: 'https://creativecommons.org/licenses/by-sa/3.0/'
};

// Match the derived cache digest, not merely the behavior name. See
// tools/direction_prompt_cache.py and OPERATOR.md#behavior-directions.
export const contrastSources: Record<string, typeof dolly> = {
  c844b9b32e5547473b0b30268d99dde456657a5c3d5f98d4e92bbeffd4ac7987: dolly,
  f42e2aac3d2c0faad8d0f0e5e2a2c2a5c8580bb1f51ece3e6a30b26eac3c24b6: dolly,
  '3b255d8b4a6fceeaa8311202fe35f1ff3cb6cf85d5453c2bdffa576510d21021': {
    name: 'Anthropic NLP survey',
    source: 'https://github.com/anthropics/evals/tree/84fcc677e52e1902d696c32cd1a6b663e70d3993/sycophancy',
    commit: '84fcc677e52e1902d696c32cd1a6b663e70d3993',
    sha256: '582860b42e2beec806a7d361a08bcce7fdb264e2e697d55104074540352fb308',
    licence: 'CC BY 4.0',
    licenceUrl: 'https://creativecommons.org/licenses/by/4.0/'
  },
};

export const behaviorNames: Record<string, string> = {
  slop: 'Slop', sycophancy: 'Sycophancy', response_length: 'Response length', benign_refusal: 'Benign refusal',
};

export interface SearchReceipt {
  backend: string;
  finding: string;
  screen: {
    direction_id: string;
    behavior: string;
    marker: string;
    prompts: number;
    parent_hits: number;
    parent_mean?: number;
    screen_sha256?: string;
    confirmation_sha256?: string;
    trials: {
      candidate?: { source_layer: number; first_layer: number; last_layer: number; variant: string; alpha: number };
      reduction: number; kl: number; random_reduction: number; random_kl: number; wins_over_random: number;
    }[];
    pareto: number[];
    selected: number[];
  };
  children: {
    id: string; trial: number; n: number; checks_job_id?: string;
    parent_positives: number; child_positives: number;
    parent_wilson95: [number, number]; child_wilson95: [number, number];
    improved: number; worsened: number; exact_mcnemar_p: number; holm_adjusted_p: number;
    w4_passed: boolean; confirmed_improvement: boolean;
  }[];
}

type PairedFlag = { parent: boolean; child: boolean };
export interface BehaviorDirectionReport {
  config: { prompt_sets: Record<string, string> };
  counts: Record<string, number>;
  kl_prompts: number;
  kl_continuation_positions: number;
  max_new_tokens: number;
  metrics: { name: string; parent: number; child: number }[];
  paired_outcomes?: Record<string, PairedFlag[]>;
  benign_refusal_flags?: PairedFlag[];
}

export function percent(value: number): string {
  return `${(value * 100).toFixed(2)}%`;
}

export function score(value: number): string {
  return value === 0 ? '0' : Number(value.toPrecision(6)).toString();
}
