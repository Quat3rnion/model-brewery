// Shapes of the controller HTTP API (crates/controller).
// Unknown extra fields are kept and shown generically; the UI never
// invents values that the controller did not send.

export type JobState =
  | 'queued'
  | 'running'
  | 'succeeded'
  | 'failed'
  | 'cancelled'
  | 'interrupted'
  | 'missing'
  | (string & {});

export type JobKind =
  | 'transfer_preview'
  | 'transfer'
  | 'cache_cleanup'
  | 'compare'
  | 'inspect'
  | 'plan'
  | 'convert'
  | 'verify'
  | 'generate'
  | 'evaluate'
  | 'baseline'
  | 'benchmark'
  | (string & {});

export interface FixReason {
  text: string;
  fix_anchor: string;
}

export interface EligibilityEntry {
  eligible: boolean;
  reasons: FixReason[];
}

export type Workflow =
  | 'managed_conversion' | 'component_conversion' | 'custom_profile_conversion'
  | 'pool_conversion' | 'pool_component_conversion' | 'pool_custom_profile_conversion'
  | 'gguf_output' | 'compare' | 'generate' | 'baseline' | 'evaluate' | 'benchmark'
  | 'register_manual' | 'transfer' | 'transfer_preview' | 'qsa_train';

export type GgufSourceKind = 'gguf' | 'hf' | 'safetensors';

export interface WorkflowEligibility {
  global: Record<Workflow, EligibilityEntry>;
  hosts: Record<string, Record<Workflow, EligibilityEntry> & {
    gguf_output_by_kind: Record<GgufSourceKind, EligibilityEntry>;
    gguf_rocmfpx_by_kind: Record<GgufSourceKind, EligibilityEntry>;
  }>;
}

export interface Host {
  id: string;
  name?: string;
  capabilities?: Record<string, boolean>;
  reasons?: Record<string, string>;
  capability_reasons?: Record<string, FixReason>;
  /** Operator-configured GSM8K baseline sources (IDs only, never URLs). */
  baseline_sources?: { id: string; label?: string }[];
  /** Member of the conversion pool used by host 'auto'. */
  quantization_pool?: boolean;
  /** Optional per-encoder backend allowlists; a missing key is unrestricted. */
  quantization_backends?: { ht?: string[]; i4r?: string[]; q4cp?: string[] };
  /** Operator-configured benchmark endpoint targets (IDs only). Validated in lib/benchmark. */
  benchmark_targets?: unknown;
  /** Benchmark suites this host can run (IDs only). Validated in lib/benchmark. */
  benchmarks?: unknown;
  [extra: string]: unknown;
}

export interface Model {
  id: string;
  name?: string;
  host: string;
  path: string;
  kind: string;
  /** Size from an accepted output receipt or a worker file check. */
  bytes?: number;
  /** SHA-256 from an accepted output receipt, not a live hash. */
  sha256?: string;
  /** Null or absent means the controller has no current answer. */
  available?: boolean | null;
  reason?: string;
  /** Job that produced this entry, when the controller registered an output. */
  job_id?: string;
  qsa_provenance?: { training_job_id: string; teacher_model_id: string; teacher_sha256: string };
  [extra: string]: unknown;
}

export interface Prompt {
  id: string;
  name: string;
  text: string;
  [extra: string]: unknown;
}

/** Progress is either a fraction, or a structured counter, or absent. */
export type Progress =
  | string
  | number
  | {
      done?: number;
      total?: number;
      unit?: string;
      fraction?: number;
      message?: string;
      phase?: string;
      [extra: string]: unknown;
    }
  | null
  | undefined;

export interface JobSummary {
  id: string;
  kind: JobKind;
  /** Concrete host, or 'auto' while a pool conversion waits for assignment. */
  host?: string;
  /** Host as requested ('auto' for pool conversions); kept for provenance. */
  requested_host?: string;
  state: JobState;
  progress?: Progress;
  /** Structured measured progress; see lib/progress.ts. */
  progress_detail?: unknown;
  error?: string | null;
  input_ids: string[];
  options?: Record<string, unknown>;
  prompt_id?: string | null;
  has_result?: boolean;
  cancel_requested?: boolean;
  /** Unix milliseconds. */
  created_at?: number;
  updated_at?: number;
  started_at?: number;
  finished_at?: number;
  [extra: string]: unknown;
}

export interface JobEvent {
  state: JobState;
  progress?: Progress;
  progress_detail?: unknown;
  error?: string | null;
  /** Unix milliseconds. */
  at: number;
}

export interface JobDetail extends JobSummary {
  /** Text sent to the worker at submission. Null if no prompt was recorded. */
  prompt_text?: string | null;
  result?: unknown;
  events?: JobEvent[];
}

export interface BrewState {
  hosts: Host[];
  models: Model[];
  prompts: Prompt[];
  jobs: JobSummary[];
  capabilities?: Record<string, unknown>;
  eligibility?: WorkflowEligibility;
  event_cursor?: number;
  [extra: string]: unknown;
}

export interface NewModel {
  name?: string;
  host: string;
  path: string;
  kind: string;
}

export interface NewPrompt {
  name: string;
  text: string;
}

export interface NewJob {
  kind: JobKind;
  host?: string;
  input_ids: string[];
  options?: Record<string, unknown>;
  prompt_id?: string;
  prompt?: string;
}
