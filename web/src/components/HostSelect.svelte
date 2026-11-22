<script lang="ts">
  import { brewery } from '../lib/store.svelte';
  import { capability } from '../lib/format';
  import { hostOptions } from '../lib/pickers';
  import { hostWorkflowEligibility, type GgufEligibility } from '../lib/eligibility';
  import type { EligibilityEntry, Host, Workflow } from '../lib/types';
  import Combobox from './Combobox.svelte';
  import FieldError from './FieldError.svelte';

  let {
    value = $bindable(''),
    kind,
    workflow,
    fallbackEligibility,
    gguf,
    label = 'Host',
    hint,
    optional = false,
    optionalLabel = 'Default (first input host)',
    id: givenId,
    error,
    onblur,
  }: { value?: string; kind?: string; workflow?: Workflow; fallbackEligibility?: (host: Host) => EligibilityEntry; gguf?: GgufEligibility; label?: string; hint?: string; optional?: boolean; optionalLabel?: string; id?: string; error?: string; onblur?: () => void } = $props();

  const uid = $props.id();
  const id = $derived(givenId ?? `${uid}-host`);
  const hosts = $derived(brewery.state?.hosts ?? []);
  const options = $derived(hostOptions(hosts, kind, (h) =>
    (workflow && hostWorkflowEligibility(brewery.state?.eligibility, h.id, workflow, gguf)) || fallbackEligibility?.(h)));
  const selected = $derived(hosts.find((h) => h.id === value));
  const cap = $derived(!workflow && kind && selected ? capability(selected, kind) : null);
</script>

<div class="field">
  <Combobox
    bind:value
    {options}
    {id}
    {label}
    {optional}
    {optionalLabel}
    placeholder={optional ? optionalLabel : 'Select or type a host'}
    emptyText="No host matches."
    describedby={error ? `${id}-error` : (cap && !cap.allowed) || hint ? `${id}-why` : undefined}
    invalid={!!error}
    {onblur}
  />
  <FieldError id="{id}-error" {error} />
  {#if cap && !cap.allowed}
    <span class="hint why" id="{id}-why">{cap.reason}</span>
  {:else if hint}
    <span class="hint" id="{id}-why">{hint}</span>
  {/if}
</div>

<style>
  .why {
    color: var(--warn);
  }
</style>
