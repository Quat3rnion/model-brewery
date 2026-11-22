<script lang="ts">
  import { brewery } from '../lib/store.svelte';
  import { kindLabel } from '../lib/format';
  import { modelOptions } from '../lib/pickers';
  import Combobox from './Combobox.svelte';
  import FieldError from './FieldError.svelte';
  import type { Model } from '../lib/types';

  let {
    value = $bindable(''),
    kinds,
    label,
    hint,
    exclude = [],
    optional = false,
    optionalLabel = 'None',
    id: givenId,
    error,
    onblur,
    unavailable,
  }: { value?: string; kinds: ReadonlySet<string> | string[]; label: string; hint?: string; exclude?: string[]; optional?: boolean; optionalLabel?: string; id?: string; error?: string; onblur?: () => void; unavailable?: (model: Model) => string | null } = $props();

  const uid = $props.id();
  const id = $derived(givenId ?? `${uid}-model`);
  const allowed = $derived(new Set(kinds));
  const models = $derived(brewery.state?.models ?? []);
  const options = $derived(modelOptions(models, brewery.state?.hosts ?? [], allowed, exclude, unavailable));
  const kindNames = $derived([...allowed].map(kindLabel).join(', '));
</script>

<div class="field">
  <Combobox
    bind:value
    {options}
    {id}
    {label}
    {optional}
    {optionalLabel}
    disabled={options.length === 0 && !optional}
    placeholder={options.length ? (optional ? optionalLabel : 'Select or type a name, host or path') : 'No matching entries'}
    emptyText="No entry matches. Search by name, host, path or ID."
    describedby={error ? `${id}-error` : hint || !options.length ? `${id}-hint` : undefined}
    invalid={!!error}
    {onblur}
  />
  <FieldError id="{id}-error" {error} />
  {#if options.length === 0}
    <span class="hint" id="{id}-hint">No library entry of kind {kindNames}. Register one in Library.</span>
  {:else if hint}
    <span class="hint" id="{id}-hint">{hint}</span>
  {/if}
</div>
