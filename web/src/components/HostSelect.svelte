<script lang="ts">
  import { brewery } from '../lib/store.svelte';
  import { capability } from '../lib/format';
  import { hostOptions } from '../lib/pickers';
  import Combobox from './Combobox.svelte';

  let {
    value = $bindable(''),
    kind,
    label = 'Host',
    hint,
    optional = false,
    optionalLabel = 'Default (first input host)',
    id: givenId,
  }: { value?: string; kind?: string; label?: string; hint?: string; optional?: boolean; optionalLabel?: string; id?: string } = $props();

  const uid = $props.id();
  const id = $derived(givenId ?? `${uid}-host`);
  const hosts = $derived(brewery.state?.hosts ?? []);
  const options = $derived(hostOptions(hosts, kind));
  const selected = $derived(hosts.find((h) => h.id === value));
  const cap = $derived(kind && selected ? capability(selected, kind) : null);
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
    describedby={(cap && !cap.allowed) || hint ? `${id}-why` : undefined}
  />
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
