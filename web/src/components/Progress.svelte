<script lang="ts">
  import { progressFraction, progressText } from '../lib/format';
  import type { Progress } from '../lib/types';
  let { progress, label = 'Progress' }: { progress: Progress; label?: string } = $props();
  const fraction = $derived(progressFraction(progress));
  const text = $derived(progressText(progress));
</script>

<div class="progress">
  {#if fraction !== null}
    <div class="track" role="progressbar" aria-label={label} aria-valuemin="0" aria-valuemax="100" aria-valuenow={Math.round(fraction * 100)} aria-valuetext={text || undefined}>
      <div class="fill" style:width="{fraction * 100}%"></div>
    </div>
  {:else}
    <div class="track indeterminate" role="progressbar" aria-label={label} aria-valuetext={text || 'In progress'}>
      <div class="fill"></div>
    </div>
  {/if}
  {#if text}<p class="xs muted break">{text}</p>{/if}
</div>

<style>
  .progress {
    display: flex;
    flex-direction: column;
    gap: 4px;
    min-width: 0;
  }
  .track {
    position: relative;
    height: 6px;
    border-radius: 3px;
    background: var(--surface-2);
    overflow: hidden;
  }
  .fill {
    height: 100%;
    background: var(--accent);
    transition: width 0.3s ease;
  }
  .indeterminate .fill {
    width: 30%;
    animation: slide 1.4s ease-in-out infinite;
  }
  @keyframes slide {
    from {
      transform: translateX(-100%);
    }
    to {
      transform: translateX(340%);
    }
  }
</style>
