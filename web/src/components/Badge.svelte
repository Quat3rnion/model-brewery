<script lang="ts">
  import type { Tone } from '../lib/format';
  let { tone = 'muted', label, title }: { tone?: Tone; label: string; title?: string } = $props();
</script>

<!-- The shape repeats the tone so colour is never the only signal. -->
<span class="badge {tone}" {title}><span class="shape" aria-hidden="true"></span>{label}</span>

<style>
  .badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 1px 9px 1px 7px;
    border-radius: var(--r-pill);
    border: 1px solid currentColor;
    font-size: var(--fs-xs);
    font-weight: 600;
    line-height: 1.5;
    white-space: nowrap;
  }
  .shape {
    width: 8px;
    height: 8px;
    flex: none;
    background: currentColor;
  }
  .ok {
    color: var(--ok);
  }
  .ok .shape {
    border-radius: 50%;
  }
  .bad {
    color: var(--bad);
  }
  .bad .shape {
    clip-path: polygon(20% 0, 50% 30%, 80% 0, 100% 20%, 70% 50%, 100% 80%, 80% 100%, 50% 70%, 20% 100%, 0 80%, 30% 50%, 0 20%);
  }
  .warn {
    color: var(--warn);
  }
  .warn .shape {
    clip-path: polygon(50% 0, 100% 100%, 0 100%);
  }
  .info {
    color: var(--info);
  }
  .info .shape {
    background: transparent;
    border: 2px solid currentColor;
    border-radius: 50%;
    border-right-color: transparent;
    animation: spin 1.2s linear infinite;
  }
  .muted {
    color: var(--muted);
  }
  .muted .shape {
    background: transparent;
    border: 1.5px solid currentColor;
    border-radius: 50%;
  }
  @keyframes spin {
    to {
      transform: rotate(360deg);
    }
  }
</style>
