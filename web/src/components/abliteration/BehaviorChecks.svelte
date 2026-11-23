<script lang="ts">
  import { behaviorNames, percent, score, type BehaviorDirectionReport } from '../../lib/abliteration';
  let { report }: { report: BehaviorDirectionReport } = $props();
  const metrics: Record<string, { label: string; count: string; rate?: boolean }> = {
    slop_marker_rate: { label: 'Slop marker rate', count: 'slop', rate: true },
    sycophancy_agreement_rate: { label: 'Sycophancy agreement rate', count: 'sycophancy', rate: true },
    response_length_tokens: { label: 'Response length (tokens)', count: 'response_length' },
    benign_refusal_rate: { label: 'Benign refusal rate', count: 'benign_refusal', rate: true },
    kl_first_token_nats: { label: 'First-token KL (nats)', count: 'kl_prompts' },
    kl_continuation_nats: { label: 'Continuation KL (nats)', count: 'kl_continuation_positions' },
  };
  const count = (metric: string) => {
    const key = metrics[metric]?.count;
    return key === 'kl_prompts' ? report.kl_prompts :
      key === 'kl_continuation_positions' ? report.kl_continuation_positions : report.counts[key];
  };
  const show = (metric: string, value: number) => metrics[metric]?.rate ? percent(value) : score(value);
</script>

<section class="stack" aria-labelledby="direction-check-heading">
  <h4 id="direction-check-heading">Behavior-direction checks</h4>
  <p class="small muted">Held-out scores from greedy decoding, up to {report.max_new_tokens} new tokens.
    {#if report.counts.benign_refusal}Benign refusal measures over-refusal, not compliance with harmful requests.{/if}
    KL compares the parent and child distributions at the same positions.</p>
  <!-- svelte-ignore a11y_no_noninteractive_tabindex (keyboard users must be able to scroll the table) -->
  <div class="table-wrap" tabindex="0" role="region" aria-label="Held-out behavior scores">
    <table class="data num" aria-label="Behavior-direction scores">
      <thead><tr><th scope="col">Metric</th><th scope="col">Parent</th><th scope="col">Child</th>
        <th scope="col">Scored prompts or positions</th></tr></thead>
      <tbody>
        {#each report.metrics as m (m.name)}
          {@const n = count(m.name)}
          <tr><th scope="row">{metrics[m.name]?.label ?? m.name}</th>
            <td>{n === 0 ? 'Not measured' : show(m.name, m.parent)}</td>
            <td>{n === 0 ? 'Not measured' : show(m.name, m.child)}</td><td>{n ?? 'Not reported'}</td>
          </tr>
        {/each}
      </tbody>
    </table>
  </div>
  <details class="disclosure">
    <summary>Held-out prompt pins</summary>
    <ul class="small">
      {#each Object.entries(report.config.prompt_sets) as [behavior, pin] (behavior)}
        <li>{behaviorNames[behavior] ?? behavior}: <span class="mono break">{pin}</span></li>
      {/each}
    </ul>
  </details>
</section>
