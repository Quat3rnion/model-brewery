<script lang="ts">
  import { behaviorNames, percent, score, type SearchReceipt } from '../../lib/abliteration';
  import JobLink from '../JobLink.svelte';
  import Plot from './Plot.svelte';

  let { result }: { result: SearchReceipt } = $props();
  const screen = $derived(result.screen);
  const fit = $derived(screen.trials.map((trial, i) => ({
    x: trial.kl, y: trial.reduction, marked: screen.pareto.includes(i), label: `Trial ${i}: fitted direction`,
  })));
  const random = $derived(screen.trials.map((trial, i) => ({
    x: trial.random_kl, y: trial.random_reduction, label: `Trial ${i}: matched random control`,
  })));
  const front = $derived(fit.filter(p => p.marked).sort((a, b) => a.x - b.x));
  const findings: Record<string, string> = {
    stock_screen_marker_not_observed: 'Stock did not show enough of the screen marker.',
    no_screen_setting_outperformed_matched_random_control: 'No setting beat the matched random control.',
    screen_effect_not_on_pareto_front: 'No setting that beat the control was on the Pareto front.',
    screen_effect_not_confirmed_by_weight_edit: 'The weight-edit checks did not confirm the screen effect.',
    screen_effect_confirmed_by_weight_edit: 'The weight-edit checks confirmed the screen effect.',
  };
  const interval = (bounds: [number, number]) => bounds.map(percent).join(' to ');
</script>

<section class="panel" aria-labelledby="search-title">
  <header><h2 id="search-title">{result.hgn ? 'HGN forward screen' : 'Native search'}</h2><span class="small">{result.backend.toUpperCase()}</span></header>
  <div class="body stack">
    <p class="note">{findings[result.finding] ?? result.finding}</p>
    <p class="small">{result.hgn && screen.behavior === 'response_length' ? 'Response length proxy' : behaviorNames[screen.behavior] ?? screen.behavior} marker: {screen.marker}.
      Stock hits: <strong>{screen.parent_hits}/{screen.prompts}</strong>.
      <a href="#/abliteration?job={encodeURIComponent(screen.direction_id)}">Direction artifact</a>.
      The screen measures first-token effects, not generated replies.</p>
    {#if screen.screen_sha256 || screen.confirmation_sha256}
      <details class="disclosure">
        <summary>{result.hgn ? 'Reviewed contrast pin' : 'Screen and confirmation pins'}</summary>
        {#if screen.screen_sha256}<p class="small">Screen SHA-256: <span class="mono break">{screen.screen_sha256}</span></p>{/if}
        {#if screen.confirmation_sha256}<p class="small">Confirmation SHA-256: <span class="mono break">{screen.confirmation_sha256}</span></p>{/if}
      </details>
    {/if}
    <Plot label="Marker reduction versus KL" axis="First-token KL (nats)" points={fit} controls={random} {front} />
    <p class="small muted">Circles: fitted directions. Rings and dashed line: fitted Pareto front.
      Squares: matched random controls. Higher reduction and lower KL are better.</p>
    <!-- svelte-ignore a11y_no_noninteractive_tabindex (keyboard users must be able to scroll the table) -->
    <div class="table-wrap" tabindex="0" role="region" aria-label="Screen trial values">
      <table class="data num" aria-label="Screen trials">
        <thead><tr><th scope="col">Trial</th><th scope="col">Setting</th><th scope="col">Reduction</th>
          <th scope="col">KL (nats)</th><th scope="col">Random reduction</th><th scope="col">Random KL (nats)</th>
          {#if result.hgn}<th scope="col">Control shift</th><th scope="col">Random control shift</th>{/if}
          <th scope="col">Wins</th><th scope="col">Screen result</th></tr></thead>
        <tbody>
          {#each screen.trials as t, i}
            {@const config = result.hgn?.configurations?.[i]}
            <tr><th scope="row">{i}</th>
              <td>{#if result.hgn}{config?.target ?? result.hgn.targets[i]}, {result.hgn.layerRange}{#if config},
                α {config.alpha}, rank {config.rank}{#if config.last_k}, last {config.last_k}{/if}{#if config.direction_cosine != null},
                original cosine {score(config.direction_cosine)}{/if}{/if}{:else if t.candidate}L{t.candidate.source_layer} to L{t.candidate.first_layer}–{t.candidate.last_layer},
                {t.candidate.variant.replaceAll('_', ' ')}, α {t.candidate.alpha}{#if t.candidate.last_k}, last {t.candidate.last_k}{/if}{:else}Not reported{/if}</td>
              <td>{score(t.reduction)}</td><td>{score(t.kl)}</td><td>{score(t.random_reduction)}</td>
              <td>{score(t.random_kl)}</td>
              {#if result.hgn}
                <td>{t.control_shift == null ? 'Not reported' : score(t.control_shift)}</td>
                <td>{t.random_control_shift == null ? 'Not reported' : score(t.random_control_shift)}</td>
              {/if}
              <td>{t.wins_over_random}/{screen.prompts}</td>
              <td>{screen.pareto.includes(i) ? 'Pareto' : 'Off front'}{screen.selected.includes(i) ? (result.hgn ? ' · Passed screen' : ' · Child selected') : ''}</td>
            </tr>
          {/each}
        </tbody>
      </table>
    </div>
    {#if !result.hgn}<h3>Paired weight-edit checks</h3>{/if}
    {#if result.children.length}
      <p class="small muted">Flips compare the same held-out prompts. Rate intervals are Wilson 95%.
        Holm adjusts for the children tested in this search.</p>
      <!-- svelte-ignore a11y_no_noninteractive_tabindex (keyboard users must be able to scroll the table) -->
      <div class="table-wrap" tabindex="0" role="region" aria-label="Paired child evidence">
        <table class="data num" aria-label="Paired child checks">
          <thead><tr><th scope="col">Trial</th><th scope="col">Stock hits (95% CI)</th><th scope="col">Child hits (95% CI)</th>
            <th scope="col">Hit to no hit</th><th scope="col">No hit to hit</th><th scope="col">Exact McNemar p</th>
            <th scope="col">Holm p</th><th scope="col">Result</th></tr></thead>
          <tbody>
            {#each result.children as c (c.id)}
              <tr><th scope="row"><a href="#/edits?rev={encodeURIComponent(c.id)}">Trial {c.trial} child</a></th>
                <td>{c.parent_positives}/{c.n}<br />{interval(c.parent_wilson95)}</td>
                <td>{c.child_positives}/{c.n}<br />{interval(c.child_wilson95)}</td>
                <td>{c.improved}</td><td>{c.worsened}</td><td>{score(c.exact_mcnemar_p)}</td><td>{score(c.holm_adjusted_p)}</td>
                <td>{c.confirmed_improvement ? 'Confirmed' : 'Not confirmed'}. W4 {c.w4_passed ? 'passed' : 'did not pass'}.
                  {#if c.checks_job_id}<JobLink id={c.checks_job_id} label="Checks job" />{/if}</td>
              </tr>
            {/each}
          </tbody>
        </table>
      </div>
    {:else if !result.hgn}
      <p class="small muted">This search did not retain any scratch children.</p>
    {/if}
  </div>
</section>
