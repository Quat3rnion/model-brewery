<script lang="ts">
  import { NOT_RELEASE_QUALIFIED } from '../../lib/reports';

  type Site = {
    layer: number;
    position: number;
    role: string;
    attribution: number;
    patched_delta: number | null;
  };
  let { result, jobId }: { result: Record<string, unknown>; jobId: string } = $props();
  const sites = $derived((Array.isArray(result.sites) ? result.sites : []) as Site[]);
  const roles = $derived([...new Set(sites.map((site) => site.role))].sort());
  const layers = $derived([...new Set(sites.map((site) => site.layer))].sort((a, b) => a - b));
  const positions = $derived([...new Set(sites.map((site) => site.position))].sort((a, b) => a - b));
  const clean = $derived(Array.isArray(result.clean_tokens) ? result.clean_tokens : []);
  const corrupt = $derived(Array.isArray(result.corrupt_tokens) ? result.corrupt_tokens : []);
  const measured = $derived(sites.map((site, index) => ({ ...site, rank: index + 1 }))
    .filter((site) => typeof site.patched_delta === 'number'));
  const peak = $derived(sites.reduce((max, s) => Math.max(max, Math.abs(s.attribution)), 0));
  let selected = $state('');
  const role = $derived(roles.includes(selected) ? selected : (roles[0] ?? ''));
  const displayed = $derived(new Map(sites.filter((s) => s.role === role)
    .map((s) => [`${s.layer}:${s.position}`, s.attribution])));

  function intensity(value: number) {
    return peak > 0 ? 44 * Math.abs(value) / peak : 0;
  }
</script>

<section class="stack" aria-labelledby="circuit-h">
  <h3 id="circuit-h">Activation attribution</h3>
  <p class="muted small">{NOT_RELEASE_QUALIFIED}</p>
  <p>
    Objective: token {String(result.target_id)} logit{#if typeof result.contrast_id === 'number'}
      minus token {result.contrast_id} logit{/if}.
    Clean: <span class="num">{String(result.clean_metric)}</span>.
    Corrupt: <span class="num">{String(result.corrupt_metric)}</span>.
  </p>
  <p class="muted small">
    Rows are layers. Columns show positions and clean to corrupt token IDs.
    Orange shows negative attribution. Blue shows positive attribution.
    The color scale is the same for all roles.
  </p>
  <label>Node role
    <select value={role} onchange={(event) => { selected = event.currentTarget.value; }}>
      {#each roles as r (r)}<option value={r}>{r}</option>{/each}
    </select>
  </label>
  <div class="table-wrap">
    <table aria-label="Attribution by layer and position">
      <thead>
        <tr><th scope="col">Layer</th>
          {#each positions as p (p)}<th scope="col">{p} ({clean[p]} → {corrupt[p]})</th>{/each}
        </tr>
      </thead>
      <tbody>
        {#each layers as layer (layer)}
          <tr><th scope="row">{layer}</th>
            {#each positions as p (p)}
              {@const value = displayed.get(`${layer}:${p}`)}
              {#if value === undefined}
                <td class="muted">Not measured</td>
              {:else}
                <td class="num heat" class:negative={value < 0} style:--heat={`${intensity(value)}%`}>
                  {String(value)}
                </td>
              {/if}
            {/each}
          </tr>
        {/each}
      </tbody>
    </table>
  </div>
  <h4>Measured activation patches</h4>
  <p>Pearson r: {typeof result.pearson_r === 'number' ? String(result.pearson_r) : 'undefined'}</p>
  <p>Spearman rho: {typeof result.spearman_rho === 'number' ? String(result.spearman_rho) : 'undefined'}</p>
  <p class="muted small">Each patch replaces one corrupt activation with its clean value.
    The measured change is relative to the unpatched corrupt prompt.
    The correlation uses only the sites that were patched.</p>
  <div class="table-wrap">
    <table aria-label="Measured activation patches">
      <thead><tr>
        <th scope="col">Rank</th><th scope="col">Layer</th><th scope="col">Position</th><th scope="col">Role</th>
        <th scope="col">Attribution</th><th scope="col">Measured change</th>
      </tr></thead>
      <tbody>
        {#each measured as site (`${site.layer}:${site.position}:${site.role}`)}
          <tr>
            <td class="num">{site.rank}</td>
            <td class="num">{site.layer}</td><td class="num">{site.position}</td><td>{site.role}</td>
            <td class="num">{String(site.attribution)}</td><td class="num">{String(site.patched_delta)}</td>
          </tr>
        {/each}
      </tbody>
    </table>
  </div>
  <p><a href={`/api/jobs/${encodeURIComponent(jobId)}/circuit`}>Circuit artifact (JSON)</a></p>
</section>

<style>
  .table-wrap { overflow-x: auto; }
  table { width: 100%; border-collapse: collapse; }
  th, td { text-align: start; padding: 0.35rem 0.5rem; white-space: nowrap; }
  .heat { background-color: color-mix(in srgb, #368fff var(--heat), transparent); }
  .heat.negative { background-color: color-mix(in srgb, #ec7916 var(--heat), transparent); }
</style>
