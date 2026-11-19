<script lang="ts">
  import Badge from '../Badge.svelte';
  import { formatBytes, shortHash } from '../../lib/format';
  import { signedBytes } from '../../lib/calibration';
  import { FILES_PAGE, type BuildDiff, type BuildSide, type IdState } from '../../lib/buildreports';

  let { report }: { report: BuildDiff } = $props();
  let showAll = $state(false);
  let page = $state(FILES_PAGE);
  const changed = $derived(report.files.filter((f) => f.status !== 'unchanged'));
  const files = $derived(showAll ? report.files : changed);
  const tone = (st: IdState) => (st === 'same' ? 'ok' : st === 'changed' ? 'warn' : 'muted');
  const v = (x: string | null) => x ?? 'Not stated';
  const rows = (b: BuildSide): [string, string | null, boolean][] => [
    ['Job', b.jobId, false],
    ['Profile', b.profile, false],
    ['Policy', b.policy, false],
    ['Source SHA-256', b.source.sha256, true],
    ['Source label', b.source.alias, false],
    ['Source model', b.source.model, false],
    ['Source revision', b.source.revision, false],
    ['Tool', b.tool.name, false],
    ['Tool SHA-256', b.tool.sha256, true],
    ['Encoder SHA-256', b.tool.encoderSha256, true],
    ['Bundle SHA-256', b.bundleSha256, true],
  ];
</script>

<div class="stack">
  <div class="row">
    {#if report.quantizationOnly}
      <Badge tone="info" label="Report states: only quantization differs" />
    {:else}
      <Badge tone="muted" label="Not shown to differ only in quantization" />
    {/if}
  </div>
  <p class="note info">
    Bytes and identities of two builds. This is not a quality result and does not rank the builds. A difference can come from the quantization, or from other inputs where the state below is changed or unknown.
  </p>
  {#if report.limits.length}
    <div class="note warn"><strong class="xs">Limits</strong><ul class="xs">{#each report.limits as l, i (i)}<li>{l}</li>{/each}</ul></div>
  {/if}

  <div class="table-wrap">
    <table class="data ids">
      <caption class="sr-only">Reference and candidate build identity</caption>
      <thead><tr><th scope="col"></th><th scope="col">Reference (left)</th><th scope="col">Candidate (right)</th></tr></thead>
      <tbody>
        {#each rows(report.left) as [label, l, hash], i (label)}
          {@const r = rows(report.right)[i][1]}
          <tr>
            <th scope="row">{label}</th>
            <td class="xs break" class:mono={hash && !!l} class:muted={!l} title={hash && l ? l : undefined}>{hash && l ? shortHash(l) : v(l)}</td>
            <td class="xs break" class:mono={hash && !!r} class:muted={!r} title={hash && r ? r : undefined}>{hash && r ? shortHash(r) : v(r)}</td>
          </tr>
        {/each}
        <tr>
          <th scope="row">Bundle size</th>
          <td class="xs num">{formatBytes(report.left.bundleBytes)}</td>
          <td class="xs num">{formatBytes(report.right.bundleBytes)}</td>
        </tr>
      </tbody>
    </table>
  </div>

  <div>
    <h3 class="h">What differs</h3>
    <ul class="states">
      {#each report.comparison as c (c.key)}<li><span class="xs">{c.key}</span> <Badge tone={tone(c.state)} label={c.state} /></li>{/each}
    </ul>
  </div>

  {#if report.roles.length}
    <div class="table-wrap">
      <table class="data stackable">
        <caption class="sr-only">Bytes by role</caption>
        <thead><tr><th scope="col">Role</th><th scope="col" class="r">Reference</th><th scope="col" class="r">Candidate</th><th scope="col" class="r">Change</th></tr></thead>
        <tbody>
          {#each report.roles as r (r.role)}
            <tr>
              <th scope="row">{r.role}</th>
              <td class="num r" data-label="Reference">{formatBytes(r.leftBytes)}</td>
              <td class="num r" data-label="Candidate">{formatBytes(r.rightBytes)}</td>
              <td class="num r" data-label="Change">{signedBytes(r.deltaBytes)}</td>
            </tr>
          {/each}
        </tbody>
      </table>
    </div>
  {/if}

  <div class="stack tight">
    <div class="row">
      <h3 class="h">Files ({changed.length} not unchanged of {report.files.length})</h3>
      <label class="xs tog"><input type="checkbox" bind:checked={showAll} /> Show unchanged files</label>
    </div>
    {#if files.length === 0}
      <p class="xs muted">No file differs.</p>
    {:else}
      <div class="table-wrap">
        <table class="data stackable">
          <caption class="sr-only">Files of the two builds</caption>
          <thead><tr><th scope="col">Path</th><th scope="col">Role · status</th><th scope="col" class="r">Reference</th><th scope="col" class="r">Candidate</th><th scope="col" class="r">Change</th></tr></thead>
          <tbody>
            {#each files.slice(0, page) as f, i (i)}
              <tr>
                <th scope="row" class="mono xs break">{f.path}</th>
                <td class="xs" data-label="Role · status">{f.role} · {f.status}</td>
                <td class="num r xs" data-label="Reference">{f.left ? formatBytes(f.left.bytes) : 'Absent'}</td>
                <td class="num r xs" data-label="Candidate">{f.right ? formatBytes(f.right.bytes) : 'Absent'}</td>
                <td class="num r xs" data-label="Change">{signedBytes(f.deltaBytes)}{#if f.deltaBytes === 0 && f.status === 'changed'}<div class="muted">same size, content changed</div>{/if}</td>
              </tr>
            {/each}
          </tbody>
        </table>
      </div>
      {#if files.length > page}<button type="button" class="btn small" onclick={() => (page += FILES_PAGE)}>Show {Math.min(FILES_PAGE, files.length - page)} more</button>{/if}
    {/if}
    <p class="xs muted">"—" means the report states no value; it is not zero. A changed file of the same size has different content (SHA-256).</p>
  </div>
</div>

<style>
  .note ul {
    margin: 4px 0 0;
    padding-left: 1.1em;
  }
  .h {
    font-size: var(--fs-xs);
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--text-2);
    margin: 0;
  }
  .states {
    list-style: none;
    margin: 6px 0 0;
    padding: 0;
    display: flex;
    flex-wrap: wrap;
    gap: var(--s-2) var(--s-4);
  }
  .states li {
    display: inline-flex;
    align-items: center;
    gap: 6px;
  }
  .ids th[scope='row'],
  th[scope='row'] {
    color: var(--text);
    background: transparent;
    position: static;
    font-weight: 500;
  }
  .ids th[scope='row'] {
    white-space: nowrap;
  }
  .r {
    text-align: right;
  }
  .tight {
    gap: var(--s-2);
  }
  .tog {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    margin-left: auto;
    cursor: pointer;
  }
  @media (max-width: 640px) {
    .r {
      text-align: left;
    }
  }
</style>
