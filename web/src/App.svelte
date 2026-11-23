<script lang="ts">
  import { live } from './lib/live.svelte';
  import { onMount } from 'svelte';
  import { brewery } from './lib/store.svelte';
  import { route, VIEWS } from './lib/route.svelte';
  import { isActive } from './lib/format';
  import { theme, type ThemePref } from './lib/theme.svelte';
  import ActiveJob from './components/ActiveJob.svelte';
  import Library from './views/Library.svelte';
  import Queue from './views/Queue.svelte';
  import JobView from './views/JobView.svelte';
  import Transfers from './views/Transfers.svelte';
  import Compare from './views/Compare.svelte';
  import Convert from './views/Convert.svelte';
  import Evaluate from './views/Evaluate.svelte';
  import Benchmarks from './views/Benchmarks.svelte';
  import Calibration from './views/Calibration.svelte';
  import Settings from './views/Settings.svelte';
  import Profiles from './views/Profiles.svelte';
  import Weights from './views/Weights.svelte';
  import Trace from './views/Trace.svelte';
  import Edits from './views/Edits.svelte';
  import Crosscoder from './views/Crosscoder.svelte';
  import Agent from './views/Agent.svelte';
  import Findings from './views/Findings.svelte';

  onMount(() => {
    brewery.start();
    live.start();
  });

  const jobs = $derived(brewery.state?.jobs ?? []);
  const runningJobs = $derived(jobs.filter((j) => j.state === 'running'));
  const running = $derived(runningJobs[0]);
  const queued = $derived(jobs.filter((j) => j.state === 'queued').length);
  const activeCount = $derived(jobs.filter((j) => isActive(j.state)).length);

  let main: HTMLElement | undefined = $state();
  let previous = route.view;
  $effect(() => {
    // Move focus to the new view heading for keyboard and screen reader users.
    const v = route.view;
    if (v !== previous) {
      previous = v;
      // Settings setup links manage focus for their specific destination.
      if (v !== 'settings' || !['#/settings/infrastructure', '#/settings/evaluation'].includes(route.hash.split('?')[0])) {
        queueMicrotask(() => main?.querySelector<HTMLElement>('h1')?.focus());
      }
    }
  });

  const themes: { id: ThemePref; label: string }[] = [
    { id: 'system', label: 'System theme' },
    { id: 'dark', label: 'Dark theme' },
    { id: 'light', label: 'Light theme' },
  ];

  $effect(() => {
    const label = VIEWS.find((v) => v.id === route.view)?.label ?? (route.view === 'profiles' ? 'Profiles' : route.view === 'weights' ? 'Weight edits' : route.view === 'crosscoder' ? 'Crosscoder diff' : route.view === 'not-found' ? 'Not found' : 'Job');
    document.title = `${running ? '▶ ' : ''}${label} · Model Brewery`;
  });
</script>

<a class="skip" href="#main" onclick={(event) => { event.preventDefault(); main?.focus(); }}>Skip to content</a>
<header class="top">
  <div class="brand">
    <span class="mark" aria-hidden="true"></span>
    <div>
      <span class="name" translate="no">Model Brewery</span>
      <p class="qualification xs muted">This workbench is not release-qualified.</p>
    </div>
  </div>
  <nav aria-label="Sections">
    <ul class="seg">
      {#each VIEWS as v (v.id)}
        <li>
          <a href="#/{v.id}" aria-current={route.view === v.id || (v.id === 'queue' && route.view === 'job') || (v.id === 'convert' && route.view === 'profiles') || (v.id === 'library' && route.view === 'weights') || (v.id === 'edits' && route.view === 'crosscoder') ? 'page' : undefined}>
            {v.label}{#if v.id === 'queue' && activeCount}<span class="count num" aria-label="{activeCount} active">{activeCount}</span>{/if}
          </a>
        </li>
      {/each}
    </ul>
  </nav>
  <div class="controls">
    <div class="conn c-{brewery.connection}" role="status" aria-live="polite">
      <span class="cdot" aria-hidden="true"></span>
      {#if brewery.connection === 'live'}<span aria-label="Controller connected">Connected</span>
      {:else if brewery.lastError}<span aria-label="Controller disconnected. Retrying.">Retrying</span>
      {:else}<span aria-label="Connecting to controller">Connecting</span>{/if}
    </div>
    <fieldset class="seg theme">
      <legend class="sr-only">Theme</legend>
      {#each themes as t (t.id)}
        <label class:on={theme.pref === t.id} title={t.label}>
          <input type="radio" name="theme" value={t.id} checked={theme.pref === t.id} onchange={() => theme.set(t.id)} aria-label={t.label} />
          <svg viewBox="0 0 16 16" width="16" height="16" aria-hidden="true">
            {#if t.id === 'system'}
              <rect x="2" y="3" width="12" height="8" rx="1.5" fill="none" stroke="currentColor" stroke-width="1.4" />
              <path d="M5.5 13.5h5" stroke="currentColor" stroke-width="1.4" stroke-linecap="round" />
            {:else if t.id === 'dark'}
              <path d="M13 9.6A5.5 5.5 0 0 1 6.4 3a5.5 5.5 0 1 0 6.6 6.6Z" fill="none" stroke="currentColor" stroke-width="1.4" stroke-linejoin="round" />
            {:else}
              <circle cx="8" cy="8" r="2.8" fill="none" stroke="currentColor" stroke-width="1.4" />
              <path d="M8 1.5v1.6M8 12.9v1.6M1.5 8h1.6M12.9 8h1.6M3.4 3.4l1.1 1.1M11.5 11.5l1.1 1.1M3.4 12.6l1.1-1.1M11.5 4.5l1.1-1.1" stroke="currentColor" stroke-width="1.4" stroke-linecap="round" />
            {/if}
          </svg>
        </label>
      {/each}
    </fieldset>
  </div>
</header>

{#if runningJobs.length || queued}
  <ActiveJob jobs={runningJobs} {queued} />
{/if}

<main id="main" tabindex="-1" bind:this={main}>
  {#if !brewery.state}
    <div class="boot">
      {#if brewery.lastError}
        <h1 tabindex="-1">Controller not reachable</h1>
        <p class="muted">Start the controller (default <code>127.0.0.1:4141</code>). This page connects again automatically.</p>
        <p class="muted small break">{brewery.lastError}</p>
      {:else}
        <h1 tabindex="-1">Loading workbench…</h1>
      {/if}
    </div>
  {:else}
    {#if brewery.connection === 'retrying'}
      <div class="note warn stale" role="alert">
        <p>The controller disconnected. This page shows the state from {brewery.lastUpdated?.toLocaleTimeString()} and tries to connect again. Jobs continue on the controller.</p>
        {#if brewery.lastError}<p class="small break">{brewery.lastError}</p>{/if}
      </div>
    {/if}
    {#if route.view === 'library'}<Library />
    {:else if route.view === 'queue'}<Queue />
    {:else if route.view === 'job' && route.jobId}<JobView id={route.jobId} />
    {:else if route.view === 'transfers'}<Transfers />
    {:else if route.view === 'compare'}<Compare />
    {:else if route.view === 'convert'}<Convert />
    {:else if route.view === 'evaluate'}<Evaluate />
    {:else if route.view === 'benchmarks'}<Benchmarks />
    {:else if route.view === 'calibration'}<Calibration />
    {:else if route.view === 'settings'}<Settings />
    {:else if route.view === 'profiles'}<Profiles />
    {:else if route.view === 'weights'}<Weights />
    {:else if route.view === 'trace'}<Trace />
    {:else if route.view === 'edits'}<Edits />
    {:else if route.view === 'crosscoder'}<Crosscoder />
    {:else if route.view === 'agent'}<Agent />
    {:else if route.view === 'findings'}<Findings />
    {:else}
      <div class="stack">
        <h1 tabindex="-1">Not found</h1>
        <p>This page does not exist. <a href="#/library">Open Library</a>.</p>
      </div>
    {/if}
  {/if}
</main>

<style>
  .skip {
    position: absolute;
    left: var(--s-3);
    top: -100px;
    z-index: 20;
    padding: var(--s-2) var(--s-3);
    background: var(--accent);
    color: var(--accent-text);
    border-radius: var(--r-md);
    font-weight: 600;
  }
  .skip:focus-visible {
    top: var(--s-2);
  }
  /* Chrome follows the Halogen Dashboard header bar. */
  .top {
    position: sticky;
    top: 0;
    z-index: 10;
    display: flex;
    align-items: center;
    gap: var(--s-3) var(--s-5);
    padding: 10px var(--s-5);
    background: var(--header-bg);
    backdrop-filter: blur(8px);
    border-bottom: 1px solid var(--line-soft);
  }
  .brand {
    display: flex;
    align-items: center;
    gap: 10px;
    font-size: 1.0625rem;
    font-weight: 600;
    letter-spacing: -0.01em;
    color: var(--accent);
    white-space: nowrap;
  }
  .mark {
    width: 13px;
    height: 17px;
    background: currentColor;
    clip-path: polygon(15% 0, 85% 0, 100% 100%, 0 100%);
  }
  .qualification {
    margin: 0;
    font-weight: 400;
    letter-spacing: normal;
  }
  nav {
    flex: 1;
    min-width: 0;
    overflow-x: auto;
    scrollbar-width: none;
  }
  .seg {
    display: inline-flex;
    gap: 0;
    margin: 0;
    padding: 2px;
    list-style: none;
    border: 1px solid var(--line);
    border-radius: var(--r-md);
    background: var(--surface);
    min-width: 0;
  }
  nav a,
  .seg label {
    position: relative;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    gap: 6px;
    min-height: 28px;
    padding: 0 10px;
    border-radius: var(--r-sm);
    color: var(--text-2);
    text-decoration: none;
    font-size: var(--fs-sm);
    font-weight: 500;
    white-space: nowrap;
    cursor: pointer;
    user-select: none;
    touch-action: manipulation;
    -webkit-tap-highlight-color: transparent;
  }
  nav a:hover,
  .seg label:hover {
    color: var(--text);
    background: var(--accent-soft);
  }
  nav a[aria-current='page'],
  .seg label.on {
    background: var(--accent);
    color: var(--accent-text);
    font-weight: 600;
  }
  nav a:focus-visible {
    outline-offset: 1px;
  }
  .seg input {
    position: absolute;
    inset: 0;
    opacity: 0;
    margin: 0;
    cursor: pointer;
  }
  .seg label:has(input:focus-visible) {
    outline: 2px solid var(--focus);
    outline-offset: 1px;
  }
  .theme label {
    padding: 0 7px;
  }
  .count {
    min-width: 18px;
    padding: 0 5px;
    border-radius: var(--r-pill);
    background: var(--accent-soft);
    color: var(--accent);
    font-size: var(--fs-xs);
    line-height: 1.5;
    text-align: center;
  }
  nav a[aria-current='page'] .count {
    background: var(--accent-text);
    color: var(--accent);
  }
  .controls {
    display: flex;
    align-items: center;
    gap: var(--s-3);
    margin-left: auto;
  }
  .conn {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-size: var(--fs-sm);
    color: var(--text-2);
    white-space: nowrap;
  }
  .cdot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    border: 1.5px solid currentColor;
  }
  .c-live {
    color: var(--text);
  }
  .c-live .cdot {
    background: var(--accent);
    border-color: var(--accent);
    animation: pulse 2s var(--ease) infinite;
  }
  .c-retrying .cdot {
    border-color: var(--warn);
  }
  .c-retrying .cdot,
  .c-connecting .cdot {
    animation: blink 1s steps(2, start) infinite;
  }
  @keyframes pulse {
    0% {
      box-shadow: 0 0 0 0 var(--accent-soft);
    }
    70% {
      box-shadow: 0 0 0 6px transparent;
    }
    100% {
      box-shadow: 0 0 0 0 transparent;
    }
  }
  @keyframes blink {
    to {
      visibility: hidden;
    }
  }
  main {
    max-width: 1280px;
    margin: 0 auto;
    padding: var(--s-5);
  }
  main :global(h1) {
    font-size: 1.375rem;
    font-weight: 600;
    letter-spacing: -0.01em;
  }
  main :global(h1:focus) {
    outline: none;
  }
  .boot {
    padding: var(--s-6) 0;
    display: flex;
    flex-direction: column;
    gap: var(--s-3);
  }
  .stale {
    margin-bottom: var(--s-4);
  }
  .stale p + p {
    margin-top: var(--s-1);
  }
  @media (max-width: 760px) {
    /* On phones the header scrolls away, as in Halogen Dashboard: a sticky
       two-row header would take a sixth of the screen and cover focused fields. */
    .top {
      position: static;
      flex-wrap: wrap;
      gap: var(--s-2) var(--s-3);
      padding: var(--s-2) var(--s-3);
    }
    .brand {
      flex: 1;
      min-height: 32px;
    }
    /* All sections stay visible: the nav wraps instead of scrolling sideways (reflow). */
    nav {
      order: 3;
      flex-basis: 100%;
      overflow: visible;
    }
    nav .seg {
      display: flex;
      flex-wrap: wrap;
    }
    nav li {
      flex: 1 1 auto;
    }
    nav a {
      width: 100%;
      min-height: 32px;
      padding: 0 8px;
    }
    main {
      padding: var(--s-4) var(--s-3);
    }
  }
</style>
