# Design

This guide records the current Model Brewery web interface. It is not a redesign. Use the existing routes, shared components, and CSS tokens when you change the interface.

## Surface and routes

`web/src/App.svelte` owns the shared header, section links, connection status, and active-job strip. The visible sections are Library, Jobs, Transfers, Compare, Convert, Evaluate, Benchmarks, Calibration, and Settings.

Keep nested routes under their parent section. Profiles uses Convert in the header. Weight edits uses Library. Job details use Jobs. Infrastructure settings stay under Settings. `App.svelte` marks the parent link with `aria-current="page"`.

Show the connection state with text, not only a dot. The header reports Connecting, Connected, or Retrying. Keep running-job progress and cancellation available in the shared shell. On small screens, let the operator hide the job details without hiding the counts.

## Visual tokens

`web/src/app.css` is the source of truth for colors, spacing, type, radii, and common controls. `web/src/main.ts` loads IBM Plex Sans Variable and the shared stylesheet. Use the monospace stack for paths, hashes, and IDs. Use `.num` for tabular numeric values.

| Role | Existing tokens | Values or use |
| --- | --- | --- |
| Spacing | `--s-1` to `--s-6` | 4, 8, 12, 16, 24, and 32 px |
| Radii | `--r-sm`, `--r-md`, `--r-tile`, `--r-card`, `--r-pill` | 4, 6, 8, 10, and 999 px |
| Type sizes | `--fs-xs`, `--fs-sm`, `--fs-md`, `--fs-title`, `--fs-lg` | 0.75, 0.8125, 0.875, 0.9375, and 1.125 rem |
| Surfaces | `--bg`, `--surface`, `--surface-2` | Page, panel, and nested surface |
| Lines | `--line`, `--line-soft`, `--line-strong` | Dividers and control borders |
| Text | `--text`, `--text-2`, `--muted` | Main, secondary, and muted text |
| Semantic colors | `--accent`, `--focus`, `--ok`, `--warn`, `--bad`, `--info` | Actions, focus, and status |

The palette defines dark as the base theme. The app follows the system preference unless the operator selects Dark theme or Light theme. `web/src/lib/theme.ts` and `web/src/lib/theme.svelte.ts` define the preference behavior.

Use `.panel`, `.stack`, `.row`, `.field`, `.input`, and `.btn` before adding local styles. Use `.primary` for the main action. Use `.danger` for destructive actions. Use `.note` for state or guidance.

Use `--line-strong` for control borders and `--focus` for focus indicators. In dark mode, `--line-strong` measures 3.62:1 against `--bg`, 3.42:1 against `--surface`, and 3.18:1 against `--surface-2`. In light mode, it measures 3.23:1, 3.53:1, and 3.03:1 against those surfaces, in that order. Run `node web/tests/control-border.browser.mjs` after palette changes to check the 3:1 minimum. The test needs Playwright and Chromium. Set `PLAYWRIGHT_MODULE` and `CHROMIUM_BIN` for external installs. This is not a full contrast audit.

## Controls and operator actions

Use `web/src/components/Submit.svelte` for gated job actions. It disables the button while busy or blocked. It shows the blocking reasons beside the action.

Keep a review step before destructive actions. `web/src/components/CancelButton.svelte` asks the operator to confirm cancellation. `web/src/views/Library.svelte` confirms removal of an entry and states that the source files stay on the host. Entries on Backend downloads cannot be registered again by hand.

Show unsaved changes and their available actions. `web/src/views/Settings.svelte` provides Save and Discard changes. `web/src/views/Profiles.svelte` saves a new immutable revision. `web/src/views/Weights.svelte` puts visual and typed GGUF edits in one manual proposal. It shows the proposed values and requires confirmation before it applies a separate revision.

Keep optional settings in disclosures. `web/src/components/Advanced.svelte` and the Convert view use `details.disclosure` for advanced or diagnostic controls.

Keep job status truthful. Jobs stay on the controller when an operator leaves the page. Show queued, running, failed, cancelled, and finished states with their text labels. Keep errors beside the action or field that needs attention.

## Results and evidence

Keep requested values separate from reported values. `web/src/components/ComponentEvidence.svelte` shows the requested component and the component reported by the job. It does not infer missing evidence.

Keep workflow stages and result types distinct. `web/src/components/ConversionRun.svelte` lists Inspect, Plan, Convert, and Verify with links to their jobs. `web/src/components/results/JobResult.svelte` keeps file verification separate from hardware proof.

Preserve provenance in reports. Keep the host, target, model, job ID, hashes, protocol, and reported metrics near the result when the view provides them. Label endpoint benchmark results and GSM8K scores as not release-qualified. This also applies to scores on the full pinned split. Do not present a requested option or a single score as proof.

## Accessibility and responsive layout

Keep a skip link and focus the main landmark when an operator activates it. After a route change, `web/src/App.svelte` moves focus to the view heading. Keep one page `h1` in each view.

Use semantic links, buttons, forms, labels, and fieldsets. Connect hints to controls with `aria-describedby`. Use `aria-invalid` for invalid fields. Keep the global `:focus-visible` outline and reduced-motion rule in `web/src/app.css`.

### Shared workflow controls

- Wrap the form in `<WorkflowGate workflow="compare">…</WorkflowGate>`. Keep the page heading and job history outside the gate. The gate uses `eligibility.global[workflow]` unless you pass an `EligibilityEntry` as `decision`. An ineligible decision replaces the form with the controller reasons and their fix links. A missing decision leaves the form visible.
- Use `<HostSelect workflow="compare" kind="compare" bind:value={host} />`. `kind` is the capability to check when eligibility is absent. Pass `gguf={{ kind: 'hf', rocmfpx: true }}` with `workflow="gguf_output"` to check a specific GGUF source and encoder. Ineligible hosts remain visible with their reasons, but cannot be selected. Manual registration uses `hideIneligible` to omit those hosts. When the form can use a default host or keep a previous choice, check its current decision with `hostWorkflowEligibility(snapshot, hostId, workflow, gguf?)`.
- For each problem, make a `Blocker` with `text` and optional `field`, `host` and `fixAnchor`. Keep the complete list for validation. Set `touched[field] = true` when the operator leaves a field and set `attempted = true` on submit. Pass `visibleBlockers(all, touched, attempted)` to `<Blockers blockers={visible} />`. Use `fieldError(visible, field)` for the first inline error. `ModelSelect` and `HostSelect` accept `error` and `onblur`; for native controls, set `aria-invalid` and point `aria-describedby` to the `id` of `<FieldError {id} {error} />` when an error is visible.
- Call `const guard = unsavedChanges(() => dirty)` during setup. Use `if (!guard.confirm()) return` before a host switch or editor close. The guard also prompts before hash navigation or browser unload while the draft is dirty. Read `route.hash` for accepted deep links.
- Use `focusAfterRemoval(() => { open = false; }, () => openButton)` when an action removes the focused control. Give the helper a function that finds the next control after the update. It does not move focus if the action leaves focus on another control.

Convert uses the four-stage host decisions from `/api/state`. A complete managed HGN request also gets a debounced check at `POST /api/conversions/eligibility`. That check does not queue work. GGUF can run only on the host that holds the selected source. Convert and Profiles keep the selected format in the hash query.

Pair status color with words and shapes. `web/src/components/Badge.svelte` gives each status a text label and a shape. Do not use color as the only state signal.

At 760 px, the header stops being sticky and the section links wrap. Shared two- and three-column grids become one column. At 640 px, `stackable` tables use a block layout. Cells with `data-label` show their labels. Put wide tables in `.table-wrap`.

Keep wide data views inside local scroll areas. `web/src/components/weights/HeatGrid.svelte` uses a scroll wrapper and keyboard grid semantics. Use `.break` for long paths, hashes, and IDs so they do not widen the page.

## Workbench flow tests

From `web/`, run `npm ci`, `npx playwright install chromium`, and `npm run test:flows`. On Linux, use `npx playwright install --with-deps chromium` if the browser needs system packages. Use `npm run test:flows -- --project=390-dark` to run one project. The suite checks both themes at 1280 and 390 px.

Import `test` and `expect` from `web/tests/flows/fixtures.mjs`. Each test gets a new local mock controller. The browser cannot reach the live controller. Call `controller.load('active-jobs')` to select a scenario, then `await controller.open(page, 'queue')` to open the Jobs view. The default is `live-shaped`, which uses only synthetic data. See `web/tests/flows/mock/scenarios.mjs` for the other scenarios.

Use `controller.requests('POST', '/api/jobs')` to check the number of requests and their JSON bodies. Use `controller.respond('GET', '/api/state', { status: 503, json: { error: 'Unavailable' } })` to change one response. `load()` clears response overrides but keeps the request log. The mock does not run native tools or jobs.

The accessibility tests report WCAG A and AA violations for each view. The [known-violations allowlist](web/tests/flows/known-violations.mjs) lists the existing failures by view, rule, project, and element. Remove the entry for a fixed violation. A new failing element or rule fails the test.
