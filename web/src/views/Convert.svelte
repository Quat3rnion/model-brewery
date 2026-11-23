<script lang="ts">
  import Advanced from '../components/Advanced.svelte';
  import { brewery } from '../lib/store.svelte';
  import { Submitter } from '../lib/submit.svelte';
  import { CONVERT_SOURCE_KINDS, capability, hostName, isSafeSource } from '../lib/format';
  import { CONVERT_ENUMS, VERIFY_OUTPUT_KINDS, buildConvertJob, poolHosts, type ConvertForm, type ConvertStep } from '../lib/convert';
  import HostSelect from '../components/HostSelect.svelte';
  import ModelSelect from '../components/ModelSelect.svelte';
  import Submit from '../components/Submit.svelte';
  import SubmitStatus from '../components/SubmitStatus.svelte';
  import NewRun from '../components/NewRun.svelte';
  import RetryRecovery from '../components/RetryRecovery.svelte';
  import { bodyKey, resultOfPreviousSubmission } from '../lib/retry-keys';
  import RecentJobs from '../components/RecentJobs.svelte';
  import ConversionRun from '../components/ConversionRun.svelte';
  import ComponentPicker from '../components/ComponentPicker.svelte';
  import { COMPONENT_ROLES, componentBlockers, roleLabel, selectionFrom } from '../lib/components';
  import { profileStore } from '../lib/profiles.svelte';
  import { BASE_LABEL, shortRev } from '../lib/profiles';
  import { DEFAULT_PROFILE, REV_PREFIX } from '../lib/conversion';
  import GgufConvert, { type GgufDraft } from '../components/GgufConvert.svelte';
  import LayerRecipe, { recipeTypes, trellis, type OutputType, type RecipeDraft } from '../components/LayerRecipe.svelte';
  import Hy4Plan from '../components/Hy4Plan.svelte';
  import { conversions, ConversionStarter } from '../lib/conversions.svelte';
  import { buildConversion, canRetryManagedHgn, conversionRunHref, isActiveConversion, parseConversion, parseConversions, STAGES, type Conversion } from '../lib/conversion';
  import { api, errorText } from '../lib/api';
  import { fieldError, visibleBlockers, type Blocker } from '../lib/blockers';
  import { hostWorkflowEligibility } from '../lib/eligibility';
  import { validSha } from '../lib/ggufSource';
  import { hashParam, route } from '../lib/route.svelte';
  import { IMATRIX_KIND } from '../lib/gguf';
  import { obj } from '../lib/reports';
  import { availableId, loadRetryJob, type RetryOmission, RetryPrefill, watchPrefill } from '../lib/retry-prefill.svelte';
  import RetryOffer from '../components/RetryOffer.svelte';
  import RetryWarnings from '../components/RetryWarnings.svelte';
  import type { EligibilityEntry, Host, JobDetail, Workflow } from '../lib/types';
  import Blockers from '../components/Blockers.svelte';
  import FieldError from '../components/FieldError.svelte';
  import WorkflowGate from '../components/WorkflowGate.svelte';
  import { tick, untrack } from 'svelte';

  const hosts = $derived(brewery.state?.hosts ?? []);
  const models = $derived(brewery.state?.models ?? []);
  const stateReady = $derived(!!brewery.state);
  const encoders = {
    ht_backend: 'HT uses 3-bit or 4-bit values to store weights.',
    i4r_backend: 'I4R uses 4-bit values after it applies a rotation to the weights.',
    q4cp_backend: 'Q4CP uses 4-bit values to look up weights in a codebook.',
  };
  const seedHint = `Enter a whole number from 0 to ${Number.MAX_SAFE_INTEGER.toLocaleString('en-US')}.`;

  let step = $state<ConvertStep>('inspect');
  const emptyForm = (): ConvertForm => ({
    source: '',
    bundle: '',
    extras: { assets: '', ngram: '', vision: '', mtp: '' },
    host: '',
    anyHost: false,
    profile: '',
    ht_backend: '',
    i4r_backend: '',
    q4cp_backend: '',
    decode_backend: '',
    seed: null,
    threads: null,
    ht_beam: null
  });
  let form = $state(emptyForm());
  let diagnostic = $state(emptyForm());

  const STEPS: { id: ConvertStep; label: string; text: string }[] = [
    { id: 'inspect', label: 'Inspect', text: 'Read the checkpoint and report what the converter would write. Writes nothing.' },
    { id: 'plan', label: 'Plan', text: 'Plan the conversion and check disk space on the host. Writes nothing.' },
    { id: 'convert', label: 'Convert', text: 'Write a new bundle to a fresh directory on the host. The source is not changed.' },
    { id: 'verify', label: 'Verify', text: 'Check a complete bundle against its original source. File verification only; it does not prove GPU use or quality.' },
  ];

  const built = $derived(buildConvertJob(step, diagnostic, models, hosts));
  const blockers = $derived.by(() => {
    const out = [...built.problems];
    if (step !== 'verify' && !diagnostic.profile) out.push('Choose a profile.');
    if (!(step === 'convert' && diagnostic.anyHost)) {
      const target = diagnostic.host || models.find((m) => m.id === diagnostic.source)?.host;
      if (target) {
        const c = capability(hosts.find((h) => h.id === target), step);
        if (!c.allowed && c.reason) out.push(c.reason);
      }
    }
    return out;
  });
  const eligible = $derived(poolHosts(hosts));

  // ---- Full workflow (default): one durable controller record ----
  let wfHost = $state('');
  let wfAny = $state(false);
  const wfForm = $derived<ConvertForm>({ ...form, host: wfHost, anyHost: wfAny });
  const wfBuilt = $derived(buildConversion(wfForm, models));
  const wfSource = $derived(models.find((m) => m.id === form.source));
  const MODES = [
    { id: 'hgn', title: 'Managed HGN', help: 'Run four steps to make and verify an HGN bundle.' },
    { id: 'gguf', title: 'GGUF', help: 'Make and verify a GGUF file on the source host.' },
    { id: 'recipe', title: 'Layer recipe', help: 'Set the encoding for each layer of a SafeTensors checkpoint.' },
  ] as const;
  type Mode = (typeof MODES)[number]['id'];
  const formatParam = $derived(hashParam(route.hash, 'format'));
  const mode = $derived(formatParam === 'gguf' || formatParam === 'recipe' ? formatParam : 'hgn');
  function setMode(value: Mode) {
    if (value === mode) return;
    location.hash = value === 'hgn' ? '#/convert' : `#/convert?format=${value}`;
  }
  let ggufSource = $state('');
  const emptyGguf = (): GgufDraft => ({ choice: '', assets: '', imatrix: '', threads: null });
  let ggufDraft = $state(emptyGguf());
  let layerSource = $state('');
  const emptyRecipe = (): RecipeDraft => ({ defaultType: 'BF16', overrides: {}, assetId: '' });
  let recipeDraft = $state(emptyRecipe());
  let hy4Source = $state('');
  let hy4Encode = $state(false);
  const recipeFormat = $derived(hashParam(route.hash, 'output') === 'gguf' ? 'gguf' : 'hgn');
  let diagnosticsOpen = $state(false);
  const loadStage = loadRetryJob(STAGES);
  type Prior = { run: Conversion } | { job: JobDetail; parent?: Conversion };
  type RetryContext = { imatrix?: string; recipeSha?: string; source?: string;
    managedExtras?: ConvertForm['extras']; diagnosticExtras?: ConvertForm['extras'] };
  const defaults = {
    form: emptyForm(), diagnostic: emptyForm(), wfHost: '', wfAny: false, step: 'inspect' as ConvertStep,
    ggufSource: '', ggufDraft: emptyGguf(), layerSource: '', recipeDraft: emptyRecipe(), hy4Source: '', hy4Encode: false, diagnosticsOpen: false,
  };
  const retry = new RetryPrefill<typeof defaults, Prior, RetryContext>({
    slot: 'convert', defaults,
    read: () => ({ form, diagnostic, wfHost, wfAny, step, ggufSource, ggufDraft, layerSource, recipeDraft, hy4Source, hy4Encode, diagnosticsOpen }),
    write: value => { ({ form, diagnostic, wfHost, wfAny, step, ggufSource, ggufDraft, layerSource, recipeDraft, hy4Source, hy4Encode, diagnosticsOpen } = value); },
    load: async (id, signal) => {
      if (id.startsWith('run:')) {
        const runId = id.slice(4);
        const result = await api.conversion(runId, signal);
        const run = parseConversion(result);
        if (!run || run.id !== runId || !canRetryManagedHgn(run)) throw new Error('Cannot reopen this conversion.');
        return { run };
      }
      const job = await loadStage(id.slice(4), signal);
      // GGUF stage records store the preset, not the full conversion request.
      const parent = job.format === 'gguf' || job.profile_revision || job.options?.layer_recipe
        ? parseConversions(await api.conversions(signal)).find(run => Object.values(run.jobIds).includes(job.id))
        : undefined;
      return { job, parent };
    },
    restore: prior => {
      if ('run' in prior) {
        const { run } = prior;
        const target = run.requestedHost ?? run.host ?? '';
        return {
          actions: ['managed'], failedId: run.id,
          patch: { ...managedPatch(run.inputIds, run.options, run.profileRevision, target, run.decodeBackend), diagnosticsOpen: false },
          context: { managedExtras: selectionFrom(run.inputIds, run.options) },
          focus: 'wf-source', autoFocus: 'wf-h',
          omissions: draft => convertOmissions(run.inputIds, run.options, target, draft.form, draft.wfAny ? 'auto' : draft.wfHost),
          retains: draft => !!draft.form.source && draft.form.source === run.inputIds[0],
        };
      }
      const { job, parent } = prior;
      const verify = !parent && job.kind === 'verify';
      const inputs = parent?.inputIds ?? (verify ? [job.input_ids[0] ?? '', ...job.input_ids.slice(2)] : job.input_ids);
      const options = parent?.options ?? job.options ?? {};
      const profile = parent?.profileRevision ?? (typeof job.profile_revision === 'string' ? job.profile_revision : '');
      const source = models.find(m => m.id === inputs[0]);
      const assets = models.find(m => m.kind === 'assets' && m.host === source?.host && m.available !== false && m.id === inputs[1])?.id ?? '';
      const recipe = obj(options.layer_recipe);
      if (recipe) {
        const types = [...recipeTypes(job.format === 'gguf' ? 'gguf' : 'hgn'), ...trellis];
        const type = (value: unknown): value is OutputType => types.some(option => option === value);
        const defaultType = type(recipe.default_type) ? recipe.default_type : 'BF16';
        const overrides = Object.fromEntries(Object.entries(obj(recipe.layers) ?? {})
          .filter(([index, value]) => /^(0|[1-9]\d*)$/.test(index) && type(value))) as RecipeDraft['overrides'];
        const validSource = (id: string) => availableId(id, models.filter(m => isSafeSource(m.kind)));
        const asset = inputs[1];
        return {
          actions: ['recipe'], failedId: parent?.id ?? job.id, patch: { layerSource: validSource(inputs[0]), recipeDraft: { defaultType, overrides, assetId: assets } },
          context: { recipeSha: typeof recipe.source_sha256 === 'string' ? recipe.source_sha256 : undefined, source: inputs[0] },
          focus: 'recipe-source',
          omissions: draft => [
            ...missing(inputs[0], validSource(draft.layerSource), 'source checkpoint', true),
            ...(draft.layerSource === inputs[0] ? missing(asset, assetOn(draft.layerSource, draft.recipeDraft.assetId), 'assets', true) : []),
          ],
          retains: draft => !!draft.layerSource && draft.layerSource === inputs[0],
        };
      }
      if (job.format === 'gguf') {
        // The controller records inputs in this order, even if the matrix was deleted.
        const standalone = source?.kind === 'safetensors' || inputs.length > 2 || models.some(m => m.id === inputs[1] && m.kind === 'assets');
        const previousMatrix = inputs[standalone ? 2 : 1];
        const imatrix = models.find(m => m.id === previousMatrix && m.kind === IMATRIX_KIND && m.host === source?.host && m.available !== false)?.id ?? '';
        const ggufDraft = {
          choice: profile ? `rev:${profile}` : typeof options.preset === 'string' ? `preset:${options.preset}` : '',
          threads: typeof options.threads === 'number' ? options.threads : null, assets, imatrix,
        };
        const validSource = (id: string) => availableId(id, models.filter(m => ['hf', 'gguf', 'safetensors'].includes(m.kind)));
        return {
          actions: ['gguf'], failedId: parent?.id ?? job.id, patch: { ggufSource: validSource(inputs[0]), ggufDraft },
          context: { source: inputs[0], imatrix: previousMatrix }, focus: 'gg-source',
          omissions: draft => [
            ...missing(inputs[0], validSource(draft.ggufSource), 'source checkpoint', true),
            ...(draft.ggufSource === inputs[0] ? [
              ...(standalone ? missing(inputs[1], assetOn(draft.ggufSource, draft.ggufDraft.assets), 'assets', true) : []),
              ...(previousMatrix && !componentOn(draft.ggufSource, draft.ggufDraft.imatrix, IMATRIX_KIND)
                ? [{ id: `matrix:${previousMatrix}`, text: `Previous importance matrix ${previousMatrix} was not restored. It was not available on the source host.` }] : []),
            ] : []),
          ],
          retains: draft => !!draft.ggufSource && draft.ggufSource === inputs[0],
        };
      }
      if (job.options?.hy4_overrides && (job.kind === 'plan' || job.kind === 'convert')) {
        const validSource = (id: string) => availableId(id, models.filter(m => m.kind === 'hf'));
        const hy4Encode = job.kind === 'convert';
        return { actions: [hy4Encode ? 'hy4:encode' : 'hy4:plan'], failedId: job.id,
          patch: { diagnosticsOpen: true, hy4Source: validSource(inputs[0]), hy4Encode },
          focus: 'hy4-source',
          omissions: draft => missing(inputs[0], validSource(draft.hy4Source), 'Hy4 source', true) };
      }
      // A saved custom profile can only be restored in the managed form.
      const target = job.requested_host ?? job.host ?? '';
      if (profile) {
        if (!parent) throw new Error('Cannot identify the managed conversion for this stage.');
        const worker = parent.requestedHost ?? parent.host ?? target;
        return {
          actions: ['managed'], failedId: parent.id, patch: managedPatch(inputs, options, profile, worker, parent.decodeBackend),
          context: { managedExtras: selectionFrom(inputs, options) },
          focus: 'wf-source',
          omissions: draft => convertOmissions(inputs, options, worker, draft.form, draft.wfAny ? 'auto' : draft.wfHost),
          retains: draft => !!draft.form.source && draft.form.source === inputs[0],
        };
      }
      const diagnostic = restoredForm(job.input_ids, job.options ?? {}, null, target);
      if (verify) diagnostic.bundle = availableId(job.input_ids[1], models.filter(m => VERIFY_OUTPUT_KINDS.has(m.kind)));
      return {
        actions: STAGES.map(stage => `diagnostic:${stage}`), retryActions: [`diagnostic:${job.kind}`], failedId: job.id,
        patch: { step: STAGES.find(stage => stage === job.kind) ?? 'inspect', diagnosticsOpen: true, diagnostic },
        context: { diagnosticExtras: selectionFrom(job.input_ids, job.options ?? {}) },
        focus: 'cv-source',
        omissions: draft => [
          ...convertOmissions(job.input_ids, job.options ?? {}, target, draft.diagnostic,
            draft.diagnostic.anyHost ? 'auto' : draft.diagnostic.host),
          ...(verify && draft.step === 'verify' ?
            missing(job.input_ids[1], availableId(draft.diagnostic.bundle, models.filter(m => VERIFY_OUTPUT_KINDS.has(m.kind))), 'bundle', true) : []),
        ],
        retains: draft => !!draft.diagnostic.source && draft.diagnostic.source === job.input_ids[0],
      };
    },
  });
  watchPrefill(retry, () => stateReady, hash => {
    const run = hashParam(hash, 'retry_run');
    const job = hashParam(hash, 'retry');
    return run ? `run:${run}` : job ? `job:${job}` : null;
  });
  const retryRunLoading = $derived(retry.loading && !!hashParam(route.hash, 'retry_run'));
  const managedRetry = $derived(retry.action('managed'));
  const ggufRetry = $derived(retry.action('gguf'));
  const recipeRetry = $derived(retry.action('recipe'));
  const hy4Action = $derived(hy4Encode ? 'hy4:encode' : 'hy4:plan');
  const restoredHy4 = $derived(retry.restoredAction('hy4:encode') || retry.restoredAction('hy4:plan'));
  const diagnosticAction = $derived(`diagnostic:${step}`);
  const diagnosticRetry = $derived(retry.action(diagnosticAction));
  $effect(() => {
    if (profileStore.gguf) void tick().then(() => retry.resumeFocus());
  });
  const sourceId = (id: string) => availableId(id, models.filter(m => CONVERT_SOURCE_KINDS.has(m.kind)));
  function componentOn(source: string, id: string, kind: string) {
    const host = models.find(m => m.id === source)?.host;
    return availableId(id, models.filter(m => m.kind === kind && m.host === host));
  }
  const assetOn = (source: string, id: string) => componentOn(source, id, 'assets');
  function missing(before: string | undefined, now: string, name: string, required: boolean): RetryOmission[] {
    return before && !now ? [{ id: `${name}:${before}`, text: `Previous ${name} ${before} is not available. Select another entry.`, required }] : [];
  }
  function convertOmissions(ids: string[], opts: Record<string, unknown>, target: string, draft: ConvertForm, host: string) {
    const original = selectionFrom(ids, opts);
    return [
      ...missing(ids[0], sourceId(draft.source), 'source checkpoint', true),
      ...(target && target !== 'auto' && !availableId(target, hosts) ?
        missing(target, host === 'auto' ? host : availableId(host, hosts), 'host', true) : []),
      ...COMPONENT_ROLES.flatMap(({ kind }) =>
        !draft.source || draft.source === ids[0]
          ? missing(original[kind], availableId(draft.extras[kind], models.filter(m => m.kind === kind)), roleLabel(kind), false) : []),
    ];
  }
  function restoredForm(inputIds: string[], opts: Record<string, unknown>, profile: string | null, target: string): ConvertForm {
    const draft = emptyForm();
    draft.source = sourceId(inputIds[0]);
    draft.host = target === 'auto' ? '' : availableId(target, hosts);
    draft.anyHost = target === 'auto';
    draft.profile = profile ? `${REV_PREFIX}${profile}` : typeof opts.profile === 'string' ? opts.profile : '';
    for (const key of ['ht_backend', 'i4r_backend', 'q4cp_backend', 'decode_backend'] as const) {
      const value = opts[key];
      if (typeof value === 'string') draft[key] = value;
    }
    for (const key of ['threads', 'seed', 'ht_beam'] as const) {
      const value = opts[key];
      if (typeof value === 'number') draft[key] = value;
    }
    const original = selectionFrom(inputIds, opts);
    for (const { kind } of COMPONENT_ROLES) draft.extras[kind] = availableId(original[kind], models.filter(m => m.kind === kind));
    return draft;
  }
  function managedPatch(inputIds: string[], opts: Record<string, unknown>, profile: string | null, target: string, decode: string | null) {
    const draft = restoredForm(inputIds, opts, profile, target);
    if (!draft.profile) draft.profile = DEFAULT_PROFILE;
    if (decode) draft.decode_backend = decode;
    const { anyHost: wfAny, host: wfHost } = draft;
    draft.host = '';
    draft.anyHost = false;
    return { form: draft, wfAny, wfHost };
  }
  $effect(() => {
    const hash = route.hash;
    const source = hashParam(hash, 'model');
    const host = hashParam(hash, 'host');
    const profile = hashParam(hash, 'profile');
    const requestedStep = STAGES.find((value) => value === hashParam(hash, 'step'));
    // Resolve IDs against inventory before putting them in visible pickers.
    if ((source || host || profile || requestedStep) && stateReady && !hashParam(hash, 'retry') && !hashParam(hash, 'retry_run')) {
      untrack(() => {
        if (requestedStep) {
          step = requestedStep;
          diagnosticsOpen = true;
        }
        const draft = requestedStep ? diagnostic : form;
        const model = models.find((m) => m.id === source);
        if (model && !requestedStep && formatParam === 'gguf') {
          if (['hf', 'gguf', 'safetensors'].includes(model.kind)) ggufSource = model.id;
        } else if (model && CONVERT_SOURCE_KINDS.has(model.kind)) draft.source = model.id;
        if (host && hosts.some((h) => h.id === host)) {
          if (requestedStep) diagnostic.host = host;
          else wfHost = host;
        }
        if (profile === 'v2' || profile === 'ht43') draft.profile = profile;
      });
    }
  });
  const recipeStarter = new ConversionStarter('convert.recipe');
  function selectRecipeSource(id: string) {
    if (id === layerSource) return;
    layerSource = id;
    recipeDraft = emptyRecipe();
  }
  const recipeSource = $derived(models.find((m) => m.id === layerSource && isSafeSource(m.kind)));
  const recipeKey = $derived(recipeSource && JSON.stringify([
    recipeSource.id, recipeSource.kind, recipeSource.host, recipeSource.path, recipeSource.sha256, recipeSource.bundle_sha256,
  ]));
  $effect(() => {
    if (!stateReady) return;
    if (form.source && !wfSource) form.source = '';
    if (layerSource && !recipeSource) layerSource = '';
    if (ggufSource && !models.some((m) => m.id === ggufSource)) ggufSource = '';
    if (diagnostic.source && !models.some((m) => m.id === diagnostic.source)) diagnostic.source = '';
  });
  $effect(() => {
    if (!stateReady) return;
    // Historical selections have an omission policy. Keep a later
    // replacement in the draft so the builder can block and name it.
    for (const [original, draft] of [[retry.context?.managedExtras, form], [retry.context?.diagnosticExtras, diagnostic]] as const) {
      if (!original) continue;
      for (const { kind } of COMPONENT_ROLES) {
        const id = draft.extras[kind];
        if (id && id === original[kind] && !availableId(id, models.filter(m => m.kind === kind))) draft.extras[kind] = '';
      }
    }
  });
  $effect(() => {
    if (profileStore.pendingUse?.startsWith('gguf:') && mode !== 'gguf') setMode('gguf');
  });
  // Saved custom profiles (read-only GET). A revision is chosen as 'rev:<hex>'.
  $effect(() => {
    void profileStore.load();
  });
  const savedProfiles = $derived(profileStore.saved.filter((p) => p.compatible));
  const chosenRev = $derived(form.profile.startsWith(REV_PREFIX) ? profileStore.saved.find((p) => p.revision === form.profile.slice(REV_PREFIX.length)) : undefined);
  $effect(() => {
    // "Use in Convert" from the Profiles page.
    const rev = profileStore.pendingUse;
    if (rev && !rev.startsWith('gguf:') && profileStore.saved.some((p) => p.revision === rev)) {
      form.profile = `${REV_PREFIX}${rev}`;
      profileStore.pendingUse = null;
    }
  });
  const hasRevision = $derived(form.profile.startsWith(REV_PREFIX));
  // The component workflow flag covers HF replacements, not GGUF assets.
  const hasComponents = $derived(wfSource?.kind === 'hf' && Object.values(form.extras).some(Boolean));
  const workflow = $derived<Workflow>(hasRevision ? 'custom_profile_conversion' : hasComponents ? 'component_conversion' : 'managed_conversion');
  const poolWorkflow = $derived<Workflow>(hasRevision ? 'pool_custom_profile_conversion' : hasComponents ? 'pool_component_conversion' : 'pool_conversion');
  const snapshot = $derived(brewery.state?.eligibility);
  const targetId = $derived((wfAny ? '' : wfHost) || wfSource?.host);
  const hostChecks = $derived.by(() => {
    if (wfAny && eligible.length) return eligible.map((host) => checkHost(host.id, poolWorkflow));
    return targetId ? [checkHost(targetId, wfAny ? poolWorkflow : workflow)] : [];
  });
  function legacyHostCheck(host: Host): EligibilityEntry {
    const fix_anchor = `#/settings/infrastructure?host=${encodeURIComponent(host.id)}&section=policy`;
    const reasons = host.scheduling_paused ? [{ text: 'Resume job scheduling.', fix_anchor }] : [];
    for (const stage of [...STAGES, ...(hasRevision ? ['custom_profiles'] : []), ...(hasComponents ? ['convert_components'] : [])]) {
      const cap = capability(host, stage);
      if (!cap.allowed) reasons.push(host.capability_reasons?.[stage] ?? { text: cap.reason ?? `Enable the ${stage} step.`, fix_anchor });
    }
    return { eligible: !reasons.length, reasons };
  }
  function checkHost(id: string, kind: Workflow): Blocker[] {
    const host = hosts.find((h) => h.id === id);
    const display = hostName(hosts, id);
    if (!host) return [{ host: display, field: 'host', text: 'The controller did not report this host.', fixAnchor: '#/settings/infrastructure' }];
    const decision = hostWorkflowEligibility(snapshot, id, kind) ?? legacyHostCheck(host);
    const reasons = decision.reasons.map((reason) => ({ text: reason.text, fixAnchor: reason.fix_anchor }));
    if (decision.eligible && hasRevision) reasons.push(...componentBlockers(wfSource, form.extras, host).map((text) => ({
      text, fixAnchor: host.capability_reasons?.convert_components?.fix_anchor ?? `#/settings/infrastructure?host=${encodeURIComponent(id)}&section=policy`,
    })));
    return reasons.map((item) => ({
      ...item, host: display, field: 'host',
    }));
  }
  const sourceSha = $derived(validSha(wfSource?.sha256));
  const localBlockers = $derived.by((): Blocker[] => {
    const out = wfBuilt.problems.map((text): Blocker => ({
      text, field: !form.profile && text === 'Choose the output profile.' ? 'profile'
        : (!wfSource || wfSource.available === false) && (text === 'Select a source checkpoint.' || text === (wfSource?.reason ?? 'Select an available source checkpoint.')) ? 'source' : undefined,
    }));
    if (hasRevision && !chosenRev?.compatible) out.push({ text: 'Select a compatible saved profile.', field: 'profile' });
    const catalogSha = profileStore.catalog?.sourceSha256;
    if (chosenRev && sourceSha && catalogSha && sourceSha !== catalogSha) out.push({ text: 'This profile was made for a different source checkpoint.', field: 'profile' });
    if (hostChecks.length && hostChecks.every((reasons) => reasons.length)) out.push(...hostChecks.flat());
    return out;
  });
  const checkBody = $derived(mode === 'hgn' && !managedRetry.blocked && conversions.status !== 'unavailable' && !localBlockers.length && wfBuilt.body ? JSON.stringify(wfBuilt.body) : null);
  // Compare by value: polling an unchanged snapshot must not repeat the check.
  const checkState = $derived(checkBody ? JSON.stringify([hosts, snapshot,
    models.filter((model) => wfBuilt.body?.input_ids.includes(model.id))]) : '');
  let checked = $state<{ key: string; decision?: EligibilityEntry; error?: string }>();
  const checkKey = $derived(checkBody && JSON.stringify([checkBody, checkState]));
  let retryCheck = $state(0);
  $effect(() => {
    const body = checkBody;
    const key = checkKey;
    void retryCheck;
    checked = undefined;
    if (!body || !key) return;
    const abort = new AbortController();
    const timer = window.setTimeout(async () => {
      try {
        const decision = await api.conversionEligibility(JSON.parse(body), abort.signal);
        if (!abort.signal.aborted) checked = { key, decision };
      } catch (error) {
        if (!abort.signal.aborted) checked = { key, error: errorText(error) };
      }
    }, 250);
    return () => { window.clearTimeout(timer); abort.abort(); };
  });
  const checking = $derived(checkKey !== null && checked?.key !== checkKey);
  const wfBlockers = $derived.by((): Blocker[] => {
    if (conversions.status === 'unavailable') return [{ text: 'This controller does not provide managed conversions. Use the individual steps below.' }];
    if (!checkBody || checking) return localBlockers;
    if (checked?.error) return [{ text: `Cannot check this conversion. ${checked.error}` }];
    return (checked?.decision?.reasons ?? []).map((r) => ({ text: r.text, fixAnchor: r.fix_anchor }));
  });
  const touched = $state<Record<string, boolean>>({});
  let attempted = $state(false);
  const visible = $derived(visibleBlockers(wfBlockers, touched, attempted || !!checkBody || (!!wfSource && !!form.profile)));
  const starter = new ConversionStarter('convert.managed');
  async function startConversion() {
    // Submit only on a click, never from an effect.
    attempted = true;
    if (!wfBuilt.body || starter.busy || !checkKey || checked?.key !== checkKey || !checked.decision?.eligible) return;
    const { hash } = route;
    const visit = route.capture();
    const run = await starter.start(wfBuilt.body, managedRetry);
    if (run) {
      if (visit.current() && starter.acceptedBodyKey === bodyKey(wfBuilt.body)) retry.markClean();
      if (!starter.replayed && visit.canPresent() && hashParam(hash, 'retry_run')) route.replaceQuery('#/convert');
    }
  }
  $effect(() => conversions.watch());
  const runs = $derived(conversions.list);
  const activeRuns = $derived(runs.filter(isActiveConversion));
  let showOlder = $state(false);
  const finishedRuns = $derived(runs.filter((c) => !isActiveConversion(c)));
  const linkedRun = $derived(hashParam(route.hash, 'run'));
  const visibleFinished = $derived(showOlder ? finishedRuns : finishedRuns.filter((c, i) => i < 5 || c.id === linkedRun));
  const hasLinkedRun = $derived(!!linkedRun && runs.some((c) => c.id === linkedRun));
  const linkedVisit = $derived(route.capture());
  $effect(() => {
    const id = linkedRun;
    const visit = linkedVisit;
    if (id && hasLinkedRun) void tick().then(() => {
      if (!visit.canPresent()) return;
      const target = document.getElementById(`run-${id}`);
      target?.focus({ preventScroll: true });
      target?.scrollIntoView({ block: 'nearest' });
    });
  });
  const optsChangedWf = $derived(
    [
      ...(['ht_backend', 'i4r_backend', 'q4cp_backend'] as const).filter((k) => form[k]).map((k) => `${k.replace('_backend', '').toUpperCase()} ${form[k]}`),
      form.seed !== null ? `seed ${form.seed}` : '',
      form.threads !== null ? `${form.threads} threads` : '',
      form.ht_beam !== null ? `beam ${form.ht_beam}` : '',
      form.decode_backend ? `verify decode ${form.decode_backend}` : '',
    ].filter(Boolean),
  );

  const s = new Submitter(job => `convert.${job.kind}`);
  function submit() {
    if (built.job && !blockers.length && !s.busy) s.submit(built.job, diagnosticRetry);
  }
  const usesBackends = $derived(step === 'inspect' || step === 'plan' || step === 'convert');
  const usesPlanOpts = $derived(step === 'plan' || step === 'convert');
  const optsChanged = $derived(
    [
      ...(usesBackends ? (['ht_backend', 'i4r_backend', 'q4cp_backend'] as const).filter((k) => diagnostic[k]).map((k) => `${k.replace('_backend', '').toUpperCase()} ${diagnostic[k]}`) : []),
      ...(usesPlanOpts ? [diagnostic.seed !== null ? `seed ${diagnostic.seed}` : '', diagnostic.threads !== null ? `${diagnostic.threads} threads` : '', diagnostic.ht_beam !== null ? `beam ${diagnostic.ht_beam}` : ''] : []),
      step === 'verify' && diagnostic.decode_backend ? `decode ${diagnostic.decode_backend}` : '',
    ].filter(Boolean),
  );
</script>

<div class="stack" oninputcapture={retry.touchField} onchangecapture={retry.touchField} onclickcapture={retry.touchField} onkeydowncapture={retry.touchField}>
  <div class="head">
    <h1 tabindex="-1">Convert</h1>
    {#if retry.failed}<RetryRecovery id="cv-retry-loading" reason={retry.failure} reload={retry.reload} startNew={retry.startNew} />{/if}
    {#if retry.loadingReason}<p id="cv-retry-loading" class="hint" role="status">{retryRunLoading ? 'Loading the previous conversion.' : retry.loadingReason}</p>{/if}
    {#if retry.offered}<RetryOffer apply={retry.usePrevious} />{/if}
    <p class="muted small">Convert a source checkpoint to an HGN bundle or a GGUF file.</p>
  </div>

  <section class="panel" aria-labelledby="wf-h">
    <header>
      <h2 id="wf-h" tabindex="-1">Convert a checkpoint</h2>
      <div class="fmt" role="radiogroup" aria-label="Conversion mode">
        {#each MODES as item (item.id)}
          <label class:on={mode === item.id}>
            <input type="radio" name="convert-format" checked={mode === item.id} onchange={() => setMode(item.id)} aria-describedby="convert-mode-help" />
            {item.title}
          </label>
        {/each}
      </div>
    </header>
    <p id="convert-mode-help" class="body small muted">{MODES.find((item) => item.id === mode)?.help}</p>
    {#if mode === 'recipe'}
      <div class="body stack">
        <RetryWarnings items={retry.omissions('recipe')} />
        <div class="grid-2">
          <ModelSelect bind:value={() => layerSource, selectRecipeSource} kinds={['hf', 'safetensors']} label="Source checkpoint" hint="Choose a SafeTensors file or an indexed HF directory." id="recipe-source" />
          <div class="field">
            <label for="recipe-format">Recipe output</label>
            <select class="input" id="recipe-format" value={recipeFormat} onchange={(event) => {
              const output = event.currentTarget.value;
              if (hashParam(route.hash, 'retry')) {
                // Output is part of this recipe draft, not a different retry.
                const query = new URLSearchParams(route.hash.split('?')[1]);
                if (output === 'gguf') query.set('output', 'gguf');
                else query.delete('output');
                route.replaceQuery(`#/convert?${query}`);
              } else {
                location.hash = output === 'gguf' ? '#/convert?format=recipe&output=gguf' : '#/convert?format=recipe';
              }
            }}>
              <option value="hgn">Structural HGN archive</option>
              <option value="gguf">GGUF file</option>
            </select>
          </div>
        </div>
        {#if recipeSource}
          {#key recipeKey}
            <LayerRecipe source={recipeSource} format={recipeFormat} starter={recipeStarter} bind:draft={recipeDraft}
              retry={recipeRetry} originalSha={retry.context?.source === layerSource ? retry.context?.recipeSha : undefined} loading={retry.unavailable} />
          {/key}
        {:else}
          <p class="hint">Select a source to see if it supports a layer recipe.</p>
        {/if}
      </div>
    {:else if mode === 'gguf'}
    <WorkflowGate workflow="gguf_output">
    <div class="body">
      {#if profileStore.gguf}<GgufConvert catalog={profileStore.gguf} bind:source={ggufSource} bind:draft={ggufDraft}
        retry={ggufRetry} omissions={retry.omissions('gguf')} acknowledged={retry.acknowledged('gguf')}
        acknowledge={() => retry.acknowledge('gguf')}
        previousMatrix={retry.context?.source === ggufSource ? retry.context?.imatrix : undefined} loading={retry.unavailable} />
      {:else if profileStore.ggufStatus === 'unavailable'}<p class="note warn">This controller does not offer GGUF output.</p>
      {:else if profileStore.status === 'loading' || profileStore.status === 'idle'}<p class="muted">Loading GGUF types…</p>
      {:else}<p class="note bad" role="alert">The GGUF catalog could not be loaded.</p>{/if}
    </div>
    </WorkflowGate>
    {:else}
    <WorkflowGate workflow="managed_conversion">
    <div class="body stack">
      <RetryWarnings items={retry.omissions('managed')} acknowledged={retry.acknowledged('managed')} acknowledge={() => {
        retry.acknowledge('managed'); document.getElementById('wf-source')?.focus();
      }} />
      <div class="grid-2">
        <ModelSelect bind:value={form.source} kinds={CONVERT_SOURCE_KINDS} label="Source checkpoint" hint="Hugging Face BF16 directory, GGUF or HGN." id="wf-source" error={fieldError(visible, 'source')} onblur={() => (touched.source = true)} />
        <div class="field">
          <label for="wf-profile">Output profile</label>
          <select id="wf-profile" class="input" bind:value={form.profile} onblur={() => (touched.profile = true)}
            aria-invalid={!!fieldError(visible, 'profile')} aria-describedby={fieldError(visible, 'profile') ? 'wf-profile-error wf-profile-hint' : 'wf-profile-hint'} aria-required="true">
            <option value="" disabled>Choose a profile</option>
            <option value={DEFAULT_PROFILE}>Converter default profile</option>
            <optgroup label="Built-in">
              {#each CONVERT_ENUMS.profile as p (p)}<option value={p}>{p === 'v2' ? 'HGN v2' : p === 'ht43' ? 'HT43' : p}</option>{/each}
            </optgroup>
            {#if savedProfiles.length}
              <optgroup label="Saved custom profiles">
                {#each savedProfiles as p (p.revision)}<option value="{REV_PREFIX}{p.revision}">{p.name} · {shortRev(p.revision)} (from {BASE_LABEL[p.base]})</option>{/each}
              </optgroup>
            {/if}
          </select>
          <FieldError id="wf-profile-error" error={fieldError(visible, 'profile')} />
          <span class="hint" id="wf-profile-hint">
            {#if chosenRev}Custom revision <span class="mono" title={chosenRev.revision}>{shortRev(chosenRev.revision)}</span>: {Object.keys(chosenRev.overrides).length} tensors changed from {BASE_LABEL[chosenRev.base]}.{:else}Select the encoding for the HGN bundle. <abbr title="HGN profile with 4-bit and 3-bit HT tensor types">HT43</abbr> is a built-in profile.{/if}
            {#if wfSource && !sourceSha}<span>Source checkpoint unverified.</span>{/if}
            <a href="#/profiles">{savedProfiles.length ? 'Edit profiles' : 'Create a custom profile'}</a>
          </span>
        </div>
      </div>
      <div class="grid-2">
        <div class="field">
          {#if wfAny}
            <span class="flabel">Run on</span>
            <p class="hint">
              The controller assigns the conversion to an idle pool worker and keeps all four stages on it.
              {eligible.length ? `Pool: ${eligible.map((h) => h.name ?? h.id).join(', ')}.` : 'No pool is configured, so it runs on the source host.'}
            </p>
          {:else}
            <HostSelect bind:value={wfHost} {workflow} fallbackEligibility={legacyHostCheck} label="Run on" optional optionalLabel="Host of the source" id="wf-host" error={fieldError(visible, 'host')} onblur={() => (touched.host = true)} />
          {/if}
          <label class="inline"><input type="checkbox" bind:checked={wfAny} /> Any eligible pool worker</label>
        </div>
      </div>

      {#if wfSource}<ComponentPicker bind:value={form.extras} source={wfSource} idPrefix="wf-comp" />{/if}
      <Advanced changed={optsChangedWf}>
        <div class="grid-3">
          {#each ['ht_backend', 'i4r_backend', 'q4cp_backend'] as const as key (key)}
            <div class="field">
              <label for="wf-{key}">{key.replace('_backend', '').toUpperCase()} encoder backend</label>
              <select id="wf-{key}" class="input" bind:value={form[key]} aria-describedby="wf-{key}-hint">
                <option value="">Converter default</option>
                {#each CONVERT_ENUMS.backend as b (b)}<option value={b}>{b}</option>{/each}
              </select>
              <span class="hint" id="wf-{key}-hint">{encoders[key]}</span>
            </div>
          {/each}
          <div class="field">
            <label for="wf-seed">Seed <span class="muted">(optional)</span></label>
            <input id="wf-seed" class="input num" type="number" min="0" max={Number.MAX_SAFE_INTEGER} step="1" bind:value={form.seed} inputmode="numeric" aria-describedby="wf-seed-hint" />
            <span class="hint" id="wf-seed-hint">{seedHint}</span>
          </div>
          <div class="field">
            <label for="wf-threads">Threads <span class="muted">(optional)</span></label>
            <input id="wf-threads" class="input num" type="number" min="1" max="64" step="1" bind:value={form.threads} inputmode="numeric" />
            <span class="hint">1 to 64.</span>
          </div>
          <div class="field">
            <label for="wf-beam">HT beam <span class="muted">(optional)</span></label>
            <input id="wf-beam" class="input num" type="number" min="1" max="256" step="1" bind:value={form.ht_beam} inputmode="numeric" />
            <span class="hint">1 to 256.</span>
          </div>
          <div class="field">
            <label for="wf-dec">Verify decode backend</label>
            <select id="wf-dec" class="input" bind:value={form.decode_backend}>
              <option value="">Converter default</option>
              {#each CONVERT_ENUMS.backend as b (b)}<option value={b}>{b}</option>{/each}
            </select>
          </div>
        </div>
        <p class="xs muted">The receipt records the backends that actually ran. A requested backend is not evidence of GPU use.</p>
      </Advanced>

      <Blockers blockers={visible} />
      <div class="row">
        <button type="button" class="btn primary" disabled={!checked?.decision?.eligible || !checkBody || checking || !!wfBlockers.length || starter.busy} onclick={startConversion}>Start conversion</button>
        {#if checking && conversions.status !== 'unavailable'}<span class="hint" role="status">Checking this conversion.</span>{/if}
        {#if checkKey && checked?.key === checkKey && checked.error}
          <button type="button" class="btn small" onclick={() => { checked = undefined; retryCheck++; }}>Check again</button>
        {/if}
      </div>
      <div aria-live="polite" class="xs">
        {#if starter.error}<p class="note bad" role="alert">{starter.error} Pressing Start again retries the same request; it will not start a second conversion.</p>
        {:else if starter.last}
          {#if resultOfPreviousSubmission(starter.acceptedBodyKey, wfBuilt.body)}<p class="muted">Result of the previous submission</p>{/if}
          <p class="muted">{starter.replayed ? 'This request was already queued as job' : 'Started conversion'} <a href={conversionRunHref(starter.last.id)}>{starter.last.id}</a>. {starter.replayed ? 'No second conversion was started.' : 'The run continues if you leave this page.'}</p>
        {/if}
      </div>
      {#if !starter.error && starter.replayed && starter.newRun}<NewRun prepare={starter.newRun} disabled={starter.busy} />{/if}
    </div>
    </WorkflowGate>
    {/if}
  </section>

  <section class="panel" id="conversion-runs" aria-labelledby="runs-h">
    <header><h2 id="runs-h">Conversions <span class="muted num small">({runs.length})</span></h2></header>
    {#if !conversions.loaded}
      <p class="empty">Loading…</p>
    {:else if conversions.status === 'unavailable'}
      <p class="empty">This controller does not provide the conversion workflow.</p>
    {:else if conversions.status === 'error' && runs.length === 0}
      <p class="note bad" role="alert" style="margin:12px 16px">Conversions could not be loaded: {conversions.error}</p>
    {:else if runs.length === 0}
      <p class="empty">No conversions yet.</p>
    {:else}
      {#if conversions.status === 'error'}<p class="xs muted" style="margin:8px 16px 0">Showing the last loaded list ({conversions.error}).</p>{/if}
      <div>
        {#each activeRuns as c (c.id)}<div id="run-{c.id}" tabindex="-1"><ConversionRun {c} /></div>{/each}
        {#each visibleFinished as c (c.id)}<div id="run-{c.id}" tabindex="-1"><ConversionRun {c} /></div>{/each}
      </div>
      {#if finishedRuns.length > 5}
        <div class="more"><button type="button" class="btn small ghost" aria-expanded={showOlder} onclick={() => (showOlder = !showOlder)}>{showOlder ? 'Show fewer' : `Show ${finishedRuns.length - 5} older`}</button></div>
      {/if}
    {/if}
  </section>

  <details class="disclosure manual" bind:open={diagnosticsOpen}>
    <summary><span class="t">Individual steps (diagnostics)</span><span class="s xs">Run inspect, plan, convert or verify as a single job</span></summary>
    <div class="stack manual-body">

  <div class="steps" role="tablist" aria-label="Converter step">
    {#each STEPS as st, i (st.id)}
      <button
        type="button"
        role="tab"
        id="tab-{st.id}"
        aria-selected={step === st.id}
        aria-controls="conv-panel"
        tabindex={step === st.id ? 0 : -1}
        class="step"
        onclick={() => (step = st.id)}
        onkeydown={(e) => {
          if (e.key !== 'ArrowRight' && e.key !== 'ArrowLeft') return;
          const next = STEPS[(i + (e.key === 'ArrowRight' ? 1 : STEPS.length - 1)) % STEPS.length];
          step = next.id;
          document.getElementById(`tab-${next.id}`)?.focus();
        }}
      >
        <span class="n num" aria-hidden="true">{i + 1}</span>{st.label}
      </button>
    {/each}
  </div>

  <div class="panel" id="conv-panel" role="tabpanel" aria-labelledby="tab-{step}" tabindex="-1">
    <div class="body stack">
      <p class="small">{STEPS.find((x) => x.id === step)?.text}</p>

      <div class="grid-2">
        <ModelSelect id="cv-source" bind:value={diagnostic.source} kinds={CONVERT_SOURCE_KINDS} label="Source checkpoint" hint="Select a source for this diagnostic job." />
        {#if step === 'verify'}
          <ModelSelect bind:value={diagnostic.bundle} kinds={VERIFY_OUTPUT_KINDS} label="Bundle to verify" hint="Choose the HGN bundle." exclude={diagnostic.source ? [diagnostic.source] : []} />
        {:else}
          <div class="field">
            {#if step === 'convert' && diagnostic.anyHost}
              <span class="flabel">Run on</span>
              <p class="hint">
                The controller assigns this job to an idle pool worker when it starts, and never moves it. Each job converts one complete checkpoint on one host; queue more jobs to use more workers.
                {eligible.length ? `Pool: ${eligible.map((h) => h.name ?? h.id).join(', ')}.` : 'No pool is configured, so the job runs on the source host.'}
              </p>
            {:else}
              <HostSelect bind:value={diagnostic.host} kind={step} label="Run on" optional optionalLabel="Host of the source" />
            {/if}
            {#if step === 'convert'}
              <label class="inline"><input type="checkbox" bind:checked={diagnostic.anyHost} /> Any eligible pool worker</label>
            {/if}
          </div>
        {/if}
      </div>
      {#if step === 'verify'}
        <div class="grid-2">
          <HostSelect bind:value={diagnostic.host} kind="verify" label="Run on" optional optionalLabel="Host of the source" />
        </div>
      {/if}

      {#if usesBackends}
        <div class="grid-2">
          <div class="field">
            <label for="cv-profile">Output profile</label>
            <select id="cv-profile" class="input" bind:value={diagnostic.profile} aria-describedby="cv-profile-hint" aria-required="true">
              <option value="" disabled>Choose a profile</option>
              {#each CONVERT_ENUMS.profile as p (p)}<option value={p}>{p === 'v2' ? 'HGN v2' : p === 'ht43' ? 'HT43' : p}</option>{/each}
            </select>
            <span class="hint" id="cv-profile-hint">Select the encoding for the HGN bundle.</span>
          </div>
        </div>
      {/if}

      {#each models.filter((m) => m.id === diagnostic.source) as source (source.id)}
        <ComponentPicker bind:value={diagnostic.extras} {source} idPrefix="cv-comp" />
      {/each}

      {#if usesBackends || usesPlanOpts || step === 'verify'}
        <Advanced changed={optsChanged}>
          <div class="grid-3">
            {#if usesBackends}
              {#each ['ht_backend', 'i4r_backend', 'q4cp_backend'] as const as key (key)}
                <div class="field">
                  <label for="cv-{key}">{key.replace('_backend', '').toUpperCase()} encoder backend</label>
                  <select id="cv-{key}" class="input" bind:value={diagnostic[key]} aria-describedby="cv-{key}-hint">
                    <option value="">Converter default</option>
                    {#each CONVERT_ENUMS.backend as b (b)}<option value={b}>{b}</option>{/each}
                  </select>
                  <span class="hint" id="cv-{key}-hint">{encoders[key]}</span>
                </div>
              {/each}
            {/if}
            {#if usesPlanOpts}
              <div class="field">
                <label for="cv-seed">Seed <span class="muted">(optional)</span></label>
                <input id="cv-seed" class="input num" type="number" min="0" max={Number.MAX_SAFE_INTEGER} step="1" bind:value={diagnostic.seed} inputmode="numeric" aria-describedby="cv-seed-hint" />
                <span class="hint" id="cv-seed-hint">{seedHint}</span>
              </div>
              <div class="field">
                <label for="cv-threads">Threads <span class="muted">(optional)</span></label>
                <input id="cv-threads" class="input num" type="number" min="1" max="64" step="1" bind:value={diagnostic.threads} inputmode="numeric" />
                <span class="hint">1 to 64.</span>
              </div>
              <div class="field">
                <label for="cv-beam">HT beam <span class="muted">(optional)</span></label>
                <input id="cv-beam" class="input num" type="number" min="1" max="256" step="1" bind:value={diagnostic.ht_beam} inputmode="numeric" />
                <span class="hint">1 to 256.</span>
              </div>
            {/if}
            {#if step === 'verify'}
              <div class="field">
                <label for="cv-dec">Decode backend</label>
                <select id="cv-dec" class="input" bind:value={diagnostic.decode_backend}>
                  <option value="">Converter default</option>
                  {#each CONVERT_ENUMS.backend as b (b)}<option value={b}>{b}</option>{/each}
                </select>
              </div>
            {/if}
          </div>
          <p class="xs muted">The receipt records the backends that actually ran. A requested backend is not evidence of GPU use.</p>
        </Advanced>
      {/if}

      <RetryWarnings items={retry.omissions(diagnosticAction)} acknowledged={retry.acknowledged(diagnosticAction)} acknowledge={() => {
        retry.acknowledge(diagnosticAction); document.getElementById('cv-source')?.focus();
      }} />
      <Submit {blockers} busy={s.busy} disabled={diagnosticRetry.blocked} onclick={submit}>Queue {STEPS.find((x) => x.id === step)?.label.toLowerCase()}</Submit>
      <SubmitStatus {s} />
    </div>
  </div>
      <RetryWarnings items={retry.omissions(hy4Action)} />
      <Hy4Plan bind:sourceId={hy4Source} bind:encode={hy4Encode} restored={restoredHy4} retry={retry.action(hy4Action)} loading={retry.unavailable} />
    </div>
  </details>

  <RecentJobs kinds={['inspect', 'plan', 'convert', 'verify']} title="Recent converter jobs" limit={8} />
</div>

<style>
  .head {
    display: flex;
    flex-direction: column;
    gap: 4px;
  }
  /* The four steps wrap on narrow screens instead of scrolling sideways. */
  .steps {
    display: flex;
    flex-wrap: wrap;
    gap: 4px;
  }
  .step {
    flex: 1 1 auto;
    justify-content: center;
    display: inline-flex;
    align-items: center;
    gap: 8px;
    min-height: 40px;
    padding: 0 var(--s-4);
    background: transparent;
    border: 1px solid var(--line);
    border-radius: var(--r-md);
    color: var(--text-2);
    font-weight: 500;
    cursor: pointer;
    white-space: nowrap;
  }
  .step:hover {
    color: var(--text);
  }
  .step[aria-selected='true'] {
    background: var(--accent-soft);
    border-color: var(--accent);
    color: var(--text);
  }
  .n {
    display: inline-grid;
    place-items: center;
    width: 20px;
    height: 20px;
    border-radius: 50%;
    border: 1px solid currentColor;
    font-size: var(--fs-xs);
  }
  .fmt {
    display: inline-flex;
    flex-wrap: wrap;
    padding: 2px;
    border: 1px solid var(--line);
    border-radius: var(--r-md);
    background: var(--surface);
  }
  .fmt label {
    position: relative;
    display: inline-flex;
    align-items: center;
    min-height: 28px;
    padding: 0 10px;
    border-radius: var(--r-sm);
    font-size: var(--fs-sm);
    color: var(--text-2);
    cursor: pointer;
    white-space: nowrap;
  }
  .fmt label.on {
    background: var(--accent);
    color: var(--accent-text);
    font-weight: 600;
  }
  .fmt input {
    position: absolute;
    inset: 0;
    opacity: 0;
    margin: 0;
    cursor: pointer;
  }
  .fmt label:has(input:focus-visible) {
    outline: 2px solid var(--focus);
    outline-offset: 1px;
  }
  .manual > summary .t {
    font-weight: 500;
    font-size: var(--fs-sm);
  }
  .manual > summary .s {
    color: var(--text-2);
  }
  .manual-body {
    padding: var(--s-2) 0 var(--s-3);
  }
  .more {
    padding: var(--s-2) var(--s-4) var(--s-3);
  }
  .flabel {
    font-size: var(--fs-sm);
    font-weight: 500;
    color: var(--text-2);
  }
  .inline {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    min-height: 36px;
    font-size: var(--fs-sm);
    cursor: pointer;
  }
</style>
