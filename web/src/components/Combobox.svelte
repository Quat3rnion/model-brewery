<script lang="ts">
  // Searchable picker (ARIA 1.2 combobox with a listbox popup).
  // Select mode: `value` is an option value (an ID); typed text only filters.
  //   Blur keeps the old value unless the text names exactly one option.
  // Free mode: `value` is the text; options are suggestions.
  // Enter on an open list picks the active option and never submits the form.
  import { exactMatch, filterOptions, stepIndex, type ComboOption } from '../lib/combobox';

  let {
    value = $bindable(''),
    options,
    id: givenId,
    label,
    labelHidden = false,
    placeholder = '',
    optional = false,
    optionalLabel = 'None',
    free = false,
    filter = true,
    disabled = false,
    mono = false,
    describedby,
    invalid = false,
    emptyText = 'No matches',
    status,
    loading = false,
    onpick,
    onfocus,
    onblur: onleave,
    ontextinput,
  }: {
    value?: string;
    options: ComboOption[];
    id?: string;
    label: string;
    labelHidden?: boolean;
    placeholder?: string;
    optional?: boolean;
    optionalLabel?: string;
    /** Free text with suggestions instead of a fixed choice. */
    free?: boolean;
    /** Filter options locally by the typed text. Off when the caller already filtered (remote suggestions). */
    filter?: boolean;
    disabled?: boolean;
    mono?: boolean;
    describedby?: string;
    invalid?: boolean;
    emptyText?: string;
    /** Extra line under the options, for example "Loading" or "List truncated". */
    status?: string | null;
    loading?: boolean;
    /** Called after a pick. Return true to keep the list open (directory completion). */
    onpick?: (o: ComboOption) => boolean | void;
    onfocus?: () => void;
    onblur?: () => void;
    /** Free mode: called on every user edit with the new text. */
    ontextinput?: (text: string) => void;
  } = $props();

  const uid = $props.id();
  const id = $derived(givenId ?? `${uid}-in`);
  const listId = $derived(`${id}-list`);
  const NONE = '\u0000none';

  let input: HTMLInputElement | undefined = $state();
  let list: HTMLUListElement | undefined = $state();
  let open = $state(false);
  let text = $state('');
  let typed = $state(false);
  let active = $state(-1);

  const selected = $derived(options.find((o) => o.value === value));
  const shownText = $derived(free ? value : (selected?.label ?? (value ? value : '')));
  // Keep the input text in step with the value while the user is not typing.
  $effect(() => {
    const t = shownText;
    if (!typed) text = t;
  });

  const all = $derived<ComboOption[]>(optional && !free ? [{ value: NONE, label: optionalLabel }, ...options] : options);
  const shown = $derived(filter && typed ? filterOptions(all, text) : all);

  $effect(() => {
    if (active >= 0 && list) list.querySelector<HTMLElement>(`[data-i="${active}"]`)?.scrollIntoView({ block: 'nearest' });
  });

  function openList() {
    if (disabled) return;
    open = true;
    const cur = shown.findIndex((o) => o.value === (value || NONE));
    active = cur >= 0 ? cur : stepIndex(shown, -1, 1);
  }
  function close(restore: boolean) {
    open = false;
    active = -1;
    if (restore && !free) {
      typed = false;
      text = shownText;
    }
  }
  function pick(o: ComboOption) {
    if (o.disabled) return;
    const v = o.value === NONE ? '' : o.value;
    value = v;
    typed = false;
    text = free ? value : value === v ? (o.value === NONE ? '' : o.label) : shownText;
    const keep = onpick?.(o) === true;
    if (keep) {
      if (free) ontextinput?.(v);
      open = true;
      active = -1;
    } else close(false);
    input?.focus();
  }
  function commitTyped() {
    if (free || !typed) return close(true);
    const t = text.trim();
    if (!t && optional) {
      value = '';
      return close(true);
    }
    const hit = exactMatch(options, t);
    if (hit) value = hit.value;
    close(true);
  }

  function onkeydown(e: KeyboardEvent) {
    if (e.isComposing) return;
    switch (e.key) {
      case 'ArrowDown':
        e.preventDefault();
        if (!open) openList();
        else active = stepIndex(shown, active, 1);
        break;
      case 'ArrowUp':
        e.preventDefault();
        if (!open) openList();
        else active = stepIndex(shown, active, -1);
        break;
      case 'Enter':
        if (!open) {
          // Select mode: a typed exact name is taken; the form is not submitted with half-typed text.
          if (!free && typed) {
            e.preventDefault();
            commitTyped();
          }
          return;
        }
        e.preventDefault();
        if (active >= 0 && shown[active]) pick(shown[active]);
        else if (free) close(false);
        else {
          const only = shown.filter((o) => !o.disabled);
          if (only.length === 1) pick(only[0]);
        }
        break;
      case 'Escape':
        if (open || typed) {
          e.preventDefault();
          e.stopPropagation();
          close(true);
        }
        break;
      case 'Tab':
        if (open) commitTyped();
        break;
    }
  }
  function oninput() {
    typed = true;
    if (free) {
      value = text;
      ontextinput?.(text);
    }
    if (!open) open = true;
    active = free ? -1 : stepIndex(shown, -1, 1);
  }
  function onblur(e: FocusEvent) {
    // Focus moving into the popup (scrollbar click) keeps it open.
    const to = e.relatedTarget as Node | null;
    if (to && list?.contains(to)) return;
    commitTyped();
    onleave?.();
  }
</script>

<div class="field combo">
  <label for={id} class:sr-only={labelHidden}>{label}</label>
  <div class="box" class:open>
    <input
      bind:this={input}
      {id}
      class="input"
      class:mono
      type="text"
      role="combobox"
      autocomplete="off"
      autocapitalize="off"
      spellcheck="false"
      aria-autocomplete="list"
      aria-expanded={open}
      aria-controls={listId}
      aria-activedescendant={open && active >= 0 ? `${id}-o${active}` : undefined}
      aria-describedby={describedby}
      aria-invalid={invalid ? 'true' : undefined}
      placeholder={placeholder || (optional && !free ? optionalLabel : '')}
      {disabled}
      bind:value={text}
      {oninput}
      {onkeydown}
      {onblur}
      onfocus={() => onfocus?.()}
      onclick={() => (open ? null : openList())}
    />
    <button type="button" class="toggle" tabindex="-1" aria-label="{open ? 'Hide' : 'Show'} options for {label}" {disabled} onmousedown={(e) => e.preventDefault()} onclick={() => (open ? close(true) : (openList(), input?.focus()))}>
      <svg viewBox="0 0 16 16" width="14" height="14" aria-hidden="true"><path d="M4 6l4 4 4-4" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" /></svg>
    </button>
    {#if open}
      <div class="pop">
        <ul bind:this={list} id={listId} role="listbox" aria-label={label}>
          {#each shown as o, i (o.value)}
            <!-- Keyboard selection runs on the input through aria-activedescendant (combobox pattern). -->
            <!-- svelte-ignore a11y_click_events_have_key_events -->
            <li
              id="{id}-o{i}"
              data-i={i}
              role="option"
              aria-selected={(o.value === NONE ? '' : o.value) === value && (value !== '' || o.value === NONE)}
              aria-disabled={o.disabled ? 'true' : undefined}
              class:active={i === active}
              onmousedown={(e) => e.preventDefault()}
              onclick={() => pick(o)}
              onmousemove={() => (o.disabled ? null : (active = i))}
            >
              <span class="l" class:mono class:none={o.value === NONE}>{o.label}</span>
              {#if o.detail}<span class="d">{o.detail}</span>{/if}
            </li>
          {/each}
        </ul>
        {#if !shown.length && !loading}<p class="msg">{emptyText}</p>{/if}
        {#if status}<p class="msg">{status}</p>{/if}
      </div>
    {/if}
  </div>
  <p class="sr-only" aria-live="polite">{open ? (loading ? 'Loading suggestions' : `${shown.length} option${shown.length === 1 ? '' : 's'}`) : ''}</p>
</div>

<style>
  .box {
    position: relative;
  }
  .box .input {
    padding-right: 32px;
    text-overflow: ellipsis;
  }
  .toggle {
    position: absolute;
    top: 0;
    right: 0;
    bottom: 0;
    width: 32px;
    display: grid;
    place-items: center;
    border: none;
    background: transparent;
    color: var(--text-2);
    cursor: pointer;
  }
  .toggle:disabled {
    cursor: not-allowed;
    opacity: 0.6;
  }
  .open .toggle svg {
    transform: rotate(180deg);
  }
  .pop {
    position: absolute;
    z-index: 30;
    top: calc(100% + 4px);
    left: 0;
    right: 0;
    background: var(--surface);
    border: 1px solid var(--line);
    border-radius: var(--r-md);
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.28);
    overflow: hidden;
  }
  ul {
    list-style: none;
    margin: 0;
    padding: 4px;
    max-height: min(320px, 50vh);
    overflow-y: auto;
  }
  li {
    display: flex;
    flex-direction: column;
    gap: 1px;
    padding: 6px 8px;
    border-radius: var(--r-sm);
    cursor: pointer;
    font-size: var(--fs-sm);
    min-width: 0;
  }
  li.active {
    background: var(--accent-soft);
    outline: 1px solid var(--accent);
    outline-offset: -1px;
  }
  li[aria-selected='true'] .l {
    font-weight: 600;
    color: var(--accent);
  }
  li[aria-disabled='true'] {
    cursor: not-allowed;
    color: var(--muted);
  }
  .l {
    overflow-wrap: anywhere;
  }
  .l.none {
    color: var(--text-2);
    font-style: italic;
  }
  .d {
    font-size: var(--fs-xs);
    color: var(--text-2);
    overflow-wrap: anywhere;
  }
  .msg {
    margin: 0;
    padding: 6px 12px 8px;
    font-size: var(--fs-xs);
    color: var(--text-2);
    border-top: 1px solid var(--line-soft);
  }
  ul:empty {
    display: none;
  }
  ul:empty + .msg {
    border-top: none;
  }
</style>
