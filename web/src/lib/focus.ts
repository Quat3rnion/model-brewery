import { tick } from 'svelte';

/** Focus the next control after an update removes the focused control. */
export async function focusAfterRemoval(update: () => void, next: () => HTMLElement | null | undefined) {
  const previous = document.activeElement;
  update();
  await tick();
  if (previous && !previous.isConnected && document.activeElement === document.body) next()?.focus();
}
