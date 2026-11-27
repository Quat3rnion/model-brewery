import { tick } from 'svelte';
import { route } from './route.svelte';

/** Move focus from a submit control only if the operator has not moved on. */
export function captureResultFocus(event: MouseEvent) {
  const trigger = event.currentTarget;
  if (trigger !== document.activeElement) return;
  const visit = route.capture();
  const preventScroll = event.detail > 0;
  return async (result: () => HTMLElement | null | undefined) => {
    await tick();
    if (visit.canPresent() && (document.activeElement === trigger || document.activeElement === document.body)) {
      result()?.focus({ preventScroll });
    }
  };
}

/** Focus the next control after an update removes the focused control. */
export async function focusAfterRemoval(update: () => void, next: () => HTMLElement | null | undefined) {
  const previous = document.activeElement;
  update();
  await tick();
  if (previous && !previous.isConnected && document.activeElement === document.body) next()?.focus();
}
