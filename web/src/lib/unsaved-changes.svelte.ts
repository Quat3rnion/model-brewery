import { route } from './route.svelte';

/** Call during component setup. Call confirm() before changing hosts or closing the editor. */
export function unsavedChanges(isDirty: () => boolean) {
  function confirm() {
    return !isDirty() || window.confirm('Discard unsaved changes?');
  }
  $effect(() => {
    if (!isDirty()) return;
    const unregister = route.guardHashChange(confirm);
    function beforeUnload(event: BeforeUnloadEvent) {
      if (!isDirty()) return;
      event.preventDefault();
      event.returnValue = true;
    }
    window.addEventListener('beforeunload', beforeUnload);
    return () => {
      unregister();
      window.removeEventListener('beforeunload', beforeUnload);
    };
  });
  return { confirm };
}
