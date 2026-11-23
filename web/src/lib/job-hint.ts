export function jobFixHint(error: string | null | undefined): { text: string; href: string; label: string } | null {
  if (!error) return null;
  if (/no space left on device|not enough (free |disk )?space|insufficient (free |disk )?space/i.test(error)) {
    return { text: 'Free space on the host before you try again.', href: '#/settings/infrastructure', label: 'Infrastructure settings' };
  }
  if (/safetensors source requires a valid immutable publication/i.test(error)) {
    return { text: 'Select a verified SafeTensors source in Weights before you try again.', href: '#/weights', label: 'Weights' };
  }
  if (/permission denied/i.test(error)) {
    return { text: 'Check that the worker has permission to read the input and write the output.', href: '#/settings/infrastructure', label: 'Infrastructure settings' };
  }
  return null;
}
