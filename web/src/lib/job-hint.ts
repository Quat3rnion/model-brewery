export function jobFixHint(error: string | null | undefined, host?: string): { text: string; href: string; label: string } | null {
  if (!error) return null;
  const infrastructure = host ? `#/settings/infrastructure?host=${encodeURIComponent(host)}&section=policy` : '#/settings/infrastructure';
  if (/no space left on device|not enough (free |disk )?space|insufficient (free |disk )?space|disk space is insufficient/i.test(error)) {
    return { text: 'Free space on the host before you try again.', href: infrastructure, label: 'Infrastructure settings' };
  }
  if (/safetensors (source requires a valid immutable publication|immutable publication could not be verified)/i.test(error)) {
    return { text: 'Select a verified SafeTensors source in Weights before you try again.', href: '#/weights', label: 'Weights' };
  }
  if (/permission denied/i.test(error)) {
    return { text: 'Check that the worker has permission to read the input and write the output.', href: infrastructure, label: 'Infrastructure settings' };
  }
  return null;
}
