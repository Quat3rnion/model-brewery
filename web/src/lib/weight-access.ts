import { capability, isSafeSource } from './format';
import type { FixReason, Host, Model } from './types';

export function weightCapability(host: Host | undefined, kind: string, hostId = host?.id ?? '') {
  const cap = capability(host, kind);
  const anchor = `#/settings/infrastructure?host=${encodeURIComponent(hostId)}&section=policy`;
  const blocked = (reason: FixReason) => ({ allowed: false, reason: reason.text, fixAnchor: reason.fix_anchor });
  if (!host) return blocked({ text: 'The source host is not registered.', fix_anchor: anchor });
  if (!cap.allowed) return blocked(host.capability_reasons?.[kind] ?? { text: cap.reason!, fix_anchor: anchor });
  if (host.scheduling_paused) return blocked({ text: 'Resume job scheduling on the source host.', fix_anchor: anchor });
  return { allowed: true, reason: null, fixAnchor: anchor };
}

export function weightFormat(model: Model, host: Host | undefined) {
  if (isSafeSource(model.kind)) return weightCapability(host, 'weight_safetensors', model.host);
  if (model.kind === 'hgn') return weightCapability(host, 'weight_hgn', model.host);
}

/** A GGUF proposal can use text entry without a visual reader. */
export function weightAccess(model: Model, host: Host | undefined) {
  const manual = weightCapability(host, 'weight_propose', model.host);
  if (model.available === false) return { ...manual, allowed: false, reason: model.reason || 'The source file is not available.' };
  const format = weightFormat(model, host);
  if (format && !format.allowed) return format;
  if (model.kind !== 'gguf') {
    for (const kind of ['weight_inspect', 'weight_visual']) {
      const read = weightCapability(host, kind, model.host);
      if (!read.allowed) return read;
    }
  }
  if (manual.allowed) return manual;
  const tune = weightCapability(host, 'weight_tune', model.host);
  return tune.allowed ? tune : manual;
}
