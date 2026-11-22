import { scenario } from './scenarios.mjs';

export function convertState() {
  const data = scenario('conversions', '');
  const reason = { text: 'Enable Verify on Worker B', fix_anchor: '#/settings/infrastructure?host=worker-b&section=policy' };
  const worker = data.hosts[1];
  worker.capabilities = { ...data.hosts[0].capabilities, verify: false };
  worker.capability_reasons.verify = reason;
  const entry = data.eligibility.hosts[worker.id];
  for (const key of Object.keys(entry).filter((key) => key.endsWith('conversion') || key === 'gguf_output')) {
    entry[key] = { eligible: false, reasons: [reason] };
  }
  entry.gguf_output_by_kind.gguf = entry.gguf_output;
  for (const host of data.hosts) host.quantization_pool = false;
  for (const key of Object.keys(data.eligibility.hosts['worker-a']).filter((key) => key.startsWith('pool_'))) {
    data.eligibility.hosts['worker-a'][key] = { eligible: true, reasons: [] };
  }
  return data;
}
