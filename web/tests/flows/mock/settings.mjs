export const gib = 1024 ** 3;
export const fingerprint = `SHA256:${'A'.repeat(43)}`;
export const sshHost = { id: 'test-ssh', name: 'Test host', host: 'old.example', user: 'operator', port: 22,
  fingerprint, revision: 1, status: 'defined' };

export function adminState(revision = 1, allowPool = false) {
  return {
    hosts: ['Worker A', 'Worker B'].map((name, i) => ({
      id: `worker-${i ? 'b' : 'a'}`, name, revision, paused: false, draining: false,
      installed_jobs: ['convert'], allowed_jobs: ['convert'], disabled_jobs: [],
      resources: {}, resource_bounds: { min_free_ram_bytes: { min: 0, max: 16 * gib } },
      weight_editor: { available: false, enabled: false, capabilities: {} },
      model_directories: { available: true, configured: [], granted: [] },
    })),
    workers: [{ id: 'worker-a', name: 'Worker A', status: 'active', allow_pool: allowPool, pool_ceiling: true, runtimes: [] }],
    audit: [{ id: 1, action: 'set_host_resources', worker: 'worker-a', resources: { min_free_ram_bytes: 2 * gib }, outcome: 'applied' }],
  };
}
