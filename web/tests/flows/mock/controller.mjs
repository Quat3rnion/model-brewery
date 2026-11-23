import { once } from 'node:events';
import { readFileSync } from 'node:fs';
import { createServer } from 'node:http';
import { catalog, ggufCatalog, conversion, conversionEligibility, detail, job, scenario, weightRead } from './scenarios.mjs';

const hgnCatalog = JSON.parse(readFileSync(catalog, 'utf8'));
const gguf = JSON.parse(readFileSync(ggufCatalog, 'utf8'));
const json = (body, status = 200) => ({ status, json: body });
const error = (status, message) => json({ error: message }, status);
const summary = ({ result, events, prompt_text, ...rest }) => rest;
const canonical = body => JSON.stringify(body, (_, value) =>
  value && typeof value === 'object' && !Array.isArray(value)
    ? Object.fromEntries(Object.keys(value).sort().map(key => [key, value[key]])) : value);

// Only model the response and state transitions that browser tests need.
// Tests can set a response for additional endpoints without changing defaults.
export async function startController(assets) {
  const log = [];
  const responses = new Map();
  const submissions = new Map();
  const fingerprints = new Map();
  let data = scenario('live-shaped', hgnCatalog.catalog_sha256);
  let seq = 0;
  const id = (prefix) => `${prefix}-mock-${++seq}`;
  const key = (method, path) => `${method} ${path}`;
  function submit(path, body, headers, create) {
    const header = headers['idempotency-key'];
    if (header !== undefined && (!header.length || header.length > 128 || /[^A-Za-z0-9._-]/.test(header))) return error(400, 'invalid Idempotency-Key');
    const scope = key(path, header);
    const fingerprint = canonical(body);
    const prior = header === undefined ? undefined : submissions.get(scope);
    if (prior) {
      if (prior.fingerprint !== fingerprint) return error(409, 'Idempotency-Key already used for a different request');
      return json(path === '/api/jobs' ? { id: prior.record.id } : prior.record);
    }
    const group = path === '/api/jobs' ? data.jobs : data.conversions;
    const now = Date.now();
    const groupKey = key(path, fingerprint);
    const active = id => ['queued', 'running', 'cancelling'].includes(group.find(row => row.id === id)?.state);
    const candidates = (fingerprints.get(groupKey) ?? []).filter(entry => now - entry.at < 10_000 || active(entry.id));
    const match = headers['brewery-allow-duplicate'] === '1' ? undefined : candidates[0];
    if (match) {
      const record = group.find(row => row.id === match.id);
      if (header !== undefined) submissions.set(scope, { fingerprint, record });
      return {
        ...json({ ...(path === '/api/jobs' ? { id: match.id } : record), duplicate_of: true }),
        headers: { 'Brewery-Duplicate': match.id },
      };
    }
    const record = create();
    fingerprints.set(groupKey, [...candidates, { id: record.id, at: now }]);
    if (header !== undefined) submissions.set(scope, { fingerprint, record });
    return json(record, 201);
  }
  const api = async (method, path, body, headers) => {
    switch (key(method, path)) {
      case 'GET /api/state':
        return json({ hosts: data.hosts, models: data.models, prompts: data.prompts,
          jobs: data.jobs.map(summary), capabilities: data.capabilities, eligibility: data.eligibility, event_cursor: data.eventCursor });
      case 'GET /api/conversions': return json(data.conversions);
      case 'GET /api/profiles': return json(data.profiles);
      case 'GET /api/profiles/catalog': return json(hgnCatalog);
      case 'GET /api/profiles/gguf-catalog': return json(gguf);
      case 'GET /api/settings': return json({ hosts: data.settings });
      case 'GET /api/settings/hf-tokens': return json({ hosts: [] });
      case 'GET /api/settings/agent-provider': return json({ provider: null, key_configured: false });
      case 'GET /api/agent/sessions': return json([]);
      case 'GET /api/admin/state':
        return json({ workers: [], audit: [], unavailable: {}, hosts: data.hosts.map((host) => {
          const jobs = Object.keys(host.capabilities).filter((kind) => host.capabilities[kind]);
          return {
            id: host.id, name: host.name, revision: 1, paused: false, draining: false,
            installed_jobs: jobs, allowed_jobs: jobs, disabled_jobs: [], resources: {},
            weight_editor: { available: host.capabilities.weight_inspect, enabled: false, capabilities: host.capabilities },
            model_directories: { available: true, configured: ['/fixtures'], granted: ['/fixtures'] },
          };
        }) });
      case 'GET /api/admin/ssh-hosts': return json({ hosts: [], available: true });
      case 'GET /api/admin/ssh-identities': return json({ identities: [], default_configured: false });
      case 'POST /api/suggestions/paths': {
        const paths = data.models.filter((m) => m.host === body.host && m.path.startsWith(body.prefix));
        return json({ items: paths.map((m) => ({ path: m.path, kind: m.kind === 'hf' ? 'directory' : 'file' })), truncated: false });
      }
      case 'POST /api/models': {
        const model = { ...body, id: id('model'), name: body.name ?? body.path.split('/').at(-1) };
        data.models.push(model);
        return json(model, 201);
      }
      case 'POST /api/prompts': {
        const prompt = { ...body, id: id('prompt') };
        data.prompts.push(prompt);
        return json(prompt, 201);
      }
      case 'POST /api/jobs': return submit(path, body, headers, () => {
        const jobId = id('job');
        const model = data.models.find((m) => m.id === body.input_ids[0]);
        const { prompt, ...request } = body;
        const entry = data.finishWeightReads && body.kind === 'weight_inspect' && model
          ? weightRead(jobId, body, model)
          : detail(job(jobId, body.kind, {
            ...request, prompt_text: body.prompt_id ? data.prompts.find((p) => p.id === body.prompt_id)?.text ?? null : prompt ?? null,
          }));
        data.jobs.push(entry);
        return summary(entry);
      });
      case 'POST /api/conversions/eligibility': return json(conversionEligibility(data, body));
      case 'POST /api/conversions': return submit(path, body, headers, () => {
        const entry = conversion(id('conversion'), body);
        const child = detail(job(id('job'), 'inspect', { host: entry.host, input_ids: entry.input_ids }));
        entry.job_ids.inspect = child.id;
        data.conversions.push(entry);
        data.jobs.push(child);
        return entry;
      });
      case 'POST /api/profiles': {
        const profile = { ...body, revision: (++seq).toString(16).padStart(64, '0'), created_at: Date.now() };
        data.profiles.push(profile);
        return json(profile, 201);
      }
      case 'POST /api/admin/model-directories/plan':
        return json({ ...body, confirmation: 'fixture-confirmation' });
    }

    const match = /^\/api\/(jobs|conversions|models|settings)\/([^/]+)(\/cancel)?$/.exec(path);
    if (match) {
      const [, group, encoded, cancel] = match;
      const entries = data[group];
      const index = entries.findIndex((entry) => entry.id === decodeURIComponent(encoded));
      if (index === -1) return error(404, `${group === 'settings' ? 'host' : group.slice(0, -1)} not found`);
      const entry = entries[index];
      if (cancel && method === 'POST' && ['jobs', 'conversions'].includes(group)) {
        if (entry.state === 'queued') entry.state = 'cancelled';
        else if (entry.state === 'running') entry.cancel_requested = true;
        return json(group === 'jobs' ? summary(entry) : entry);
      }
      if (!cancel && method === 'GET' && ['jobs', 'conversions'].includes(group)) return json(entry);
      if (!cancel && method === 'DELETE' && group === 'models') {
        entries.splice(index, 1);
        return { status: 204 };
      }
      if (!cancel && method === 'PUT' && group === 'settings') {
        if (body.revision !== entry.revision) return error(409, 'Settings revision changed');
        entries[index] = { ...entry, ...body, revision: entry.revision + 1 };
        return json(entries[index]);
      }
    }
    return error(404, 'API endpoint not found');
  };

  const server = createServer((req, res) => {
    const { pathname: path, search } = new URL(req.url, 'http://127.0.0.1');
    if (path !== '/api' && !path.startsWith('/api/') && path !== '/live-progress/current.json') {
      assets(req, res, () => { res.writeHead(404).end(); });
      return;
    }
    const chunks = [];
    req.on('data', (chunk) => chunks.push(chunk));
    req.on('end', async () => {
      const method = req.method;
      const raw = Buffer.concat(chunks).toString();
      let body;
      try { body = raw ? JSON.parse(raw) : undefined; } catch { body = raw; }
      const request = { method, path, search, body, headers: { ...req.headers } };
      log.push(request);
      try {
        const override = responses.get(key(method, path));
        const answer = override
          ? (typeof override === 'function' ? await override(request) : override)
          : data.down ? { down: true }
          : data.status ? error(data.status, `Fixture HTTP ${data.status}`)
          : await api(method, path, body, request.headers);
        if (answer.down) req.socket.destroy();
        else if (answer.status === 204) res.writeHead(204).end();
        else res.writeHead(answer.status ?? 200, {
          'Content-Type': 'application/json', 'Cache-Control': 'no-store',
          ...(typeof answer.json?.error === 'string' ? { 'Brewery-Error': '1' } : {}),
          ...answer.headers,
        })
          .end(JSON.stringify(answer.json));
      } catch (cause) {
        res.writeHead(500, { 'Content-Type': 'application/json' }).end(JSON.stringify({ error: `Mock error: ${cause.message}` }));
      }
    });
  });
  server.listen(0, '127.0.0.1');
  await once(server, 'listening');
  const origin = `http://127.0.0.1:${server.address().port}`;
  return {
    origin,
    open: (page, route = 'library') => page.goto(`${origin}/#/${route}`),
    load(name) {
      data = scenario(name, hgnCatalog.catalog_sha256);
      responses.clear();
      submissions.clear();
      fingerprints.clear();
    },
    respond: (method, path, response) => responses.set(key(method, path), response),
    requests: (method, path) => structuredClone(log.filter((r) => (!method || r.method === method) && (!path || r.path === path))),
    close: () => new Promise((resolve, reject) => {
      server.close((err) => err ? reject(err) : resolve());
      server.closeAllConnections();
    }),
  };
}
