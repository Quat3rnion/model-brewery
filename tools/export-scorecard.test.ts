import { afterEach, expect, test } from 'bun:test';
import { mkdtempSync, rmSync, truncateSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const cli = join(import.meta.dir, 'export-scorecard');
const scratch: string[] = [];
afterEach(() => {
  for (const dir of scratch.splice(0)) rmSync(dir, { recursive: true, force: true });
});

function tempDir() {
  const dir = mkdtempSync(join(process.env.TMPDIR ?? '/arboretum', 'export-scorecard.'));
  scratch.push(dir);
  return dir;
}

function jsonFile(dir: string, name: string, body: unknown) {
  const path = join(dir, name);
  writeFileSync(path, JSON.stringify(body));
  return path;
}

function run(...args: string[]) {
  return Bun.spawnSync([cli, ...args], { timeout: 5_000 });
}

const detail = {
  id: '7591bc43-d790-4fd4-b9d7-5c236285c5e9',
  kind: 'benchmark',
  host: 'compute',
  state: 'failed',
  progress: '',
  input_ids: [],
  options: { benchmark_target: 'local', benchmarks: ['ran', 'omitted'] },
  pool_slot: false,
  cancel_requested: false,
  has_result: true,
  created_at: 1761000000000,
  updated_at: 1761000000001,
  events: [{ state: 'failed', progress: '', at: 1761000000001 }],
  result: {
    schema: 'brewery-benchmark-report-v1',
    target: 'local',
    status: 'failed',
    runs: [{ benchmark: 'ran', status: 'failed', metrics: [] }],
  },
};

test('builds one canonical scorecard for mixed benchmark and evaluate detail exports', () => {
  const dir = tempDir();
  const a = jsonFile(dir, 'bench.json', detail);
  const b = jsonFile(dir, 'eval.json', {
    ...detail,
    id: '7591bc43-d790-4fd4-a9d7-5c236285c5ea',
    kind: 'evaluate',
    state: 'succeeded',
    options: {},
    input_ids: ['example-model'],
    events: [{ state: 'succeeded', progress: '', at: 1761000000001 }],
    result: { protocol_id: 'example', identity: { synthetic: true } },
  });
  const proc = run(a, b);
  expect(proc.stderr.toString()).toBe('');
  expect(proc.exitCode).toBe(0);
  const scorecard = JSON.parse(proc.stdout.toString());
  expect(new Date(scorecard.generated_at).toISOString()).toBe(scorecard.generated_at);
  expect(scorecard).toMatchObject({
    schema: 'brewery-workbench-scorecard-v1',
    release_qualified: false,
    benchmark_jobs: [{
      job_id: detail.id,
      host: 'compute',
      host_name: null,
      result_loaded: true,
      target_matches_request: true,
      failed_benchmarks: ['ran'],
      missing_benchmarks: ['omitted'],
      report: detail.result,
    }],
    gsm8k_jobs: [{
      kind: 'evaluate',
      input_ids: ['example-model'],
      result_loaded: true,
      protocol_id: 'example',
      synthetic: true,
    }],
  });
});

test('accepts a job detail without a result and orders jobs newest first', () => {
  const dir = tempDir();
  const { result, ...withoutResult } = detail;
  const newId = '33c1e38a-d882-44f2-b8b4-3aaceccec223';
  const old = jsonFile(dir, 'old.json', detail);
  const recent = jsonFile(dir, 'recent.json', {
    ...withoutResult,
    id: newId,
    created_at: detail.created_at + 10,
    updated_at: detail.updated_at + 10,
    has_result: false,
  });
  const proc = run(old, recent);
  expect(proc.exitCode).toBe(0);
  const entries = JSON.parse(proc.stdout.toString()).benchmark_jobs;
  expect(entries).toMatchObject([
    { job_id: newId, result_loaded: true, report: null, missing_benchmarks: ['ran', 'omitted'] },
    { job_id: detail.id, report: result },
  ]);
});

function expectFailure(proc: ReturnType<typeof run>, message: string) {
  expect(proc.exitCode).toBe(1);
  expect(proc.stdout.toString()).toBe('');
  expect(proc.stderr.toString()).toContain(message);
}

test('rejects a bad file after a valid detail without writing a partial scorecard', () => {
  const dir = tempDir();
  const valid = jsonFile(dir, 'valid.json', detail);
  const other = { ...detail, id: 'c5cdaccd-d2a4-418b-b378-879e38f156ad' };
  const { events, result, ...summary } = other;
  const bad = jsonFile(dir, 'bad.json', summary);
  expectFailure(run(valid, bad), 'not a job summary');
  jsonFile(dir, 'bad.json', { ...summary, events });
  expectFailure(run(valid, bad), 'has_result must agree');
  jsonFile(dir, 'bad.json', { ...other, has_result: false, result });
  expectFailure(run(valid, bad), 'has_result must agree');
  jsonFile(dir, 'bad.json', { ...other, kind: 'verify' });
  expectFailure(run(valid, bad), 'benchmark or evaluate');
  jsonFile(dir, 'bad.json', [other]);
  expectFailure(run(valid, bad), 'one job detail object');
  jsonFile(dir, 'bad.json', { ...other, state: 'finished' });
  expectFailure(run(valid, bad), 'invalid job state');
  jsonFile(dir, 'bad.json', { ...other, id: 'abc' });
  expectFailure(run(valid, bad), 'invalid job ID');
  writeFileSync(bad, '{"id":');
  expectFailure(run(valid, bad), 'expected valid UTF-8 JSON');
  expectFailure(run(valid, jsonFile(dir, 'copy.json', { ...detail, kind: 'evaluate' })), 'duplicate job ID');
});

test('limits file count, individual size, and combined size', () => {
  const dir = tempDir();
  expectFailure(run(...Array(257).fill('missing.json')), 'expected 1–256');
  const max = 16 * 1024 * 1024;
  const tooLarge = jsonFile(dir, 'too-large.json', detail);
  truncateSync(tooLarge, max + 1);
  expectFailure(run(tooLarge), 'larger than 16 MiB');
  expectFailure(run(dir), 'expected a regular file');

  const paths = Array.from({ length: 5 }, (_, i) => {
    const json = JSON.stringify({ ...detail, id: `9cb24ef${i}-1c5a-4f29-a14e-83c6d48a5020` });
    const body = Buffer.alloc(max, ' ');
    body.write(json);
    const path = join(dir, `${i}.json`);
    writeFileSync(path, body);
    return path;
  });
  expectFailure(run(...paths), 'exceed 64 MiB');
});
