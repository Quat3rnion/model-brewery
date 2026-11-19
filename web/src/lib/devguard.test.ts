import { describe, expect, it } from 'vitest';
import type { IncomingMessage } from 'node:http';
import { devRequestAllowed } from '../../vite.config';

const req = (method: string, headers: Record<string, string | string[]>) =>
  ({ method, headersDistinct: Object.fromEntries(Object.entries(headers).map(([k, v]) => [k, Array.isArray(v) ? v : [v]])) }) as unknown as IncomingMessage;
const host = { host: '127.0.0.1:5174' };

describe('dev proxy guard mirrors the controller guard', () => {
  it('allows same-origin browser and plain CLI GET', () => {
    expect(devRequestAllowed(req('GET', { ...host, origin: 'http://127.0.0.1:5174', 'sec-fetch-site': 'same-origin' }), 5174)).toBe(true);
    expect(devRequestAllowed(req('GET', host), 5174)).toBe(true);
  });
  it('rejects foreign host, origin, fetch site and repeated headers', () => {
    expect(devRequestAllowed(req('GET', { host: 'evil:5174' }), 5174)).toBe(false);
    expect(devRequestAllowed(req('GET', { ...host, origin: 'http://evil.example' }), 5174)).toBe(false);
    expect(devRequestAllowed(req('GET', { ...host, 'sec-fetch-site': 'cross-site' }), 5174)).toBe(false);
    expect(devRequestAllowed(req('GET', { ...host, origin: ['http://127.0.0.1:5174', 'http://127.0.0.1:5174'] }), 5174)).toBe(false);
  });
  it('requires JSON for a POST without browser same-origin proof', () => {
    expect(devRequestAllowed(req('POST', host), 5174)).toBe(false);
    expect(devRequestAllowed(req('POST', { ...host, 'content-type': 'application/json' }), 5174)).toBe(true);
    expect(devRequestAllowed(req('POST', { ...host, 'sec-fetch-site': 'same-origin' }), 5174)).toBe(true);
  });
});
