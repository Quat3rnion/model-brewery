/// <reference types="vitest/config" />
import { defineConfig, type Plugin } from 'vite';
import { svelte } from '@sveltejs/vite-plugin-svelte';
import type { IncomingMessage, ServerResponse } from 'node:http';

// Production: the controller serves web/dist and /api on one loopback origin.
//
// Development only: Vite serves the UI on DEV_HOST and proxies /api to the
// controller. The controller accepts only its own Host and Origin. The dev
// server first applies the same checks to its own origin and rejects any
// other request. Only then does the proxy set the upstream Host and Origin.
// The controller guard is not changed.
const controller = new URL(process.env.BREWERY_CONTROLLER ?? 'http://127.0.0.1:4141');
const DEV_PORT = 5174;
const PREVIEW_PORT = 5175;

function single(req: IncomingMessage, name: string): string | undefined | null {
  const raw = req.headersDistinct[name];
  if (!raw) return undefined;
  return raw.length === 1 ? raw[0] : null; // null: repeated header
}

/** Same rules as the controller guard, applied to the dev server origin. */
export function devRequestAllowed(req: IncomingMessage, port: number): boolean {
  const host = single(req, 'host');
  if (host !== `127.0.0.1:${port}` && host !== `localhost:${port}`) return false;
  const origin = single(req, 'origin');
  if (origin === null || (origin !== undefined && origin !== `http://${host}`)) return false;
  const site = single(req, 'sec-fetch-site');
  if (site === null || (site !== undefined && site !== 'same-origin' && site !== 'none')) return false;
  if (req.method === 'POST' && origin === undefined && site !== 'same-origin') {
    const type = single(req, 'content-type')?.split(';')[0].trim().toLowerCase();
    if (type !== 'application/json') return false;
  }
  return true;
}

function guard(port: number) {
  return (req: IncomingMessage, res: ServerResponse, next: () => void) => {
    if (!(req.url === '/api' || req.url?.startsWith('/api/'))) return next();
    if (devRequestAllowed(req, port)) return next();
    res.statusCode = 403;
    res.setHeader('Content-Type', 'application/json');
    res.end(JSON.stringify({ error: 'host or origin is not allowed (dev proxy)' }));
  };
}

const devGuard: Plugin = {
  name: 'brewery-dev-api-guard',
  // Plain use() runs before the Vite proxy middleware.
  configureServer(server) {
    server.middlewares.use(guard(DEV_PORT));
  },
  configurePreviewServer(server) {
    server.middlewares.use(guard(PREVIEW_PORT));
  },
};

const proxy = {
  '/api': {
    target: controller.origin,
    changeOrigin: true, // upstream Host = controller host
    configure(p: { on(event: 'proxyReq', fn: (proxyReq: { getHeader(k: string): unknown; setHeader(k: string, v: string): void }) => void): void }) {
      p.on('proxyReq', (proxyReq) => {
        // The request passed devRequestAllowed, so a present Origin is the dev origin.
        if (proxyReq.getHeader('origin') !== undefined) proxyReq.setHeader('origin', controller.origin);
      });
    },
  },
};

export default defineConfig({
  plugins: [devGuard, svelte()],
  base: './',
  build: { outDir: 'dist', emptyOutDir: true, sourcemap: false },
  server: { host: '127.0.0.1', port: DEV_PORT, strictPort: true, proxy },
  preview: { host: '127.0.0.1', port: PREVIEW_PORT, strictPort: true, proxy },
  test: { include: ['src/**/*.test.ts'], environment: 'node' },
});
