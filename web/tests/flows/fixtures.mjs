import { test as base } from '@playwright/test';
import { svelte } from '@sveltejs/vite-plugin-svelte';
import { fileURLToPath } from 'node:url';
import { createServer } from 'vite';
import { startController } from './mock/controller.mjs';

export { expect } from '@playwright/test';

export const test = base.extend({
  assets: [async ({}, use) => {
    const vite = await createServer({
      root: fileURLToPath(new URL('../../', import.meta.url)),
      configFile: false, // Do not use the development proxy or BREWERY_CONTROLLER.
      plugins: [svelte()],
      server: { middlewareMode: true, ws: false },
    });
    try {
      await use(vite.middlewares);
    } finally {
      await vite.close();
    }
  }, { scope: 'worker' }],

  controller: async ({ assets }, use) => {
    const controller = await startController(assets);
    try {
      await use(controller);
    } finally {
      await controller.close();
    }
  },

  context: async ({ context, controller }, use) => {
    // Mock pages must never make a request to a real controller or other origin.
    await context.route('**/*', (route) => (
      new URL(route.request().url()).origin === controller.origin ? route.continue() : route.abort()
    ));
    await use(context);
  },
});
