import { defineConfig } from '@playwright/test';

export default defineConfig({
  testDir: './tests/flows',
  forbidOnly: !!process.env.CI,
  workers: 2,
  outputDir: `${process.env.TMPDIR ?? 'dist'}/brewery-flow-results`,
  use: {
    browserName: 'chromium',
    reducedMotion: 'reduce',
    serviceWorkers: 'block',
    trace: 'retain-on-failure',
  },
  projects: [1280, 390].flatMap((width) => ['light', 'dark'].map((colorScheme) => ({
    name: `${width}-${colorScheme}`,
    use: { viewport: { width, height: 900 }, colorScheme },
  }))),
});
