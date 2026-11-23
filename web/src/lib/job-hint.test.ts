import { expect, it } from 'vitest';
import { jobFixHint } from './job-hint';

it('gives specific guidance and direct view links only for known errors', () => {
  expect(jobFixHint('write: No space left on device (os error 28)')).toMatchObject({
    text: 'Free space on the host before you try again.',
    href: '#/settings/infrastructure',
  });
  expect(jobFixHint('SafeTensors source requires a valid immutable publication or an explicit verification allowance within the size limit')).toMatchObject({
    text: 'Select a verified SafeTensors source in Weights before you try again.',
    href: '#/weights',
  });
  expect(jobFixHint('Unknown error')).toBeNull();
});
