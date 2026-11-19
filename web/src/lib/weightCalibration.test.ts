import { expect, it, vi } from 'vitest';
import { uploadBatch } from './weightCalibration';

it('registers only the two matrices and display name, and returns the backend entry ID', async () => {
  const file = new File(['{"activations":[[1,2]],"targets":[[3]]}'], 'fit.json');
  const send = vi.fn(async () => Response.json({ id: 'fit-1', kind: 'weight-calibration', host: 'backend-hf' }, { status: 201 }));
  expect(await uploadBatch(file, send)).toBe('fit-1');
  expect(send).toHaveBeenCalledWith('/api/models/weight-calibration', expect.objectContaining({
    method: 'POST', body: '{"name":"fit.json","activations":[[1,2]],"targets":[[3]]}',
  }));
});

it.each([
  '{"activations":[[1],[2,3]],"targets":[[4],[5]]}',
  '{"activations":[[1e999]],"targets":[[0]]}',
  '{"activations":[[1]],"targets":[]}',
  '{"activations":[[1]],"targets":[[2]],"path":"/not-an-upload"}',
  ' '.repeat(256 * 1024 + 1),
])('refuses invalid or oversized data without sending a request', async content => {
  const send = vi.fn();
  await expect(uploadBatch(new File([content], 'bad.json'), send)).rejects.toThrow();
  expect(send).not.toHaveBeenCalled();
});

it('directs the user to the existing Library entry after a duplicate upload', async () => {
  const file = new File(['{"activations":[[1]],"targets":[[2]]}'], 'again.json');
  await expect(uploadBatch(file, async () => Response.json(
    { error: 'this calibration batch is already registered' }, { status: 409 },
  ))).rejects.toThrow(/select its Library entry/i);
});
