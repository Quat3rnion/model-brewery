import { expect, it } from 'vitest';
import { directoryReceipt, readDirectories } from './model-directories';

it('accepts confirmation only for the exact reviewed operation, worker, revision and paths', () => {
  const request = { worker:'local-eval', revision:3, action:'replace' as const, path:'/models/new', previous_path:'/models/old' };
  const reply = {...request, confirmation:'opaque'};
  expect(directoryReceipt(reply,request)).toEqual(reply);
  for (const patch of [{worker:'other'}, {revision:4}, {action:'add'}, {path:'/elsewhere'}, {previous_path:undefined}, {confirmation:''}])
    expect(directoryReceipt({...reply,...patch}, request)).toBeNull();
});

it('requires explicit configured and granted sets, including when both are empty', () => {
  expect(readDirectories({available:true, configured:[], granted:[]})).toEqual({available:true, configured:[], granted:[]});
  expect(readDirectories({available:true, configured:[]})).toBeNull();
});
