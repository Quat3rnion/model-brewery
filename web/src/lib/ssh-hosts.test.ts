import { afterEach, expect, it, vi } from 'vitest';
import { sshAction, saveHost, removeHost, type SshHost } from './ssh-hosts';

const host: SshHost = { id:'lab', name:'Lab', host:'lab.example', port:22, user:'operator', fingerprint:'SHA256:'+'A'.repeat(43), revision:3, status:'planned', confirmation:'a'.repeat(64) };
afterEach(() => vi.unstubAllGlobals());
it('sends only revision and confirmed current plan for install', async () => {
  const fetch = vi.fn().mockResolvedValue(Response.json(host));
  vi.stubGlobal('fetch', fetch);
  await sshAction(host,'install',true);
  expect(JSON.parse(fetch.mock.calls[0][1].body)).toEqual({action:'install', revision:3, confirmation:host.confirmation});
  await expect(sshAction(host,'install',false)).rejects.toThrow();
  await expect(sshAction({...host,status:'defined'},'install',true)).rejects.toThrow();
  expect(fetch).toHaveBeenCalledTimes(1);
});
it('updates definition without leaking saved plan fields; removal sends only revision', async () => {
  const fetch = vi.fn().mockResolvedValueOnce(Response.json(host)).mockResolvedValueOnce(new Response(null,{status:204}));
  vi.stubGlobal('fetch',fetch);
  await saveHost(host,host.revision);
  await removeHost(host);
  expect(JSON.parse(fetch.mock.calls[0][1].body)).toEqual({name:host.name,host:host.host,user:host.user,port:22,fingerprint:host.fingerprint,revision:3});
  expect(fetch.mock.calls[1]).toEqual(['/api/admin/ssh-hosts/lab',expect.objectContaining({method:'DELETE',body:'{"revision":3}',mode:'same-origin',redirect:'error'})]);
});
