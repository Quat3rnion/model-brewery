export interface Definition { id: string; name: string; host: string; user: string; port: number; fingerprint: string; identity_file?: string|null }
export interface SshHost extends Definition { revision: number; status: 'defined'|'verified'|'planned'|'installed'; confirmation?: string }
export type Action = 'test'|'plan'|'install';
const base = '/api/admin/ssh-hosts';
const path = (id: string) => `${base}/${encodeURIComponent(id)}`;
const fp = /^SHA256:[A-Za-z0-9+/]{43}$/;
export function validEndpoint(host:string,port:number) {
  return !!host && !/[\s/@]/.test(host) && !host.startsWith('-') && Number.isInteger(port) && port >= 1 && port <= 65535;
}
export interface HostKey {host:string; port:number; algorithm:'ssh-ed25519'; fingerprint:string; public_key:string}
export async function probeHostKey(host:string,port:number):Promise<HostKey> {
  if (!validEndpoint(host,port)) throw new Error('Enter a host address and valid SSH port.');
  const key = await request('/api/admin/ssh-host-key','POST',{host,port}) as HostKey;
  if (key?.host !== host || key.port !== port || key.algorithm !== 'ssh-ed25519' ||
      typeof key.fingerprint !== 'string' || !fp.test(key.fingerprint) ||
      typeof key.public_key !== 'string' || !/^ssh-ed25519 [A-Za-z0-9+/]{68}$/.test(key.public_key))
    throw new Error('The probe did not return a valid host key for this address and port.');
  return key;
}
export function problem(d: Definition) {
  if (!d.id.trim() || !d.name.trim()) return 'Enter a host ID and display name.';
  if (!d.host || /[\s/@]/.test(d.host) || d.host.startsWith('-')) return 'Enter a host address, not a URL.';
  if (!d.user.trim() || /[\s/@]/.test(d.user)) return 'Enter the SSH account name.';
  if (!Number.isInteger(d.port) || d.port < 1 || d.port > 65535) return 'Use a port from 1 to 65535.';
  if (!fp.test(d.fingerprint)) return 'Enter the complete SHA256 fingerprint from a trusted source.';
  if (d.identity_file != null && (!d.identity_file.startsWith('/') || d.identity_file.endsWith('/') ||
    d.identity_file.trim() !== d.identity_file || /[\u0000-\u001f\u007f\u202a-\u202e\u2066-\u2069]/u.test(d.identity_file)))
    return 'Enter an absolute path to a login key file on the controller.';
  return '';
}
async function request(url: string, method = 'GET', body?: unknown): Promise<unknown> {
  const r = await fetch(url,{method,mode:'same-origin',referrerPolicy:'same-origin',credentials:'same-origin',redirect:'error',cache:'no-store',
    headers:body ? {'Content-Type':'application/json'} : undefined,body:body ? JSON.stringify(body) : undefined});
  if (!r.ok) {
    if (body && typeof body === 'object' && 'identity_file' in body && typeof body.identity_file === 'string' && [400,422].includes(r.status))
      throw new Error(`HTTP ${r.status}: Check the host fields and login identity. Select a listed key or enter the absolute path to a private key the controller can read.`);
    if (url === '/api/admin/ssh-host-key') {
      const issue: Record<number,string> = {
        429:'Another host key probe is running. Try again when it finishes.',
        502:'No Ed25519 host key was received. Check the address, port and SSH server.',
        503:'The controller cannot run ssh-keyscan. Install the OpenSSH client on the controller or enter a known fingerprint.',
      };
      if (issue[r.status]) throw new Error(`HTTP ${r.status}: ${issue[r.status]}`);
    }
    const reason: Record<number,string> = {400:'Check the host fields.',401:'Sign in to Settings.',403:'Check your Settings access.',409:'The host changed, is busy, or cannot perform this action. Refresh and review it again.',502:'The SSH helper refused the action. Check the address, account and trusted fingerprint.',503:'The SSH helper is unavailable.'};
    throw new Error(`HTTP ${r.status}: ${reason[r.status] ?? 'Refresh the host list and check the controller configuration.'}`);
  }
  return r.status === 204 ? null : await r.json();
}
function row(value: unknown): SshHost {
  const h = value as SshHost;
  if (!h || !['id','name','host','user','fingerprint'].every(k => typeof (value as Record<string,unknown>)[k] === 'string') ||
    !Number.isInteger(h.port) || !Number.isSafeInteger(h.revision) || !['defined','verified','planned','installed'].includes(h.status)) throw new Error('Unrecognized SSH host receipt. Refresh the host list.');
  return h;
}
export async function listHosts() {
  const result = await request(base) as {hosts:unknown[]; available:boolean};
  if (!Array.isArray(result?.hosts) || typeof result.available !== 'boolean') throw new Error('The SSH host list is unavailable.');
  return {hosts:result.hosts.map(row),available:result.available};
}
export async function saveHost(d:Definition, revision?:number) {
  const issue = problem(d);
  if (issue) throw new Error(issue);
  const {name,host,user,port,fingerprint,identity_file} = d;
  const body = {name,host,user,port,fingerprint,identity_file};
  return row(await (revision === undefined ? request(base,'POST',{...body,id:d.id}) : request(path(d.id),'PUT',{...body,revision})));
}
export function permitted(h:SshHost, action:Action) {
  if (action === 'test') return h.status === 'defined' || h.status === 'verified';
  if (action === 'plan') return h.status === 'verified' || h.status === 'planned';
  return h.status === 'planned' && /^[0-9a-f]{64}$/.test(h.confirmation ?? '');
}
export async function sshAction(h:SshHost, action:Action, confirmed=false) {
  if (!permitted(h,action) || (action === 'install' && !confirmed)) throw new Error('Review and confirm the current installation plan.');
  return row(await request(`${path(h.id)}/action`,'POST',{action,revision:h.revision,...(action === 'install' ? {confirmation:h.confirmation} : {})}));
}
export const removeHost = (h:SshHost) => request(path(h.id),'DELETE',{revision:h.revision});

export async function listIdentities() {
  const result = await request('/api/admin/ssh-identities') as {identities:{path:string;label:string}[];default_configured:boolean};
  if (!Array.isArray(result?.identities) || typeof result.default_configured !== 'boolean' ||
    !result.identities.every(i=>typeof i?.path === 'string' && i.path.startsWith('/') && typeof i.label === 'string'))
    throw new Error('The controller did not report SSH login identities.');
  return result;
}
