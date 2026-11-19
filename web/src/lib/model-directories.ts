import { obj } from './reports';

export interface Directories { available:boolean; reason?:string; configured:string[]; granted:string[] }
export interface DirectoryChange { worker:string; revision:number; action:'add'|'replace'|'remove'; path:string; previous_path?:string }
export interface DirectoryReceipt extends DirectoryChange { confirmation:string }
const paths = (v:unknown):v is string[] => Array.isArray(v) && v.every(p => typeof p === 'string' && p.startsWith('/'));
export function readDirectories(value:unknown):Directories|null {
  const d = obj(value);
  if (!d || typeof d.available !== 'boolean' || !paths(d.configured) || !paths(d.granted)) return null;
  return {available:d.available, configured:d.configured, granted:d.granted, ...(typeof d.reason === 'string' ? {reason:d.reason} : {})};
}
export function directoryReceipt(value:unknown, request:DirectoryChange):DirectoryReceipt|null {
  const d = obj(value);
  if (!d || d.worker !== request.worker || d.revision !== request.revision || d.action !== request.action ||
      d.path !== request.path || (d.previous_path ?? undefined) !== request.previous_path ||
      typeof d.confirmation !== 'string' || !d.confirmation) return null;
  return {...request, confirmation:d.confirmation};
}
