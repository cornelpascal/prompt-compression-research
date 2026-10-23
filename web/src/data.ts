import { useEffect, useState } from 'react'
import type { Catalog, Literature, PrivacyRow } from './types'

const cache = new Map<string, Promise<unknown>>()
export function resource<T>(file: string): Promise<T> {
  if (!cache.has(file)) cache.set(file, fetch(`${import.meta.env.BASE_URL}data/${file}`).then(r => {
    if (!r.ok) throw new Error('The research snapshot could not be loaded. Please retry.')
    return r.json()
  }).catch(error => { cache.delete(file); throw error }))
  return cache.get(file) as Promise<T>
}
export function useCatalog() {
  const [data,setData]=useState<Catalog>(); const [error,setError]=useState('')
  useEffect(()=>{ resource<Catalog>('catalog.json').then(setData).catch(e=>setError(e.message)) },[])
  return {data,error}
}
export function useLiterature() {
  const [data,setData]=useState<Literature>(); const [error,setError]=useState('')
  useEffect(()=>{ resource<Literature>('literature.json').then(setData).catch(e=>setError(e.message)) },[])
  return {data,error}
}
export const pct = (n: number|null|undefined, digits=0) => typeof n === 'number' && Number.isFinite(n) ? `${(n*100).toFixed(digits)}%` : 'Not measured'
export const fmt = (n: number) => new Intl.NumberFormat('en').format(n)
export const date = (s:string) => new Date(s.slice(0,10)+'T12:00:00Z').toLocaleDateString('en-GB',{day:'numeric',month:'short',year:'numeric',timeZone:'UTC'})
export const sourceUrl = (path:string) => `https://github.com/cornelpascal/prompt-compression-research/blob/main/${path}`
export const label = (key:string) => ({attention_frozen:'Frozen attention',attention_sft:'Trained Q/K',rewrite:'Rewrite · RL',rewrite_sft:'Rewrite · SFT',delete:'Deletion · RL',delete_sft:'Deletion · SFT',original:'Original',truncate:'Prefix truncation',llmlingua2:'LLMLingua-2',empty:'Empty context',redact:'Redaction',redact_compress:'Redact + compress',independent:'Independent',correlated:'Correlated',essential:'Task-essential'}[key] || key.replaceAll('_',' '))
export function privacySeries(rows:PrivacyRow[], regime:string, domain:string, access:string, attacker:string) {
  return rows.filter(r=>r.regime===regime && r.domain===domain && r.access===access && r.attacker===attacker)
}
export function download(name:string, content:string, mime='text/plain') {
  const url=URL.createObjectURL(new Blob([content],{type:mime})); const a=document.createElement('a');a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000)
}
export function csv(rows:Record<string,unknown>[]) {
  const keys=[...new Set(rows.flatMap(r=>Object.keys(r)))];const cell=(v:unknown)=>'"'+String(typeof v==='object'?JSON.stringify(v):v??'').replaceAll('"','""')+'"'
  return [keys.map(cell).join(','),...rows.map(r=>keys.map(k=>cell(r[k])).join(','))].join('\n')
}
