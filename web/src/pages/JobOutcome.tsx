import { useEffect, useState } from 'react'
import { CartesianGrid, Legend, ResponsiveContainer, Scatter, ScatterChart, Tooltip, XAxis, YAxis } from 'recharts'
import { Badge, Figure, Notice } from '../components'
import { download, label, resource } from '../data'
import type { Evidence } from '../types'

type RecordData={id:string;task:string;status:string;evidence:Evidence;commit:string;elapsedSeconds:number;
  summary?:{unit:string;rows:Record<string,unknown>[];interpretation:string};
  suites?:{name:string;tests:number;failures:number;errors:number;skipped:number}[]}
const mean=(v:unknown)=>typeof v==='number'?v:Number((v as {mean:number})?.mean)
export default function JobOutcome({file}:{file:string}){
  const [data,setData]=useState<RecordData>();const [error,setError]=useState('')
  useEffect(()=>{setData(undefined);setError('');resource<RecordData>(file).then(setData).catch(e=>setError(e.message))},[file])
  if(error)return <Notice>{error}</Notice>
  if(!data)return <p>Loading the published outcome…</p>
  let rows:Record<string,unknown>[]=[];let xLabel='',yLabel='',scope=''
  const raw=data.summary?.rows||[]
  if(data.task==='theory'){
    rows=raw.filter(r=>r.error_budget===0&&r.status!=='infeasible').map(r=>({group:String(r.regime),x:r.length_budget,y:r.leakage_bits}))
    xLabel='Expected length budget';yLabel='Leakage (bits)';scope='Zero task-error slice; infeasible points omitted. All grid points remain in the JSON export.'
  }else if(data.task.includes('demo')){
    const target=raw.find(r=>r.target!=='compressor_direct')?.target
    rows=raw.filter(r=>r.target===target).map(r=>({group:String(r.method),x:r.request_retained_fraction,y:mean(r.f1)}))
    xLabel='Achieved receiver-token retention';yLabel='Answer F1';scope=`Receiver alias: ${target}. Random tiny-model fixture; no pretrained quality claim.`
  }else if(data.task==='privacy-pilot'){
    const first=raw[0]
    if(first){rows=raw.filter(r=>['regime','domain','access','attacker'].every(k=>r[k]===first[k])).map(r=>({group:String(r.method),x:mean(r.utility),y:mean(r.secret_recovery)}))
      scope=`${first.regime} · ${first.domain} · ${first.access} access · ${first.attacker}. Length uses UTF-8 bytes; other conditions are in the JSON export.`}
    xLabel='Task utility';yLabel='Secret recovery'
  }
  const groups=[...new Set(rows.map(r=>String(r.group)))],colors=['#285d4b','#bd6b4d','#7b79aa','#779333']
  return <section className="published-result"><div className="panel prose"><Badge kind={data.evidence}/><h3>Published result · {data.id}</h3><p>{scope||'Engineering test outcomes from the isolated CPU runner.'}</p><p className="caption">Commit {data.commit.slice(0,12)} · {data.elapsedSeconds.toFixed(1)} seconds · {data.status}. Timings include the preset workload and are not compressor inference benchmarks.</p><button className="button subtle small" onClick={()=>download(`outcome-${data.id}.json`,JSON.stringify(data,null,2),'application/json')}>Download complete result JSON</button></div>
  {data.suites&&<div className="table-scroll panel"><table><thead><tr><th>Suite</th><th>Passed</th><th>Failures</th><th>Errors</th><th>Skipped</th></tr></thead><tbody>{data.suites.map(s=><tr key={s.name}><td>{s.name}</td><td>{s.tests-s.failures-s.errors-s.skipped}</td><td>{s.failures}</td><td>{s.errors}</td><td>{s.skipped}</td></tr>)}</tbody></table><p className="caption">Distributed tests are skipped in the default compression suite and exercised separately. Skips are not passes.</p></div>}
  {rows.length>0&&<Figure title="Workflow measurements" subtitle={`${yLabel} versus ${xLabel.toLowerCase()}`} evidence={data.evidence} rows={rows} observations={`${rows.length} points in this declared slice.`} interpretation={data.summary?.interpretation||''} limitations={scope} reproduce={`gh workflow run research.yml -f task=${data.task} -f request_id=repeat-${data.id}`}><ResponsiveContainer width="100%" height={320}><ScatterChart margin={{top:12,right:25,left:15,bottom:28}}><CartesianGrid strokeDasharray="3 4"/><XAxis type="number" dataKey="x" name={xLabel} label={{value:xLabel,position:'bottom',offset:10}}/><YAxis type="number" dataKey="y" name={yLabel}/><Tooltip/><Legend verticalAlign="top"/>{groups.map((group,i)=><Scatter key={group} name={label(group)} data={rows.filter(r=>r.group===group).sort((a,b)=>Number(a.x)-Number(b.x))} fill={colors[i%colors.length]} line/>)}</ScatterChart></ResponsiveContainer></Figure>}
  </section>
}
