export const OWNER='cornelpascal'
export const REPOSITORY='prompt-compression-research'
const API=`https://api.github.com/repos/${OWNER}/${REPOSITORY}`
export type Run = {id:number;display_title:string;status:string;conclusion:string|null;html_url:string;created_at:string;updated_at:string;event:string;head_sha:string}
export type Job = {id:number;name:string;status:string;conclusion:string|null;steps?:{number:number;name:string;status:string;conclusion:string|null}[]}
export const TASKS=['attention-demo','compression-demo','privacy-pilot','theory','tests'] as const
export type Task = typeof TASKS[number]

class GitHubSession {
  #token=''
  get connected(){return Boolean(this.#token)}
  async request(path:string, method='GET', body?:unknown, token=this.#token){
    if(!path.startsWith('/actions/')&&path!=='/user'&&path!=='')throw new Error('Unsupported GitHub operation')
    const url=path==='/user'?'https://api.github.com/user':API+path
    const response=await fetch(url,{method,credentials:'omit',headers:{Accept:'application/vnd.github+json','X-GitHub-Api-Version':'2026-03-10',...(token?{Authorization:`Bearer ${token}`}:{}) ,...(body?{'Content-Type':'application/json'}:{})},body:body?JSON.stringify(body):undefined})
    if(!response.ok){
      if(response.status===401)throw new Error('GitHub authentication failed. Reconnect with a valid token.')
      if(response.status===403)throw new Error('GitHub denied this request or rate-limited it. Check repository Actions permissions and retry later.')
      if(response.status===404)throw new Error('This repository or workflow is not available to the current session.')
      throw new Error(`GitHub returned ${response.status}. The operation was not confirmed.`)
    }
    return response.status===204?null:response.json()
  }
  async connect(token:string){
    this.#token=''
    const user=await this.request('/user','GET',undefined,token)
    if(user.login!==OWNER)throw new Error(`Owner controls are restricted to ${OWNER}.`)
    const repository=await this.request('','GET',undefined,token)
    if(!repository.permissions?.push)throw new Error('Repository write permission is required.')
    this.#token=token
  }
  disconnect(){this.#token=''}
  async runs():Promise<Run[]>{const result=await this.request('/actions/workflows/research.yml/runs?per_page=15');return result.workflow_runs}
  async jobs(id:number):Promise<Job[]>{if(!Number.isSafeInteger(id)||id<=0)throw new Error('Invalid run ID');const result=await this.request(`/actions/runs/${id}/jobs`);return result.jobs}
  async dispatch(task:Task, requestId:string){
    if(!this.connected)throw new Error('Connect an owner session before launching a workflow.')
    if(!TASKS.includes(task)||!/^[-a-zA-Z0-9]{8,80}$/.test(requestId))throw new Error('Invalid workflow request')
    return this.request('/actions/workflows/research.yml/dispatches','POST',{ref:'main',inputs:{task,request_id:requestId}})
  }
  async cancel(id:number){if(!this.connected||!Number.isSafeInteger(id)||id<=0)throw new Error('An owner session and valid run ID are required.');return this.request(`/actions/runs/${id}/cancel`,'POST')}
}
export const github=new GitHubSession()
