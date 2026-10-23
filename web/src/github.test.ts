import { afterEach, describe, expect, it, vi } from 'vitest'
import { github } from './github'

const ok = (value:unknown, status=200) => new Response(status===204?null:JSON.stringify(value), {status})
afterEach(()=>{ github.disconnect(); vi.unstubAllGlobals() })

describe('owner workflow controls',()=>{
  it('rejects unauthenticated launches and cancellations',async()=>{
    await expect(github.dispatch('theory','request-123')).rejects.toThrow('Connect')
    await expect(github.cancel(123)).rejects.toThrow('owner session')
  })
  it('verifies identity and repository permission before retaining the token',async()=>{
    vi.stubGlobal('fetch',vi.fn().mockResolvedValue(ok({login:'visitor'})))
    await expect(github.connect('example-token')).rejects.toThrow('restricted')
    expect(github.connected).toBe(false)
    vi.stubGlobal('fetch',vi.fn().mockResolvedValueOnce(ok({login:'cornelpascal'})).mockResolvedValueOnce(ok({permissions:{push:false}})))
    await expect(github.connect('example-token')).rejects.toThrow('write permission')
    expect(github.connected).toBe(false)
  })
  it('dispatches only a fixed preset on main; disconnect removes authentication',async()=>{
    const store=vi.spyOn(Storage.prototype,'setItem')
    const fetch=vi.fn().mockResolvedValueOnce(ok({login:'cornelpascal'}))
      .mockResolvedValueOnce(ok({permissions:{push:true}})).mockResolvedValueOnce(ok(null,204))
      .mockResolvedValueOnce(ok({workflow_runs:[]}))
    vi.stubGlobal('fetch',fetch)
    await github.connect('example-token')
    await github.dispatch('theory','request-123')
    const [url,request]=fetch.mock.calls[2]
    expect(url).toBe('https://api.github.com/repos/cornelpascal/prompt-compression-research/actions/workflows/research.yml/dispatches')
    expect(JSON.parse(request.body)).toEqual({ref:'main',inputs:{task:'theory',request_id:'request-123'}})
    expect(request.headers.Authorization).toBe('Bearer example-token')
    expect(store).not.toHaveBeenCalled()
    github.disconnect()
    await github.runs()
    expect(fetch.mock.calls[3][1].headers.Authorization).toBeUndefined()
  })
  it('rejects unsafe IDs, presets, and unrelated endpoints',async()=>{
    vi.stubGlobal('fetch',vi.fn().mockResolvedValueOnce(ok({login:'cornelpascal'})).mockResolvedValueOnce(ok({permissions:{push:true}})))
    await github.connect('example-token')
    await expect(github.dispatch('theory','shell;code')).rejects.toThrow('Invalid')
    await expect(github.dispatch('shell' as 'theory','request-123')).rejects.toThrow('Invalid')
    await expect(github.jobs(-1)).rejects.toThrow('Invalid')
    await expect(github.request('https://example.org')).rejects.toThrow('Unsupported')
  })
  it.each([401,403,404,500])('handles HTTP %s without exposing response credentials',async(status)=>{
    vi.stubGlobal('fetch',vi.fn().mockResolvedValue(ok({message:'sensitive-response-example'},status)))
    await expect(github.runs()).rejects.not.toThrow('sensitive-response-example')
  })
})
