import { describe, expect, it } from 'vitest'
import { csv, pct, privacySeries } from './data'
import type { PrivacyRow } from './types'

describe('measurement integrity',()=>{
  it('distinguishes missing measurements from measured zero',()=>{
    expect(pct(undefined)).toBe('Not measured')
    expect(pct(NaN)).toBe('Not measured')
    expect(pct(0)).toBe('0%')
    expect(pct(.98)).toBe('98%')
  })
  it('keeps incompatible attacker access conditions separate',()=>{
    const a={regime:'independent',domain:'scheduling',access:'text',attacker:'copy'} as PrivacyRow
    const b={...a,access:'context'}
    const c={...a,attacker:'attribute_nb'}
    expect(privacySeries([a,b,c],'independent','scheduling','text','copy')).toEqual([a])
  })
  it('exports commas, multiline cells, quotes and missing values losslessly',()=>{
    expect(csv([{name:'one,"two"\nthree',value:0},{name:'second'}])).toBe('"name","value"\n"one,""two""\nthree","0"\n"second",""')
  })
})
