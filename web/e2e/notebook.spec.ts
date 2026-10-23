import { expect, test } from '@playwright/test'

test('all research sections render without browser errors',async({page})=>{
  const errors:string[]=[];page.on('pageerror',e=>errors.push(e.message))
  for(const path of ['/','/methods','/methods/attention','/literature','/datasets','/experiments','/results','/validation','/run','/reproduce','/about']){
    await page.goto('#'+path)
    await expect(page.locator('h1')).toBeVisible()
    await expect(page.locator('main')).not.toContainText('Unable to load')
    expect(await page.evaluate(()=>document.documentElement.scrollWidth<=window.innerWidth)).toBe(true)
  }
  expect(errors).toEqual([])
})

test('graphs expose underlying values, filters and exports',async({page})=>{
  await page.goto('#/results')
  await expect(page.getByRole('heading',{name:'Answer quality versus retained context'})).toBeVisible()
  const svg=page.waitForEvent('download');await page.getByRole('button',{name:'SVG',exact:true}).click()
  expect((await svg).suggestedFilename()).toContain('.svg')
  await page.getByRole('button',{name:'Data table',exact:true}).click()
  await expect(page.locator('table tbody tr').first()).toBeVisible()
  const csv=page.waitForEvent('download');await page.getByRole('button',{name:'CSV',exact:true}).click()
  expect((await csv).suggestedFilename()).toContain('.csv')
  await page.getByRole('tab',{name:'Privacy & utility',exact:true}).click()
  await expect(page.locator('h3').filter({hasText:'Utility'}).first()).toBeVisible()
  await page.getByRole('tab',{name:'Finite-source frontier',exact:true}).click()
  await expect(page.locator('main')).toContainText('bits')
})

test('search, literature filtering, and owner boundary work',async({page})=>{
  await page.goto('#/')
  await page.getByRole('button',{name:'Search research /'}).click()
  await page.getByRole('textbox',{name:'Search research',exact:true}).fill('attention')
  await expect(page.getByRole('dialog')).toContainText('attention')
  await page.keyboard.press('Escape')
  await expect(page.getByRole('dialog')).not.toBeVisible()
  await page.goto('#/literature')
  await page.getByRole('textbox',{name:'Search papers'}).fill('nonexistent-paper-0xyz')
  await expect(page.getByRole('heading',{name:'No matching papers'})).toBeVisible()
  await page.goto('#/run')
  await expect(page.getByRole('button',{name:'Launch workflow'})).toBeDisabled()
  await expect(page.locator('main')).toContainText('browser memory only')
})
