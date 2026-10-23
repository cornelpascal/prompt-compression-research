import { defineConfig } from '@playwright/test'

export default defineConfig({
  testDir: './e2e', timeout: 30000, retries: 1, workers: 2,
  use: { baseURL: 'http://127.0.0.1:4173/prompt-compression-research/', screenshot:'only-on-failure', trace:'retain-on-failure' },
  webServer: { command: 'npm run preview -- --port 4173', url: 'http://127.0.0.1:4173/prompt-compression-research/', reuseExistingServer: !process.env.CI },
  projects: [{name:'desktop',use:{viewport:{width:1440,height:1000}}},{name:'mobile',use:{viewport:{width:390,height:844}}}],
})
