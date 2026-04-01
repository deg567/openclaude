// @ts-nocheck
import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs'
import { dirname, join, resolve } from 'node:path'
import { spawnSync } from 'node:child_process'
import { homedir } from 'node:os'

type CheckResult = {
  ok: boolean
  label: string
  detail?: string
}

type CliOptions = {
  json: boolean
  outFile: string | null
}

function pass(label: string, detail?: string): CheckResult {
  return { ok: true, label, detail }
}

function fail(label: string, detail?: string): CheckResult {
  return { ok: false, label, detail }
}

function isTruthy(value: string | undefined): boolean {
  if (!value) return false
  const normalized = value.trim().toLowerCase()
  return normalized !== '' && normalized !== '0' && normalized !== 'false' && normalized !== 'no'
}

function parseOptions(argv: string[]): CliOptions {
  const options: CliOptions = {
    json: false,
    outFile: null,
  }

  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i]
    if (arg === '--json') {
      options.json = true
      continue
    }

    if (arg === '--out') {
      const next = argv[i + 1]
      if (next && !next.startsWith('--')) {
        options.outFile = next
        i++
      }
    }
  }

  return options
}

function checkNodeVersion(): CheckResult {
  const raw = process.versions.node
  const major = Number(raw.split('.')[0] ?? '0')
  if (Number.isNaN(major)) {
    return fail('Node.js version', `Could not parse version: ${raw}`)
  }

  if (major < 20) {
    return fail('Node.js version', `Detected ${raw}. Require >= 20.`)
  }

  return pass('Node.js version', raw)
}

function checkBunRuntime(): CheckResult {
  const bunVersion = (globalThis as { Bun?: { version?: string } }).Bun?.version
  if (!bunVersion) {
    return pass('Bun runtime', 'Not running inside Bun (this is acceptable for Node startup).')
  }
  return pass('Bun runtime', bunVersion)
}

function checkBuildArtifacts(): CheckResult {
  const distCli = resolve(process.cwd(), 'dist', 'cli.mjs')
  if (!existsSync(distCli)) {
    return fail('Build artifacts', `Missing ${distCli}. Run: bun run build`)
  }
  return pass('Build artifacts', distCli)
}

function isLocalBaseUrl(baseUrl: string): boolean {
  try {
    const url = new URL(baseUrl)
    return url.hostname === 'localhost' || url.hostname === '127.0.0.1' || url.hostname === '::1'
  } catch {
    return false
  }
}

function currentBaseUrl(): string {
  return process.env.OPENAI_BASE_URL ?? 'https://api.openai.com/v1'
}

function getCodexAuthPath(): string {
  return join(process.env.CODEX_HOME || join(homedir(), '.codex'), 'auth.json')
}

function readCodexAuthSession(): { accessToken: string; accountId: string } | null {
  const authPath = getCodexAuthPath()
  if (!existsSync(authPath)) return null

  try {
    const parsed = JSON.parse(readFileSync(authPath, 'utf8'))
    const accessToken = parsed?.tokens?.access_token
    const accountId =
      parsed?.tokens?.account_id ||
      parsed?.tokens?.id_token?.['https://api.openai.com/auth']?.chatgpt_account_id
    if (!accessToken || !accountId) return null
    return { accessToken, accountId }
  } catch {
    return null
  }
}

function checkOpenAIEnv(): CheckResult[] {
  const results: CheckResult[] = []
  const useOpenAI = isTruthy(process.env.CLAUDE_CODE_USE_OPENAI)

  if (!useOpenAI) {
    results.push(pass('Provider mode', 'Anthropic login flow enabled (CLAUDE_CODE_USE_OPENAI is off).'))
    return results
  }

  const baseUrl = process.env.OPENAI_BASE_URL ?? 'https://api.openai.com/v1'
  const model = process.env.OPENAI_MODEL
  const key = process.env.OPENAI_API_KEY
  const wantsCodexOAuth =
    process.env.OPENAI_AUTH_MODE === 'codex' || isTruthy(process.env.OPENAI_USE_CODEX_OAUTH)
  const codexAuthPath = getCodexAuthPath()

  results.push(pass('Provider mode', 'OpenAI-compatible provider enabled.'))

  if (!model) {
    results.push(pass('OPENAI_MODEL', 'Not set. Runtime fallback model will be used.'))
  } else {
    results.push(pass('OPENAI_MODEL', model))
  }

  results.push(pass('OPENAI_BASE_URL', baseUrl))

  if (wantsCodexOAuth) {
    if (!existsSync(codexAuthPath)) {
      results.push(fail('Codex auth cache', `Missing ${codexAuthPath}. Run codex login first.`))
    } else {
      results.push(pass('Codex auth cache', codexAuthPath))
    }
  } else if (key === 'SUA_CHAVE') {
    results.push(fail('OPENAI_API_KEY', 'Placeholder value detected: SUA_CHAVE.'))
  } else if (!key && !isLocalBaseUrl(baseUrl)) {
    results.push(fail('OPENAI_API_KEY', 'Missing key for non-local provider URL.'))
  } else if (!key) {
    results.push(pass('OPENAI_API_KEY', 'Not set (allowed for local providers like Ollama/LM Studio).'))
  } else {
    results.push(pass('OPENAI_API_KEY', 'Configured.'))
  }

  return results
}

async function checkBaseUrlReachability(): Promise<CheckResult> {
  if (!isTruthy(process.env.CLAUDE_CODE_USE_OPENAI)) {
    return pass('Provider reachability', 'Skipped (OpenAI-compatible mode disabled).')
  }

  const baseUrl = currentBaseUrl()
  const key = process.env.OPENAI_API_KEY
  const wantsCodexOAuth =
    process.env.OPENAI_AUTH_MODE === 'codex' || isTruthy(process.env.OPENAI_USE_CODEX_OAUTH)
  const endpoint = wantsCodexOAuth
    ? `${baseUrl.replace(/\/$/, '')}/responses`
    : `${baseUrl.replace(/\/$/, '')}/models`
  const codexSession = wantsCodexOAuth ? readCodexAuthSession() : null

  const controller = new AbortController()
  const timeout = setTimeout(() => controller.abort(), 4000)

  try {
    const headers: Record<string, string> = {}
    if (wantsCodexOAuth) {
      headers.Accept = 'text/event-stream'
      if (codexSession) {
        headers.Authorization = `Bearer ${codexSession.accessToken}`
        headers['ChatGPT-Account-ID'] = codexSession.accountId
      }
    }
    if (key) {
      headers.Authorization = `Bearer ${key}`
    }

    const response = await fetch(endpoint, {
      method: wantsCodexOAuth ? 'POST' : 'GET',
      headers,
      ...(wantsCodexOAuth
        ? {
            body: JSON.stringify({
              model: process.env.OPENAI_MODEL || 'gpt-5',
              instructions: 'Health check',
              input: [{ type: 'message', role: 'user', content: [{ type: 'input_text', text: 'ping' }] }],
              tools: [],
              tool_choice: 'auto',
              parallel_tool_calls: true,
              store: false,
              stream: true,
              include: [],
            }),
          }
        : {}),
      signal: controller.signal,
    })

    if (response.status === 200 || response.status === 401 || response.status === 403) {
      return pass('Provider reachability', `Reached ${endpoint} (status ${response.status}).`)
    }

    return fail('Provider reachability', `Unexpected status ${response.status} from ${endpoint}.`)
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error)
    return fail('Provider reachability', `Failed to reach ${endpoint}: ${message}`)
  } finally {
    clearTimeout(timeout)
  }
}

function checkOllamaProcessorMode(): CheckResult {
  if (!isTruthy(process.env.CLAUDE_CODE_USE_OPENAI)) {
    return pass('Ollama processor mode', 'Skipped (OpenAI-compatible mode disabled).')
  }

  const baseUrl = currentBaseUrl()
  if (!isLocalBaseUrl(baseUrl)) {
    return pass('Ollama processor mode', 'Skipped (provider URL is not local).')
  }

  const result = spawnSync('ollama', ['ps'], {
    cwd: process.cwd(),
    encoding: 'utf8',
    shell: true,
  })

  if (result.status !== 0) {
    const detail = (result.stderr || result.stdout || 'Unable to run ollama ps').trim()
    return fail('Ollama processor mode', detail)
  }

  const output = (result.stdout || '').trim()
  if (!output) {
    return fail('Ollama processor mode', 'ollama ps returned empty output.')
  }

  const lines = output.split(/\r?\n/).map(line => line.trim()).filter(Boolean)
  const modelLine = lines.find(line => line.includes(':') && !line.startsWith('NAME'))
  if (!modelLine) {
    return pass('Ollama processor mode', 'No loaded model found (run a prompt first).')
  }

  if (modelLine.includes('CPU')) {
    return pass('Ollama processor mode', 'Detected CPU mode. This is valid but can be slow for larger models.')
  }

  return pass('Ollama processor mode', `Detected non-CPU mode: ${modelLine}`)
}

function serializeSafeEnvSummary(): Record<string, string | boolean> {
  return {
    CLAUDE_CODE_USE_OPENAI: isTruthy(process.env.CLAUDE_CODE_USE_OPENAI),
    OPENAI_AUTH_MODE: process.env.OPENAI_AUTH_MODE ?? '(unset)',
    OPENAI_MODEL: process.env.OPENAI_MODEL ?? '(unset)',
    OPENAI_BASE_URL: process.env.OPENAI_BASE_URL ?? 'https://api.openai.com/v1',
    OPENAI_API_KEY_SET: Boolean(process.env.OPENAI_API_KEY),
  }
}

function printResults(results: CheckResult[]): void {
  for (const result of results) {
    const icon = result.ok ? 'PASS' : 'FAIL'
    const suffix = result.detail ? ` - ${result.detail}` : ''
    console.log(`[${icon}] ${result.label}${suffix}`)
  }
}

function writeJsonReport(
  options: CliOptions,
  results: CheckResult[],
): void {
  const payload = {
    timestamp: new Date().toISOString(),
    cwd: process.cwd(),
    summary: {
      total: results.length,
      passed: results.filter(result => result.ok).length,
      failed: results.filter(result => !result.ok).length,
    },
    env: serializeSafeEnvSummary(),
    results,
  }

  if (options.json) {
    console.log(JSON.stringify(payload, null, 2))
  }

  if (options.outFile) {
    const outputPath = resolve(process.cwd(), options.outFile)
    mkdirSync(dirname(outputPath), { recursive: true })
    writeFileSync(outputPath, JSON.stringify(payload, null, 2), 'utf8')
    if (!options.json) {
      console.log(`Report written to ${outputPath}`)
    }
  }
}

async function main(): Promise<void> {
  const options = parseOptions(process.argv.slice(2))
  const results: CheckResult[] = []

  results.push(checkNodeVersion())
  results.push(checkBunRuntime())
  results.push(checkBuildArtifacts())
  results.push(...checkOpenAIEnv())
  results.push(await checkBaseUrlReachability())
  results.push(checkOllamaProcessorMode())

  if (!options.json) {
    printResults(results)
  }

  writeJsonReport(options, results)

  const hasFailure = results.some(result => !result.ok)
  if (hasFailure) {
    process.exitCode = 1
    return
  }

  if (!options.json) {
    console.log('\nRuntime checks completed successfully.')
  }
}

await main()

export {}
