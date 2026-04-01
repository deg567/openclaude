// @ts-nocheck
import { writeFileSync } from 'node:fs'
import { resolve } from 'node:path'

type ProviderProfile = 'codex' | 'openai' | 'ollama'

type ProfileFile = {
  profile: ProviderProfile
  env: {
    OPENAI_BASE_URL?: string
    OPENAI_MODEL?: string
    OPENAI_API_KEY?: string
    OPENAI_AUTH_MODE?: string
    OPENAI_USE_CODEX_OAUTH?: string
  }
  createdAt: string
}

function parseArg(name: string): string | null {
  const args = process.argv.slice(2)
  const idx = args.indexOf(name)
  if (idx === -1) return null
  return args[idx + 1] ?? null
}

function parseProviderArg(): ProviderProfile | 'auto' {
  const p = parseArg('--provider')?.toLowerCase()
  if (p === 'codex' || p === 'openai' || p === 'ollama') return p
  return 'auto'
}

async function hasLocalOllama(): Promise<boolean> {
  const endpoint = 'http://localhost:11434/api/tags'
  const controller = new AbortController()
  const timeout = setTimeout(() => controller.abort(), 1200)

  try {
    const response = await fetch(endpoint, {
      method: 'GET',
      signal: controller.signal,
    })
    return response.ok
  } catch {
    return false
  } finally {
    clearTimeout(timeout)
  }
}

function sanitizeApiKey(key: string | null): string | undefined {
  if (!key || key === 'SUA_CHAVE') return undefined
  return key
}

async function main(): Promise<void> {
  const provider = parseProviderArg()
  const argModel = parseArg('--model')
  const argBaseUrl = parseArg('--base-url')
  const argApiKey = parseArg('--api-key')

  let selected: ProviderProfile
  if (provider === 'auto') {
    selected = (await hasLocalOllama()) ? 'ollama' : 'openai'
  } else {
    selected = provider
  }

  const env: ProfileFile['env'] = {}
  if (selected === 'ollama') {
    env.OPENAI_BASE_URL = argBaseUrl || 'http://localhost:11434/v1'
    env.OPENAI_MODEL = argModel || process.env.OPENAI_MODEL || 'llama3.1:8b'
    const key = sanitizeApiKey(argApiKey || process.env.OPENAI_API_KEY || null)
    if (key) env.OPENAI_API_KEY = key
  } else if (selected === 'codex') {
    env.OPENAI_BASE_URL = argBaseUrl || process.env.OPENAI_BASE_URL || 'https://chatgpt.com/backend-api/codex'
    env.OPENAI_MODEL = argModel || process.env.OPENAI_MODEL || 'gpt-5'
    env.OPENAI_AUTH_MODE = 'codex'
    env.OPENAI_USE_CODEX_OAUTH = '1'
  } else {
    env.OPENAI_BASE_URL = argBaseUrl || process.env.OPENAI_BASE_URL || 'https://api.openai.com/v1'
    env.OPENAI_MODEL = argModel || process.env.OPENAI_MODEL || 'gpt-4o'
    const key = sanitizeApiKey(argApiKey || process.env.OPENAI_API_KEY || null)
    if (!key) {
      console.error('OpenAI profile requires a real API key. Use --api-key or set OPENAI_API_KEY.')
      process.exit(1)
    }
    env.OPENAI_API_KEY = key
  }

  const profile: ProfileFile = {
    profile: selected,
    env,
    createdAt: new Date().toISOString(),
  }

  const outputPath = resolve(process.cwd(), '.openclaude-profile.json')
  writeFileSync(outputPath, JSON.stringify(profile, null, 2), 'utf8')

  console.log(`Saved profile: ${selected}`)
  console.log(`Path: ${outputPath}`)
  console.log('Next: bun run dev:profile')
}

await main()

export {}
