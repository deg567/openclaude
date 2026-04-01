import { existsSync, readFileSync } from 'node:fs'
import { homedir } from 'node:os'
import { join } from 'node:path'

type CodexAuthJson = {
  auth_mode?: string | null
  tokens?: {
    access_token?: string | null
    account_id?: string | null
    id_token?: {
      'https://api.openai.com/auth'?: {
        chatgpt_account_id?: string | null
        chatgpt_plan_type?: string | null
      }
    } | null
  } | null
}

export type CodexAuthSession = {
  accessToken: string
  accountId: string
  planType?: string
}

function getCodexHome(): string {
  return process.env.CODEX_HOME || join(homedir(), '.codex')
}

export function getCodexAuthFilePath(): string {
  return join(getCodexHome(), 'auth.json')
}

export function getCodexAuthSession(): CodexAuthSession | null {
  const authPath = getCodexAuthFilePath()
  if (!existsSync(authPath)) return null

  try {
    const parsed = JSON.parse(readFileSync(authPath, 'utf8')) as CodexAuthJson
    const accessToken = parsed.tokens?.access_token?.trim()
    const accountId =
      parsed.tokens?.account_id?.trim() ||
      parsed.tokens?.id_token?.['https://api.openai.com/auth']?.chatgpt_account_id?.trim()

    if (!accessToken || !accountId) {
      return null
    }

    const planType =
      parsed.tokens?.id_token?.['https://api.openai.com/auth']?.chatgpt_plan_type?.trim() ||
      undefined

    return {
      accessToken,
      accountId,
      planType,
    }
  } catch {
    return null
  }
}
