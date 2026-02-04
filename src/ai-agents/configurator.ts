import langchainClient from '../lib/langchain-client';
import type { AIInvokeResult } from '../lib/langchain-client';
import { z } from 'zod';

const ConfiguratorOutputSchema = z.object({
    suggestedPrompt: z.string(),
});

export type ConfiguratorOutput = z.infer<typeof ConfiguratorOutputSchema>;

/** Generator config snapshot (what was used for a run). Temperature stays consistent; configurator only adjusts the prompt. */
export type GeneratorConfigSnapshot = {
    temperature: number;
    maxTokens: number;
    timeout?: number;
};

/** One entry in configurator history: a past run and its outcome. */
export type ConfiguratorHistoryEntry = {
    generatorPrompt: string;
    generatorConfig: GeneratorConfigSnapshot;
    score: number;
    /** Prompt we suggested for the *next* run after this (if any). */
    suggestedNextPrompt?: string;
};

const DEFAULT_MODEL_NAME = 'gpt-4o-mini';
const DEFAULT_MODEL_PROVIDER = 'openai';
const DEFAULT_SYSTEM_PROMPT = `You are a configurator for a GAN-like setup where the generator picks 3 experiences from a CV that best meet job requirements and summarizes them; the discriminator scores how well the choice and summaries match the job (0–10; 10 = perfect, 0 = very bad).

- The generator receives the job requirements, the CV, and an instruction (prompt). Temperature is fixed; only the instruction changes.
- Your job: given the generator's instruction that was used, the score from the discriminator, the job requirements, and CV context, suggest a NEW instruction (prompt) for the generator to use on the NEXT run so that the next score is likely to be higher.

Consider:
- Rephrasing the instruction to stress job requirements (e.g. platform engineering, DevOps, observability, CI/CD, scale).
- Asking for different formatting or emphasis (e.g. "highlight technologies from the job description", "focus on developer experience and reliability").
- Use history of past runs: if a certain phrasing led to a higher score, steer toward that; if the discriminator penalized missing criteria, make the next prompt mention those criteria explicitly.
- Output only the new instruction text, ready to be used as the generator prompt. Keep it concise (one or two sentences). Do not include the job requirements or CV in your output, only have the instruction.`;

const DEFAULT_SETTINGS = {
    temperature: 0.2,
    maxTokens: 128,
    timeout: 30_000,
};

/** In-memory history of generator runs and scores for the configurator. */
let history: ConfiguratorHistoryEntry[] = [];

/**
 * Returns the current configurator history (read-only snapshot).
 */
export function getHistory(): readonly ConfiguratorHistoryEntry[] {
    return history;
}

/**
 * Clears the in-memory history (e.g. for tests or a new session).
 */
export function clearHistory(): void {
    history = [];
}

/**
 * Configurator: receives generator prompt + discriminator score, optional CV and job context,
 * uses in-memory history, and suggests the next generator prompt to maximise score. Temperature stays consistent.
 */
export async function suggestNextPrompt(params: {
    generatorPrompt: string;
    generatorConfig: GeneratorConfigSnapshot;
    score: number;
    /** Optional: source CV content for context. */
    cvContent?: string;
    /** Optional: job requirements for context. */
    jobRequirements?: string;
}): Promise<AIInvokeResult<string>> {
    const { generatorPrompt, generatorConfig, score, cvContent, jobRequirements } = params;

    const entry: ConfiguratorHistoryEntry = {
        generatorPrompt,
        generatorConfig: { ...generatorConfig },
        score,
    };
    history.push(entry);

    const historyBlock =
        history.length === 0
            ? 'No previous runs yet.'
            : history
                .map((h, i) => {
                    const part = `Run ${i + 1}: prompt="${h.generatorPrompt.slice(0, 120)}${h.generatorPrompt.length > 120 ? '...' : ''}", score=${h.score}`;
                    if (h.suggestedNextPrompt != null) {
                        return part + ` (suggested next prompt: "${h.suggestedNextPrompt.slice(0, 60)}...")`;
                    }
                    return part;
                })
                .join('\n');

    const cvBlock =
        cvContent != null
            ? `\n\nSource CV context (excerpt):\n${cvContent.slice(0, 1500)}${cvContent.length > 1500 ? '...' : ''}`
            : '';
    const jobBlock =
        jobRequirements != null
            ? `\n\nJob requirements (excerpt):\n${jobRequirements.slice(0, 1500)}${jobRequirements.length > 1500 ? '...' : ''}`
            : '';
    const userPrompt = `History of runs (most recent last):\n${historyBlock}\n\nCurrent run we just got the score for: prompt="${generatorPrompt}", score=${score}.${jobBlock}${cvBlock}\n\nSuggest the next generator prompt (instruction only, no CV or job text) to maximise the score. Reply with only the new instruction text.`;

    const result = await langchainClient.invoke(
        DEFAULT_MODEL_NAME,
        DEFAULT_SYSTEM_PROMPT,
        userPrompt,
        DEFAULT_SETTINGS,
        ConfiguratorOutputSchema,
        DEFAULT_MODEL_PROVIDER
    );

    if (result.success) {
        const suggested = result.response.suggestedPrompt.trim();
        if (history.length > 0 && suggested) {
            history[history.length - 1].suggestedNextPrompt = suggested;
        }
        return { success: true, response: suggested || generatorPrompt };
    }
    return { success: false, error: result.error };
}

export { ConfiguratorOutputSchema };
