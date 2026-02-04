import langchainClient from '../lib/langchain-client';
import type { AIInvokeResult } from '../lib/langchain-client';
import { z } from 'zod';

const GeneratorOutputSchema = z.object({ text: z.string() });

export type GeneratorOutput = z.infer<typeof GeneratorOutputSchema>;

const DEFAULT_MODEL_NAME = 'gpt-4o-mini';
const DEFAULT_MODEL_PROVIDER = 'openai';
const DEFAULT_SYSTEM_PROMPT = 'You are a generator. Respond with the requested text only.';

const DEFAULT_SETTINGS = {
    temperature: 0.7,
    maxTokens: 1024,
    timeout: 30_000,
};

export type GeneratorSettings = {
    temperature?: number;
    maxTokens?: number;
    timeout?: number;
};

export async function run(
    prompt: string,
    overrides?: GeneratorSettings
): Promise<AIInvokeResult<string>> {
    const settings = {
        ...DEFAULT_SETTINGS,
        ...overrides,
    };
    const result = await langchainClient.invoke(
        DEFAULT_MODEL_NAME,
        DEFAULT_SYSTEM_PROMPT,
        prompt,
        settings,
        GeneratorOutputSchema,
        DEFAULT_MODEL_PROVIDER
    );

    if (result.success) {
        return { success: true, response: result.response.text };
    }
    return { success: false, error: result.error };
}

export { GeneratorOutputSchema };
