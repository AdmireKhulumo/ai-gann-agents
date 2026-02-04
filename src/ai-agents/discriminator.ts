import langchainClient from '../lib/langchain-client';
import type { AIInvokeResult } from '../lib/langchain-client';
import { z } from 'zod';

const DiscriminatorOutputSchema = z.object({
    score: z.number().min(0).max(10),
});

export type DiscriminatorOutput = z.infer<typeof DiscriminatorOutputSchema>;

const DEFAULT_MODEL_NAME = 'gpt-4o-mini';
const DEFAULT_MODEL_PROVIDER = 'openai';
const DEFAULT_SYSTEM_PROMPT = `You are a discriminator that evaluates how well a generator picked 3 experiences from a CV that best meet given job requirements, and summarized them.

You receive:
1. The job requirements (what the role is looking for)
2. The original instruction to the generator (e.g. pick 3 experiences that best meet the job requirements)
3. The full CV (source text)
4. The generator's output (exactly 3 experiences with 2-line summaries each)
5. Optionally: "Expected experiences" — if provided, these are the experiences deemed most correct for this job; treat them as the ultimate ground truth of what is a good selection.

Scoring rules (0–10):
- The \"expected experiences\" (when provided) are the primary standard for what counts as a good selection. The closer the generator's 3 chosen experiences are to these, the higher the score.
- Do NOT give a score above 6 unless the generator's chosen experiences significantly overlap with, or are very close variants of, the expected experiences.
- 10 = perfect: the generator chose 3 experiences that match the expected experiences extremely well, summaries are accurate to the CV, and the instruction is followed (exactly 3, 2-line summaries).
- 0 = very bad: wrong or irrelevant experiences, inaccurate summaries, or instruction not followed (e.g. not 3 experiences, or not tied to job requirements).

Be consistent and critical. Consider: fit to job requirements, accuracy vs source CV, and, above all, whether the right experiences were selected according to the expected experiences. When expected experiences are provided, treat them as the ultimate source of truth for what is correct. Only output the score. Be strict; scores above 8 should mean the output is really very good and closely aligned with the expected experiences.`;

const DEFAULT_SETTINGS = {
    temperature: 0.1,
    maxTokens: 64,
    timeout: 30_000,
};

/** Context the discriminator needs to evaluate choice of experiences and summary quality. */
export type DiscriminatorContext = {
    /** The instruction given to the generator (e.g. pick 3 experiences that best meet job requirements). */
    generatorPrompt: string;
    /** The source text (CV) that was summarised. */
    cvContent: string;
    /** The job requirements the experiences should match. */
    jobRequirements: string;
    /**
     * Optional: experiences you deem most correct for this job.
     * When provided, the discriminator uses these as the gold standard and scores alignment with them.
     */
    expectedExperiences?: string;
};

/**
 * GANN discriminator: takes the generator's summary output and scores how well it summarised
 * and picked relevant experiences according to the prompt (0–10). Uses the source CV and
 * generator prompt as context.
 */
export async function run(
    generatorOutput: string,
    context: DiscriminatorContext
): Promise<AIInvokeResult<number>> {
    const { generatorPrompt, cvContent, jobRequirements, expectedExperiences } = context;
    const expectedBlock =
        expectedExperiences != null && expectedExperiences.trim() !== ''
            ? `\n\n---\nExpected experiences (deemed most correct for this job; score alignment with these):\n${expectedExperiences}`
            : '';
    const userPrompt = `Job requirements:\n${jobRequirements}\n\n---\nOriginal instruction to the generator:\n${generatorPrompt}\n\n---\nSource text (CV):\n${cvContent}${expectedBlock}\n\n---\nGenerator's output (3 experiences + 2-line summaries):\n${generatorOutput}\n\nScore how well the generator picked 3 experiences that best meet the job requirements and summarised them (0–10).`;

    const result = await langchainClient.invoke(
        DEFAULT_MODEL_NAME,
        DEFAULT_SYSTEM_PROMPT,
        userPrompt,
        DEFAULT_SETTINGS,
        DiscriminatorOutputSchema,
        DEFAULT_MODEL_PROVIDER
    );

    if (result.success) {
        return { success: true, response: result.response.score };
    }
    return { success: false, error: result.error };
}

export { DiscriminatorOutputSchema };
