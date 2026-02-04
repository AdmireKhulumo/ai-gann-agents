import 'dotenv/config';
import * as fs from 'fs';
import * as path from 'path';
import { run as runGenerator, type GeneratorSettings } from './ai-agents/generator';
import { run as runDiscriminator } from './ai-agents/discriminator';
import { suggestNextPrompt, type GeneratorConfigSnapshot } from './ai-agents/configurator';

const CV_PATH = path.join(process.cwd(), 'cv.txt');
const JOB_REQUIREMENTS_PATH = path.join(process.cwd(), 'job-requirements.txt');
const EXPECTED_EXPERIENCES_PATH = path.join(process.cwd(), 'expected-experiences.txt');
const NUM_ROUNDS = 5;

const INITIAL_GENERATOR_PROMPT =
  'Tell me a joke';

/** Fixed generator settings; temperature stays consistent. */
const GENERATOR_CONFIG: GeneratorConfigSnapshot = {
  temperature: 0.7,
  maxTokens: 1024,
  timeout: 30_000,
};

async function main(): Promise<void> {
  console.log('AI LinkedIn GANN\n');

  const cvContent = fs.readFileSync(CV_PATH, 'utf-8');
  const jobRequirements = fs.readFileSync(JOB_REQUIREMENTS_PATH, 'utf-8');
  const expectedExperiences = fs.readFileSync(EXPECTED_EXPERIENCES_PATH, 'utf-8');

  let currentPrompt = INITIAL_GENERATOR_PROMPT;
  const promptWithContext = (instruction: string) =>
    `Job requirements:\n${jobRequirements}\n\n---\nCV:\n${cvContent}\n\n---\n${instruction}`;

  const discriminatorContext = {
    generatorPrompt: currentPrompt,
    cvContent,
    jobRequirements,
    expectedExperiences,
  };

  for (let round = 1; round <= NUM_ROUNDS; round++) {
    console.log(`--- Round ${round} ---`);
    console.log('Generator prompt:', currentPrompt);

    const fullGeneratorPrompt = promptWithContext(currentPrompt);
    const genResult = await runGenerator(fullGeneratorPrompt, GENERATOR_CONFIG as GeneratorSettings);
    if (!genResult.success) {
      console.error('Generator error:', genResult.error);
      return;
    }

    const text = genResult.response;
    console.log('\n  [Generator]\n  ', text);

    const discResult = await runDiscriminator(text, { ...discriminatorContext, generatorPrompt: currentPrompt });
    if (!discResult.success) {
      console.error('Discriminator error:', discResult.error);
      return;
    }

    const score = discResult.response;
    console.log('\n  [Discriminator]\n  ', 'Score:', score);

    if (score >= 9) {
      console.log('\nTarget score (9+) reached. Stopping.');
      break;
    }

    const configResult = await suggestNextPrompt({
      generatorPrompt: currentPrompt,
      generatorConfig: GENERATOR_CONFIG,
      score,
      cvContent,
      jobRequirements,
    });

    if (!configResult.success) {
      console.error('Configurator error:', configResult.error);
      return;
    }

    const suggestedPrompt = configResult.response;
    console.log('\n  [Configurator]\n  ', 'Suggested prompt:', suggestedPrompt.slice(0, 80) + (suggestedPrompt.length > 80 ? '...' : ''));
    currentPrompt = suggestedPrompt;
    discriminatorContext.generatorPrompt = currentPrompt;
    console.log('');
  }

  console.log('Done.');
}

main();
