import { createAgent, HumanMessage, initChatModel } from 'langchain';
import { z, ZodObject } from 'zod';

type AIInvokeSuccess<T> = {
    success: true;
    response: T;
};

type AIInvokeFailure = {
    success: false;
    error: Error;
};

export type AIInvokeResult<T> = AIInvokeSuccess<T> | AIInvokeFailure;

type ModelSettings = {
    temperature: number;
    maxTokens: number;
    timeout: number;
};

const invoke = async <T extends ZodObject<any>>(
    modelName: string,
    systemPrompt: string,
    userPrompt: string,
    settings: ModelSettings,
    responseFormat: T,
    modelProvider?: string
): Promise<AIInvokeResult<z.infer<T>>> => {
    try {
        console.debug({ modelName, modelProvider, temperature: settings.temperature }, 'Invoking AI model');

        const model = await initChatModel(modelName, {
            modelProvider,
            temperature: settings.temperature,
            maxTokens: settings.maxTokens,
            timeout: settings.timeout,
        });

        const agent = createAgent({
            model,
            systemPrompt,
            responseFormat,
        });

        const response = await agent.invoke({
            messages: [new HumanMessage(userPrompt)],
        });

        const structuredResponse = response.structuredResponse;
        if (!structuredResponse) {
            console.error('AI invocation returned no structured response');
            return { success: false, error: Error('No structured response from AI invocation.') };
        }

        console.debug('AI invocation successful');
        return {
            success: true,
            response: structuredResponse as z.infer<T>,
        };
    } catch (error) {
        console.error({ error }, 'AI invocation threw an error');
        return {
            success: false,
            error: error as Error,
        };
    }
};

export default { invoke };
