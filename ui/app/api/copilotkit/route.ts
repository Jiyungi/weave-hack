import {
  CopilotRuntime,
  OpenAIAdapter,
  copilotRuntimeNextJSAppRouterEndpoint,
} from "@copilotkit/runtime";
import OpenAI from "openai";
import { NextRequest } from "next/server";

const openai = new OpenAI({
  baseURL: process.env.OPENMIRROR_BRAIN_BASE_URL ?? "http://localhost:8001/v1",
  apiKey: process.env.OPENMIRROR_BRAIN_API_KEY ?? "sk-no-key",
});

const model =
  process.env.OPENMIRROR_BRAIN_MODEL ?? "Qwen/Qwen2.5-14B-Instruct";

const serviceAdapter = new OpenAIAdapter({ openai, model });
const runtime = new CopilotRuntime();

const { handleRequest } = copilotRuntimeNextJSAppRouterEndpoint({
  runtime,
  serviceAdapter,
  endpoint: "/api/copilotkit",
});

export const POST = async (req: NextRequest) => handleRequest(req);
