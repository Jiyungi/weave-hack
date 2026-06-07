/// <reference types="vitest/config" />
import { resolve } from "node:path";
import { defineConfig, loadEnv, type Plugin } from "vite";
import react from "@vitejs/plugin-react";

const RUNTIME_ENDPOINT = "/api/copilotkit";

/**
 * Vite dev-server middleware that hosts the REAL CopilotKit runtime endpoint at
 * /api/copilotkit. This is the simplest host that works for `npm run dev`: a
 * single command serves the React app and the runtime. The runtime module is
 * imported lazily so test/config loads stay light. For production use the
 * standalone Express host (`npm run server`).
 */
function copilotKitRuntimePlugin(inferenceApiUrl: string, redisUrl: string | undefined): Plugin {
  return {
    name: "weaveself-copilotkit-runtime",
    apply: "serve",
    async configureServer(server) {
      const { createCopilotKitNodeHandler } = await import(
        "./src/server/copilotkit-runtime.js"
      );
      const handler = createCopilotKitNodeHandler({
        inferenceApiUrl,
        redisUrl,
        endpoint: RUNTIME_ENDPOINT,
      });

      server.middlewares.use((req, res, next) => {
        if (req.url && req.url.startsWith(RUNTIME_ENDPOINT)) {
          void handler(req, res);
          return;
        }
        next();
      });
    },
  };
}

export default defineConfig(({ mode }) => {
  // Read the shared repo-root .env (one level up from app/).
  const envDir = resolve(__dirname, "..");
  const env = loadEnv(mode, envDir, "");
  const inferenceApiUrl =
    env.INFERENCE_API_URL ?? env.VITE_INFERENCE_API_URL ?? "http://127.0.0.1:8000";
  const redisUrl = env.REDIS_URL ?? "redis://127.0.0.1:6379";

  return {
    envDir,
    plugins: [react(), copilotKitRuntimePlugin(inferenceApiUrl, redisUrl)],
    server: {
      port: Number(env.FRONTEND_PORT ?? 3000),
    },
    test: {
      environment: "node",
      include: ["src/**/*.test.ts"],
    },
  };
});
