import express from "express";
import { createCopilotKitNodeHandler } from "./copilotkit-runtime.js";

/**
 * Standalone Node/Express host for the CopilotKit runtime endpoint.
 *
 * Run it for production (after `npm run build`) or alongside `vite preview`:
 *
 *   node --env-file=../.env --experimental-strip-types src/server/server.ts
 *
 * The dev workflow does NOT need this — `npm run dev` mounts the same runtime as
 * Vite middleware. This server is the simplest "real" host for the built app.
 */
const PORT = Number(process.env.FRONTEND_PORT ?? process.env.PORT ?? 3000);
const INFERENCE_API_URL = process.env.INFERENCE_API_URL ?? "http://127.0.0.1:8000";
const REDIS_URL = process.env.REDIS_URL ?? "redis://127.0.0.1:6379";
const ENDPOINT = "/api/copilotkit";

const app = express();

// CORS-friendly: the browser app and runtime may be served from different ports.
app.use((req, res, next) => {
  res.setHeader("Access-Control-Allow-Origin", req.headers.origin ?? "*");
  res.setHeader("Access-Control-Allow-Methods", "GET,POST,OPTIONS");
  res.setHeader(
    "Access-Control-Allow-Headers",
    "content-type, x-weaveself-adapter-id, x-weaveself-unit-label",
  );
  if (req.method === "OPTIONS") {
    res.statusCode = 204;
    res.end();
    return;
  }
  next();
});

const copilotHandler = createCopilotKitNodeHandler({
  inferenceApiUrl: INFERENCE_API_URL,
  redisUrl: REDIS_URL,
  endpoint: ENDPOINT,
});

app.use(ENDPOINT, (req, res) => {
  void copilotHandler(req, res);
});

// Serve the built static app (vite build output) when present.
app.use(express.static("dist"));

app.listen(PORT, () => {
  // eslint-disable-next-line no-console
  console.log(
    `WeaveSelf CopilotKit runtime on http://127.0.0.1:${PORT}${ENDPOINT} ` +
      `→ Inference_API ${INFERENCE_API_URL}`,
  );
});
