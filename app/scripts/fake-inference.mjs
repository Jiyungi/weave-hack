// Minimal fake WeaveSelf Inference_API for local smoke testing.
// POST /generate { prompt, adapter_id, max_new_tokens } -> { text, tokens, latency_ms }
import { createServer } from "node:http";

const PORT = Number(process.env.PORT ?? 8000);

createServer((req, res) => {
  if (req.method === "POST" && req.url === "/generate") {
    let body = "";
    req.on("data", (chunk) => (body += chunk));
    req.on("end", () => {
      let parsed = {};
      try {
        parsed = JSON.parse(body || "{}");
      } catch {
        parsed = {};
      }
      const text = `[adapter=${parsed.adapter_id ?? "base"}] echo: ${parsed.prompt ?? ""}`;
      res.setHeader("content-type", "application/json");
      res.end(JSON.stringify({ text, tokens: 7, latency_ms: 5 }));
    });
    return;
  }
  res.statusCode = 404;
  res.end("not found");
}).listen(PORT, () => {
  console.log(`fake inference on http://127.0.0.1:${PORT}/generate`);
});
