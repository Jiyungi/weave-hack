// End-to-end smoke test: POST a real CopilotKit generateCopilotResponse mutation
// to the running runtime endpoint and confirm the proxied Inference_API text
// (and the forwarded adapter_id) comes back.
const ENDPOINT = process.env.ENDPOINT ?? "http://localhost:3000/api/copilotkit";
const ADAPTER_ID = process.env.ADAPTER_ID ?? "stackexchange_cooking_v3";

const query = `
  mutation generateCopilotResponse($data: GenerateCopilotResponseInput!, $properties: JSONObject) {
    generateCopilotResponse(data: $data, properties: $properties) {
      threadId
      messages {
        __typename
        ... on TextMessageOutput { content role }
      }
    }
  }`;

const variables = {
  data: {
    metadata: { requestType: "Chat" },
    frontend: { actions: [], url: "http://localhost:3000" },
    messages: [
      {
        id: "msg-1",
        createdAt: new Date().toISOString(),
        textMessage: { content: "How do I fix an over-salted soup?", role: "user" },
      },
    ],
  },
  properties: {},
};

const res = await fetch(ENDPOINT, {
  method: "POST",
  headers: {
    "content-type": "application/json",
    accept: "application/json, multipart/mixed",
    "x-weaveself-adapter-id": ADAPTER_ID,
  },
  body: JSON.stringify({ operationName: "generateCopilotResponse", query, variables }),
});

console.log("HTTP", res.status, res.headers.get("content-type"));
const text = await res.text();
console.log("---- response body (truncated) ----");
console.log(text.slice(0, 1200));
console.log("---- checks ----");
console.log("contains echoed prompt:", text.includes("over-salted soup"));
console.log("contains adapter id:", text.includes(ADAPTER_ID));
