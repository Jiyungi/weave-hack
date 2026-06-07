# `src/runtime` - CopilotKit Node/TS runtime

`ag-ui-bridge.ts` implements the AG-UI SSE bridge boundary used by the
CopilotKit runtime. It streams Python LangGraph agent events, records connection
errors when the bridge is unavailable, and clears them after a successful stream
connection.
