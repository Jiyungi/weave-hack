# `src/frontend` - CopilotKit React app surface

Track C currently exposes framework-independent chat and dashboard modules so the
CopilotKit React shell can wire them without changing contract behavior:

- `chat.ts`: Unit selection, `/generate` calls, and message state.
- `eval-results.ts`: adapter-library filtering, heatmap view model, examples, and size chart.
- `mock-eval-results.ts`: fallback demo fixture for task 10.5.
