import type { AdapterMeta } from "../contracts/index.js";
import type { EvalResults } from "./eval-results.js";

export const MOCK_ADAPTERS: AdapterMeta[] = [
  {
    adapter_id: "stackexchange_cooking_v3",
    base_model: "Qwen/Qwen2.5-7B-Instruct",
    unit_type: "category",
    unit_label: "cooking",
    train_rows: 812,
    trained_at: "2026-06-06T21:00:00Z",
    day_index: 3,
    size_bytes: 102400,
  },
  {
    adapter_id: "stackexchange_diy_v3",
    base_model: "Qwen/Qwen2.5-7B-Instruct",
    unit_type: "category",
    unit_label: "diy",
    train_rows: 744,
    trained_at: "2026-06-06T21:05:00Z",
    day_index: 3,
    size_bytes: 101888,
  },
  {
    adapter_id: "empty_pending_adapter",
    base_model: "Qwen/Qwen2.5-7B-Instruct",
    unit_type: "category",
    unit_label: "pending",
    train_rows: 0,
    trained_at: "2026-06-06T20:00:00Z",
    day_index: 0,
    size_bytes: 0,
  },
];

export const MOCK_EVAL_RESULTS: EvalResults = {
  perplexity: {
    base: 12.4,
    adapter: 8.1,
    context_memory: 8.3,
  },
  confusion_matrix: {
    labels: ["cooking", "diy", "money"],
    matrix: [
      [18, 1, 0],
      [0, 16, 2],
      [1, 1, 17],
    ],
  },
  size_bytes: {
    nktmirror: 102400,
    lora: 161480704,
  },
  examples: [
    {
      prompt: "How should I rescue an over-salted soup?",
      base: "Dilute the soup with water or broth and adjust seasoning.",
      adapter: "Add unsalted stock, a potato if it fits the dish, and rebalance with acid at the end.",
      reference: "Stretch it with unsalted stock, then finish with lemon once the salt level settles.",
    },
  ],
};
