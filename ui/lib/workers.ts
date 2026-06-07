/** Mirror of agents/workers.py — keep grant routing in sync for UI register/seed. */

export const RESEARCH_AGENT = "research-agent";
export const OPS_AGENT = "ops-agent";
export const SUPPORT_AGENT = "support-agent";

const RESEARCH = new Set([
  "web_search", "http_fetch", "wikipedia", "news", "doc_search", "doc_index",
  "pdf_read", "dictionary", "synonyms", "geocode", "country_info", "translate",
  "brightdata_scrape", "ip_info",
]);

const OPS = new Set([
  "python", "calculator", "datetime_now", "shell", "read_file", "write_file",
  "list_dir", "apply_patch", "csv_query", "sql_query", "hash_text", "base64_tool",
  "uuid_gen", "password_gen", "json_format", "regex_test", "roman",
  "number_base", "morse", "slugify", "epoch_convert", "unit_convert",
  "currency", "stock_price", "crypto_price",
]);

const SUPPORT = new Set(["weather", "calendar", "forecast", "timezone"]);

export function skillOwners(skill: string): string[] {
  if (RESEARCH.has(skill)) return [RESEARCH_AGENT];
  if (OPS.has(skill)) return [OPS_AGENT];
  if (SUPPORT.has(skill)) return [SUPPORT_AGENT];
  if (skill === "http_fetch") return [RESEARCH_AGENT, OPS_AGENT];
  const n = skill.toLowerCase();
  if (/search|wiki|news|doc|pdf|fetch|dict/.test(n)) return [RESEARCH_AGENT];
  if (/python|calc|shell|file|sql|csv|hash/.test(n)) return [OPS_AGENT];
  if (/weather|calendar|forecast/.test(n)) return [SUPPORT_AGENT];
  return [RESEARCH_AGENT, OPS_AGENT];
}

export function grantsForSkill(skill: string): Record<string, string[]> {
  const out: Record<string, string[]> = {};
  for (const p of skillOwners(skill)) out[p] = [skill];
  return out;
}

/** Policies applied by seed demo (must match ensure_workers_seeded defaults). */
export const SEED_POLICIES: Record<string, string[]> = {
  [SUPPORT_AGENT]: ["weather"],
  [OPS_AGENT]: ["calendar"],
  [RESEARCH_AGENT]: ["web_search"],
};
