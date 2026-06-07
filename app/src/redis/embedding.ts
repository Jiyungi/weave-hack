const EMBEDDING_DIMENSIONS = 64;

function tokenWeight(token: string): number {
  let hash = 2166136261;
  for (let index = 0; index < token.length; index += 1) {
    hash ^= token.charCodeAt(index);
    hash = Math.imul(hash, 16777619);
  }
  return hash >>> 0;
}

export function embedText(text: string): number[] {
  const vector = Array.from({ length: EMBEDDING_DIMENSIONS }, () => 0);
  const normalized = text.toLowerCase().replace(/[^a-z0-9]+/g, " ").trim();
  const tokens = normalized.length > 0 ? normalized.split(/\s+/) : [text.toLowerCase()];

  for (const token of tokens) {
    const hash = tokenWeight(token);
    vector[hash % EMBEDDING_DIMENSIONS] += 1;
    vector[(hash >>> 8) % EMBEDDING_DIMENSIONS] += 0.5;
  }

  return vector;
}

export function cosineSimilarity(left: readonly number[], right: readonly number[]): number {
  const length = Math.max(left.length, right.length);
  let dot = 0;
  let leftMagnitude = 0;
  let rightMagnitude = 0;

  for (let index = 0; index < length; index += 1) {
    const leftValue = left[index] ?? 0;
    const rightValue = right[index] ?? 0;
    dot += leftValue * rightValue;
    leftMagnitude += leftValue * leftValue;
    rightMagnitude += rightValue * rightValue;
  }

  if (leftMagnitude === 0 || rightMagnitude === 0) {
    return 0;
  }

  return dot / (Math.sqrt(leftMagnitude) * Math.sqrt(rightMagnitude));
}
