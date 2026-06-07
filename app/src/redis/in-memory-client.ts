import type { AdapterMeta } from "../contracts/index.js";
import { validateAdapterMeta } from "../contracts/adapter-meta-validator.js";
import type { RedisClientApi } from "./client-api.js";
import { adapterBlobKey, adapterIndexKey, adapterMetaKey, interactionsKey } from "./keys.js";
import { cosineSimilarity, embedText } from "./embedding.js";

interface AdapterIndexRecord {
  adapterId: string;
  unitLabel: string;
  embedding: number[];
}

function cloneBytes(bytes: Uint8Array): Uint8Array {
  return new Uint8Array(bytes);
}

function cloneMeta(meta: AdapterMeta): AdapterMeta {
  return { ...meta };
}

export class InMemoryRedisClient implements RedisClientApi {
  private readonly blobs = new Map<string, Uint8Array>();
  private readonly metas = new Map<string, AdapterMeta>();
  private readonly index = new Map<string, AdapterIndexRecord>();
  private readonly interactions = new Map<string, object[]>();

  async storeAdapter(meta: AdapterMeta, blob: Uint8Array): Promise<void> {
    const validMeta = validateAdapterMeta(meta);
    this.blobs.set(adapterBlobKey(validMeta.adapter_id), cloneBytes(blob));
    this.metas.set(adapterMetaKey(validMeta.adapter_id), cloneMeta(validMeta));
    this.index.set(validMeta.adapter_id, {
      adapterId: validMeta.adapter_id,
      unitLabel: validMeta.unit_label,
      embedding: embedText(validMeta.unit_label),
    });
  }

  async fetchMeta(adapterId: string): Promise<AdapterMeta> {
    const meta = this.metas.get(adapterMetaKey(adapterId));
    if (meta === undefined) {
      throw new Error(`Adapter metadata not found: ${adapterId}`);
    }
    return cloneMeta(meta);
  }

  async fetchBlob(adapterId: string): Promise<Uint8Array> {
    const blob = this.blobs.get(adapterBlobKey(adapterId));
    if (blob === undefined) {
      throw new Error(`Adapter blob not found: ${adapterId}`);
    }
    return cloneBytes(blob);
  }

  async route(queryOrUser: string): Promise<string> {
    if (this.index.size === 0) {
      throw new Error(`${adapterIndexKey()} is empty`);
    }

    const queryEmbedding = embedText(queryOrUser);
    let best: AdapterIndexRecord | undefined;
    let bestScore = Number.NEGATIVE_INFINITY;

    for (const record of this.index.values()) {
      const score = cosineSimilarity(queryEmbedding, record.embedding);
      if (
        score > bestScore ||
        (score === bestScore && best !== undefined && record.adapterId < best.adapterId)
      ) {
        best = record;
        bestScore = score;
      }
    }

    if (best === undefined) {
      throw new Error(`${adapterIndexKey()} is empty`);
    }
    return best.adapterId;
  }

  async appendInteraction(unitLabel: string, interaction: object): Promise<void> {
    const key = interactionsKey(unitLabel);
    const existing = this.interactions.get(key) ?? [];
    existing.push(structuredClone(interaction));
    this.interactions.set(key, existing);
  }

  getInteractions(unitLabel: string): object[] {
    return [...(this.interactions.get(interactionsKey(unitLabel)) ?? [])];
  }

  getIndexedUnitLabels(): string[] {
    return [...this.index.values()].map((record) => record.unitLabel);
  }
}
