import type { AdapterMeta } from "../contracts/index.js";
import { validateAdapterMeta } from "../contracts/adapter-meta-validator.js";
import type { RedisClientApi } from "./client-api.js";
import { adapterBlobKey, adapterIndexKey, adapterMetaKey, interactionsKey } from "./keys.js";
import { cosineSimilarity, embedText } from "./embedding.js";

interface RedisCommandClient {
  get(key: string): Promise<string | Buffer | null>;
  set(key: string, value: string | Buffer): Promise<unknown>;
  rPush(key: string, value: string): Promise<unknown>;
  lRange(key: string, start: number, stop: number): Promise<string[]>;
}

interface AdapterIndexRecord {
  adapterId: string;
  unitLabel: string;
  embedding: number[];
}

function bytesToBase64(bytes: Uint8Array): string {
  return Buffer.from(bytes).toString("base64");
}

function base64ToBytes(value: string | Buffer): Uint8Array {
  const text = Buffer.isBuffer(value) ? value.toString("utf8") : value;
  return new Uint8Array(Buffer.from(text, "base64"));
}

export class RedisBackedClient implements RedisClientApi {
  constructor(private readonly client: RedisCommandClient) {}

  async storeAdapter(meta: AdapterMeta, blob: Uint8Array): Promise<void> {
    const validMeta = validateAdapterMeta(meta);
    await this.client.set(adapterBlobKey(validMeta.adapter_id), bytesToBase64(blob));
    await this.client.set(adapterMetaKey(validMeta.adapter_id), JSON.stringify(validMeta));

    const records = await this.readIndex();
    const nextRecords = records.filter((record) => record.adapterId !== validMeta.adapter_id);
    nextRecords.push({
      adapterId: validMeta.adapter_id,
      unitLabel: validMeta.unit_label,
      embedding: embedText(validMeta.unit_label),
    });
    await this.client.set(adapterIndexKey(), JSON.stringify(nextRecords));
  }

  async fetchMeta(adapterId: string): Promise<AdapterMeta> {
    const raw = await this.client.get(adapterMetaKey(adapterId));
    if (raw === null) {
      throw new Error(`Adapter metadata not found: ${adapterId}`);
    }
    const parsed = JSON.parse(Buffer.isBuffer(raw) ? raw.toString("utf8") : raw);
    return validateAdapterMeta(parsed);
  }

  async fetchBlob(adapterId: string): Promise<Uint8Array> {
    const raw = await this.client.get(adapterBlobKey(adapterId));
    if (raw === null) {
      throw new Error(`Adapter blob not found: ${adapterId}`);
    }
    return base64ToBytes(raw);
  }

  async route(queryOrUser: string): Promise<string> {
    const records = await this.readIndex();
    if (records.length === 0) {
      throw new Error(`${adapterIndexKey()} is empty`);
    }

    const queryEmbedding = embedText(queryOrUser);
    let best = records[0];
    let bestScore = cosineSimilarity(queryEmbedding, best.embedding);

    for (const record of records.slice(1)) {
      const score = cosineSimilarity(queryEmbedding, record.embedding);
      if (score > bestScore || (score === bestScore && record.adapterId < best.adapterId)) {
        best = record;
        bestScore = score;
      }
    }

    return best.adapterId;
  }

  async appendInteraction(unitLabel: string, interaction: object): Promise<void> {
    await this.client.rPush(interactionsKey(unitLabel), JSON.stringify(interaction));
  }

  async readInteractions(unitLabel: string): Promise<object[]> {
    const values = await this.client.lRange(interactionsKey(unitLabel), 0, -1);
    return values.map((value) => JSON.parse(value) as object);
  }

  private async readIndex(): Promise<AdapterIndexRecord[]> {
    const raw = await this.client.get(adapterIndexKey());
    if (raw === null) {
      return [];
    }
    const text = Buffer.isBuffer(raw) ? raw.toString("utf8") : raw;
    const parsed = JSON.parse(text) as AdapterIndexRecord[];
    return parsed.map((record) => ({
      adapterId: record.adapterId,
      unitLabel: record.unitLabel,
      embedding: record.embedding,
    }));
  }
}
