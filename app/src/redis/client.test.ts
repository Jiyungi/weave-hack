import { describe, expect, it } from "vitest";
import fc from "fast-check";
import type { AdapterMeta } from "../contracts/index.js";
import { InMemoryRedisClient, cosineSimilarity, embedText } from "./index.js";

function adapterMeta(adapterId: string, unitLabel: string, sizeBytes = 102400): AdapterMeta {
  return {
    adapter_id: adapterId,
    base_model: "Qwen/Qwen2.5-7B-Instruct",
    unit_type: "category",
    unit_label: unitLabel,
    train_rows: 10,
    trained_at: "2026-06-06T21:00:00Z",
    day_index: 3,
    size_bytes: sizeBytes,
  };
}

describe("Redis layer and client API (Requirement 19)", () => {
  it("Feature: weaveself, Property 25: Redis blob round-trip preserves bytes", async () => {
    await fc.assert(
      fc.asyncProperty(fc.uint8Array({ minLength: 0, maxLength: 4096 }), async (blob) => {
        const client = new InMemoryRedisClient();
        await client.storeAdapter(adapterMeta("adapter_blob", "cooking"), blob);
        const fetched = await client.fetchBlob("adapter_blob");
        expect([...fetched]).toEqual([...blob]);
        if (fetched.length > 0) {
          fetched[0] = fetched[0] ^ 1;
          expect([...(await client.fetchBlob("adapter_blob"))]).toEqual([...blob]);
        }
      }),
      { numRuns: 100 },
    );
  });

  it("round-trips the standalone demo 100 KB blob byte-identically", async () => {
    const client = new InMemoryRedisClient();
    const blob = new Uint8Array(102400);
    for (let index = 0; index < blob.length; index += 1) {
      blob[index] = index % 251;
    }

    await client.storeAdapter(adapterMeta("adapter_100kb", "cooking"), blob);
    expect([...(await client.fetchBlob("adapter_100kb"))]).toEqual([...blob]);
  });

  it("Feature: weaveself, Property 26: Metadata is retrievable independently of the blob", async () => {
    await fc.assert(
      fc.asyncProperty(fc.integer({ min: 0, max: 200000 }), async (sizeBytes) => {
        const client = new InMemoryRedisClient();
        const meta = adapterMeta("adapter_meta", "diy", sizeBytes);
        await client.storeAdapter(meta, new Uint8Array([1, 2, 3]));

        const fetched = await client.fetchMeta(meta.adapter_id);
        expect(fetched).toEqual(meta);
        fetched.unit_label = "mutated";
        expect((await client.fetchMeta(meta.adapter_id)).unit_label).toBe("diy");
      }),
      { numRuns: 100 },
    );
  });

  it("Feature: weaveself, Property 27: Route returns the top-1 adapter by vector similarity", async () => {
    await fc.assert(
      fc.asyncProperty(
        fc.uniqueArray(
          fc.string({ minLength: 1, maxLength: 16 }).filter((value) => value.trim().length > 0),
          { minLength: 2, maxLength: 8 },
        ),
        fc.integer({ min: 0, max: 7 }),
        async (labels, selectedIndex) => {
          const client = new InMemoryRedisClient();
          const selectedLabel = labels[selectedIndex % labels.length];

          for (const [index, label] of labels.entries()) {
            await client.storeAdapter(adapterMeta(`adapter_${index}`, label), new Uint8Array([index]));
          }

          const expected = labels
            .map((label, index) => ({
              adapterId: `adapter_${index}`,
              score: cosineSimilarity(embedText(selectedLabel), embedText(label)),
            }))
            .sort((left, right) => right.score - left.score || left.adapterId.localeCompare(right.adapterId))[0]
            .adapterId;

          await expect(client.route(selectedLabel)).resolves.toBe(expected);
        },
      ),
      { numRuns: 100 },
    );
  });

  it("Feature: weaveself, Property 28: Interactions append under the unit key", async () => {
    await fc.assert(
      fc.asyncProperty(fc.array(fc.string(), { minLength: 0, maxLength: 20 }), async (messages) => {
        const client = new InMemoryRedisClient();
        for (const message of messages) {
          await client.appendInteraction("cooking", { message });
        }
        expect(client.getInteractions("cooking")).toEqual(messages.map((message) => ({ message })));
        expect(client.getInteractions("diy")).toEqual([]);
      }),
      { numRuns: 100 },
    );
  });
});
