import { describe, it, expect } from "vitest";
import {
  REDIS_KEY_PREFIXES,
  adapterBlobKey,
  adapterMetaKey,
  adapterIndexKey,
  interactionsKey,
} from "./keys.js";

describe("Redis layout key builders (Requirement 3.1–3.4)", () => {
  it("builds the adapter blob key (Req 3.1)", () => {
    expect(adapterBlobKey("abc123")).toBe("adapter:blob:abc123");
  });

  it("builds the adapter metadata key (Req 3.2)", () => {
    expect(adapterMetaKey("abc123")).toBe("adapter:meta:abc123");
  });

  it("returns the fixed vector index key (Req 3.3)", () => {
    expect(adapterIndexKey()).toBe("adapter:index");
  });

  it("builds the interactions key from a unit label (Req 3.4)", () => {
    expect(interactionsKey("alice")).toBe("interactions:alice");
  });

  it("uses the shared key prefixes consistently", () => {
    expect(adapterBlobKey("x").startsWith(REDIS_KEY_PREFIXES.adapterBlob)).toBe(true);
    expect(adapterMetaKey("x").startsWith(REDIS_KEY_PREFIXES.adapterMeta)).toBe(true);
    expect(interactionsKey("x").startsWith(REDIS_KEY_PREFIXES.interactions)).toBe(true);
    expect(adapterIndexKey()).toBe(REDIS_KEY_PREFIXES.adapterIndex);
  });

  it("keeps blob and meta keys distinct for the same adapter id", () => {
    expect(adapterBlobKey("id")).not.toBe(adapterMetaKey("id"));
  });

  it("rejects an empty adapter id", () => {
    expect(() => adapterBlobKey("")).toThrow();
    expect(() => adapterMetaKey("")).toThrow();
  });

  it("rejects an empty unit label", () => {
    expect(() => interactionsKey("")).toThrow();
  });
});
