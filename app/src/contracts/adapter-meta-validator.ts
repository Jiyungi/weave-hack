import {
  ADAPTER_META_REQUIRED_FIELDS,
  type AdapterMeta,
  type AdapterMetaField,
} from "./adapter-meta.js";

export class MissingAdapterMetaFieldError extends Error {
  readonly field: AdapterMetaField;

  constructor(field: AdapterMetaField) {
    super(`Adapter metadata is missing required field: ${field}`);
    this.name = "MissingAdapterMetaFieldError";
    this.field = field;
  }
}

export function validateAdapterMeta(value: unknown): AdapterMeta {
  if (value === null || typeof value !== "object") {
    throw new MissingAdapterMetaFieldError("adapter_id");
  }

  const record = value as Record<string, unknown>;
  for (const field of ADAPTER_META_REQUIRED_FIELDS) {
    if (!(field in record)) {
      throw new MissingAdapterMetaFieldError(field);
    }
  }

  if (record.unit_type !== "category" && record.unit_type !== "user") {
    throw new Error("unit_type must be either category or user");
  }

  for (const field of ["adapter_id", "base_model", "unit_label", "trained_at"] as const) {
    if (typeof record[field] !== "string" || record[field].length === 0) {
      throw new Error(`${field} must be a non-empty string`);
    }
  }

  for (const field of ["train_rows", "day_index", "size_bytes"] as const) {
    if (typeof record[field] !== "number" || !Number.isInteger(record[field]) || record[field] < 0) {
      throw new Error(`${field} must be a non-negative integer`);
    }
  }

  const adapterId = record.adapter_id as string;
  const baseModel = record.base_model as string;
  const unitType = record.unit_type;
  const unitLabel = record.unit_label as string;
  const trainRows = record.train_rows as number;
  const trainedAt = record.trained_at as string;
  const dayIndex = record.day_index as number;
  const sizeBytes = record.size_bytes as number;

  return {
    adapter_id: adapterId,
    base_model: baseModel,
    unit_type: unitType,
    unit_label: unitLabel,
    train_rows: trainRows,
    trained_at: trainedAt,
    day_index: dayIndex,
    size_bytes: sizeBytes,
  };
}
