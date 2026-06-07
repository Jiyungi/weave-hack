import { NextRequest } from "next/server";
import { proxyRequest } from "@/lib/proxy";

const AG_URL = process.env.AG_URL ?? "http://localhost:8200";

async function handler(
  req: NextRequest,
  { params }: { params: { path: string[] } },
) {
  return proxyRequest(req, AG_URL, params.path);
}

export const GET = handler;
export const POST = handler;
