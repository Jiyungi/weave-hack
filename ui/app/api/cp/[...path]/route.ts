import { NextRequest } from "next/server";
import { proxyRequest } from "@/lib/proxy";

const CP_URL = process.env.CP_URL ?? "http://localhost:8100";

async function handler(
  req: NextRequest,
  { params }: { params: { path: string[] } },
) {
  return proxyRequest(req, CP_URL, params.path);
}

export const GET = handler;
export const POST = handler;
