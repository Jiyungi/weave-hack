import { NextRequest, NextResponse } from "next/server";

/** Forward a request to a backend base URL, preserving method/body/query. */
export async function proxyRequest(
  req: NextRequest,
  baseUrl: string,
  pathSegments: string[],
): Promise<NextResponse> {
  const path = "/" + pathSegments.join("/");
  const url = new URL(path, baseUrl);
  req.nextUrl.searchParams.forEach((v, k) => url.searchParams.set(k, v));

  const headers = new Headers();
  const ct = req.headers.get("content-type");
  if (ct) headers.set("content-type", ct);

  const init: RequestInit = {
    method: req.method,
    headers,
  };
  if (req.method !== "GET" && req.method !== "HEAD") {
    init.body = await req.text();
  }

  try {
    const upstream = await fetch(url.toString(), init);
    const body = await upstream.text();
    return new NextResponse(body, {
      status: upstream.status,
      headers: {
        "content-type": upstream.headers.get("content-type") || "application/json",
      },
    });
  } catch (e) {
    const msg = e instanceof Error ? e.message : String(e);
    return NextResponse.json(
      { detail: `proxy to ${baseUrl} failed: ${msg}` },
      { status: 502 },
    );
  }
}
