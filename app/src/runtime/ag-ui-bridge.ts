export interface AgUiEvent {
  event: string;
  data: string;
}

export interface AgUiBridgeState {
  connected: boolean;
  connectionError: string | null;
}

export type AgUiFetchLike = (input: string, init?: RequestInit) => Promise<Response>;

export class AgUiBridge {
  private state: AgUiBridgeState = {
    connected: false,
    connectionError: null,
  };

  constructor(
    private readonly endpointUrl: string,
    private readonly fetchImpl: AgUiFetchLike = fetch,
  ) {}

  getState(): AgUiBridgeState {
    return { ...this.state };
  }

  async *stream(request: object): AsyncGenerator<AgUiEvent> {
    const response = await this.fetchImpl(this.endpointUrl, {
      method: "POST",
      headers: {
        accept: "text/event-stream",
        "content-type": "application/json",
      },
      body: JSON.stringify(request),
    });

    if (!response.ok || response.body === null) {
      const connectionError = `AG-UI bridge unavailable: HTTP ${response.status}`;
      this.state = {
        connected: false,
        connectionError,
      };
      throw new Error(connectionError);
    }

    this.state = { connected: true, connectionError: null };

    for await (const event of parseSseStream(response.body)) {
      yield event;
    }
  }
}

export async function* parseSseStream(body: ReadableStream<Uint8Array>): AsyncGenerator<AgUiEvent> {
  const reader = body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  try {
    while (true) {
      const chunk = await reader.read();
      if (chunk.done) {
        break;
      }
      buffer += decoder.decode(chunk.value, { stream: true });

      let separatorIndex = buffer.indexOf("\n\n");
      while (separatorIndex >= 0) {
        const rawEvent = buffer.slice(0, separatorIndex);
        buffer = buffer.slice(separatorIndex + 2);
        const parsed = parseSseEvent(rawEvent);
        if (parsed !== null) {
          yield parsed;
        }
        separatorIndex = buffer.indexOf("\n\n");
      }
    }

    buffer += decoder.decode();
    if (buffer.trim().length > 0) {
      const parsed = parseSseEvent(buffer);
      if (parsed !== null) {
        yield parsed;
      }
    }
  } finally {
    reader.releaseLock();
  }
}

function parseSseEvent(rawEvent: string): AgUiEvent | null {
  const lines = rawEvent.split(/\r?\n/);
  let event = "message";
  const data: string[] = [];

  for (const line of lines) {
    if (line.startsWith("event:")) {
      event = line.slice("event:".length).trim();
    }
    if (line.startsWith("data:")) {
      data.push(line.slice("data:".length).trimStart());
    }
  }

  if (data.length === 0) {
    return null;
  }

  return { event, data: data.join("\n") };
}
