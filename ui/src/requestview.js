// Pure normalizers for rendering a request detail. Framework-free + unit-tested.
//
// The OpenAI-compatible surface is permissive (extra fields tolerated, `input`
// may be a string OR an array, `content` may be a string OR parts), so the SPA
// must normalize defensively before rendering.

/** Coerce a value to an array (scalar -> [scalar], missing -> []). */
export function asArray(value) {
  if (value === undefined || value === null) return [];
  return Array.isArray(value) ? value : [value];
}

/**
 * Normalize a Responses-API `input` into renderable item objects.
 * A string input becomes a single user message; non-object array entries are
 * treated as user text.
 */
export function normalizeResponsesInput(input) {
  if (input === undefined || input === null) return [];
  const items = Array.isArray(input) ? input : [input];
  return items.map((item) => {
    if (item && typeof item === "object" && !Array.isArray(item)) return item;
    return { type: "message", role: "user", content: String(item) };
  });
}

/** Chat `messages` as a safe array of objects. */
export function normalizeChatMessages(messages) {
  return asArray(messages).filter((m) => m && typeof m === "object");
}

/** `tools` as a safe array of objects. */
export function normalizeTools(tools) {
  return asArray(tools).filter((t) => t && typeof t === "object");
}

/**
 * Flatten a tool definition into the fields the Tools panel renders.
 *
 * Chat Completions nests the function under `tool.function`
 * (`{type, function: {name, description, parameters}}`), while the Responses
 * API sends the function fields flat (`{type, name, description, parameters}`).
 * Both shapes are accepted; anything missing falls back to safe defaults so a
 * malformed tool can never crash the panel.
 */
export function toolDisplay(tool) {
  if (!tool || typeof tool !== "object") {
    return { name: "function", description: "", parameters: {} };
  }
  const fn = tool.function && typeof tool.function === "object" ? tool.function : tool;
  return {
    name: fn.name || "function",
    description: typeof fn.description === "string" ? fn.description : "",
    parameters: fn.parameters && typeof fn.parameters === "object" ? fn.parameters : {},
  };
}

/** Everything the detail view needs, pre-computed and crash-safe. */
export function normalizeDetail(detail) {
  const body = (detail && detail.body) || {};
  return {
    body,
    parsed: (detail && detail.parsed) || {},
    templates: asArray(detail && detail.templates),
    streamMode: (detail && detail.stream_mode) || "word-chunk",
    inputItems: normalizeResponsesInput(body.input),
    chatMessages: normalizeChatMessages(body.messages),
    tools: normalizeTools(body.tools),
  };
}

/** Whether a request can still be answered (only pending requests can). */
export function isAnswerable(req) {
  return Boolean(req) && req.state === "pending";
}

/**
 * Whether the live word-chunk answer form applies. The operator's effective
 * stream mode governs the answer form: word-chunk gets the live typing form,
 * once gets the plain submit form. The client's `stream` flag is deliberately
 * NOT part of this decision — a non-streaming request can still be answered
 * word-by-word (the deltas are simply not delivered), and the queue's answer
 * area must match whatever mode is configured in Settings.
 */
export function usesLiveWordChunk(streamMode) {
  return streamMode === "word-chunk";
}

/**
 * Append a canned template body to the current draft. A newline is inserted
 * between existing text and the template; an empty draft takes the body as-is.
 */
export function appendTemplate(current, body) {
  const base = current || "";
  const add = body || "";
  if (!base) return add;
  if (!add) return base;
  return `${base}\n${add}`;
}