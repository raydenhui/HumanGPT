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
 * Whether the live word-chunk answer form applies. Only *streaming* requests
 * whose effective stream mode is word-chunk use it; non-streaming requests and
 * once-mode use the plain submit form.
 */
export function usesLiveWordChunk(req, streamMode) {
  return Boolean(req && req.stream) && streamMode === "word-chunk";
}