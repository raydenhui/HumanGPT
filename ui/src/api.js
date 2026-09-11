// Thin JSON/SDK-agnostic client for the HumanGPT operator API.
// The /v1/* OpenAI surface stays the contract for typed SDK clients; the SPA
// talks to /api/* JSON endpoints that the same FastAPI app serves.

const OP_KEY = "humangpt_operator";

export function operatorName() {
  try {
    return localStorage.getItem(OP_KEY) || "anonymous";
  } catch (e) {
    return "anonymous";
  }
}

export function setOperatorName(name) {
  try {
    localStorage.setItem(OP_KEY, name.trim() || "anonymous");
  } catch (e) {
    /* ignore */
  }
}

async function request(path, opts = {}) {
  const res = await fetch(path, opts);
  if (res.status === 204) return null;
  let body;
  try {
    body = await res.json();
  } catch (e) {
    body = null;
  }
  if (!res.ok) {
    const msg = (body && (body.error || body.detail)) || res.statusText;
    throw new Error(typeof msg === "string" ? msg : JSON.stringify(msg));
  }
  return body;
}

export function getJSON(path) {
  return request(path);
}

export function postForm(path, data) {
  const fd = new FormData();
  for (const [k, v] of Object.entries(data || {})) {
    if (v !== undefined && v !== null) fd.append(k, String(v));
  }
  return request(path, { method: "POST", body: fd });
}

export function postJSON(path, data) {
  return request(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  });
}

export function postEmpty(path) {
  return request(path, { method: "POST" });
}

// ---- typed helpers ------------------------------------------------------

export const api = {
  meta: () => getJSON("/api/meta"),
  queue: () => getJSON("/api/requests"),
  request: (id) => getJSON(`/api/requests/${encodeURIComponent(id)}`),

  answer: (id, payload) =>
    postJSON(`/api/requests/${encodeURIComponent(id)}/answer`, payload),
  streamWord: (id, delta) =>
    postJSON(`/api/requests/${encodeURIComponent(id)}/stream-word`, { delta }),
  returnToPending: (id) => postEmpty(`/api/requests/${encodeURIComponent(id)}/return`),
  timeout: (id) => postEmpty(`/api/requests/${encodeURIComponent(id)}/timeout`),
  discard: (id) => postEmpty(`/api/requests/${encodeURIComponent(id)}/discard`),

  templates: () => getJSON("/api/templates"),
  saveTemplate: (name, body) => postJSON("/api/templates", { name, body }),
  deleteTemplate: (name) => postEmpty(`/api/templates/${encodeURIComponent(name)}/delete`),

  settings: () => getJSON("/api/settings"),
  saveSettings: (data) => postJSON("/api/settings", data),
  saveModel: (data) => postJSON("/api/models", data),
  deleteModel: (id) => postEmpty(`/api/models/${encodeURIComponent(id)}/delete`),
};

export function fmtTs(epoch) {
  if (!epoch) return "—";
  return new Date(epoch * 1000).toLocaleString();
}
export function fmtAge(epoch) {
  const s = Math.max(0, Math.floor(Date.now() / 1000) - epoch);
  if (s < 60) return `${s}s`;
  if (s < 3600) return `${Math.floor(s / 60)}m`;
  return `${Math.floor(s / 3600)}h`;
}

export const STATUS_LABEL = {
  pending: "pending",
  answered: "answered",
  streamed: "streamed",
  timed_out: "timed out",
  interrupted: "interrupted",
  discarded: "discarded",
};

export const STATUS_COLOR = {
  pending: "bg-amber-500/15 text-amber-400 border-amber-500",
  answered: "bg-emerald-500/15 text-emerald-400 border-emerald-500",
  streamed: "bg-emerald-500/15 text-emerald-400 border-emerald-500",
  timed_out: "bg-rose-500/15 text-rose-400 border-rose-500",
  interrupted: "bg-rose-500/15 text-rose-400 border-rose-500",
  discarded: "bg-zinc-600/15 text-zinc-400 border-zinc-600",
};