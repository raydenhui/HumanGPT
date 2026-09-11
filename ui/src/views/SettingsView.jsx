import React, { useState, useEffect } from "react";
import { api } from "../api.js";

export function SettingsView({ refreshToken }) {
  const [data, setData] = useState(null);
  const [streamMode, setStreamMode] = useState("word-chunk");
  const [chunkDelay, setChunkDelay] = useState(0);
  const [modelId, setModelId] = useState("");
  const [modelDesc, setModelDesc] = useState("");
  const [modelPricing, setModelPricing] = useState("");
  const [flash, setFlash] = useState(null);
  const [copied, setCopied] = useState(false);
  const [initialized, setInitialized] = useState(false);

  useEffect(() => {
    // Initial load seeds the form; later refreshes only update the model list
    // / API URL so in-progress form edits aren't clobbered.
    api.settings().then((d) => {
      setData(d);
      if (!initialized) {
        setStreamMode(d.stream_mode);
        setChunkDelay(d.stream_chunk_delay_ms);
        setInitialized(true);
      }
    }).catch((e) => setFlash({ kind: "err", msg: String(e) }));
  }, [refreshToken]);

  function flashMsg(msg) {
    setFlash({ kind: "ok", msg });
    setTimeout(() => setFlash(null), 2500);
  }

  async function saveSettings(e) {
    e.preventDefault();
    try {
      await api.saveSettings({ stream_mode: streamMode, stream_chunk_delay_ms: chunkDelay });
      flashMsg("Stream settings saved");
    } catch (err) {
      setFlash({ kind: "err", msg: String(err) });
    }
  }

  async function saveModel(e) {
    e.preventDefault();
    try {
      await api.saveModel({ model_id: modelId, description: modelDesc, pricing: modelPricing });
      setModelId("");
      setModelDesc("");
      setModelPricing("");
      setData(null);
      await api.settings().then(setData).catch(() => {});
      flashMsg(`Model "${modelId}" saved`);
    } catch (err) {
      setFlash({ kind: "err", msg: String(err) });
    }
  }

  async function removeModel(id) {
    if (!window.confirm(`Delete model metadata for "${id}"?`)) return;
    try {
      await api.deleteModel(id);
      await api.settings().then(setData).catch(() => {});
      flashMsg(`Removed "${id}"`);
    } catch (err) {
      setFlash({ kind: "err", msg: String(err) });
    }
  }

  function copyUrl() {
    const url = (data && data.api_base_url) || window.location.origin + "/v1";
    const done = () => {
      setCopied(true);
      setTimeout(() => setCopied(false), 1600);
    };
    if (navigator.clipboard) navigator.clipboard.writeText(url).then(done).catch(() => done());
    else done();
  }

  return (
    <section className="max-w-3xl mx-auto space-y-5">
      <h1 className="text-2xl font-bold mb-1">Settings</h1>
      {flash && (
        <div className={`rounded-lg p-3 border ${flash.kind === "ok" ? "border-emerald-500 bg-emerald-500/10 text-emerald-400" : "border-rose-500 bg-rose-500/10 text-rose-400"}`}>
          {flash.msg}
        </div>
      )}

      {/* API base URL */}
      <div className="panel">
        <h3 className="text-[15px] font-semibold mb-1.5">API base URL</h3>
        <p className="text-dim text-[13px] mb-2">Set this as your SDK <code className="text-[12px]">base_url</code>.</p>
        <div className="flex items-center gap-2.5">
          <code className="flex-1 bg-panel2 border border-line rounded-lg px-3 py-2 text-[14px] user-select-all">
            {(data && data.api_base_url) || "…"}
          </code>
          <button className="btn" onClick={copyUrl}>{copied ? "Copied ✓" : "Copy"}</button>
        </div>
        <p className="text-dim text-[13px] mt-2">
          API key mode: <strong className="text-ink">{data ? (data.configured_key ? "fixed" : "any") : "…"}</strong> —
          {data?.configured_key ? "the SDK must send the configured key." : "any Authorization: Bearer value is accepted."}
        </p>
      </div>

      {/* stream defaults */}
      <form onSubmit={saveSettings} className="panel space-y-3">
        <h3 className="text-[15px] font-semibold">Stream defaults</h3>
        <label className="flex items-center gap-2.5 text-[14px]">
          <span className="text-dim w-44 text-right">Stream mode</span>
          <select className="input flex-1" value={streamMode} onChange={(e) => setStreamMode(e.target.value)}>
            <option value="word-chunk">word-chunk (live typing, locked words)</option>
            <option value="once">once (submit, whole message)</option>
          </select>
        </label>
        <label className="flex items-center gap-2.5 text-[14px]">
          <span className="text-dim w-44 text-right">Chunk delay (ms)</span>
          <input type="number" min="0" step="1" className="input flex-1" value={chunkDelay} onChange={(e) => setChunkDelay(Number(e.target.value))} />
        </label>
        <button type="submit" className="btn primary">Save stream settings</button>
      </form>

      {/* model catalog */}
      <div className="panel">
        <h3 className="text-[15px] font-semibold mb-1.5">Model list</h3>
        <p className="text-dim text-[13px] mb-2">
          Model IDs come from <code className="text-[12px]">models.json</code>. Description and pricing set here are
          stored in the database and served by <code className="text-[12px]">/v1/models</code>.
        </p>

        <details className="mb-3">
          <summary className="cursor-pointer text-dim text-[14px]">Add / edit a model's details</summary>
          <form onSubmit={saveModel} className="mt-2 space-y-2">
            <input className="input" placeholder="model id (e.g. human-gpt)" value={modelId} onChange={(e) => setModelId(e.target.value)} />
            <input className="input" placeholder="description" value={modelDesc} onChange={(e) => setModelDesc(e.target.value)} />
            <input className="input" placeholder='pricing JSON (e.g. {"input":1,"output":2,"unit":"USD/1K tokens"})' value={modelPricing} onChange={(e) => setModelPricing(e.target.value)} />
            <button type="submit" className="btn primary">Save model</button>
          </form>
        </details>

        <table className="w-full text-[14px]">
          <thead>
            <tr className="text-left text-dim text-[13px] border-b border-line">
              <th className="py-1.5">ID</th><th>Description</th><th>Pricing</th><th></th>
            </tr>
          </thead>
          <tbody>
            {data?.models?.map((m) => (
              <tr key={m.id} className="border-b border-line">
                <td className="py-2 font-mono text-[13px]">{m.id}</td>
                <td className="py-2 text-dim">{m.description || "—"}</td>
                <td className="py-2 text-dim text-[13px]">{m.pricing ? JSON.stringify(m.pricing) : "—"}</td>
                <td className="py-2">
                  <button className="btn small danger" onClick={() => removeModel(m.id)}>Remove</button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}