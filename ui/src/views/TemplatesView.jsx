import React, { useState, useEffect } from "react";
import { api } from "../api.js";

export function TemplatesView({ refreshToken }) {
  const [templates, setTemplates] = useState([]);
  const [name, setName] = useState("");
  const [body, setBody] = useState("");
  const [flash, setFlash] = useState(null);
  const [copied, setCopied] = useState(null);

  const refresh = () => api.templates().then((d) => setTemplates(d.templates || [])).catch(() => {});
  useEffect(() => { refresh(); /* keeps the form state on refresh */ }, [refreshToken]);

  async function save(e) {
    e.preventDefault();
    if (!name.trim()) {
      setFlash({ kind: "err", msg: "Template name is required" });
      return;
    }
    try {
      await api.saveTemplate(name.trim(), body);
      setFlash({ kind: "ok", msg: `Saved "${name.trim()}"` });
      setName("");
      setBody("");
      refresh();
      setTimeout(() => setFlash(null), 2500);
    } catch (err) {
      setFlash({ kind: "err", msg: String(err) });
    }
  }

  async function remove(n) {
    if (!window.confirm(`Delete template "${n}"?`)) return;
    try {
      await api.deleteTemplate(n);
      refresh();
    } catch (err) {
      setFlash({ kind: "err", msg: String(err) });
    }
  }

  function copyBody(t) {
    const done = () => {
      setCopied(t.name);
      setTimeout(() => setCopied(null), 1500);
    };
    if (navigator.clipboard) navigator.clipboard.writeText(t.body).then(done).catch(() => done());
    else done();
  }

  return (
    <section className="max-w-3xl mx-auto space-y-5">
      <h1 className="text-2xl font-bold mb-1">Canned responses</h1>
      <p className="text-dim text-sm">Canned assistant replies. Save them here and insert them into an answer on the Queue page.</p>

      {flash && (
        <div className={`rounded-lg p-3 border ${flash.kind === "ok" ? "border-emerald-500 bg-emerald-500/10 text-emerald-400" : "border-rose-500 bg-rose-500/10 text-rose-400"}`}>
          {flash.msg}
        </div>
      )}

      <form onSubmit={save} className="panel space-y-3">
        <h3 className="text-[15px] font-semibold">New template</h3>
        <input className="input" placeholder="template name (unique)" value={name} onChange={(e) => setName(e.target.value)} />
        <textarea className="input min-h-[90px] font-mono" placeholder="template body" value={body} onChange={(e) => setBody(e.target.value)} />
        <button type="submit" className="btn primary w-full">Save / update template</button>
      </form>

      <div className="panel">
        <h3 className="text-[15px] font-semibold mb-1">All templates ({templates.length})</h3>
        {templates.length === 0 && <p className="text-dim text-sm py-3">No templates yet.</p>}
        <div className="space-y-2.5 mt-1">
          {templates.map((t) => (
            <div key={t.name} className="rounded-lg border border-line bg-panel2 p-3">
              <div className="flex items-center justify-between">
                <span className="font-semibold text-ink">{t.name}</span>
                <span className="flex gap-2">
                  <button className="btn small" onClick={() => copyBody(t)}>{copied === t.name ? "Copied ✓" : "Copy body"}</button>
                  <button className="btn small danger" onClick={() => remove(t.name)}>Delete</button>
                </span>
              </div>
              <pre className="bg-black/40 border border-line rounded-lg p-2 mt-2 text-[12px] whitespace-pre-wrap max-h-36 overflow-auto">{t.body}</pre>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}