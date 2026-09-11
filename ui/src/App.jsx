import React, { useState, useEffect } from "react";
import { api, operatorName, setOperatorName } from "./api.js";
import { QueueView } from "./views/QueueView.jsx";
import { HistoryView } from "./views/HistoryView.jsx";
import { TemplatesView } from "./views/TemplatesView.jsx";
import { SettingsView } from "./views/SettingsView.jsx";

const TABS = [
  { id: "queue", label: "Queue" },
  { id: "history", label: "History" },
  { id: "templates", label: "Templates" },
  { id: "settings", label: "Settings" },
];

export function App() {
  const [tab, setTab] = useState("queue");
  const [opName, setOp] = useState(operatorName());
  const [meta, setMeta] = useState(null);
  const [refreshToken, setRefreshToken] = useState(0);
  const [copied, setCopied] = useState(false);

  useEffect(() => {
    api.meta().then(setMeta).catch(() => setMeta({ api_base_url: window.location.origin + "/v1" }));
  }, []);

  // SSE live updates -> bump the queue refresh
  useEffect(() => {
    if (!window.EventSource) return;
    const es = new EventSource("/events");
    es.addEventListener("update", (ev) => {
      let data;
      try {
        data = JSON.parse(ev.data);
      } catch (e) {
        return;
      }
      if (data.type === "request-changed" || data.type === "templates-changed" || data.type === "models-changed") {
        setRefreshToken((n) => n + 1);
      }
    });
    return () => es.close();
  }, []);

  function copyApiUrl() {
    const url = (meta && meta.api_base_url) || window.location.origin + "/v1";
    const done = () => {
      setCopied(true);
      setTimeout(() => setCopied(false), 1600);
    };
    if (navigator.clipboard) {
      navigator.clipboard.writeText(url).then(done).catch(() => done());
    } else {
      done();
    }
  }

  const pendingCount = meta ? meta.pending_count : 0;

  return (
    <div className="min-h-screen bg-base text-ink flex flex-col">
      <header className="bg-panel border-b border-line sticky top-0 z-20">
        <div className="flex items-center gap-4 px-6 py-3">
          <div className="leading-tight">
            <a href="/" className="text-xl font-bold text-ink hover:underline">HumanGPT</a>
            <p className="text-xs text-dim">human-in-the-loop OpenAI-compatible mock</p>
          </div>
          <nav className="flex gap-1">
            {TABS.map((t) => (
              <button
                key={t.id}
                onClick={() => setTab(t.id)}
                className={`px-3 py-1.5 rounded-lg text-[15px] font-medium transition-colors ${tab === t.id ? "bg-panel2 text-ink" : "text-dim hover:text-ink hover:bg-panel2"}`}
              >
                {t.label}
                {t.id === "queue" && pendingCount > 0 && (
                  <span className="ml-1.5 inline-block min-w-[18px] rounded-full bg-amber-500 text-black text-[11px] px-1 text-center">{pendingCount}</span>
                )}
              </button>
            ))}
          </nav>
          <div className="ml-auto flex items-center gap-3">
            <button
              onClick={copyApiUrl}
              title="Copy the API base URL (SDK base_url)"
              className="text-[13px] px-3 py-1.5 rounded-lg border border-line bg-panel2 text-ink hover:border-accent"
            >
              {copied ? "Copied ✓" : "Copy API URL"}
            </button>
            <label className="flex items-center gap-2 text-[13px] text-dim">
              Operator
              <input
                type="text"
                defaultValue={opName}
                placeholder="your name"
                className="bg-panel2 border border-line rounded-lg px-2.5 py-1.5 text-sm text-ink focus:outline-none focus:border-accent w-32"
                onBlur={(e) => {
                  setOperatorName(e.target.value);
                  setOp(operatorName());
                }}
                onKeyDown={(e) => {
                  if (e.key === "Enter") e.currentTarget.blur();
                }}
              />
            </label>
          </div>
        </div>
      </header>

      <main className="flex-1 px-6 py-5">
        {tab === "queue" && <QueueView refreshToken={refreshToken} />}
        {tab === "history" && <HistoryView refreshToken={refreshToken} />}
        {tab === "templates" && <TemplatesView refreshToken={refreshToken} />}
        {tab === "settings" && <SettingsView refreshToken={refreshToken} />}
      </main>

      <footer className="text-center text-dim text-[13px] py-5 border-t border-line">
        HumanGPT · <a className="text-accent hover:underline" href="/v1/models">/v1/models</a>
      </footer>
    </div>
  );
}