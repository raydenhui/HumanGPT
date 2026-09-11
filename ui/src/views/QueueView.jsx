import React, { useState, useEffect, useRef } from "react";
import { operatorName } from "../api.js";
import { RequestDetail } from "./RequestDetail.jsx";

const STATUS = {
  pending: { label: "pending", cls: "bg-amber-500/15 text-amber-400 border-amber-500" },
  answered: { label: "answered", cls: "bg-emerald-500/15 text-emerald-400 border-emerald-500" },
  streamed: { label: "streamed", cls: "bg-emerald-500/15 text-emerald-400 border-emerald-500" },
  timed_out: { label: "timed out", cls: "bg-rose-500/15 text-rose-400 border-rose-500" },
  interrupted: { label: "interrupted", cls: "bg-rose-500/15 text-rose-400 border-rose-500" },
  discarded: { label: "discarded", cls: "bg-zinc-600/15 text-zinc-400 border-zinc-600" },
};

function Badge({ state }) {
  const s = STATUS[state] || STATUS.discarded;
  return (
    <span className={`inline-block px-2 py-0.5 rounded-full text-[12px] border ${s.cls}`}>{s.label}</span>
  );
}

export function QueueView({ refreshToken }) {
  const [queue, setQueue] = useState([]);
  const [selectedId, setSelectedId] = useState(null);
  const [detail, setDetail] = useState(null);
  const [detailLoading, setDetailLoading] = useState(false);
  const [error, setError] = useState(null);
  const selectedRef = useRef(null);

  async function loadQueue() {
    try {
      const resp = await fetch("/api/requests");
      const data = await resp.json();
      setQueue(data.pending || data);
    } catch (e) {
      /* transient */
    }
  }

  useEffect(() => {
    loadQueue();
    const iv = setInterval(loadQueue, 2000);
    return () => clearInterval(iv);
  }, [refreshToken]);

  function select(id) {
    selectedRef.current = id;
    setSelectedId(id);
  }

  function fetchDetail(id, { claim = false } = {}) {
    setDetailLoading(true);
    setError(null);
    fetch(`/api/requests/${encodeURIComponent(id)}`)
      .then((r) => r.json())
      .then((d) => {
        // Ignore a stale response if the operator already selected another item.
        if (selectedRef.current !== id) return;
        setDetail(d);
        setDetailLoading(false);
        // soft-claim only when the operator *opens* a fresh pending request
        if (
          claim &&
          d.request &&
          d.request.state === "pending" &&
          !d.request.claimed_by
        ) {
          fetch(`/api/requests/${encodeURIComponent(id)}/claim`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ operator: operatorName() }),
          })
            .then(() => loadQueue())
            .catch(() => {});
        }
      })
      .catch((e) => {
        if (selectedRef.current !== id) return;
        setError(String(e));
        setDetailLoading(false);
      });
  }

  // Selection changes -> load the detail (one click shows everything).
  useEffect(() => {
    if (!selectedId) {
      setDetail(null);
      return;
    }
    fetchDetail(selectedId, { claim: true });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedId]);

  // Background refreshes must NOT drop the open selection; just refresh the
  // detail in place (without re-claiming).
  useEffect(() => {
    if (selectedId) fetchDetail(selectedId, { claim: false });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [refreshToken]);

  return (
    <div className="grid grid-cols-[minmax(0,26rem)_1fr] gap-6 items-start h-full">
      {/* ---------------- queue list (left) ---------------- */}
      <section className="bg-panel border border-line rounded-xl p-4">
        <h2 className="text-[17px] font-semibold mb-3">
          Queue <span className="text-dim text-[13px]">({queue.length})</span>
        </h2>
        {queue.length === 0 ? (
          <p className="text-dim text-sm py-6 text-center">Queue is empty — the human has answered everything.</p>
        ) : (
          <ul className="space-y-1.5">
            {queue.map((r) => (
              <li key={r.id}>
                <button
                  onClick={() => select(r.id)}
                  className={`w-full text-left rounded-lg p-2.5 transition-colors ${selectedId === r.id ? "bg-accent/15 border border-accent/40" : "bg-panel2 border border-line hover:border-accent/40"}`}
                >
                  <div className="flex items-center gap-2">
                    <Badge state={r.state} />
                    <span className="font-mono text-[12px] text-dim truncate">{r.id}</span>
                    {r.stream && <span className="text-[11px] text-amber-400 px-1 rounded bg-panel2">SSE</span>}
                    {r.has_images && <span className="text-[11px] text-amber-400 px-1 rounded bg-panel2">img</span>}
                  </div>
                  <div className="mt-1 text-sm text-ink line-clamp-2">{r.snippet || "—"}</div>
                  <div className="mt-0.5 flex justify-between text-[12px] text-dim">
                    <span>{r.model}{r.claimed_by ? ` · 👤 ${r.claimed_by}` : ""}</span>
                    <span>{age(r.created_at)}</span>
                  </div>
                </button>
              </li>
            ))}
          </ul>
        )}
      </section>

      {/* ---------------- detail pane (right) ---------------- */}
      <section className="min-w-0">
        {error && <div className="bg-rose-500/10 border border-rose-500 rounded-lg p-3 mb-4 text-rose-400">{error}</div>}
        {detailLoading ? (
          <div className="bg-panel border border-line rounded-xl p-10 text-dim">Loading request…</div>
        ) : detail ? (
          <RequestDetail
            key={detail.request.id}
            detail={detail}
            onChanged={() => {
              loadQueue();
              if (selectedId) {
                // re-fetch to reflect new state after an answer/action
                fetch(`/api/requests/${encodeURIComponent(selectedId)}`)
                  .then((r) => r.json())
                  .then(setDetail)
                  .catch(() => {});
              }
            }}
          />
        ) : (
          <div className="bg-panel border border-line rounded-xl p-12 text-center text-dim">
            <p className="text-lg font-semibold text-ink/80 mt-2">Operator queue</p>
            <p className="mt-2 max-w-md mx-auto">
              Select a pending request on the left to answer it. The request stays in this
              pane — no switching between pages.
            </p>
          </div>
        )}
      </section>
    </div>
  );
}

function age(epoch) {
  const s = Math.max(0, Math.floor(Date.now() / 1000) - epoch);
  if (s < 60) return `${s}s`;
  if (s < 3600) return `${Math.floor(s / 60)}m`;
  return `${Math.floor(s / 3600)}h`;
}