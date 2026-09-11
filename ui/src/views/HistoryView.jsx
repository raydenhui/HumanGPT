import React, { useState, useEffect } from "react";
import { api, fmtTs, STATUS_LABEL, STATUS_COLOR } from "../api.js";

export function HistoryView({ refreshToken }) {
  const [rows, setRows] = useState([]);
  const [selected, setSelected] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetch("/api/history")
      .then((r) => r.json())
      .then((d) => setRows(d.rows || []))
      .catch((e) => setError(String(e)));
  }, [refreshToken]);

  function open(id) {
    fetch(`/api/history/${encodeURIComponent(id)}`)
      .then((r) => r.json())
      .then(setSelected)
      .catch((e) => setError(String(e)));
  }

  return (
    <div className="grid grid-cols-[minmax(0,30rem)_1fr] gap-6 items-start">
      <section className="panel">
        <h2 className="text-[17px] font-semibold mb-3">
          History <span className="text-dim text-[13px]">({rows.length})</span>
        </h2>
        <table className="w-full text-[14px]">
          <thead>
            <tr className="text-left text-dim text-[13px] border-b border-line">
              <th className="py-1.5">Status</th>
              <th>Model</th>
              <th>Answered by</th>
              <th>When</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.id} className="border-b border-line cursor-pointer hover:bg-panel2" onClick={() => open(r.id)}>
                <td className="py-2">
                  <span className={`inline-block px-2 py-0.5 rounded-full text-[12px] border ${STATUS_COLOR[r.state] || STATUS_COLOR.discarded}`}>
                    {STATUS_LABEL[r.state] || r.state}
                  </span>
                </td>
                <td className="py-2">{r.model}</td>
                <td className="py-2 text-dim">{r.answered_by || "—"}</td>
                <td className="py-2 text-dim whitespace-nowrap">{r.answered_at ? fmtTs(r.answered_at) : fmtTs(r.created_at)}</td>
              </tr>
            ))}
          </tbody>
        </table>
        {!rows.length && !error && <p className="text-dim text-sm py-5 text-center">Nothing here yet.</p>}
      </section>

      <section className="min-w-0">
        {error && <div className="text-rose-400">{error}</div>}
        {selected ? (
          <Transcript {...selected} />
        ) : (
          <div className="panel flex flex-col items-center justify-center py-16 text-dim">
            <p className="text-lg text-ink/80 font-semibold">Transcript</p>
            <p className="mt-2 max-w-md mx-auto text-center">Select a request on the left to audit its full transcript: request JSON, the human's response envelope, finish_reason, usage, and who answered.</p>
          </div>
        )}
      </section>
    </div>
  );
}

function Transcript({ request, body, response }) {
  const params = request?.parsed?.params || {};
  return (
    <div className="space-y-4">
      <div className="flex flex-wrap gap-2 items-center">
        <span className="font-mono text-[13px]">{request?.id}</span>
        <span className="text-dim text-[13px]">{request?.endpoint}</span>
        <span className="tag">{request?.model}</span>
        <span className="text-dim text-[13px]">answered by {request?.answered_by || "—"}</span>
      </div>
      <div className="panel">
        <h3 className="text-[15px] font-semibold mb-2">Request</h3>
        <pre className="bg-black/40 border border-line rounded-lg p-2.5 text-[12px] overflow-auto whitespace-pre-wrap max-h-80">
          {JSON.stringify(body, null, 2)}
        </pre>
      </div>
      {response && (
        <div className="panel">
          <h3 className="text-[15px] font-semibold mb-2">Human response <span className="badge-ok">delivered</span></h3>
          <pre className="bg-black/40 border border-line rounded-lg p-2.5 text-[12px] overflow-auto whitespace-pre-wrap max-h-80">
            {JSON.stringify(response, null, 2)}
          </pre>
          <div className="text-dim text-[13px] mt-2">
            finish_reason={response.choices?.[0]?.finish_reason} · usage={JSON.stringify(response.usage)}
          </div>
        </div>
      )}
      <div className="panel">
        <h3 className="text-[15px] font-semibold mb-2">Generation parameters</h3>
        <pre className="bg-black/40 border border-line rounded-lg p-2 text-[13px]">{JSON.stringify(params, null, 2)}</pre>
      </div>
    </div>
  );
}