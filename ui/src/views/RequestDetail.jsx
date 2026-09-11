import React, { useState, useRef } from "react";
import { api, operatorName } from "../api.js";
import { lockOneWord } from "../wordlock.js";
import { normalizeDetail, isAnswerable, usesLiveWordChunk } from "../requestview.js";

function Part(part) {
  if (!part || typeof part !== "object") return null;
  if (part.type === "text" || part.type === "input_text") {
    return <div className="whitespace-pre-wrap break-words text-ink text-[15px] leading-relaxed">{part.text}</div>;
  }
  if (part.type === "image_url" || part.type === "input_image") {
    const url = typeof part.image_url === "string" ? part.image_url : part.image_url?.url;
    if (!url) return null;
    return (
      <div className="my-2">
        <img src={url} loading="lazy" referrerPolicy="no-referrer" alt="attachment"
             className="max-w-full max-h-72 rounded-lg border border-line" />
        <a href={url} target="_blank" rel="noreferrer" className="text-[12px] text-dim break-all hover:underline">open ↗</a>
      </div>
    );
  }
  return null;
}

function Message(msg) {
  const role = msg.role || "message";
  return (
    <div className={`rounded-lg bg-panel2 px-3.5 py-2.5 border-l-4 ${roleBorder(role)}`}>
      <div className="text-[12px] uppercase tracking-wide text-accent font-semibold mb-1">
        {role}
        {msg.name && <span className="text-dim font-normal"> name={msg.name}</span>}
        {msg.tool_call_id && <span className="text-dim font-normal"> · tool_call_id={msg.tool_call_id}</span>}
      </div>
      {Array.isArray(msg.content)
        ? msg.content.map((p, i) => <Part key={i} {...p} />)
        : msg.content !== undefined && msg.content !== null && (
          <div className="whitespace-pre-wrap break-words text-[15px] leading-relaxed">{String(msg.content)}</div>
        )}
      {msg.tool_calls?.map((call) => (
        <details key={call?.id || "tc"} className="mt-1.5">
          <summary className="cursor-pointer text-accent text-[13px]">
            <span className="tag">{call?.function?.name || "function"}</span> <span className="text-dim font-mono text-[12px]">{call?.id}</span>
          </summary>
          <pre className="bg-black/40 border border-line rounded-lg p-2 text-[12px] overflow-x-auto whitespace-pre-wrap">{call?.function?.arguments || "{}"}</pre>
        </details>
      ))}
    </div>
  );
}

function roleBorder(role) {
  return { system: "border-accent", user: "border-emerald-500", assistant: "border-amber-500", tool: "border-rose-500" }[role] || "border-line";
}

function ToolDetail(tool) {
  const fn = tool?.function || {};
  return (
    <details className="my-1">
      <summary className="cursor-pointer text-ink text-[14px]">
        <span className="inline-block px-1.5 rounded bg-panel2 text-accent text-[12px]">{fn.name || "function"}</span>
        {fn.description && <span className="text-dim text-[13px] ml-2">{fn.description.slice(0, 90)}</span>}
      </summary>
      <pre className="bg-black/40 border border-line rounded-lg p-2 text-[12px] overflow-x-auto whitespace-pre-wrap mt-1">
        {JSON.stringify(fn.parameters || {}, null, 2)}
      </pre>
    </details>
  );
}

export function RequestDetail({ detail, onChanged }) {
  const req = detail.request;
  const { body, parsed, templates, streamMode, inputItems, chatMessages, tools } = normalizeDetail(detail);

  // A request is only answerable while pending. Once answered/streamed/etc.
  // the form is replaced by the delivered response (prevents repeat submits,
  // which the server rejects with `request_not_pending`).
  const isPending = isAnswerable(req);
  // The live word-chunk form only makes sense for a *streaming* request whose
  // effective mode is word-chunk. Non-streaming requests use the plain form
  // (also used for tool-call answers).
  const live = usesLiveWordChunk(req, streamMode);

  const [flash, setFlash] = useState(null);
  const [locked, setLocked] = useState([]); // deltas already streamed
  const [tail, setTail] = useState("");
  const [toolCall, setToolCall] = useState(false);
  const [toolName, setToolName] = useState("");
  const [toolArgs, setToolArgs] = useState("");
  const [busy, setBusy] = useState(false);
  const submittingRef = useRef(false);

  const lockedText = locked.join("");

  function pushDelta(delta) {
    setLocked((prev) => [...prev, delta]);
    // fire-and-forget; failures are surfaced on finish
    api.streamWord(req.id, delta).catch(() => {});
  }

  /**
   * Live word lock — delegates to the pure `lockOneWord` (see wordlock.js) so
   * the exact typing semantics ("1 2 3" -> deltas ["1"," 2"," 3"], no double
   * spaces, no trailing space) are unit-tested. Every typed separator is
   * consumed exactly once.
   */
  function handleTailChange(freshValue) {
    const res = lockOneWord(freshValue);
    if (res) {
      pushDelta(res.delta);
      setTail(res.tail);
    } else {
      setTail(freshValue);
    }
  }

  /** Send word now: stream the current word exactly as typed. */
  function sendWordNow() {
    if (tail.trim() === "") return;
    pushDelta(tail);
    setTail("");
  }

  function finish() {
    // Ref guard closes the double-click window before React re-renders `busy`.
    if (submittingRef.current || busy) return;
    if (!isPending) {
      setFlash({ kind: "err", msg: "This request is no longer pending." });
      onChanged();
      return;
    }
    submittingRef.current = true;
    setBusy(true);
    // Snapshot the full text BEFORE side effects so the submitted envelope
    // (usage/content) exactly matches what the client received as deltas.
    // Drop any trailing whitespace the operator left in the buffer (they
    // pressed Shift+Enter without intending a final space).
    const cleanedTail = tail.replace(/\s+$/, "");
    const fullText = lockedText + cleanedTail;
    // Stream the remaining (cleaned) tail as the final delta.
    if (live && cleanedTail !== "") {
      pushDelta(cleanedTail);
      setTail("");
    }
    api
      .answer(req.id, {
        text: fullText,
        is_tool_call: toolCall,
        tool_name: toolCall ? toolName : "",
        tool_arguments: toolCall ? toolArgs : "",
        stream_mode: streamMode,
        operator: operatorName(),
      })
      .then(() => {
        setFlash({ kind: "ok", msg: "Answer submitted." });
        onChanged();
        setTimeout(() => setFlash(null), 4000);
      })
      .catch((e) => {
        const msg = String(e && e.message ? e.message : e);
        setFlash({ kind: "err", msg });
        submittingRef.current = false;
        setBusy(false);
        // If another submit won the race (or it was answered elsewhere),
        // refresh so the UI shows the delivered response instead of the form.
        if (/not pending|already answered/i.test(msg)) {
          onChanged();
        }
      });
  }

  function insertTemplate(name) {
    const t = (templates || []).find((x) => x.name === name);
    if (t) {
      setTail(tail + (tail ? "\n" : "") + t.body);
    }
  }

  async function action(kind, confirmMsg) {
    if (confirmMsg && !window.confirm(confirmMsg)) return;
    try {
      await api[kind](req.id);
      onChanged();
    } catch (e) {
      setFlash({ kind: "err", msg: String(e) });
    }
  }

  return (
    <div className="space-y-4">
      {flash && (
        <div className={`rounded-lg p-3 border ${flash.kind === "ok" ? "border-emerald-500 bg-emerald-500/10 text-emerald-400" : "border-rose-500 bg-rose-500/10 text-rose-400"}`}>
          {flash.msg}
        </div>
      )}

      <div className="flex flex-wrap gap-2 items-center">
        <span className="font-mono text-[13px] text-ink">{req.id}</span>
        <span className="px-2 py-0.5 rounded-full text-[12px] border bg-amber-500/15 text-amber-400 border-amber-500">{req.status}</span>
        <span className="px-1.5 rounded bg-panel2 text-accent text-[12px]">{req.endpoint}</span>
        <span className="px-1.5 rounded bg-panel2 text-ink text-[12px]">{req.model}</span>
        {req.stream && <span className="px-1.5 rounded bg-panel2 text-amber-400 text-[12px]">stream:true</span>}
        {req.claimed_by && <span className="text-[12px] text-amber-400">👤 {req.claimed_by}</span>}
        <span className="text-[12px] text-dim">created {fmtTime(req.created_at)}</span>
      </div>

      {/* message chain */}
      <Panel>
        <SectionTitle>Messages {parsed.message_count ? `(${parsed.message_count})` : ""}</SectionTitle>
        {inputItems.length === 0 && chatMessages.length === 0 && (
          <p className="text-dim text-sm">(empty)</p>
        )}
        {req.endpoint === "/v1/responses"
          ? inputItems.map((i, idx) =>
              i.type === "function_call_output" ? (
                <div key={idx} className="rounded-lg bg-panel2 px-3 py-2 border-l-4 border-rose-500">
                  <div className="text-[12px] uppercase text-rose-400">function_call_output</div>
                  <div className="whitespace-pre-wrap break-words">{String(i.output ?? "")}</div>
                </div>
              ) : (
                <Message key={i.id || idx} role={i.role || "user"} content={i.content} />
              )
            )
          : chatMessages.map((m, i) => <Message key={i} {...m} />)}
      </Panel>

      {/* tools */}
      {tools.length > 0 && (
        <Panel>
          <SectionTitle>Tools</SectionTitle>
          {tools.map((t, i) => <ToolDetail key={i} tool={t} />)}
        </Panel>
      )}

      {/* params */}
      <Panel>
        <SectionTitle>Generation parameters</SectionTitle>
        <pre className="bg-black/40 border border-line rounded-lg p-2 text-[13px] overflow-x-auto whitespace-pre-wrap">
          {JSON.stringify(parsed.params || {}, null, 2)}
        </pre>
      </Panel>

      {/* raw JSON */}
      <Panel>
        <details>
          <summary className="cursor-pointer text-[14px] text-dim">Raw request JSON</summary>
          <pre className="bg-black/40 border border-line rounded-lg p-2 text-[12px] overflow-auto whitespace-pre-wrap max-h-72 mt-1">
            {JSON.stringify(body, null, 2)}
          </pre>
        </details>
      </Panel>

      {/* answer form (only while pending) or the delivered response */}
      {isPending ? (
        <AnswerForm
          live={live}
          requestStream={Boolean(req.stream)}
          streamMode={streamMode}
          locked={lockedText}
          tail={tail}
          toolCall={toolCall}
          toolName={toolName}
          toolArgs={toolArgs}
          busy={busy}
          templates={templates}
          onTail={handleTailChange}
          onSendWord={sendWordNow}
          onFinish={finish}
          onToolCall={setToolCall}
          onToolName={setToolName}
          onToolArgs={setToolArgs}
          onInsert={insertTemplate}
        />
      ) : (
        <DeliveredResponse detail={detail} req={req} />
      )}

      {/* actions */}
      <Panel>
        <SectionTitle>Actions</SectionTitle>
        <div className="flex gap-3">
          <button className="btn" onClick={() => action("returnToPending", "Return this request to pending?")}>↺ Return to pending</button>
          <button className="btn" onClick={() => action("timeout", "Mark as timed out? The client receives a 504.")}>⏱ Mark timed out</button>
          <button className="btn danger" onClick={() => action("discard", "Discard this request? The client receives an error. The row is kept.")}>✕ Discard</button>
        </div>
      </Panel>
    </div>
  );
}

function AnswerForm({ live, requestStream, streamMode, locked, tail, toolCall, toolName, toolArgs, busy, templates, onTail, onSendWord, onFinish, onToolCall, onToolName, onToolArgs, onInsert }) {
  const plainHint = requestStream
    ? "Once mode. Type the reply, then Submit."
    : "Non-streaming request. Type the reply, then Submit.";
  return (
    <Panel>
      <SectionTitle>Answer as the assistant</SectionTitle>
      {live ? (
        <>
          <p className="text-dim text-[13px] leading-relaxed">
            Live word-chunk mode. Each word streams as it is typed and is then locked.{" "}
            <kbd className="px-1 rounded bg-panel2 text-[12px]">Shift+Enter</kbd> to finish.
          </p>
          <div className="flex flex-col gap-2.5 mt-1">
            <div className="rounded-lg border border-dashed border-line bg-panel2/60 px-3 py-2 min-h-[36px] text-[14px] text-dim whitespace-pre-wrap break-words">
              {locked && <span className="bg-accent/15 text-accent rounded px-1">{locked}</span>}
              {!locked && <span className="opacity-70">Sent text (locked)</span>}
            </div>
            <textarea
              className="w-full bg-panel2 border border-line rounded-lg p-3 text-[16px] leading-relaxed min-h-[110px] focus:outline-none focus:border-accent"
              placeholder="Type the assistant reply"
              value={tail}
              onChange={(e) => onTail(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && e.shiftKey) {
                  e.preventDefault();
                  onFinish();
                } else if (e.key === "Enter") {
                  // plain Enter = separator that locks the current word
                  e.preventDefault();
                  onTail(e.target.value + "\n");
                }
              }}
            />
          </div>
          <div className="flex gap-2.5 flex-wrap mt-2">
            <button className="btn" onClick={onSendWord}>Send word now</button>
            <button className="btn primary" disabled={busy} onClick={onFinish} title="Shift+Enter">Finish response (Shift+Enter)</button>
          </div>
          <details className="mt-2">
            <summary className="cursor-pointer text-dim text-[14px]">This reply is a <code className="text-[12px]">tool_call</code></summary>
            <div className="mt-2 space-y-2">
              <label className="flex items-center gap-2 text-[14px]"><input type="checkbox" className="w-4 h-4 accent-accent" checked={toolCall} onChange={(e) => onToolCall(e.target.checked)} /> tool_call answer</label>
              {toolCall && (
                <>
                  <input className="input" placeholder="function name" value={toolName} onChange={(e) => onToolName(e.target.value)} />
                  <input className="input" placeholder='arguments JSON' value={toolArgs} onChange={(e) => onToolArgs(e.target.value)} />
                </>
              )}
            </div>
          </details>
          {templates?.length > 0 && (
            <div className="flex flex-wrap gap-1.5 mt-3 items-center">
              <span className="text-dim text-[13px]">Insert canned:</span>
              {templates.map((t) => (
                <button key={t.name} className="text-[12px] px-2 py-0.5 rounded-full border border-accent text-accent bg-transparent hover:bg-accent/10" onClick={() => onInsert(t.name)}>
                  {t.name}
                </button>
              ))}
            </div>
          )}
        </>
      ) : (
        <>
          <p className="text-dim text-[13px]">{plainHint}</p>
          <textarea
            className="w-full bg-panel2 border border-line rounded-lg p-3 text-[15px] min-h-[120px] focus:outline-none focus:border-accent"
            placeholder="Type the assistant reply"
            onChange={(e) => onTail(e.target.value)}
          />
          <div className="mt-2 space-y-2">
            <label className="flex items-center gap-2 text-[14px]"><input type="checkbox" className="w-4 h-4 accent-accent" checked={toolCall} onChange={(e) => onToolCall(e.target.checked)} /> this reply is a tool_call</label>
            {toolCall && (
              <>
                <input className="input" placeholder="function name (e.g. get_weather)" value={toolName} onChange={(e) => onToolName(e.target.value)} />
                <input className="input" placeholder='arguments JSON (e.g. {"city":"Paris"})' value={toolArgs} onChange={(e) => onToolArgs(e.target.value)} />
              </>
            )}
          </div>
          <button className="btn primary w-full" disabled={busy} onClick={onFinish}>Submit</button>
        </>
      )}
    </Panel>
  );
}

function Panel({ children }) {
  return <div className="bg-panel border border-line rounded-xl p-4">{children}</div>;
}
function SectionTitle({ children }) {
  return <h3 className="text-[15px] font-semibold text-ink mb-2.5">{children}</h3>;
}
function fmtTime(epoch) {
  return new Date(epoch * 1000).toLocaleString();
}

/** Shown once a request is no longer pending: the delivered answer (or a note
 * for terminal states without a response). */
function DeliveredResponse({ detail, req }) {
  const response = detail.response;
  const tone =
    req.state === "answered" || req.state === "streamed"
      ? "border-emerald-500 bg-emerald-500/10 text-emerald-400"
      : "border-rose-500 bg-rose-500/10 text-rose-400";
  return (
    <Panel>
      <SectionTitle>Response</SectionTitle>
      <div className={`rounded-lg p-3 border mb-3 text-[13px] ${tone}`}>
        This request is <strong>{req.status}</strong>
        {req.answered_by ? ` (answered by ${req.answered_by})` : ""}. Only pending requests can be answered.
      </div>
      {response ? (
        <>
          <div className="text-[13px] text-dim mb-1">
            finish_reason={response.choices?.[0]?.finish_reason ?? "—"} · usage=
            {JSON.stringify(response.usage || {})}
          </div>
          <pre className="bg-black/40 border border-line rounded-lg p-2.5 text-[12px] overflow-auto whitespace-pre-wrap max-h-80">
            {JSON.stringify(response, null, 2)}
          </pre>
        </>
      ) : (
        <p className="text-dim text-sm">No response envelope was recorded for this request.</p>
      )}
    </Panel>
  );
}