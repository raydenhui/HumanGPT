// HumanGPT operator UI — vanilla JS. No build step.
(function () {
  "use strict";

  var OP_KEY = "humangpt_operator";

  function operatorName() {
    try { return localStorage.getItem(OP_KEY) || ""; } catch (e) { return ""; }
  }
  function saveOperator(name) {
    try { localStorage.setItem(OP_KEY, name); } catch (e) { /* ignore */ }
  }
  function prefillOperator() {
    var name = operatorName();
    if (!name) return;
    var el = document.getElementById("operator-name");
    if (el) el.value = name;
    var f = document.getElementById("operator-form-field");
    if (f) f.value = name;
  }

  // ------------------------------------------------------------ helpers
  function fmtTs(epoch) { return new Date(epoch * 1000).toLocaleString(); }
  function fmtAge(epoch) {
    var s = Math.max(0, Math.floor(Date.now() / 1000) - epoch);
    if (s < 60) return s + "s";
    if (s < 3600) return Math.floor(s / 60) + "m";
    return Math.floor(s / 3600) + "h";
  }
  function refreshTimes() {
    document.querySelectorAll(".ts").forEach(function (el) {
      var e = parseInt(el.dataset.epoch, 10);
      if (!isNaN(e)) el.textContent = fmtTs(e);
    });
    document.querySelectorAll(".age").forEach(function (el) {
      var e = parseInt(el.dataset.epoch, 10);
      if (!isNaN(e)) el.textContent = fmtAge(e);
    });
  }
  function copyText(text, done) {
    if (navigator.clipboard) {
      navigator.clipboard.writeText(text).then(done).catch(function () { fallback(text); done(); });
    } else { fallback(text); done(); }
  }
  function fallback(text) {
    var ta = document.createElement("textarea");
    ta.value = text;
    ta.style.position = "fixed";
    ta.style.left = "-9999px";
    document.body.appendChild(ta);
    ta.select();
    try { document.execCommand("copy"); } catch (e) { /* ignore */ }
    document.body.removeChild(ta);
  }

  function flashCopyButton(btn, okLabel) {
    var before = btn.dataset.origText || btn.textContent;
    btn.textContent = okLabel || "Copied ✓";
    setTimeout(function () { btn.textContent = before; }, 1500);
  }

  // ------------------------------------------------- copy API URL button
  function initCopyApiUrl() {
    var btn = document.getElementById("copy-api-url");
    if (!btn) return;
    btn.addEventListener("click", function () {
      copyText(window.location.origin + "/v1", function () { flashCopyButton(btn); });
    });
  }

  // ------------------------------------------------- templates copy
  function initTemplateCopies() {
    document.querySelectorAll(".copy-template").forEach(function (b) {
      b.addEventListener("click", function () {
        copyText(b.dataset.body || "", function () { flashCopyButton(b); });
      });
    });
    document.querySelectorAll(".copy-api-url").forEach(function (b) {
      b.addEventListener("click", function () {
        copyText(b.dataset.url || "", function () { flashCopyButton(b); });
      });
    });
  }

  // ------------------------------------------------- canned insert (once-mode + pane)
  function initTemplateInsert() {
    document.querySelectorAll(".insert-template").forEach(function (b) {
      b.addEventListener("click", function () {
        var ta = document.querySelector(".request-pane #answer-text, form.answer-form textarea[name=text]");
        if (!ta) return;
        if (typeof ta.insertText === "function") ta.insertText(b.dataset.body);
        else ta.value = (ta.value || "") + (b.dataset.body || "");
        ta.focus();
      });
    });
  }

  // -------------------------------------------------- live word-chunk typing
  var lockedText = "";
  var currentReqId = null;

  function pushDelta(delta) {
    if (!currentReqId) return;
    fetch("/requests/" + encodeURIComponent(currentReqId) + "/stream-word", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ delta: delta }),
    }).catch(function () { /* transient; next word retries */ });
  }
  function consumeWholeInputAsLocked() {
    var ta = document.getElementById("answer-text");
    if (!ta) return;
    var text = ta.value;
    ta.value = "";
    if (!text) return;
    lockedText += text;
    pushDelta(text);
    updateLocked();
  }
  function consumeLeadingWord() {
    var ta = document.getElementById("answer-text");
    if (!ta) return;
    var text = ta.value;
    // one atom = one or more dictation separators? No: one *word* (non-space
    // run incl. trailing spaces). Regex splits leading word+trailing ws.
    var m = text.match(/^(\S+\s*)/);
    if (!m) return;
    var word = m[1];
    ta.value = text.slice(word.length);
    lockedText += word;
    pushDelta(word);
    updateLocked();
  }
  function updateLocked() {
    var box = document.getElementById("locked-words");
    if (!box) return;
    var empty = box.textContent === "" && !box.children.length;
    box.textContent = "";
    var frag = document.createElement("span");
    frag.className = "locked";
    frag.textContent = lockedText;
    box.appendChild(frag);
    if (box.textContent === "") {
      box.classList.add("empty-placeholder");
    } else {
      box.classList.remove("empty-placeholder");
    }
  }
  function draftKey(id) { return "hg_draft_" + id; }
  function restoreDraft(id) {
    try { return localStorage.getItem(draftKey(id)) || ""; } catch (e) { return ""; }
  }
  function saveDraft(id, text) {
    try { localStorage.setItem(draftKey(id), text); } catch (e) { /* ignore */ }
  }

  function initRequestPane() {
    var ta = document.getElementById("answer-text");
    if (ta) {
      currentReqId = currentRequestId();
      lockedText = restoreDraft(currentReqId || "");
      ta.value = "";
      updateLocked();
    }
    var sendBtn = document.getElementById("send-word-btn");
    if (sendBtn) {
      sendBtn.addEventListener("click", function () {
        var t = document.getElementById("answer-text");
        if (t && t.value.trim()) consumeLeadingWord();
      });
    }
    var finishBtn = document.getElementById("finish-btn");
    if (finishBtn) {
      finishBtn.addEventListener("click", finishResponse);
    }
    var ta2 = document.getElementById("answer-text");
    if (ta2) {
      ta2.addEventListener("keydown", function (ev) {
        if (ev.key === "Enter" && ev.shiftKey) {
          ev.preventDefault();
          finishResponse();
        }
      });
      // live locking: whenever a separator is typed and then a space follows,
      // JS cannot know "word complete" as-you-type without a separator check.
      // We lock on the FIRST SPACE/NEWLINE after a non-space char is typed
      // (the moment the word is followed by a space). We use input events.
      ta2.addEventListener("input", onAnswerInput);
    }
  }
  function currentRequestId() {
    var pane = document.getElementById("md-request");
    if (!pane) return null;
    var p = pane.querySelector(".request-pane");
    return p ? p.dataset.requestId : null;
  }
  function finishResponse() {
    if (!currentReqId) return;
    var ta = document.getElementById("answer-text");
    if (!ta) return;
    // stream the remaining tail, then submit the *full* answer.
    var tail = ta.value;
    consumeWholeInputAsLocked();
    var fd = new FormData();
    fd.append("operator", operatorName() || "anonymous");
    fd.append("text", lockedText + tail);
    fd.append("stream_mode", "word-chunk");
    fetch("/requests/" + encodeURIComponent(currentReqId) + "/answer", {
      method: "POST",
      body: fd,
      redirect: "follow",
    }).then(function (r) {
      if (r.ok) {
        try { localStorage.removeItem(draftKey(currentReqId)); } catch (e) {}
        setTimeout(function () { loadRequestIntoPane(currentReqId); }, 200);
      } else {
        try { r.json().then(function (j) { alert("Error: " + (j.error || r.status)); }); } catch (e) {}
      }
    }).catch(function () { alert("Network error submitting answer."); });
  }
  var lastWasSpace = false;
  function onAnswerInput(ev) {
    var ta = ev.target;
    var val = ta.value;
    // lock on the FIRST whitespace encountered at the END of the text (a new
    // space typed after a non-space). Capture the previous leading word(s).
    if (val.length > 0 && /\s$/.test(val) && !lastWasSpace) {
      consumeLeadingWord();
    }
    lastWasSpace = /\s$/.test(ta.value);
    if (currentReqId) saveDraft(currentReqId, ta.value);
  }

  // -------------------------------------------------------- master–detail
  function loadRequestIntoPane(id) {
    var pane = document.getElementById("md-request");
    if (!pane) return;
    pane.innerHTML = '<p class="empty">Loading…</p>';
    fetch("_fragments/request/" + encodeURIComponent(id) + "?operator=" + encodeURIComponent(operatorName()))
      .then(function (r) { return r.text(); })
      .then(function (html) {
        pane.innerHTML = html;
        var empty = document.getElementById("md-empty");
        if (empty) empty.style.display = "none";
        document.querySelectorAll(".queue-row").forEach(function (tr) { tr.classList.remove("active"); });
        var active = document.querySelector('.queue-row[data-id="' + id + '"]');
        if (active) active.classList.add("active");
        initRequestPane();
        refreshTimes();
      })
      .catch(function () { pane.innerHTML = '<p class="empty">Failed to load request.</p>'; });
  }
  function initQueueTable() {
    document.querySelectorAll(".queue-row.clickable").forEach(function (tr) {
      tr.addEventListener("click", function () { loadRequestIntoPane(tr.dataset.id); });
    });
  }

  // ------------------------------------------------------- SSE live updates
  function refreshQueue() {
    var tableEl = document.getElementById("queue-table");
    if (!tableEl) return;
    fetch("_fragments/queue", { headers: { Accept: "text/html" } })
      .then(function (r) { return r.text(); })
      .then(function (html) {
        var wrap = document.createElement("div");
        wrap.innerHTML = html;
        var fresh = wrap.querySelector("table");
        if (fresh) {
          var old = tableEl;
          old.parentNode.replaceChild(fresh, old);
        }
        var rows = fresh ? fresh.querySelectorAll("tbody tr").length : 0;
        var pill = document.getElementById("pill-pending");
        if (pill) pill.textContent = String(rows);
        var counter = document.getElementById("queue-count");
        if (counter) counter.textContent = "(" + rows + ")";
        initQueueTable();
        refreshTimes();
      })
      .catch(function () {});
  }
  function connectEvents() {
    if (!window.EventSource) return;
    var es = new EventSource("/events");
    es.addEventListener("update", function (ev) {
      var data;
      try { data = JSON.parse(ev.data); } catch (e) { return; }
      if (data.type === "request-changed") refreshQueue();
      if (data.type === "templates-changed" || data.type === "models-changed" || data.type === "settings-changed") {
        if (location.pathname === "/settings" || location.pathname === "/templates") location.reload();
      }
    });
  }

  // ---------------------------------------------------------------- wiring
  document.addEventListener("DOMContentLoaded", function () {
    prefillOperator();
    var opEl = document.getElementById("operator-name");
    if (opEl) opEl.addEventListener("change", function () { saveOperator(opEl.value.trim()); });
    initCopyApiUrl();
    initTemplateCopies();
    initTemplateInsert();
    initQueueTable();
    refreshTimes();
    setInterval(refreshTimes, 15000);
    connectEvents();
  });
})();