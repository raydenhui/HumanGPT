// Unit tests for ui/src/requestview.js normalizers.
// Run: node --test src/requestview.test.js

import {
  asArray,
  normalizeChatMessages,
  normalizeDetail,
  normalizeResponsesInput,
  normalizeTools,
} from "./requestview.js";
import test from "node:test";
import assert from "node:assert/strict";

test("asArray coerces scalars and passes arrays through", () => {
  assert.deepEqual(asArray(undefined), []);
  assert.deepEqual(asArray(null), []);
  assert.deepEqual(asArray("x"), ["x"]);
  assert.deepEqual(asArray([1, 2]), [1, 2]);
});

test("normalizeResponsesInput handles a STRING input (the crash case)", () => {
  const items = normalizeResponsesInput("tell me something");
  assert.equal(items.length, 1);
  assert.equal(items[0].role, "user");
  assert.equal(items[0].content, "tell me something");
});

test("normalizeResponsesInput handles an item array", () => {
  const items = normalizeResponsesInput([
    { type: "message", role: "user", content: "hi" },
    { type: "message", role: "assistant", content: "hello" },
  ]);
  assert.equal(items.length, 2);
  assert.equal(items[1].role, "assistant");
});

test("normalizeResponsesInput handles non-object entries", () => {
  const items = normalizeResponsesInput(["raw string", 42]);
  assert.equal(items.length, 2);
  assert.equal(items[0].content, "raw string");
  assert.equal(items[1].content, "42");
});

test("normalizeChatMessages filters non-objects and accepts missing", () => {
  assert.deepEqual(normalizeChatMessages(undefined), []);
  assert.deepEqual(normalizeChatMessages({ role: "user" }), [{ role: "user" }]);
  const out = normalizeChatMessages([{ role: "user" }, "junk", null, { role: "tool" }]);
  assert.equal(out.length, 2);
});

test("normalizeTools is array-safe", () => {
  assert.deepEqual(normalizeTools(undefined), []);
  assert.deepEqual(normalizeTools([{ type: "function" }]), [{ type: "function" }]);
});

test("normalizeDetail on the failing responses shape does not throw", () => {
  const detail = {
    request: { id: "req_x", endpoint: "/v1/responses" },
    body: { input: "just a string", tools: undefined },
    parsed: {},
    stream_mode: "word-chunk",
  };
  const n = normalizeDetail(detail);
  assert.equal(n.inputItems.length, 1);
  assert.deepEqual(n.tools, []);
  assert.deepEqual(n.templates, []);
});

test("normalizeDetail is null-safe", () => {
  const n = normalizeDetail(null);
  assert.deepEqual(n.inputItems, []);
  assert.deepEqual(n.chatMessages, []);
  assert.deepEqual(n.tools, []);
  assert.equal(n.streamMode, "word-chunk");
});