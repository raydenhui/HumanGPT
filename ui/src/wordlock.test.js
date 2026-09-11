// Unit tests for the pure word-lock logic (ui/src/wordlock.js).
// Run: node --test src/wordlock.test.js

import { lockOneWord, simulateTyping } from "./wordlock.js";
import test from "node:test";
import assert from "node:assert/strict";

test("lockOneWord locks a completed word, keeping the separator as tail", () => {
  const res = lockOneWord("1 ");
  assert.deepEqual(res, { delta: "1", tail: " " });
});

test("lockOneWord does not lock a bare word (no separator) or mid-word", () => {
  assert.equal(lockOneWord("1"), null); // final word stays until finish
  assert.equal(lockOneWord("hel"), null); // mid-word
});

test("lockOneWord rides the leading separator onto the next word", () => {
  const res = lockOneWord(" 2 ");
  assert.deepEqual(res, { delta: " 2", tail: " " });
});

test("typing '1 2 3 4 5' sends each space exactly once and no trailing space", () => {
  const { locked, fullText } = simulateTyping("1 2 3 4 5");
  // locked = ["1"," 2"," 3"," 4"," 5"] — one space per separator, no trailing
  assert.deepEqual(locked, ["1", " 2", " 3", " 4", " 5"]);
  assert.equal(fullText, "1 2 3 4 5");
  assert.equal(locked.join("").endsWith(" "), false);
});

test("typing with an intentional trailing space still trims it on finish", () => {
  const { locked, fullText } = simulateTyping("1 2 3 ");
  assert.deepEqual(locked, ["1", " 2", " 3"]);
  assert.equal(fullText, "1 2 3");
});

test("single word with no space reconstructs exactly", () => {
  const { locked, fullText } = simulateTyping("hello");
  assert.deepEqual(locked, ["hello"]);
  assert.equal(fullText, "hello");
});

test("multiline separators lock correctly", () => {
  const { locked, fullText } = simulateTyping("a\nb\tc");
  assert.equal(fullText, "a\nb\tc");
  assert.equal(locked.join("").endsWith("c"), true);
});