// Pure live word-lock logic. Kept framework-free so it can be unit-tested.

/**
 * Lock one word from the draft text.
 *
 * The textarea value at any moment is `leading-whitespace + word-in-progress`
 * (after a lock, the tail holds only the trailing separator). When a
 * separator is typed, the word is complete and the whole buffer looks like
 * `lead-ws + word + trail-ws`. We lock `lead-ws + word` as the delta and keep
 * `trail-ws` as the new tail.
 *
 * Net effect for "1 2 3": deltas ["1", " 2", " 3"] — each space consumed
 * exactly once (no double-space) — and the trailing separator stays in the
 * tail where finish() drops it, so the answer has no trailing space.
 *
 * @param {string} draft current textarea value
 * @returns {{ delta: string, tail: string }|null} null when nothing lockable
 */
export function lockOneWord(draft) {
  const m = draft.match(/^(\s*)(\S+)(\s+)$/);
  if (!m) return null;
  return { delta: m[1] + m[2], tail: m[3] };
}

/**
 * Simulate interactive typing of `input` char-by-char through the lock logic,
 * then apply finish() semantics (drop trailing whitespace, push the rest).
 * Mirrors RequestDetail's handleTailChange + finish.
 */
export function simulateTyping(input) {
  const locked = [];
  let tail = "";
  for (const ch of input) {
    const next = tail + ch;
    const res = lockOneWord(next);
    if (res) {
      locked.push(res.delta);
      tail = res.tail;
    } else {
      tail = next;
    }
  }
  const cleanedTail = tail.replace(/\s+$/, "");
  if (cleanedTail !== "") locked.push(cleanedTail);
  return { locked, fullText: locked.join("") };
}