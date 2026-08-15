/**
 * Hotkey capture and formatting, shared by the pill and the settings window.
 *
 * Accelerators are built from `KeyboardEvent.code` rather than `.key`, because
 * `.key` reports the *shifted* character: Ctrl+Shift+1 arrives as "!", which
 * produces an accelerator the global-shortcut plugin cannot parse. Anything not
 * in the table below is rejected at capture time instead of failing later, at
 * registration, where there is no good way to explain it.
 */

const MODIFIER_CODES = new Set([
  "ControlLeft",
  "ControlRight",
  "ShiftLeft",
  "ShiftRight",
  "AltLeft",
  "AltRight",
  "MetaLeft",
  "MetaRight",
]);

/** Physical key code -> accelerator token understood by the shortcut plugin. */
const CODE_TO_KEY: Record<string, string> = {
  Space: "Space",
  Enter: "Enter",
  Tab: "Tab",
  Backspace: "Backspace",
  Delete: "Delete",
  Insert: "Insert",
  Home: "Home",
  End: "End",
  PageUp: "PageUp",
  PageDown: "PageDown",
  ArrowUp: "Up",
  ArrowDown: "Down",
  ArrowLeft: "Left",
  ArrowRight: "Right",
  Minus: "Minus",
  Equal: "Equal",
  BracketLeft: "BracketLeft",
  BracketRight: "BracketRight",
  Backslash: "Backslash",
  Semicolon: "Semicolon",
  Quote: "Quote",
  Backquote: "Backquote",
  Comma: "Comma",
  Period: "Period",
  Slash: "Slash",
};

/** How the accelerator tokens read back to a person. */
const KEY_LABELS: Record<string, string> = {
  CommandOrControl: "Ctrl",
  Minus: "-",
  Equal: "=",
  BracketLeft: "[",
  BracketRight: "]",
  Backslash: "\\",
  Semicolon: ";",
  Quote: "'",
  Backquote: "`",
  Comma: ",",
  Period: ".",
  Slash: "/",
};

function keyFromCode(code: string): string | null {
  if (/^Key[A-Z]$/.test(code)) return code.slice(3);
  if (/^Digit[0-9]$/.test(code)) return code.slice(5);
  if (/^F([1-9]|1[0-9]|2[0-4])$/.test(code)) return code;
  if (/^Numpad[0-9]$/.test(code)) return code;
  return CODE_TO_KEY[code] ?? null;
}

export type HotkeyCapture =
  /** Nothing to commit yet — no modifier held, or modifiers only. */
  | { kind: "pending" }
  | { kind: "accelerator"; accelerator: string }
  | { kind: "unsupported" };

export function captureHotkey(e: KeyboardEvent): HotkeyCapture {
  if (MODIFIER_CODES.has(e.code)) return { kind: "pending" };

  const parts: string[] = [];
  if (e.ctrlKey || e.metaKey) parts.push("CommandOrControl");
  if (e.shiftKey) parts.push("Shift");
  if (e.altKey) parts.push("Alt");

  // A global shortcut without a modifier would swallow the key everywhere.
  if (parts.length === 0) return { kind: "pending" };

  const key = keyFromCode(e.code);
  if (!key) return { kind: "unsupported" };

  parts.push(key);
  return { kind: "accelerator", accelerator: parts.join("+") };
}

export function formatHotkey(accelerator: string): string {
  return accelerator
    .split("+")
    .map((part) => KEY_LABELS[part] ?? part)
    .join(" + ");
}
