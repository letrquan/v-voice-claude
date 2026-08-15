/**
 * Session logging for the pill.
 *
 * Hold-to-talk problems are only visible in real time — the pill has no console
 * and the window is never focused while the hotkey is held — so every event is
 * mirrored to a file on disk that can be read after the fact.
 * Use `getLogPath()` to find it.
 */

import { invoke } from "@tauri-apps/api/core";

let sessionStartMs = 0;

/** Reset the "+Xs" column so timings read relative to the current session. */
export function startLogSession(): void {
  sessionStartMs = Date.now();
}

export function endLogSession(): void {
  sessionStartMs = 0;
}

function stamp(): string {
  const d = new Date();
  const pad = (n: number, width = 2) => String(n).padStart(width, "0");
  return `${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}.${pad(
    d.getMilliseconds(),
    3
  )}`;
}

export function log(event: string, data?: Record<string, unknown>): void {
  const elapsed = sessionStartMs
    ? `+${((Date.now() - sessionStartMs) / 1000).toFixed(2)}s`
    : "-";
  let detail = "";
  if (data) {
    try {
      detail = ` ${JSON.stringify(data)}`;
    } catch {
      detail = " [unserializable]";
    }
  }
  const line = `[${stamp()}] [${elapsed.padStart(8)}] ${event}${detail}`;
  console.log(line);
  // Fire and forget: logging must never be able to break a recording.
  invoke("log_event", { line }).catch(() => {});
}

export async function getLogPath(): Promise<string> {
  return invoke<string>("get_log_path");
}
