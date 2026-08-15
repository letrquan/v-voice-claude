import { useState, useEffect, useCallback, useRef } from "react";
import { listen } from "@tauri-apps/api/event";
import { invoke } from "@tauri-apps/api/core";
import { register, unregister } from "@tauri-apps/plugin-global-shortcut";
import { getCurrentWindow } from "@tauri-apps/api/window";
import { LogicalSize, LogicalPosition } from "@tauri-apps/api/dpi";
import { Waveform } from "./components/Waveform";
import { useAudioCapture } from "./hooks/useAudioCapture";
import { formatHotkey } from "./lib/hotkey";
import { log, startLogSession, endLogSession } from "./lib/log";

/** "error" holds the pill open just long enough to read what went wrong. */
type AppState = "idle" | "listening" | "processing" | "error";

type ErrorKind = "download" | "init" | "transcribe" | "mic" | "hotkey";

interface AppError {
  kind: ErrorKind;
  message: string;
}

/** Only asset problems have anything to retry; the rest just need dismissing. */
function isRetryable(error: AppError): boolean {
  return error.kind === "download" || error.kind === "init";
}

function formatError(e: unknown): string {
  const raw =
    typeof e === "string" ? e : e instanceof Error ? e.message : String(e);
  const trimmed = raw.trim();
  if (!trimmed) return "Unknown error";
  return trimmed.length > 180 ? `${trimmed.slice(0, 177)}...` : trimmed;
}

interface AppSettings {
  model: string;
  language: string;
  hotkey: string;
  quit_hotkey: string;
  microphone_id: string;
  vad_silence_threshold: number;
  vad_silence_frames: number;
  transcription_mode: string;
  cloud_provider: string;
  cloud_api_key: string;
  local_engine: string;
}

const DEFAULT_SETTINGS: AppSettings = {
  model: "tiny.en",
  language: "en",
  hotkey: "CommandOrControl+Shift+Space",
  quit_hotkey: "CommandOrControl+Shift+Q",
  microphone_id: "",
  vad_silence_threshold: 0.015,
  vad_silence_frames: 45,
  transcription_mode: "local",
  cloud_provider: "openai",
  cloud_api_key: "",
  local_engine: "whisper",
};

const VAD_DEFAULTS = {
  MIN_SPEECH_FRAMES: 8,
  RING_CIRC: 38,
};

/* ─── Sizes ─── */
const SIZE_IDLE = { w: 48, h: 48 };
const SIZE_PILL = { w: 260, h: 48 };
const SIZE_TALL = { w: 260, h: 120 };

/* ─── Timing ─── */
const TRANSITION_MS = 420;
const SHOW_TRANSCRIPT_DELAY = 300;
const ERROR_HOLD_MS = 2600; // how long a failure stays readable before collapsing
const STREAMING_CHUNK_MS = 2000; // collect audio every 2s
const STREAMING_WINDOW_DURATION = 4; // enough context without delaying first text
const STREAMING_OVERLAP_DURATION = 0.65; // preserve boundary phonemes
const STREAMING_PREROLL_DURATION = 0.5; // retain speech onset without decoding silence
const MIN_FINAL_CHUNK_DURATION = 0.25; // minimum seconds for the final chunk
/** Backstop for a hotkey release that never arrives, so the mic cannot stay open forever. */
const MAX_HOLD_MS = 120000;

function appendAudio(existing: Float32Array | null, incoming: Float32Array): Float32Array {
  if (!existing || existing.length === 0) return incoming;
  const merged = new Float32Array(existing.length + incoming.length);
  merged.set(existing);
  merged.set(incoming, existing.length);
  return merged;
}

/** Return only the newly recognized words from an overlapping window. */
function mergeTranscript(existing: string, incoming: string): string {
  const next = incoming.trim();
  if (!next) return "";
  if (!existing.trim()) return next;

  const oldWords = existing.trim().split(/\s+/);
  const newWords = next.split(/\s+/);
  const normalize = (word: string) => word.toLowerCase().replace(/[^\p{L}\p{N}]+/gu, "");
  const oldNormalized = oldWords.map(normalize);
  const newNormalized = newWords.map(normalize);
  const maxOverlap = Math.min(24, oldWords.length, newWords.length);
  let overlap = 0;

  for (let size = maxOverlap; size > 0; size -= 1) {
    const oldTail = oldNormalized.slice(-size);
    const newHead = newNormalized.slice(0, size);
    if (oldTail.every((word, index) => word && word === newHead[index])) {
      overlap = size;
      break;
    }
  }

  return newWords.slice(overlap).join(" ").trim();
}

/**
 * Resize the window while keeping its visual center-x and bottom-y anchored.
 */
async function resizeInPlace(width: number, height: number) {
  const win = getCurrentWindow();

  const oldPos = await win.outerPosition();
  const oldSize = await win.outerSize();
  const factor = await win.scaleFactor();

  const oldW = oldSize.width / factor;
  const oldH = oldSize.height / factor;
  const oldX = oldPos.x / factor;
  const oldY = oldPos.y / factor;

  const centerX = oldX + oldW / 2;
  const bottomY = oldY + oldH;

  const newX = Math.round(centerX - width / 2);
  const newY = Math.round(bottomY - height);

  await win.setSize(new LogicalSize(width, height));
  await win.setPosition(new LogicalPosition(newX, newY));
}

function App() {
  const [state, setState] = useState<AppState>("idle");
  const [transcript, setTranscript] = useState("");
  const [modelReady, setModelReady] = useState(false);
  const [downloadProgress, setDownloadProgress] = useState(0);
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [showTranscript, setShowTranscript] = useState(false);
  const [error, setError] = useState<AppError | null>(null);
  const [settings, setSettings] = useState<AppSettings>(DEFAULT_SETTINGS);

  const stateRef = useRef<AppState>("idle");
  const modelReadyRef = useRef(false);
  const settingsRef = useRef<AppSettings>(DEFAULT_SETTINGS);
  const vadArcRef = useRef<SVGCircleElement>(null);
  const errorRef = useRef<AppError | null>(null);

  // Bumped on every new listening session so a teardown that is still awaiting
  // a timer cannot stomp the session that replaced it.
  const sessionRef = useRef(0);

  const applyError = useCallback((next: AppError | null) => {
    errorRef.current = next;
    setError(next);
  }, []);

  // ─── Hold-to-talk state ───
  // The hotkey is hold-to-talk, so while the keys are physically down the
  // session must survive silence: VAD auto-stop only applies once released.
  const hotkeyHeldRef = useRef(false);
  // Windows repeats WM_HOTKEY while a shortcut is held; counted, not acted on.
  const pressRepeatsRef = useRef(0);
  // A release that lands before the session finished opening the mic would
  // otherwise be dropped, leaving the pill listening with nobody holding a key.
  const startingRef = useRef(false);
  const pendingReleaseRef = useRef(false);
  const holdWatchdogRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // VAD state refs
  const silenceStartMsRef = useRef<number>(0);
  const speechFramesRef = useRef(0);
  const isSpeakingRef = useRef(false);
  const speechDetectedRef = useRef(false);
  const vadWarmupUntilRef = useRef(0); // timestamp: ignore VAD auto-stop until this time
  const noiseFloorRef = useRef(0.005);

  const { start, stop, consumeBuffer, analyserNode } = useAudioCapture();

  // Streaming transcription refs
  const streamingIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const streamingBusyRef = useRef(false);
  const streamingTaskRef = useRef<Promise<void> | null>(null);
  const committedTextRef = useRef("");
  const streamingPendingRef = useRef<Float32Array | null>(null);
  const streamingSampleRateRef = useRef(16000);
  const discardStreamingRef = useRef(false);
  // Streaming retries quietly, so the last failure is only worth showing if the
  // whole session ended up producing nothing.
  const lastStreamErrorRef = useRef<string | null>(null);

  const transcriptScrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    stateRef.current = state;
  }, [state]);

  // Auto-scroll transcript
  useEffect(() => {
    if (transcriptScrollRef.current) {
      transcriptScrollRef.current.scrollTop = transcriptScrollRef.current.scrollHeight;
    }
  }, [transcript]);

  useEffect(() => {
    modelReadyRef.current = modelReady;
  }, [modelReady]);

  useEffect(() => {
    settingsRef.current = settings;
  }, [settings]);

  // ─── VAD processing ───
  const processVAD = useCallback((rms: number) => {
    if (!analyserNode) return;

    const s = settingsRef.current;
    
    // Dynamic sound-reactive glow
    const pill = document.getElementById("pill");
    if (pill) {
      const glowOpacity = Math.max(0.2, Math.min(0.9, 0.2 + rms * 10));
      pill.style.setProperty("--glow-opacity", String(glowOpacity));
    }

    // Learn the local noise floor during mic startup instead of using a fixed
    // threshold that behaves poorly across microphones and rooms.
    const now = Date.now();
    if (now < vadWarmupUntilRef.current) {
      noiseFloorRef.current = noiseFloorRef.current * 0.92 + rms * 0.08;
      return;
    }

    const adaptiveThreshold = Math.max(
      s.vad_silence_threshold * 0.6,
      noiseFloorRef.current * 2.2
    );
    const speechThreshold = isSpeakingRef.current
      ? adaptiveThreshold * 0.72
      : adaptiveThreshold;
    const speech = rms > speechThreshold;

    if (!isSpeakingRef.current && rms < adaptiveThreshold) {
      noiseFloorRef.current = noiseFloorRef.current * 0.98 + rms * 0.02;
    }
    const targetSilenceMs = s.vad_silence_frames * (1000 / 60);

    if (speech) {
      speechFramesRef.current++;
      silenceStartMsRef.current = 0;
      if (!isSpeakingRef.current && speechFramesRef.current >= VAD_DEFAULTS.MIN_SPEECH_FRAMES) {
        isSpeakingRef.current = true;
        speechDetectedRef.current = true;
        setIsSpeaking(true);
      }
    } else {
      speechFramesRef.current = 0;
      if (silenceStartMsRef.current === 0) {
        silenceStartMsRef.current = now;
      }

      const elapsedMs = now - silenceStartMsRef.current;
      // While the hotkey is held the countdown ring would only promise a stop
      // that is not going to happen, so it stays empty.
      const held = hotkeyHeldRef.current;
      const prog = held ? 0 : Math.min(elapsedMs / targetSilenceMs, 1);

      if (vadArcRef.current) {
        vadArcRef.current.style.strokeDashoffset = (VAD_DEFAULTS.RING_CIRC * (1 - prog)).toFixed(2);
      }

      if (isSpeakingRef.current && elapsedMs >= targetSilenceMs) {
        isSpeakingRef.current = false;
        silenceStartMsRef.current = 0;
        setIsSpeaking(false);
        if (vadArcRef.current) {
          vadArcRef.current.style.strokeDashoffset = String(VAD_DEFAULTS.RING_CIRC);
        }
        if (held) {
          // Hold-to-talk: a pause between sentences is not the end of the take.
          log("vad.pause-ignored-while-held", {
            silenceMs: Math.round(elapsedMs),
            targetSilenceMs: Math.round(targetSilenceMs),
            rms: Number(rms.toFixed(4)),
          });
          return;
        }
        if (stateRef.current === "listening") {
          log("vad.auto-stop", {
            silenceMs: Math.round(elapsedMs),
            targetSilenceMs: Math.round(targetSilenceMs),
            rms: Number(rms.toFixed(4)),
            noiseFloor: Number(noiseFloorRef.current.toFixed(4)),
            threshold: Number(adaptiveThreshold.toFixed(4)),
          });
          toggleRef.current();
        }
      }
    }
  }, [analyserNode]);

  // ─── Close button (cancel listening) ───
  const handleClose = useCallback(async () => {
    if (stateRef.current === "listening") {
      const session = sessionRef.current;
      log("session.cancelled", { session });
      hotkeyHeldRef.current = false;
      if (holdWatchdogRef.current) {
        clearTimeout(holdWatchdogRef.current);
        holdWatchdogRef.current = null;
      }
      discardStreamingRef.current = true;
      // Stop streaming
      if (streamingIntervalRef.current) {
        clearInterval(streamingIntervalRef.current);
        streamingIntervalRef.current = null;
      }
      streamingBusyRef.current = false;
      committedTextRef.current = "";
      streamingPendingRef.current = null;
      speechDetectedRef.current = false;
      lastStreamErrorRef.current = null;

      if (streamingTaskRef.current) {
        await streamingTaskRef.current;
      }
      await stop();
      if (sessionRef.current !== session) return;
      applyError(null);
      setState("idle");
      stateRef.current = "idle";
      setShowTranscript(false);
      setIsSpeaking(false);
      silenceStartMsRef.current = 0;
      speechFramesRef.current = 0;
      isSpeakingRef.current = false;
      speechDetectedRef.current = false;
      await new Promise((r) => setTimeout(r, TRANSITION_MS));
      if (sessionRef.current !== session) return;
      await resizeInPlace(SIZE_IDLE.w, SIZE_IDLE.h);
      setTranscript("");
    }
  }, [stop, applyError]);

  // ─── Quit app ───
  const handleQuit = useCallback(async () => {
    if (stateRef.current === "listening") {
      discardStreamingRef.current = true;
      if (streamingIntervalRef.current) {
        clearInterval(streamingIntervalRef.current);
        streamingIntervalRef.current = null;
      }
      if (streamingTaskRef.current) {
        await streamingTaskRef.current;
      }
      await stop();
    }
    await getCurrentWindow().close();
  }, [stop]);

  // ─── Show a failure in the expanded pill, then settle back to idle ───
  // The pill is 48px when idle, so a message only fits while it is expanded.
  const showErrorThenIdle = useCallback(
    async (session: number, appError: AppError) => {
      applyError(appError);
      setState("error");
      stateRef.current = "error";
      setIsSpeaking(false);
      setShowTranscript(true);
      await resizeInPlace(SIZE_TALL.w, SIZE_TALL.h);
      await new Promise((r) => setTimeout(r, ERROR_HOLD_MS));
      if (sessionRef.current !== session) return;

      setState("idle");
      stateRef.current = "idle";
      setShowTranscript(false);
      await new Promise((r) => setTimeout(r, TRANSITION_MS));
      if (sessionRef.current !== session) return;
      await resizeInPlace(SIZE_IDLE.w, SIZE_IDLE.h);
    },
    [applyError]
  );

  // ─── Toggle: idle -> listening -> processing -> idle ───
  const handleToggle = useCallback(async () => {
    const currentState = stateRef.current;

    // "error" is just idle holding a message, so a new session can start from it.
    if ((currentState === "idle" || currentState === "error") && modelReadyRef.current) {
      const session = (sessionRef.current += 1);
      startLogSession();
      log("session.start", { session, hotkeyHeld: hotkeyHeldRef.current });
      startingRef.current = true;
      pendingReleaseRef.current = false;
      silenceStartMsRef.current = 0;
      speechFramesRef.current = 0;
      isSpeakingRef.current = false;
      setIsSpeaking(false);
      setTranscript("");
      setShowTranscript(false);
      applyError(null);
      lastStreamErrorRef.current = null;
      streamingBusyRef.current = false;
      vadWarmupUntilRef.current = Date.now() + 1200; // 1.2s grace period for mic warm-up
      noiseFloorRef.current = Math.max(0.003, settingsRef.current.vad_silence_threshold / 3);
      streamingPendingRef.current = null;
      streamingSampleRateRef.current = 16000;
      discardStreamingRef.current = false;
      speechDetectedRef.current = false;

      await resizeInPlace(SIZE_PILL.w, SIZE_PILL.h);
      setState("listening");
      stateRef.current = "listening";

      // Pass microphone deviceId from settings
      try {
        await start(settingsRef.current.microphone_id || undefined);
        log("mic.started");
      } catch (e) {
        startingRef.current = false;
        pendingReleaseRef.current = false;
        log("mic.error", { message: formatError(e) });
        if (sessionRef.current !== session) return;
        await showErrorThenIdle(session, { kind: "mic", message: formatError(e) });
        return;
      }

      setTimeout(async () => {
        if (stateRef.current === "listening") {
          await resizeInPlace(SIZE_TALL.w, SIZE_TALL.h);
          setShowTranscript(true);
        }
      }, SHOW_TRANSCRIPT_DELAY);

      // ─── Start overlapping streaming (transcribe + type stable words) ───
      committedTextRef.current = "";
      streamingIntervalRef.current = setInterval(() => {
        if (stateRef.current !== "listening") return;
        if (streamingBusyRef.current) return;

        const buf = consumeBuffer();
        if (buf) {
          streamingSampleRateRef.current = buf.sampleRate;
          streamingPendingRef.current = appendAudio(streamingPendingRef.current, buf.samples);
        }

        const pending = streamingPendingRef.current;
        if (!pending) return;
        const sampleRate = streamingSampleRateRef.current;
        if (!speechDetectedRef.current) {
          const prerollSamples = Math.floor(sampleRate * STREAMING_PREROLL_DURATION);
          if (pending.length > prerollSamples) {
            streamingPendingRef.current = pending.slice(-prerollSamples);
          }
          return;
        }
        const windowSamples = Math.floor(sampleRate * STREAMING_WINDOW_DURATION);
        const overlapSamples = Math.floor(sampleRate * STREAMING_OVERLAP_DURATION);
        if (pending.length < windowSamples) return;

        const window = pending.slice(0, windowSamples);

        const task = (async () => {
          streamingBusyRef.current = true;
          try {
            const chunkText = await invoke<string>("transcribe_streaming", {
              samples: Array.from(window),
              sampleRate,
              prompt: committedTextRef.current.slice(-250),
            });
            log("stream.chunk", {
              seconds: Number((window.length / sampleRate).toFixed(2)),
              text: chunkText.trim().slice(0, 120),
            });
            const delta = mergeTranscript(committedTextRef.current, chunkText);
            if (delta && !discardStreamingRef.current) {
              const textToType = committedTextRef.current ? " " + delta : delta;

              // Type directly into active textbox
              await invoke("type_text", { text: textToType });

              // Track committed text
              committedTextRef.current += textToType;
              setTranscript(committedTextRef.current);
            }
            // Keep a short tail so the next window can recover words split at
            // the boundary without retranscribing the entire session.
            streamingPendingRef.current = pending.slice(
              Math.max(0, windowSamples - overlapSamples)
            );
            lastStreamErrorRef.current = null;
          } catch (e) {
            // Keep the window queued so a transient failure does not lose audio,
            // but remember why in case the session never produces any text.
            lastStreamErrorRef.current = formatError(e);
            log("stream.error", { message: lastStreamErrorRef.current });
          } finally {
            streamingBusyRef.current = false;
          }
        })();
        streamingTaskRef.current = task;
        void task.finally(() => {
          if (streamingTaskRef.current === task) streamingTaskRef.current = null;
        });
      }, STREAMING_CHUNK_MS);

      // If the shortcut plugin ever loses a release, nothing else would close
      // the mic now that VAD stands down while held.
      if (holdWatchdogRef.current) clearTimeout(holdWatchdogRef.current);
      holdWatchdogRef.current = setTimeout(() => {
        holdWatchdogRef.current = null;
        if (sessionRef.current !== session) return;
        if (stateRef.current !== "listening") return;
        log("hold.watchdog-stop", { maxHoldMs: MAX_HOLD_MS });
        hotkeyHeldRef.current = false;
        toggleRef.current();
      }, MAX_HOLD_MS);

      startingRef.current = false;

      // The key was let go while the mic was still opening — honour it now
      // instead of listening on with nobody holding anything.
      if (pendingReleaseRef.current) {
        pendingReleaseRef.current = false;
        log("hotkey.released-during-start", { session });
        void toggleRef.current();
      }

    } else if (currentState === "listening") {
      const session = sessionRef.current;
      log("session.stop", {
        session,
        hotkeyHeld: hotkeyHeldRef.current,
        speechDetected: speechDetectedRef.current,
        committedChars: committedTextRef.current.length,
      });

      if (holdWatchdogRef.current) {
        clearTimeout(holdWatchdogRef.current);
        holdWatchdogRef.current = null;
      }

      // ─── Stop streaming interval ───
      if (streamingIntervalRef.current) {
        clearInterval(streamingIntervalRef.current);
        streamingIntervalRef.current = null;
      }

      setState("processing");
      stateRef.current = "processing";

      // Wait for any in-progress streaming transcription to complete before
      // flushing the remaining audio, avoiding overlapping native inference.
      if (streamingTaskRef.current) {
        await streamingTaskRef.current;
      }

      // Stop audio capture — returns remaining audio since last consume
      const remainingAudio = await stop();
      if (remainingAudio) {
        streamingPendingRef.current = appendAudio(streamingPendingRef.current, remainingAudio.samples);
      }

      // Transcribe and type remaining audio
      const pendingFinal = streamingPendingRef.current;
      const finalSampleRate = remainingAudio?.sampleRate ?? streamingSampleRateRef.current;
      let finalError: string | null = null;
      if (speechDetectedRef.current
        && pendingFinal
        && pendingFinal.length > finalSampleRate * MIN_FINAL_CHUNK_DURATION) {
        try {
          const chunkText = await invoke<string>("transcribe_streaming", {
            samples: Array.from(pendingFinal),
            sampleRate: finalSampleRate,
            prompt: committedTextRef.current.slice(-250),
          });
          const delta = mergeTranscript(committedTextRef.current, chunkText);
          if (delta) {
            const textToType = committedTextRef.current ? " " + delta : delta;
            await invoke("type_text", { text: textToType });
            committedTextRef.current += textToType;
            setTranscript(committedTextRef.current);
          }
          streamingPendingRef.current = null;
        } catch (e) {
          console.error("Final chunk transcription error:", e);
          finalError = formatError(e);
          log("final.error", { message: finalError });
        }
      }
      log("session.done", { session, transcript: committedTextRef.current.slice(0, 200) });
      endLogSession();

      if (sessionRef.current !== session) return;

      // Report the final chunk's failure, or — when the session produced no text
      // at all — the streaming failure that has been retrying invisibly.
      const failure =
        finalError ??
        (committedTextRef.current ? null : lastStreamErrorRef.current);

      if (failure) {
        await showErrorThenIdle(session, { kind: "transcribe", message: failure });
      } else {
        setState("idle");
        stateRef.current = "idle";
        setShowTranscript(false);
        setIsSpeaking(false);
        await new Promise((r) => setTimeout(r, TRANSITION_MS));
        if (sessionRef.current !== session) return;
        await resizeInPlace(SIZE_IDLE.w, SIZE_IDLE.h);
      }

      if (sessionRef.current !== session) return;

      // Save pill position after settling back to idle
      savePillPosition();

      setTimeout(() => {
        if (sessionRef.current !== session) return;
        setTranscript("");
        committedTextRef.current = "";
        streamingPendingRef.current = null;
        discardStreamingRef.current = false;
      }, 2000);
    }
  }, [start, stop, consumeBuffer, applyError, showErrorThenIdle]);

  const toggleRef = useRef(handleToggle);
  useEffect(() => {
    toggleRef.current = handleToggle;
  }, [handleToggle]);

  const quitRef = useRef(handleQuit);
  useEffect(() => {
    quitRef.current = handleQuit;
  }, [handleQuit]);

  // ─── Save pill position (debounced) ───
  const savePillTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const savePillPosition = useCallback(() => {
    if (savePillTimerRef.current) clearTimeout(savePillTimerRef.current);
    savePillTimerRef.current = setTimeout(async () => {
      try {
        const win = getCurrentWindow();
        const pos = await win.outerPosition();
        const factor = await win.scaleFactor();
        const x = pos.x / factor;
        const y = pos.y / factor;
        await invoke("save_pill_position", { x, y });
      } catch {
        // Best effort
      }
    }, 500);
  }, []);

  // ─── Listen for window move (drag) to save position ───
  useEffect(() => {
    const win = getCurrentWindow();
    const unlisten = win.onMoved(() => {
      // Only save when idle (pill is at its natural position)
      if (stateRef.current === "idle") {
        savePillPosition();
      }
    });
    return () => {
      unlisten.then((fn) => fn());
    };
  }, [savePillPosition]);

  // ─── Start the local model download, reporting why it failed ───
  const startModelDownload = useCallback(() => {
    const s = settingsRef.current;
    setDownloadProgress(0);
    const downloadCmd =
      s.local_engine === "zipformer" ? "download_zipformer_model" : "download_model";
    invoke(downloadCmd)
      .then(() => setModelReady(true))
      .catch((e) => {
        console.error("Download error:", e);
        applyError({ kind: "download", message: `Download failed: ${formatError(e)}` });
      });
  }, [applyError]);

  // ─── Click the error pill: retry what can be retried, dismiss the rest ───
  const handleRetry = useCallback(() => {
    const current = errorRef.current;
    applyError(null);

    if (!current || !isRetryable(current)) return;
    // Cloud mode has nothing local to fetch.
    if (settingsRef.current.transcription_mode === "cloud") return;

    startModelDownload();
  }, [applyError, startModelDownload]);

  // ─── Register / re-register hotkeys ───
  const registerHotkeys = useCallback(async (s: AppSettings) => {
    // Unregister old ones first (best effort)
    try { await unregister("CommandOrControl+Shift+Space"); } catch {}
    try { await unregister("CommandOrControl+Shift+Q"); } catch {}
    // Also try to unregister the actual configured keys in case they differ
    try { await unregister(s.hotkey); } catch {}
    try { await unregister(s.quit_hotkey); } catch {}

    // Register hold-to-talk
    const failed: string[] = [];
    try {
      await register(s.hotkey, (event: any) => {
        if (event.state === "Pressed") {
          // Windows re-fires the shortcut while it is held down; only the first
          // press opens a session, the rest are auto-repeat.
          if (hotkeyHeldRef.current && (startingRef.current || stateRef.current === "listening")) {
            pressRepeatsRef.current += 1;
            return;
          }
          hotkeyHeldRef.current = true;
          pressRepeatsRef.current = 0;
          log("hotkey.pressed", { state: stateRef.current });
          if (stateRef.current === "idle" || stateRef.current === "error") {
            toggleRef.current();
          }
        } else if (event.state === "Released") {
          const wasHeld = hotkeyHeldRef.current;
          hotkeyHeldRef.current = false;
          log("hotkey.released", {
            state: stateRef.current,
            wasHeld,
            repeats: pressRepeatsRef.current,
            starting: startingRef.current,
          });
          pressRepeatsRef.current = 0;
          if (stateRef.current === "listening") {
            toggleRef.current();
          } else if (startingRef.current) {
            // The mic is still opening; handleToggle picks this up once it is up.
            pendingReleaseRef.current = true;
          }
        }
      });
    } catch (e) {
      console.error("Failed to register hold-to-talk hotkey:", e);
      failed.push(formatHotkey(s.hotkey));
    }

    // Register quit
    try {
      await register(s.quit_hotkey, (event: any) => {
        if (!event.state || event.state === "Pressed") {
          quitRef.current();
        }
      });
    } catch (e) {
      console.error("Failed to register quit hotkey:", e);
      failed.push(formatHotkey(s.quit_hotkey));
    }

    // A hotkey that silently fails to register leaves no way to talk at all,
    // so say so rather than logging it to a console nobody is watching.
    if (failed.length > 0) {
      applyError({
        kind: "hotkey",
        message: `Could not register ${failed.join(" and ")}. Another app may already use it — pick a different shortcut in Settings.`,
      });
    }
  }, [applyError]);

  // ─── Load settings + model on mount ───
  useEffect(() => {
    // Load settings first, then check model
    invoke<AppSettings>("get_settings")
      .then((s) => {
        setSettings(s);
        settingsRef.current = s;

        // Register hotkeys with loaded settings
        registerHotkeys(s).catch(console.error);

        // Check model readiness (cloud mode is always ready)
        if (s.transcription_mode === "cloud") {
          return true;
        }
        if (s.local_engine === "zipformer") {
          return invoke<boolean>("is_zipformer_model_ready");
        }
        return invoke<boolean>("is_model_ready");
      })
      .then((ready) => {
        setModelReady(ready);
        if (!ready && settingsRef.current.transcription_mode !== "cloud") {
          startModelDownload();
        }
      })
      .catch((e) => {
        console.error("Init error:", e);
        applyError({ kind: "init", message: `Could not start: ${formatError(e)}` });
        // Still register default hotkeys
        registerHotkeys(DEFAULT_SETTINGS).catch(console.error);
      });

    // Download progress listener
    const unlistenProgress = listen<number>("download-progress", (event) => {
      setDownloadProgress(event.payload);
    });

    // Settings changed listener (from settings window)
    const unlistenSettings = listen("settings-changed", async () => {
      try {
        const s = await invoke<AppSettings>("get_settings");
        const oldSettings = settingsRef.current;
        setSettings(s);
        settingsRef.current = s;

        const oldUsesWhisper = oldSettings.transcription_mode === "local"
          && oldSettings.local_engine === "whisper";
        const newUsesWhisper = s.transcription_mode === "local"
          && s.local_engine === "whisper";
        if (oldUsesWhisper && (!newUsesWhisper || s.model !== oldSettings.model)) {
          await invoke("stop_whisper_server");
        }

        // Re-register hotkeys if they changed
        if (s.hotkey !== oldSettings.hotkey || s.quit_hotkey !== oldSettings.quit_hotkey) {
          // Unregister old hotkeys
          try { await unregister(oldSettings.hotkey); } catch {}
          try { await unregister(oldSettings.quit_hotkey); } catch {}
          await registerHotkeys(s);
        }

        // Re-check model if model, mode, or engine changed
        if (s.model !== oldSettings.model || s.transcription_mode !== oldSettings.transcription_mode || s.local_engine !== oldSettings.local_engine) {
          if (s.transcription_mode === "cloud") {
            setModelReady(true);
          } else {
            const readyCmd = s.local_engine === "zipformer" ? "is_zipformer_model_ready" : "is_model_ready";
            const ready = await invoke<boolean>(readyCmd);
            setModelReady(ready);
            if (!ready) {
              startModelDownload();
            }
          }
        }
      } catch (e) {
        console.error("Failed to reload settings:", e);
      }
    });

    return () => {
      unlistenProgress.then((fn) => fn());
      unlistenSettings.then((fn) => fn());
      // Cleanup hotkeys
      const s = settingsRef.current;
      unregister(s.hotkey).catch(() => {});
      unregister(s.quit_hotkey).catch(() => {});
    };
  }, [registerHotkeys, applyError, startModelDownload]);

  // ─── Build CSS classes for #pill ───
  const pillClasses: string[] = [];

  if (!modelReady && !error) {
    pillClasses.push("downloading");
  }
  if (error && state === "idle") {
    pillClasses.push("has-error");
  }
  if (state === "listening") {
    pillClasses.push("expanded", "listening");
  }
  if (state === "processing") {
    pillClasses.push("expanded", "processing");
  }
  if (state === "error") {
    pillClasses.push("expanded");
  }
  if (showTranscript && state !== "idle") {
    pillClasses.push("show-transcript");
  }
  if (isSpeaking && state === "listening") {
    pillClasses.push("vad-active");
  }

  const downloadCircumference = 69.115;
  const downloadOffset = downloadCircumference * (1 - downloadProgress / 100);

  return (
    <div
      id="pill"
      className={pillClasses.join(" ")}
      data-tauri-drag-region
      title={
        error
          ? isRetryable(error)
            ? `${error.message} — click to retry`
            : `${error.message} — click to dismiss`
          : undefined
      }
      onClick={error && state === "idle" ? handleRetry : undefined}
    >
      {/* ─── Quit button ─── */}
      <button className="quit-btn" onClick={handleQuit} title={`Quit (${formatHotkey(settings.quit_hotkey)})`}>
        &#x2715;
      </button>

      {/* ─── Mic icon ─── */}
      <div className="mic-icon" data-tauri-drag-region>
        <svg viewBox="0 0 18 18" fill="none" data-tauri-drag-region>
          <rect x="6" y="1" width="6" height="10" rx="3" fill="var(--muted)" stroke="none" />
          <path
            d="M3 8a6 6 0 0 0 12 0"
            stroke="var(--muted)"
            strokeWidth="1.5"
            strokeLinecap="round"
            fill="none"
          />
          <line
            x1="9" y1="14" x2="9" y2="17"
            stroke="var(--muted)"
            strokeWidth="1.5"
            strokeLinecap="round"
          />
        </svg>
      </div>

      {/* ─── Download ring ─── */}
      <div className="download-bar" data-tauri-drag-region>
        <svg viewBox="0 0 28 28" data-tauri-drag-region>
          <circle className="download-ring-bg" cx="14" cy="14" r="11" />
          <circle
            className="download-ring-fg"
            cx="14"
            cy="14"
            r="11"
            style={{ strokeDashoffset: downloadOffset }}
          />
          <text className="download-pct" x="14" y="14">
            {Math.round(downloadProgress)}%
          </text>
        </svg>
      </div>

      {/* ─── Error indicator ─── */}
      <div className="error-icon" data-tauri-drag-region>
        <svg viewBox="0 0 18 18" fill="none" data-tauri-drag-region>
          <circle cx="9" cy="9" r="8" fill="none" stroke="var(--danger)" strokeWidth="1.5" />
          <line x1="9" y1="5" x2="9" y2="10" stroke="var(--danger)" strokeWidth="1.5" strokeLinecap="round" />
          <circle cx="9" cy="13" r="1" fill="var(--danger)" />
        </svg>
      </div>

      {/* ─── Inner row ─── */}
      <div className="inner-row" data-tauri-drag-region>
        <div className="canvas-wrap" data-tauri-drag-region>
          <Waveform analyserNode={analyserNode} onFrame={processVAD} />
          <div className="idle-bars" data-tauri-drag-region>
            <div className="ibar" />
            <div className="ibar" />
            <div className="ibar" />
            <div className="ibar" />
            <div className="ibar" />
          </div>
        </div>

        <svg className="vad-ring" viewBox="0 0 16 16" data-tauri-drag-region>
          <circle className="vad-bg" cx="8" cy="8" r="6" />
          <circle className="vad-fg" ref={vadArcRef} cx="8" cy="8" r="6" />
        </svg>

        <button className="close-btn" onClick={handleClose}>
          &#x2715;
        </button>
      </div>

      {/* ─── Processing overlay ─── */}
      <div className="processing-overlay" data-tauri-drag-region>
        <div className="spinner" data-tauri-drag-region />
        <span className="processing-text" data-tauri-drag-region>...</span>
      </div>

      {/* ─── Transcript row ─── */}
      <div className="transcript-row" ref={transcriptScrollRef} data-tauri-drag-region>
        <div className="tx" data-tauri-drag-region>
          {error && state !== "idle" ? (
            <span className="tx-error">{error.message}</span>
          ) : state === "processing" ? (
            <span className="tx-pending">finalizing...</span>
          ) : transcript && state === "listening" ? (
            <span className="tx-streaming">{transcript}<span className="tx-cursor">|</span></span>
          ) : transcript ? (
            transcript
          ) : (
            <span className="tx-interim">listening...</span>
          )}
        </div>
      </div>
    </div>
  );
}

export default App;
