import { useState, useEffect, useCallback, useRef } from "react";
import { listen } from "@tauri-apps/api/event";
import { invoke } from "@tauri-apps/api/core";
import { register, unregister } from "@tauri-apps/plugin-global-shortcut";
import { getCurrentWindow } from "@tauri-apps/api/window";
import { LogicalSize, LogicalPosition } from "@tauri-apps/api/dpi";
import { Waveform } from "./components/Waveform";
import { useAudioCapture } from "./hooks/useAudioCapture";

type AppState = "idle" | "listening" | "processing";

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
};

const VAD_DEFAULTS = {
  MIN_SPEECH_FRAMES: 8,
  RING_CIRC: 38,
};

/* ─── Sizes ─── */
const SIZE_IDLE = { w: 48, h: 48 };
const SIZE_PILL = { w: 260, h: 48 };
const SIZE_TALL = { w: 260, h: 100 };

/* ─── Timing ─── */
const TRANSITION_MS = 420;
const SHOW_TRANSCRIPT_DELAY = 300;
const STREAMING_CHUNK_MS = 3500; // consume + transcribe every 3.5s
const MIN_CHUNK_DURATION = 0.75; // minimum seconds of audio per chunk
const MIN_FINAL_CHUNK_DURATION = 0.25; // minimum seconds for the final chunk

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
  const [errorMsg, setErrorMsg] = useState("");
  const [settings, setSettings] = useState<AppSettings>(DEFAULT_SETTINGS);

  const stateRef = useRef<AppState>("idle");
  const modelReadyRef = useRef(false);
  const settingsRef = useRef<AppSettings>(DEFAULT_SETTINGS);
  const vadArcRef = useRef<SVGCircleElement>(null);

  // VAD state refs
  const silentFramesRef = useRef(0);
  const speechFramesRef = useRef(0);
  const isSpeakingRef = useRef(false);
  const vadWarmupUntilRef = useRef(0); // timestamp: ignore VAD auto-stop until this time

  const { start, stop, consumeBuffer, analyserNode } = useAudioCapture();

  // Streaming transcription refs
  const streamingIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const streamingBusyRef = useRef(false);
  const committedTextRef = useRef("");

  useEffect(() => {
    stateRef.current = state;
  }, [state]);

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

    // Skip VAD during warm-up period (mic initialization produces noise)
    const now = Date.now();
    if (now < vadWarmupUntilRef.current) return;

    const speech = rms > s.vad_silence_threshold;

    if (speech) {
      speechFramesRef.current++;
      silentFramesRef.current = 0;
      if (!isSpeakingRef.current && speechFramesRef.current >= VAD_DEFAULTS.MIN_SPEECH_FRAMES) {
        isSpeakingRef.current = true;
        setIsSpeaking(true);
      }
    } else {
      silentFramesRef.current++;
      if (isSpeakingRef.current) speechFramesRef.current = 0;

      const prog = Math.min(silentFramesRef.current / s.vad_silence_frames, 1);
      if (vadArcRef.current) {
        vadArcRef.current.style.strokeDashoffset = (VAD_DEFAULTS.RING_CIRC * (1 - prog)).toFixed(2);
      }

      if (isSpeakingRef.current && silentFramesRef.current >= s.vad_silence_frames) {
        isSpeakingRef.current = false;
        silentFramesRef.current = 0;
        setIsSpeaking(false);
        if (vadArcRef.current) {
          vadArcRef.current.style.strokeDashoffset = String(VAD_DEFAULTS.RING_CIRC);
        }
        if (stateRef.current === "listening") {
          toggleRef.current();
        }
      }
    }
  }, [analyserNode]);

  // ─── Close button (cancel listening) ───
  const handleClose = useCallback(async () => {
    if (stateRef.current === "listening") {
      // Stop streaming
      if (streamingIntervalRef.current) {
        clearInterval(streamingIntervalRef.current);
        streamingIntervalRef.current = null;
      }
      streamingBusyRef.current = false;
      committedTextRef.current = "";

      await stop();
      setState("idle");
      stateRef.current = "idle";
      setShowTranscript(false);
      setIsSpeaking(false);
      silentFramesRef.current = 0;
      speechFramesRef.current = 0;
      isSpeakingRef.current = false;
      await new Promise((r) => setTimeout(r, TRANSITION_MS));
      await resizeInPlace(SIZE_IDLE.w, SIZE_IDLE.h);
      setTranscript("");
    }
  }, [stop]);

  // ─── Quit app ───
  const handleQuit = useCallback(async () => {
    if (stateRef.current === "listening") {
      await stop();
    }
    await getCurrentWindow().close();
  }, [stop]);

  // ─── Toggle: idle -> listening -> processing -> idle ───
  const handleToggle = useCallback(async () => {
    const currentState = stateRef.current;

    if (currentState === "idle" && modelReadyRef.current) {
      silentFramesRef.current = 0;
      speechFramesRef.current = 0;
      isSpeakingRef.current = false;
      setIsSpeaking(false);
      setTranscript("");
      setShowTranscript(false);
      setErrorMsg("");
      streamingBusyRef.current = false;
      vadWarmupUntilRef.current = Date.now() + 1200; // 1.2s grace period for mic warm-up

      await resizeInPlace(SIZE_PILL.w, SIZE_PILL.h);
      setState("listening");
      stateRef.current = "listening";

      // Pass microphone deviceId from settings
      await start(settingsRef.current.microphone_id || undefined);

      setTimeout(async () => {
        if (stateRef.current === "listening") {
          await resizeInPlace(SIZE_TALL.w, SIZE_TALL.h);
          setShowTranscript(true);
        }
      }, SHOW_TRANSCRIPT_DELAY);

      // ─── Start chunked streaming (transcribe + type directly) ───
      committedTextRef.current = "";
      streamingIntervalRef.current = setInterval(async () => {
        if (stateRef.current !== "listening") return;
        if (streamingBusyRef.current) return;

        const buf = consumeBuffer();
        if (!buf || buf.samples.length < buf.sampleRate * MIN_CHUNK_DURATION) return;

        streamingBusyRef.current = true;
        try {
          const chunkText = await invoke<string>("transcribe_streaming", {
            samples: Array.from(buf.samples),
            sampleRate: buf.sampleRate,
            prompt: committedTextRef.current,
          });
          if (chunkText && chunkText.trim() && stateRef.current === "listening") {
            const trimmed = chunkText.trim();
            const textToType = committedTextRef.current ? " " + trimmed : trimmed;

            // Type directly into active textbox
            await invoke("type_text", { text: textToType });

            // Track committed text
            committedTextRef.current += textToType;
            setTranscript(committedTextRef.current);
          }
        } catch {
          // Non-critical, skip this chunk
        } finally {
          streamingBusyRef.current = false;
        }
      }, STREAMING_CHUNK_MS);

    } else if (currentState === "listening") {
      // ─── Stop streaming interval ───
      if (streamingIntervalRef.current) {
        clearInterval(streamingIntervalRef.current);
        streamingIntervalRef.current = null;
      }

      setState("processing");
      stateRef.current = "processing";

      // Wait for any in-progress streaming transcription to complete
      let waitCount = 0;
      while (streamingBusyRef.current && waitCount < 50) {
        await new Promise((r) => setTimeout(r, 100));
        waitCount++;
      }
      streamingBusyRef.current = false;

      // Stop audio capture — returns remaining audio since last consume
      const remainingAudio = await stop();

      // Transcribe and type remaining audio
      if (remainingAudio && remainingAudio.samples.length > remainingAudio.sampleRate * MIN_FINAL_CHUNK_DURATION) {
        try {
          const chunkText = await invoke<string>("transcribe_streaming", {
            samples: Array.from(remainingAudio.samples),
            sampleRate: remainingAudio.sampleRate,
            prompt: committedTextRef.current,
          });
          if (chunkText && chunkText.trim()) {
            const trimmed = chunkText.trim();
            const textToType = committedTextRef.current ? " " + trimmed : trimmed;
            await invoke("type_text", { text: textToType });
            committedTextRef.current += textToType;
            setTranscript(committedTextRef.current);
          }
        } catch (e) {
          console.error("Final chunk transcription error:", e);
          setErrorMsg(String(e));
        }
      }

      setState("idle");
      stateRef.current = "idle";
      setShowTranscript(false);
      setIsSpeaking(false);
      await new Promise((r) => setTimeout(r, TRANSITION_MS));
      await resizeInPlace(SIZE_IDLE.w, SIZE_IDLE.h);

      // Save pill position after settling back to idle
      savePillPosition();

      setTimeout(() => {
        setTranscript("");
        committedTextRef.current = "";
      }, 2000);
    }
  }, [start, stop, consumeBuffer]);

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

  // ─── Retry download ───
  const handleRetry = useCallback(() => {
    setErrorMsg("");
    setDownloadProgress(0);
    invoke("download_model")
      .then(() => setModelReady(true))
      .catch((e) => {
        console.error("Retry download error:", e);
        setErrorMsg("Download failed. Click to retry.");
      });
  }, []);

  // ─── Register / re-register hotkeys ───
  const registerHotkeys = useCallback(async (s: AppSettings) => {
    // Unregister old ones first (best effort)
    try { await unregister("CommandOrControl+Shift+Space"); } catch {}
    try { await unregister("CommandOrControl+Shift+Q"); } catch {}
    // Also try to unregister the actual configured keys in case they differ
    try { await unregister(s.hotkey); } catch {}
    try { await unregister(s.quit_hotkey); } catch {}

    // Register hold-to-talk
    await register(s.hotkey, (event: any) => {
      if (event.state === "Pressed") {
        if (stateRef.current === "idle") {
          toggleRef.current();
        }
      } else if (event.state === "Released") {
        if (stateRef.current === "listening") {
          toggleRef.current();
        }
      }
    });

    // Register quit
    await register(s.quit_hotkey, (event: any) => {
      if (!event.state || event.state === "Pressed") {
        quitRef.current();
      }
    });
  }, []);

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
        return invoke<boolean>("is_model_ready");
      })
      .then((ready) => {
        setModelReady(ready);
        if (!ready && settingsRef.current.transcription_mode !== "cloud") {
          invoke("download_model")
            .then(() => setModelReady(true))
            .catch((e) => {
              console.error("Download error:", e);
              setErrorMsg("Download failed. Click to retry.");
            });
        }
      })
      .catch((e) => {
        console.error("Init error:", e);
        setErrorMsg("Failed to initialize. Click to retry.");
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

        // Re-register hotkeys if they changed
        if (s.hotkey !== oldSettings.hotkey || s.quit_hotkey !== oldSettings.quit_hotkey) {
          // Unregister old hotkeys
          try { await unregister(oldSettings.hotkey); } catch {}
          try { await unregister(oldSettings.quit_hotkey); } catch {}
          await registerHotkeys(s);
        }

        // Re-check model if model or mode changed
        if (s.model !== oldSettings.model || s.transcription_mode !== oldSettings.transcription_mode) {
          if (s.transcription_mode === "cloud") {
            setModelReady(true);
          } else {
            const ready = await invoke<boolean>("is_model_ready");
            setModelReady(ready);
            if (!ready) {
              invoke("download_model")
                .then(() => setModelReady(true))
                .catch((e) => {
                  console.error("Download error:", e);
                  setErrorMsg("Download failed. Click to retry.");
                });
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
  }, [registerHotkeys]);

  // ─── Build CSS classes for #pill ───
  const pillClasses: string[] = [];

  if (!modelReady && !errorMsg) {
    pillClasses.push("downloading");
  }
  if (errorMsg && state === "idle") {
    pillClasses.push("has-error");
  }
  if (state === "listening") {
    pillClasses.push("expanded", "listening");
  }
  if (state === "processing") {
    pillClasses.push("expanded", "processing");
  }
  if (showTranscript && (state === "listening" || state === "processing")) {
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
      onClick={errorMsg && state === "idle" ? handleRetry : undefined}
    >
      {/* ─── Quit button ─── */}
      <button className="quit-btn" onClick={handleQuit} title={`Quit (${settings.quit_hotkey.replace("CommandOrControl", "Ctrl").replace(/\+/g, "+")})`}>
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
      <div className="transcript-row" data-tauri-drag-region>
        <div className="tx" data-tauri-drag-region>
          {errorMsg && state !== "idle" ? (
            <span className="tx-error">{errorMsg}</span>
          ) : state === "processing" ? (
            <span className="tx-pending">finalizing...</span>
          ) : transcript && state === "listening" ? (
            <span className="tx-streaming">{transcript.slice(-80)}<span className="tx-cursor">|</span></span>
          ) : transcript ? (
            transcript.slice(-80)
          ) : (
            <span className="tx-interim">listening...</span>
          )}
        </div>
      </div>
    </div>
  );
}

export default App;
