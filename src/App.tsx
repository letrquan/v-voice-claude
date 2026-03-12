import { useState, useEffect, useCallback, useRef } from "react";
import { listen } from "@tauri-apps/api/event";
import { invoke } from "@tauri-apps/api/core";
import { register, unregister } from "@tauri-apps/plugin-global-shortcut";
import { getCurrentWindow } from "@tauri-apps/api/window";
import { LogicalSize, LogicalPosition } from "@tauri-apps/api/dpi";
import { Waveform } from "./components/Waveform";
import { useAudioCapture } from "./hooks/useAudioCapture";

type AppState = "idle" | "listening" | "processing";

const VAD = {
  SILENCE_THRESHOLD: 0.015,
  SILENCE_FRAMES: 45,
  MIN_SPEECH_FRAMES: 8,
  RING_CIRC: 38,
};

/**
 * Resize the window while keeping its visual center-x and bottom-y anchored.
 * This way, if the user drags the pill somewhere, expand/collapse happens
 * in-place instead of snapping back to screen center.
 */
async function resizeInPlace(width: number, height: number) {
  const win = getCurrentWindow();

  // Get current position + size so we can anchor around it
  const oldPos = await win.outerPosition();
  const oldSize = await win.outerSize();
  const factor = await win.scaleFactor();

  // Convert physical → logical
  const oldW = oldSize.width / factor;
  const oldH = oldSize.height / factor;
  const oldX = oldPos.x / factor;
  const oldY = oldPos.y / factor;

  // Anchor: keep center-x and bottom-y stable
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
  const stateRef = useRef<AppState>("idle");
  const modelReadyRef = useRef(false);
  const vadArcRef = useRef<SVGCircleElement>(null);

  // VAD state refs (updated every frame, not React state)
  const silentFramesRef = useRef(0);
  const speechFramesRef = useRef(0);
  const isSpeakingRef = useRef(false);

  const { start, stop, analyserNode } = useAudioCapture();

  useEffect(() => {
    stateRef.current = state;
  }, [state]);

  useEffect(() => {
    modelReadyRef.current = modelReady;
  }, [modelReady]);

  // VAD processing — runs on analyserNode from requestAnimationFrame inside Waveform
  const processVAD = useCallback(() => {
    if (!analyserNode) return;

    const dataArr = new Uint8Array(analyserNode.fftSize);
    analyserNode.getByteTimeDomainData(dataArr);

    let sum = 0;
    for (let i = 0; i < dataArr.length; i++) {
      const s = (dataArr[i] - 128) / 128;
      sum += s * s;
    }
    const rms = Math.sqrt(sum / dataArr.length);
    const speech = rms > VAD.SILENCE_THRESHOLD;

    if (speech) {
      speechFramesRef.current++;
      silentFramesRef.current = 0;
      if (!isSpeakingRef.current && speechFramesRef.current >= VAD.MIN_SPEECH_FRAMES) {
        isSpeakingRef.current = true;
        setIsSpeaking(true);
      }
    } else {
      silentFramesRef.current++;
      if (isSpeakingRef.current) speechFramesRef.current = 0;

      // Update VAD arc
      const prog = Math.min(silentFramesRef.current / VAD.SILENCE_FRAMES, 1);
      if (vadArcRef.current) {
        vadArcRef.current.style.strokeDashoffset = (VAD.RING_CIRC * (1 - prog)).toFixed(2);
      }

      if (isSpeakingRef.current && silentFramesRef.current >= VAD.SILENCE_FRAMES) {
        isSpeakingRef.current = false;
        silentFramesRef.current = 0;
        setIsSpeaking(false);
        if (vadArcRef.current) {
          vadArcRef.current.style.strokeDashoffset = String(VAD.RING_CIRC);
        }
      }
    }
  }, [analyserNode]);

  const handleToggle = useCallback(async () => {
    const currentState = stateRef.current;

    if (currentState === "idle" && modelReadyRef.current) {
      setState("listening");
      stateRef.current = "listening";

      // Reset VAD state
      silentFramesRef.current = 0;
      speechFramesRef.current = 0;
      isSpeakingRef.current = false;
      setIsSpeaking(false);
      setTranscript("");

      await start();

      // Expand pill
      await resizeInPlace(260, 48);

      // Show transcript row after a beat
      setTimeout(async () => {
        await resizeInPlace(260, 100);
      }, 300);
    } else if (currentState === "listening") {
      setState("processing");
      stateRef.current = "processing";
      const audioData = await stop();

      if (audioData && audioData.samples.length > 0) {
        try {
          const result = await invoke<string>("transcribe", {
            samples: Array.from(audioData.samples),
            sampleRate: audioData.sampleRate,
          });
          if (result.trim()) {
            setTranscript(result.trim());
            await new Promise((r) => setTimeout(r, 400));
            await invoke("type_text", { text: result.trim() });
          }
        } catch (e) {
          console.error("Transcription error:", e);
          setTranscript("Error: " + String(e));
        }
      }

      setState("idle");
      stateRef.current = "idle";

      // Collapse pill back
      await resizeInPlace(48, 48);

      setTimeout(() => setTranscript(""), 3000);
    }
  }, [start, stop]);

  const handleClose = useCallback(async () => {
    if (stateRef.current === "listening") {
      setState("idle");
      stateRef.current = "idle";
      await stop();
      silentFramesRef.current = 0;
      speechFramesRef.current = 0;
      isSpeakingRef.current = false;
      setIsSpeaking(false);
      await resizeInPlace(48, 48);
      setTimeout(() => setTranscript(""), 400);
    }
  }, [stop]);

  const toggleRef = useRef(handleToggle);
  useEffect(() => {
    toggleRef.current = handleToggle;
  }, [handleToggle]);

  // Download model on mount
  useEffect(() => {
    invoke<boolean>("is_model_ready")
      .then((ready) => {
        setModelReady(ready);
        if (!ready) {
          invoke("download_model")
            .then(() => setModelReady(true))
            .catch(console.error);
        }
      })
      .catch(console.error);

    const unlisten = listen<number>("download-progress", (event) => {
      setDownloadProgress(event.payload);
    });

    return () => {
      unlisten.then((fn) => fn());
    };
  }, []);

  // Register global shortcut once
  useEffect(() => {
    register("CommandOrControl+Shift+Space", (event: any) => {
      if (!event.state || event.state === "Pressed") {
        toggleRef.current();
      }
    }).catch(console.error);

    return () => {
      unregister("CommandOrControl+Shift+Space").catch(console.error);
    };
  }, []);

  // Build CSS classes for #pill
  const pillClasses = [
    state === "listening" ? "expanded listening" : "",
    state === "processing" ? "expanded processing" : "",
    state === "listening" && transcript ? "show-transcript" : "",
    // Show transcript row after 300ms — handled by resize, but CSS class needed
    state === "listening" ? "show-transcript" : "",
    isSpeaking ? "vad-active" : "",
  ]
    .filter(Boolean)
    .join(" ");

  // Download progress as ring (circumference of r=11 circle ≈ 69.115)
  const downloadCircumference = 69.115;
  const downloadOffset = downloadCircumference * (1 - downloadProgress / 100);

  return (
    <div id="pill" className={pillClasses} data-tauri-drag-region>
      {/* Idle mic icon — shown when not downloading and idle */}
      {modelReady && state === "idle" && (
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
      )}

      {/* Download progress ring — shown when model not ready */}
      {!modelReady && (
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
      )}

      {/* Expanded inner row */}
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

      {/* Transcript row */}
      <div className="transcript-row" data-tauri-drag-region>
        <div className="tx" data-tauri-drag-region>
          {state === "processing" ? (
            <span className="tx-pending">processing...</span>
          ) : transcript ? (
            transcript.slice(-80)
          ) : (
            <span className="tx-interim">listening...</span>
          )}
        </div>
      </div>

      {/* Processing spinner overlay (shown on pill when processing) */}
      {state === "processing" && (
        <div className="processing-overlay" data-tauri-drag-region>
          <div className="spinner" data-tauri-drag-region />
          <span className="processing-text" data-tauri-drag-region>...</span>
        </div>
      )}
    </div>
  );
}

export default App;
