import { useState, useEffect, useCallback } from "react";
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import { getCurrentWindow } from "@tauri-apps/api/window";

/* ─── Types (mirror Rust structs) ─── */

interface ModelInfo {
  id: string;
  filename: string;
  label: string;
  size_mb: number;
  url: string;
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
}

const LANGUAGES = [
  { code: "en", label: "English" },
  { code: "vi", label: "Vietnamese (Tiếng Việt)" },
  { code: "auto", label: "Auto-detect" },
  { code: "zh", label: "Chinese" },
  { code: "de", label: "German" },
  { code: "es", label: "Spanish" },
  { code: "ru", label: "Russian" },
  { code: "ko", label: "Korean" },
  { code: "fr", label: "French" },
  { code: "ja", label: "Japanese" },
  { code: "pt", label: "Portuguese" },
  { code: "tr", label: "Turkish" },
  { code: "pl", label: "Polish" },
  { code: "it", label: "Italian" },
  { code: "nl", label: "Dutch" },
  { code: "sv", label: "Swedish" },
  { code: "th", label: "Thai" },
  { code: "id", label: "Indonesian" },
  { code: "hi", label: "Hindi" },
  { code: "ar", label: "Arabic" },
];

/* ─── Hotkey capture helpers ─── */

const MODIFIER_KEYS = new Set([
  "Control",
  "Shift",
  "Alt",
  "Meta",
]);

const KEY_MAP: Record<string, string> = {
  Control: "CommandOrControl",
  Meta: "CommandOrControl",
  " ": "Space",
  ArrowUp: "Up",
  ArrowDown: "Down",
  ArrowLeft: "Left",
  ArrowRight: "Right",
};

function keyboardEventToAccelerator(e: KeyboardEvent): string | null {
  const parts: string[] = [];
  if (e.ctrlKey || e.metaKey) parts.push("CommandOrControl");
  if (e.shiftKey) parts.push("Shift");
  if (e.altKey) parts.push("Alt");

  // Need at least one modifier
  if (parts.length === 0) return null;

  const key = e.key;
  if (MODIFIER_KEYS.has(key)) return null; // only modifiers pressed so far

  const mapped = KEY_MAP[key] || key.toUpperCase();
  parts.push(mapped);
  return parts.join("+");
}

function formatHotkey(accel: string): string {
  return accel
    .replace("CommandOrControl", "Ctrl")
    .replace(/\+/g, " + ");
}

/* ─── Component ─── */

export default function SettingsPage() {
  const [settings, setSettings] = useState<AppSettings | null>(null);
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [downloadedModels, setDownloadedModels] = useState<string[]>([]);
  const [microphones, setMicrophones] = useState<MediaDeviceInfo[]>([]);
  const [downloadingModel, setDownloadingModel] = useState<string | null>(null);
  const [downloadProgress, setDownloadProgress] = useState(0);
  const [capturingField, setCapturingField] = useState<"hotkey" | "quit_hotkey" | null>(null);
  const [saved, setSaved] = useState(false);
  const [dirty, setDirty] = useState(false);
  const [showApiKey, setShowApiKey] = useState(false);

  // ─── Load initial data ───
  useEffect(() => {
    Promise.all([
      invoke<AppSettings>("get_settings"),
      invoke<ModelInfo[]>("get_available_models"),
      invoke<string[]>("get_downloaded_models"),
    ]).then(([s, m, d]) => {
      setSettings(s);
      setModels(m);
      setDownloadedModels(d);
    });

    // Enumerate microphones
    navigator.mediaDevices
      .getUserMedia({ audio: true })
      .then((stream) => {
        stream.getTracks().forEach((t) => t.stop());
        return navigator.mediaDevices.enumerateDevices();
      })
      .then((devices) => {
        setMicrophones(devices.filter((d) => d.kind === "audioinput"));
      })
      .catch(() => {});

    // Listen for download progress
    const unlisten = listen<number>("download-progress", (e) => {
      setDownloadProgress(e.payload);
    });
    return () => {
      unlisten.then((fn) => fn());
    };
  }, []);

  // ─── Hotkey capture ───
  useEffect(() => {
    if (!capturingField || !settings) return;

    const handler = (e: KeyboardEvent) => {
      e.preventDefault();
      e.stopPropagation();

      // Escape cancels
      if (e.key === "Escape") {
        setCapturingField(null);
        return;
      }

      const accel = keyboardEventToAccelerator(e);
      if (accel) {
        setSettings({ ...settings, [capturingField]: accel });
        setCapturingField(null);
        setDirty(true);
      }
    };

    window.addEventListener("keydown", handler, true);
    return () => window.removeEventListener("keydown", handler, true);
  }, [capturingField, settings]);

  // ─── Updater helper ───
  const update = useCallback(
    (patch: Partial<AppSettings>) => {
      if (!settings) return;
      const merged = { ...settings, ...patch };

      // Auto-switch from English-only model to multilingual when selecting a non-English language
      if (patch.language && patch.language !== "en" && merged.model.endsWith(".en")) {
        const multilingualModel = merged.model.replace(".en", "");
        // Check if the multilingual variant is available
        const hasMultilingual = models.some((m) => m.id === multilingualModel);
        if (hasMultilingual) {
          merged.model = multilingualModel;
        }
      }

      setSettings(merged);
      setDirty(true);
      setSaved(false);
    },
    [settings, models]
  );

  // ─── Save ───
  const handleSave = useCallback(async () => {
    if (!settings) return;
    try {
      await invoke("set_settings", { settings });
      setDirty(false);
      setSaved(true);
      setTimeout(() => setSaved(false), 2000);
    } catch (e) {
      console.error("Failed to save settings:", e);
    }
  }, [settings]);

  // ─── Download a model ───
  const handleDownloadModel = useCallback(
    async (modelId: string) => {
      setDownloadingModel(modelId);
      setDownloadProgress(0);
      try {
        await invoke("download_specific_model", { modelId });
        const updated = await invoke<string[]>("get_downloaded_models");
        setDownloadedModels(updated);
      } catch (e) {
        console.error("Model download failed:", e);
      } finally {
        setDownloadingModel(null);
      }
    },
    []
  );

  // ─── Delete a model ───
  const handleDeleteModel = useCallback(
    async (modelId: string) => {
      try {
        await invoke("delete_model", { modelId });
        const updated = await invoke<string[]>("get_downloaded_models");
        setDownloadedModels(updated);
      } catch (e) {
        console.error("Model delete failed:", e);
      }
    },
    []
  );

  if (!settings) {
    return <div className="settings-page"><p>Loading...</p></div>;
  }

  const isEnModel = settings.model.endsWith(".en");

  return (
    <div className="settings-page">
      <div className="settings-container">
        <h1>V Voice Settings</h1>

        {/* ─── Transcription Mode ─── */}
        <section className="settings-section">
          <h2>Transcription Engine</h2>
          <div className="mode-toggle">
            <button
              className={`mode-btn ${settings.transcription_mode === "local" ? "active" : ""}`}
              onClick={() => update({ transcription_mode: "local" })}
            >
              <span className="mode-icon">💻</span>
              <span className="mode-label">Local</span>
              <span className="mode-desc">Whisper on your machine</span>
            </button>
            <button
              className={`mode-btn ${settings.transcription_mode === "cloud" ? "active" : ""}`}
              onClick={() => update({ transcription_mode: "cloud" })}
            >
              <span className="mode-icon">☁️</span>
              <span className="mode-label">Cloud</span>
              <span className="mode-desc">OpenAI / Groq API</span>
            </button>
          </div>
        </section>

        {/* ─── Cloud Config (only shown in cloud mode) ─── */}
        {settings.transcription_mode === "cloud" && (
          <section className="settings-section">
            <h2>Cloud Provider</h2>
            <div className="provider-grid">
              <button
                className={`provider-card ${settings.cloud_provider === "openai" ? "active" : ""}`}
                onClick={() => update({ cloud_provider: "openai" })}
              >
                <span className="provider-name">OpenAI</span>
                <span className="provider-desc">Whisper API · Accurate</span>
              </button>
              <button
                className={`provider-card ${settings.cloud_provider === "groq" ? "active" : ""}`}
                onClick={() => update({ cloud_provider: "groq" })}
              >
                <span className="provider-name">Groq</span>
                <span className="provider-desc">Whisper v3 Turbo · Fast</span>
              </button>
            </div>

            <div style={{ marginTop: 16 }}>
              <h2>API Key</h2>
              <div className="api-key-row">
                <input
                  type={showApiKey ? "text" : "password"}
                  className="api-key-input"
                  value={settings.cloud_api_key}
                  onChange={(e) => update({ cloud_api_key: e.target.value })}
                  placeholder={settings.cloud_provider === "groq" ? "gsk_..." : "sk-..."}
                  spellCheck={false}
                  autoComplete="off"
                />
                <button
                  className="btn btn-sm btn-select"
                  onClick={() => setShowApiKey(!showApiKey)}
                  style={{ flexShrink: 0 }}
                >
                  {showApiKey ? "Hide" : "Show"}
                </button>
              </div>
              <p className="hint" style={{ marginTop: 6 }}>
                {settings.cloud_provider === "groq"
                  ? "Get your free API key at console.groq.com"
                  : "Get your API key at platform.openai.com"}
              </p>
            </div>
          </section>
        )}

        {/* ─── Model Selection (only in local mode) ─── */}
        {settings.transcription_mode === "local" && (
          <section className="settings-section">
            <h2>Whisper Model</h2>
            <div className="model-grid">
              {models.map((m) => {
                const downloaded = downloadedModels.includes(m.id);
                const isActive = settings.model === m.id;
                const isDownloading = downloadingModel === m.id;

                return (
                  <div
                    key={m.id}
                    className={`model-card ${isActive ? "active" : ""} ${downloaded ? "downloaded" : ""}`}
                  >
                    <div className="model-card-header">
                      <span className="model-label">{m.label}</span>
                      <span className="model-size">{m.size_mb < 1000 ? `${m.size_mb} MB` : `${(m.size_mb / 1000).toFixed(1)} GB`}</span>
                    </div>
                    <div className="model-card-actions">
                      {downloaded ? (
                        <>
                          <button
                            className={`btn btn-sm ${isActive ? "btn-active" : "btn-select"}`}
                            onClick={() => update({ model: m.id })}
                            disabled={isActive}
                          >
                            {isActive ? "Active" : "Select"}
                          </button>
                          {!isActive && (
                            <button
                              className="btn btn-sm btn-danger"
                              onClick={() => handleDeleteModel(m.id)}
                            >
                              Delete
                            </button>
                          )}
                        </>
                      ) : isDownloading ? (
                        <div className="download-bar-inline">
                          <div
                            className="download-bar-fill"
                            style={{ width: `${downloadProgress}%` }}
                          />
                          <span>{Math.round(downloadProgress)}%</span>
                        </div>
                      ) : (
                        <button
                          className="btn btn-sm btn-download"
                          onClick={() => handleDownloadModel(m.id)}
                          disabled={downloadingModel !== null}
                        >
                          Download
                        </button>
                      )}
                    </div>
                  </div>
                );
              })}
            </div>
          </section>
        )}

        {/* ─── Language ─── */}
        <section className="settings-section">
          <h2>Language</h2>
          {isEnModel && settings.language === "en" && (
            <p className="hint">Selecting a non-English language will auto-switch to a multilingual model.</p>
          )}
          <select
            value={settings.language}
            onChange={(e) => update({ language: e.target.value })}
          >
            {LANGUAGES.map((l) => (
              <option key={l.code} value={l.code}>{l.label}</option>
            ))}
          </select>
          {settings.language === "vi" && (
            <p className="hint" style={{ marginTop: 8 }}>
              💡 For best Vietnamese accuracy, use the <strong>Small</strong> or <strong>Medium</strong> multilingual model.
            </p>
          )}
        </section>

        {/* ─── Microphone ─── */}
        <section className="settings-section">
          <h2>Microphone</h2>
          <select
            value={settings.microphone_id}
            onChange={(e) => update({ microphone_id: e.target.value })}
          >
            <option value="">System Default</option>
            {microphones.map((mic) => (
              <option key={mic.deviceId} value={mic.deviceId}>
                {mic.label || `Microphone ${mic.deviceId.slice(0, 8)}`}
              </option>
            ))}
          </select>
        </section>

        {/* ─── Hotkeys ─── */}
        <section className="settings-section">
          <h2>Hotkeys</h2>
          <div className="hotkey-row">
            <label>Hold-to-talk</label>
            <button
              className={`hotkey-btn ${capturingField === "hotkey" ? "capturing" : ""}`}
              onClick={() => setCapturingField("hotkey")}
            >
              {capturingField === "hotkey" ? "Press keys..." : formatHotkey(settings.hotkey)}
            </button>
          </div>
          <div className="hotkey-row">
            <label>Quit</label>
            <button
              className={`hotkey-btn ${capturingField === "quit_hotkey" ? "capturing" : ""}`}
              onClick={() => setCapturingField("quit_hotkey")}
            >
              {capturingField === "quit_hotkey" ? "Press keys..." : formatHotkey(settings.quit_hotkey)}
            </button>
          </div>
        </section>

        {/* ─── Footer ─── */}
        <div className="settings-footer">
          <button className="btn btn-save" onClick={handleSave} disabled={!dirty}>
            {saved ? "Saved!" : "Save"}
          </button>
          <button className="btn btn-cancel" onClick={() => getCurrentWindow().close()}>
            Close
          </button>
        </div>
      </div>
    </div>
  );
}
