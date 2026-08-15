import { useState, useEffect, useCallback } from "react";
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import { getCurrentWindow } from "@tauri-apps/api/window";
import { captureHotkey, formatHotkey } from "../lib/hotkey";

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
  local_engine: string;
  granite_api_port: number;
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

/* ─── Component ─── */

export default function SettingsPage() {
  const [settings, setSettings] = useState<AppSettings | null>(null);
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [downloadedModels, setDownloadedModels] = useState<string[]>([]);
  const [microphones, setMicrophones] = useState<MediaDeviceInfo[]>([]);
  const [downloadingModel, setDownloadingModel] = useState<string | null>(null);
  const [downloadProgress, setDownloadProgress] = useState(0);
  const [capturingField, setCapturingField] = useState<"hotkey" | "quit_hotkey" | null>(null);
  const [hotkeyHint, setHotkeyHint] = useState<string | null>(null);
  const [saved, setSaved] = useState(false);
  const [dirty, setDirty] = useState(false);
  const [showApiKey, setShowApiKey] = useState(false);
  const [zipformerReady, setZipformerReady] = useState(false);
  const [downloadingZipformer, setDownloadingZipformer] = useState(false);

  // ─── Load initial data ───
  useEffect(() => {
    Promise.all([
      invoke<AppSettings>("get_settings"),
      invoke<ModelInfo[]>("get_available_models"),
      invoke<string[]>("get_downloaded_models"),
      invoke<boolean>("is_zipformer_model_ready"),
    ]).then(([s, m, d, zf]) => {
      setSettings(s);
      setModels(m);
      setDownloadedModels(d);
      setZipformerReady(zf);
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
        setHotkeyHint(null);
        return;
      }

      const captured = captureHotkey(e);
      if (captured.kind === "unsupported") {
        // Keep listening — the user just has to pick a key we can register.
        setHotkeyHint("That key can't be used in a shortcut. Try a letter, number, or function key.");
        return;
      }
      if (captured.kind === "accelerator") {
        setSettings({ ...settings, [capturingField]: captured.accelerator });
        setCapturingField(null);
        setHotkeyHint(null);
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

  // ─── Download Zipformer model ───
  const handleDownloadZipformer = useCallback(async () => {
    setDownloadingZipformer(true);
    setDownloadProgress(0);
    try {
      await invoke("download_zipformer_model");
      const ready = await invoke<boolean>("is_zipformer_model_ready");
      setZipformerReady(ready);
    } catch (e) {
      console.error("Zipformer download failed:", e);
    } finally {
      setDownloadingZipformer(false);
    }
  }, []);

  if (!settings) {
    return <div className="settings-page"><p>Loading...</p></div>;
  }

  const isEnModel = settings.model.endsWith(".en");

  return (
    <div className="settings-page">
      <div className="settings-container">
        <header className="settings-header">
          <div className="settings-brand" aria-hidden="true">
            <svg viewBox="0 0 24 24" fill="none">
              <rect x="9" y="3" width="6" height="11" rx="3" />
              <path d="M6 11a6 6 0 0 0 12 0M12 17v4M9 21h6" />
            </svg>
          </div>
          <div>
            <p className="settings-eyebrow">V Voice</p>
            <h1>Settings</h1>
            <p className="settings-subtitle">
              Choose how your voice is captured, transcribed, and typed.
            </p>
          </div>
        </header>

        {/* ─── Transcription Mode ─── */}
        <section className="settings-section settings-section-featured">
          <div className="section-heading">
            <div>
              <span className="section-index">01</span>
              <h2>Transcription engine</h2>
            </div>
            <p>Pick the balance of privacy, speed, and accuracy that suits you.</p>
          </div>
          <div className="mode-toggle">
            <button
              type="button"
              className={`mode-btn ${settings.transcription_mode === "local" && settings.local_engine === "whisper" ? "active" : ""}`}
              onClick={() => update({ transcription_mode: "local", local_engine: "whisper" })}
              aria-pressed={settings.transcription_mode === "local" && settings.local_engine === "whisper"}
            >
              <span className="mode-icon">💻</span>
              <span className="mode-label">Local</span>
              <span className="mode-desc">Whisper on your machine</span>
            </button>
            <button
              type="button"
              className={`mode-btn ${settings.transcription_mode === "cloud" ? "active" : ""}`}
              onClick={() => update({ transcription_mode: "cloud" })}
              aria-pressed={settings.transcription_mode === "cloud"}
            >
              <span className="mode-icon">☁️</span>
              <span className="mode-label">Cloud</span>
              <span className="mode-desc">OpenAI / Groq API</span>
            </button>
            <button
              type="button"
              className={`mode-btn ${settings.transcription_mode === "local" && settings.local_engine === "zipformer" ? "active" : ""}`}
              onClick={() => update({ transcription_mode: "local", local_engine: "zipformer", language: "vi" })}
              aria-pressed={settings.transcription_mode === "local" && settings.local_engine === "zipformer"}
            >
              <span className="mode-icon">🇻🇳</span>
              <span className="mode-label">Zipformer</span>
              <span className="mode-desc">Vietnamese · Ultra-fast</span>
            </button>
          </div>
        </section>

        {/* ─── Cloud Config (only shown in cloud mode) ─── */}
        {settings.transcription_mode === "cloud" && (
          <section className="settings-section">
            <div className="section-heading">
              <div>
                <span className="section-index">02</span>
                <h2>Cloud provider</h2>
              </div>
              <p>Your API key is stored locally in the app settings.</p>
            </div>
            <div className="provider-grid">
              <button
                type="button"
                className={`provider-card ${settings.cloud_provider === "openai" ? "active" : ""}`}
                onClick={() => update({ cloud_provider: "openai" })}
                aria-pressed={settings.cloud_provider === "openai"}
              >
                <span className="provider-name">OpenAI</span>
                <span className="provider-desc">Whisper API · Accurate</span>
              </button>
              <button
                type="button"
                className={`provider-card ${settings.cloud_provider === "groq" ? "active" : ""}`}
                onClick={() => update({ cloud_provider: "groq" })}
                aria-pressed={settings.cloud_provider === "groq"}
              >
                <span className="provider-name">Groq</span>
                <span className="provider-desc">Whisper v3 Turbo · Fast</span>
              </button>
            </div>

            <div className="api-key-field">
              <label htmlFor="cloud-api-key">API key</label>
              <div className="api-key-row">
                <input
                  id="cloud-api-key"
                  type={showApiKey ? "text" : "password"}
                  className="api-key-input"
                  value={settings.cloud_api_key}
                  onChange={(e) => update({ cloud_api_key: e.target.value })}
                  placeholder={settings.cloud_provider === "groq" ? "gsk_..." : "sk-..."}
                  spellCheck={false}
                  autoComplete="off"
                />
                <button
                  type="button"
                  className="btn btn-sm btn-select"
                  onClick={() => setShowApiKey(!showApiKey)}
                  aria-label={showApiKey ? "Hide API key" : "Show API key"}
                >
                  {showApiKey ? "Hide" : "Show"}
                </button>
              </div>
              <p className="hint api-key-hint">
                {settings.cloud_provider === "groq"
                  ? "Get your free API key at console.groq.com"
                  : "Get your API key at platform.openai.com"}
              </p>
            </div>
          </section>
        )}

        {/* ─── Model Selection (only in whisper local mode) ─── */}
        {settings.transcription_mode === "local" && settings.local_engine === "whisper" && (
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

        {/* ─── Zipformer Model Download (only in zipformer mode) ─── */}
        {settings.transcription_mode === "local" && settings.local_engine === "zipformer" && (
          <section className="settings-section">
            <h2>Zipformer Vietnamese Model</h2>
            <div className={`model-card active ${zipformerReady ? "downloaded" : ""}`}>
              <div className="model-card-header">
                <span className="model-label">Zipformer 30M (Vietnamese)</span>
                <span className="model-size">~30 MB (int8)</span>
              </div>
              <div className="model-card-actions">
                {zipformerReady ? (
                  <button className="btn btn-sm btn-active" disabled>
                    ✅ Ready
                  </button>
                ) : downloadingZipformer ? (
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
                    onClick={handleDownloadZipformer}
                  >
                    Download
                  </button>
                )}
              </div>
            </div>
            <p className="hint">
              🏆 VLSP 2025 winner · 40× faster than Whisper · Trained on 6000h Vietnamese speech
            </p>
          </section>
        )}

        {/* ─── Language ─── */}
        <section className="settings-section">
          <div className="section-heading compact">
            <div><span className="section-index">03</span><h2>Language</h2></div>
          </div>
          {settings.local_engine === "zipformer" && settings.transcription_mode === "local" ? (
            <p className="hint">🇻🇳 Zipformer engine only supports Vietnamese. Switch to Whisper or Cloud for other languages.</p>
          ) : (
            <>
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
              {settings.language === "vi" && settings.local_engine !== "zipformer" && (
                <p className="hint" style={{ marginTop: 8 }}>
                  💡 For best Vietnamese accuracy, try the <strong>Zipformer</strong> engine above!
                </p>
              )}
            </>
          )}
        </section>

        {/* ─── Microphone ─── */}
        <section className="settings-section">
          <div className="section-heading compact">
            <div><span className="section-index">04</span><h2>Microphone</h2></div>
          </div>
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
          <div className="section-heading">
            <div><span className="section-index">05</span><h2>Hotkeys</h2></div>
            <p>Click a shortcut, then press your new key combination.</p>
          </div>
          <div className="hotkey-row">
            <label>Hold-to-talk</label>
            <button
              type="button"
              className={`hotkey-btn ${capturingField === "hotkey" ? "capturing" : ""}`}
              onClick={() => {
                setCapturingField("hotkey");
                setHotkeyHint(null);
              }}
            >
              {capturingField === "hotkey" ? "Press keys..." : formatHotkey(settings.hotkey)}
            </button>
          </div>
          <div className="hotkey-row">
            <label>Quit</label>
            <button
              type="button"
              className={`hotkey-btn ${capturingField === "quit_hotkey" ? "capturing" : ""}`}
              onClick={() => {
                setCapturingField("quit_hotkey");
                setHotkeyHint(null);
              }}
            >
              {capturingField === "quit_hotkey" ? "Press keys..." : formatHotkey(settings.quit_hotkey)}
            </button>
          </div>
          {hotkeyHint && <p className="hint hint-warn">{hotkeyHint}</p>}
        </section>

        {/* ─── Footer ─── */}
        <div className="settings-footer">
          <p className={`save-state ${dirty ? "is-dirty" : ""}`}>
            <span />{dirty ? "Unsaved changes" : "Settings are up to date"}
          </p>
          <button type="button" className="btn btn-cancel" onClick={() => getCurrentWindow().close()}>
            Close
          </button>
          <button type="button" className="btn btn-save" onClick={handleSave} disabled={!dirty}>
            {saved ? "Saved!" : "Save"}
          </button>
        </div>
      </div>
    </div>
  );
}
