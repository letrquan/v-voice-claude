use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::Mutex;

use tauri::Emitter;
use tauri_plugin_store::StoreExt;

/// All whisper.cpp models we support (from HuggingFace ggerganov/whisper.cpp)
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ModelInfo {
    pub id: String,       // e.g. "tiny.en"
    pub filename: String, // e.g. "ggml-tiny.en.bin"
    pub label: String,    // e.g. "Tiny (English)"
    pub size_mb: u32,     // approximate download size
    pub url: String,      // full download URL
}

impl ModelInfo {
    fn new(id: &str, label: &str, size_mb: u32) -> Self {
        let filename = format!("ggml-{}.bin", id);
        let url = format!(
            "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/{}",
            filename
        );
        Self {
            id: id.to_string(),
            filename,
            label: label.to_string(),
            size_mb,
            url,
        }
    }
}

/// Get the full list of available models
pub fn available_models() -> Vec<ModelInfo> {
    vec![
        ModelInfo::new("tiny.en", "Tiny (English)", 75),
        ModelInfo::new("tiny", "Tiny (Multilingual)", 75),
        ModelInfo::new("base.en", "Base (English)", 142),
        ModelInfo::new("base", "Base (Multilingual)", 142),
        ModelInfo::new("small.en", "Small (English)", 466),
        ModelInfo::new("small", "Small (Multilingual)", 466),
        ModelInfo::new("medium.en", "Medium (English)", 1500),
        ModelInfo::new("medium", "Medium (Multilingual)", 1500),
    ]
}

/// Zipformer model info (Vietnamese-optimised, uses sherpa-onnx)
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ZipformerModelInfo {
    pub id: String,
    pub label: String,
    pub size_mb: u32,
    pub encoder_url: String,
    pub decoder_url: String,
    pub joiner_url: String,
    pub tokens_url: String,
}

pub fn zipformer_model() -> ZipformerModelInfo {
    let base = "https://huggingface.co/hynt/Zipformer-30M-RNNT-6000h/resolve/main";
    ZipformerModelInfo {
        id: "zipformer-vi".to_string(),
        label: "Zipformer 30M (Vietnamese)".to_string(),
        size_mb: 30,
        encoder_url: format!("{}/encoder-epoch-20-avg-10.int8.onnx", base),
        decoder_url: format!("{}/decoder-epoch-20-avg-10.int8.onnx", base),
        joiner_url: format!("{}/joiner-epoch-20-avg-10.int8.onnx", base),
        tokens_url: format!("{}/config.json", base),
    }
}

/// Granite 4.0 1B Speech model info
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct GraniteModelInfo {
    pub id: String,
    pub label: String,
    pub model_id: String,   // HuggingFace model identifier
    pub size_mb: u32,
    pub languages: Vec<String>,
}

pub fn granite_model() -> GraniteModelInfo {
    GraniteModelInfo {
        id: "granite-speech".to_string(),
        label: "Granite 4.0 1B Speech".to_string(),
        model_id: "ibm-granite/granite-4.0-1b-speech".to_string(),
        size_mb: 2000,
        languages: vec![
            "en".to_string(), "fr".to_string(), "de".to_string(),
            "es".to_string(), "pt".to_string(), "ja".to_string(),
        ],
    }
}

/// Saved pill position on a specific monitor (logical coordinates relative to monitor origin)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PillPosition {
    pub x: f64,
    pub y: f64,
}

/// Persistent settings schema
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppSettings {
    pub model: String,              // model id, e.g. "tiny.en"
    pub language: String,           // language code, e.g. "en", "auto"
    pub hotkey: String,             // hold-to-talk hotkey, e.g. "CommandOrControl+Shift+Space"
    pub quit_hotkey: String,        // quit hotkey, e.g. "CommandOrControl+Shift+Q"
    pub microphone_id: String,      // "" = system default
    pub vad_silence_threshold: f64, // RMS threshold
    pub vad_silence_frames: u32,    // frames of silence before auto-stop
    #[serde(default = "default_transcription_mode")]
    pub transcription_mode: String, // "local" or "cloud"
    #[serde(default = "default_cloud_provider")]
    pub cloud_provider: String,     // "openai" or "groq"
    #[serde(default)]
    pub cloud_api_key: String,      // API key for cloud provider
    #[serde(default)]
    pub pill_positions: std::collections::HashMap<String, PillPosition>, // monitor fingerprint -> position
    #[serde(default = "default_local_engine")]
    pub local_engine: String,       // "whisper", "zipformer", or "granite"
    #[serde(default = "default_granite_port")]
    pub granite_api_port: u16,      // port for the local Granite inference server
}

fn default_transcription_mode() -> String { "local".to_string() }
fn default_cloud_provider() -> String { "openai".to_string() }
fn default_local_engine() -> String { "whisper".to_string() }
fn default_granite_port() -> u16 { 8976 }

impl Default for AppSettings {
    fn default() -> Self {
        Self {
            model: "tiny.en".to_string(),
            language: "en".to_string(),
            hotkey: "CommandOrControl+Shift+Space".to_string(),
            quit_hotkey: "CommandOrControl+Shift+Q".to_string(),
            microphone_id: String::new(),
            vad_silence_threshold: 0.015,
            vad_silence_frames: 45,
            transcription_mode: "local".to_string(),
            cloud_provider: "openai".to_string(),
            cloud_api_key: String::new(),
            pill_positions: std::collections::HashMap::new(),
            local_engine: "whisper".to_string(),
            granite_api_port: 8976,
        }
    }
}

/// In-memory cache of settings, synced with the store
pub struct SettingsState(pub Mutex<AppSettings>);

const STORE_FILENAME: &str = "settings.json";
const STORE_KEY: &str = "settings";

/// Load settings from the tauri store (or return defaults)
pub fn load_settings(app: &tauri::AppHandle) -> AppSettings {
    let store = app.store(STORE_FILENAME).ok();
    if let Some(store) = store {
        if let Some(val) = store.get(STORE_KEY) {
            if let Ok(settings) = serde_json::from_value::<AppSettings>(val.clone()) {
                return settings;
            }
        }
    }
    AppSettings::default()
}

/// Save settings to the tauri store
fn save_settings(app: &tauri::AppHandle, settings: &AppSettings) -> Result<(), String> {
    let store = app
        .store(STORE_FILENAME)
        .map_err(|e| format!("Failed to open store: {}", e))?;
    let val = serde_json::to_value(settings).map_err(|e| format!("Serialize error: {}", e))?;
    store.set(STORE_KEY, val);
    store
        .save()
        .map_err(|e| format!("Store save error: {}", e))?;
    Ok(())
}

/// Returns the base data directory
fn data_dir() -> PathBuf {
    let base = dirs::data_dir().unwrap_or_else(|| PathBuf::from("."));
    base.join("v-voice-claude")
}

// ─── Tauri Commands ───

#[tauri::command]
pub fn get_settings(state: tauri::State<'_, SettingsState>) -> AppSettings {
    state.0.lock().unwrap().clone()
}

#[tauri::command]
pub fn set_settings(
    app: tauri::AppHandle,
    state: tauri::State<'_, SettingsState>,
    settings: AppSettings,
) -> Result<(), String> {
    save_settings(&app, &settings)?;
    let mut current = state.0.lock().unwrap();
    *current = settings;
    // Emit event so other windows (main pill) can react
    let _ = app.emit("settings-changed", ());
    Ok(())
}

#[tauri::command]
pub fn get_available_models() -> Vec<ModelInfo> {
    available_models()
}

/// Check which models are already downloaded
#[tauri::command]
pub fn get_downloaded_models() -> Vec<String> {
    let models_dir = data_dir().join("models");
    available_models()
        .into_iter()
        .filter(|m| models_dir.join(&m.filename).exists())
        .map(|m| m.id)
        .collect()
}

/// Check if a specific model is downloaded
#[tauri::command]
pub fn is_model_downloaded(model_id: String) -> bool {
    let info = available_models().into_iter().find(|m| m.id == model_id);
    if let Some(info) = info {
        data_dir().join("models").join(&info.filename).exists()
    } else {
        false
    }
}

/// Delete a downloaded model file
#[tauri::command]
pub fn delete_model(model_id: String) -> Result<(), String> {
    let info = available_models()
        .into_iter()
        .find(|m| m.id == model_id)
        .ok_or_else(|| format!("Unknown model: {}", model_id))?;
    let path = data_dir().join("models").join(&info.filename);
    if path.exists() {
        std::fs::remove_file(&path).map_err(|e| format!("Failed to delete model: {}", e))?;
    }
    Ok(())
}

/// Get Zipformer model info
#[tauri::command]
pub fn get_zipformer_model() -> ZipformerModelInfo {
    zipformer_model()
}

/// Check if Zipformer model files are downloaded
#[tauri::command]
pub fn is_zipformer_ready() -> bool {
    let zf_dir = data_dir().join("models").join("zipformer-vi");
    let sherpa = data_dir().join("bin").join("sherpa-onnx-streaming.exe");
    zf_dir.join("encoder.int8.onnx").exists()
        && zf_dir.join("decoder.int8.onnx").exists()
        && zf_dir.join("joiner.int8.onnx").exists()
        && zf_dir.join("tokens.txt").exists()
        && sherpa.exists()
}

/// Get Granite model info
#[tauri::command]
pub fn get_granite_model() -> GraniteModelInfo {
    granite_model()
}

/// Check if the Granite model is downloaded (model directory with config.json exists)
#[tauri::command]
pub fn is_granite_ready() -> bool {
    let model_dir = data_dir().join("models").join("granite-speech");
    model_dir.join("config.json").exists()
        && model_dir.join("model.safetensors").exists()
}
