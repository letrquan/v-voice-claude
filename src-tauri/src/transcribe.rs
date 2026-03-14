use std::io::{Cursor, Read};
use std::path::PathBuf;

use futures_util::StreamExt;
use tauri::Emitter;

use crate::settings;

const WHISPER_CLI_ZIP_URL: &str =
    "https://github.com/ggml-org/whisper.cpp/releases/download/v1.8.3/whisper-bin-x64.zip";

const WHISPER_SAMPLE_RATE: u32 = 16000;

/// Returns the base directory for all v-voice-claude data
fn data_dir() -> PathBuf {
    let base = dirs::data_dir().unwrap_or_else(|| PathBuf::from("."));
    base.join("v-voice-claude")
}

/// Path to a model file by model id (e.g. "tiny.en" -> "ggml-tiny.en.bin")
pub fn model_path(model_id: &str) -> PathBuf {
    let filename = format!("ggml-{}.bin", model_id);
    data_dir().join("models").join(filename)
}

/// Path to the whisper-cli.exe binary
fn cli_path() -> PathBuf {
    data_dir().join("bin").join("whisper-cli.exe")
}

/// Check if both the given model and CLI binary are available
pub fn is_ready(model_id: &str) -> bool {
    model_path(model_id).exists() && cli_path().exists()
}

/// Download a URL into memory, emitting progress events.
async fn download_bytes(
    app: &tauri::AppHandle,
    url: &str,
    label: &str,
) -> Result<Vec<u8>, String> {
    let response = reqwest::get(url)
        .await
        .map_err(|e| format!("{} download failed: {}", label, e))?;

    let total_size = response.content_length().unwrap_or(0);
    let mut downloaded: u64 = 0;
    let mut bytes = Vec::with_capacity(total_size as usize);

    let mut stream = response.bytes_stream();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|e| format!("{} download error: {}", label, e))?;
        downloaded += chunk.len() as u64;
        bytes.extend_from_slice(&chunk);

        if total_size > 0 {
            let progress = (downloaded as f64 / total_size as f64) * 100.0;
            let _ = app.emit("download-progress", progress);
        }
    }

    Ok(bytes)
}

/// Download the specified GGML model and whisper-cli.exe binary (if not already present).
/// Emits "download-progress" events to the frontend.
pub async fn download_model(app: tauri::AppHandle, model_id: &str) -> Result<(), String> {
    let model = model_path(model_id);
    let cli = cli_path();

    // Look up model URL from the available models list
    let model_info = settings::available_models()
        .into_iter()
        .find(|m| m.id == model_id)
        .ok_or_else(|| format!("Unknown model: {}", model_id))?;

    // If both exist, nothing to do
    if model.exists() && cli.exists() {
        let _ = app.emit("download-progress", 100.0_f64);
        return Ok(());
    }

    // --- Download model ---
    if !model.exists() {
        let _ = app.emit("download-progress", 0.0_f64);

        if let Some(parent) = model.parent() {
            tokio::fs::create_dir_all(parent)
                .await
                .map_err(|e| format!("Failed to create models dir: {}", e))?;
        }

        let bytes = download_bytes(&app, &model_info.url, "model").await?;
        tokio::fs::write(&model, &bytes)
            .await
            .map_err(|e| format!("Failed to save model: {}", e))?;
    }

    // --- Download whisper-cli.exe (~4MB zip) ---
    if !cli.exists() {
        let _ = app.emit("download-progress", 92.0_f64);

        let bin_dir = data_dir().join("bin");
        tokio::fs::create_dir_all(&bin_dir)
            .await
            .map_err(|e| format!("Failed to create bin dir: {}", e))?;

        let zip_bytes = download_bytes(&app, WHISPER_CLI_ZIP_URL, "whisper-cli").await?;

        // Extract .exe and .dll files from the zip in a single pass
        let cursor = Cursor::new(zip_bytes);
        let mut archive =
            zip::ZipArchive::new(cursor).map_err(|e| format!("Failed to open zip: {}", e))?;

        let mut found_cli = false;
        for i in 0..archive.len() {
            let mut file = archive
                .by_index(i)
                .map_err(|e| format!("Zip entry error: {}", e))?;

            let name = file.name().to_string();
            let file_name = name.rsplit('/').next().unwrap_or(&name);

            let should_extract =
                file_name == "whisper-cli.exe" || file_name.ends_with(".dll");

            if should_extract && !file.is_dir() {
                let dest = bin_dir.join(file_name);
                let mut buf = Vec::new();
                file.read_to_end(&mut buf)
                    .map_err(|e| format!("Failed to read {} from zip: {}", file_name, e))?;
                std::fs::write(&dest, &buf)
                    .map_err(|e| format!("Failed to write {}: {}", file_name, e))?;

                if file_name == "whisper-cli.exe" {
                    found_cli = true;
                }
            }
        }

        if !found_cli {
            return Err(
                "whisper-cli.exe not found in the downloaded zip archive".to_string(),
            );
        }
    }

    let _ = app.emit("download-progress", 100.0_f64);
    Ok(())
}

/// Resample audio from source rate to target rate using linear interpolation
fn resample(samples: &[f32], from_rate: u32, to_rate: u32) -> Vec<f32> {
    if from_rate == to_rate {
        return samples.to_vec();
    }
    let ratio = to_rate as f64 / from_rate as f64;
    let new_len = (samples.len() as f64 * ratio) as usize;
    let mut resampled = Vec::with_capacity(new_len);
    for i in 0..new_len {
        let src_idx = i as f64 / ratio;
        let idx = src_idx as usize;
        let frac = (src_idx - idx as f64) as f32;
        let sample = if idx + 1 < samples.len() {
            samples[idx] * (1.0 - frac) + samples[idx + 1] * frac
        } else {
            samples[idx.min(samples.len().saturating_sub(1))]
        };
        resampled.push(sample);
    }
    resampled
}

/// Transcribe audio using the specified model and language
pub async fn transcribe_audio(
    samples: Vec<f32>,
    sample_rate: u32,
    model_id: &str,
    language: &str,
) -> Result<String, String> {
    let model = model_path(model_id);
    let cli = cli_path();

    if !model.exists() {
        return Err("Model not downloaded yet".to_string());
    }
    if !cli.exists() {
        return Err("whisper-cli.exe not downloaded yet".to_string());
    }

    // Resample to 16kHz if needed
    let audio_data = if sample_rate != WHISPER_SAMPLE_RATE {
        resample(&samples, sample_rate, WHISPER_SAMPLE_RATE)
    } else {
        samples
    };

    // Write audio to a temporary WAV file (16kHz, mono, 16-bit PCM)
    let temp_dir = std::env::temp_dir();
    let wav_path = temp_dir.join("v-voice-claude-audio.wav");

    let wav_path_clone = wav_path.clone();
    tokio::task::spawn_blocking(move || {
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: WHISPER_SAMPLE_RATE,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut writer = hound::WavWriter::create(&wav_path_clone, spec)
            .map_err(|e| format!("Failed to create WAV file: {}", e))?;

        for &sample in &audio_data {
            let s = (sample * 32767.0).clamp(-32768.0, 32767.0) as i16;
            writer
                .write_sample(s)
                .map_err(|e| format!("Failed to write WAV sample: {}", e))?;
        }
        writer
            .finalize()
            .map_err(|e| format!("Failed to finalize WAV: {}", e))?;

        Ok::<(), String>(())
    })
    .await
    .map_err(|e| format!("WAV write task error: {}", e))??;

    // Build whisper-cli command
    let cli_str = cli.to_str().unwrap();
    let model_str = model.to_str().unwrap();
    let wav_str = wav_path.to_str().unwrap();

    let mut cmd = tokio::process::Command::new(cli_str);
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        cmd.creation_flags(0x08000000); // CREATE_NO_WINDOW
    }
    cmd.arg("-m").arg(model_str);
    cmd.arg("-f").arg(wav_str);
    cmd.arg("--no-timestamps");

    // Language: "auto" means let whisper detect; otherwise specify
    if language != "auto" {
        cmd.arg("-l").arg(language);
    }

    // Vietnamese-specific: provide an initial prompt to help Whisper
    // produce properly accented Vietnamese text with diacritics
    if language == "vi" {
        cmd.arg("--prompt").arg(
            "Xin chào, đây là bản ghi âm tiếng Việt. Hãy chuyển đổi chính xác với dấu thanh đầy đủ."
        );
    }

    cmd.arg("--no-prints");

    let output = cmd
        .output()
        .await
        .map_err(|e| format!("Failed to run whisper-cli: {}", e))?;

    // Clean up temp file (best effort)
    let _ = tokio::fs::remove_file(&wav_path).await;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!("whisper-cli failed: {}", stderr));
    }

    let text = String::from_utf8_lossy(&output.stdout);
    Ok(text.trim().to_string())
}

/// Transcribe audio partially (for real-time streaming preview).
/// Uses a separate temp file so it doesn't conflict with the final transcription.
pub async fn transcribe_partial(
    samples: Vec<f32>,
    sample_rate: u32,
    model_id: &str,
    language: &str,
    prompt: &str,
) -> Result<String, String> {
    let model = model_path(model_id);
    let cli = cli_path();

    if !model.exists() || !cli.exists() {
        return Err("Model or CLI not ready".to_string());
    }

    // Resample to 16kHz if needed
    let audio_data = if sample_rate != WHISPER_SAMPLE_RATE {
        resample(&samples, sample_rate, WHISPER_SAMPLE_RATE)
    } else {
        samples
    };

    // Use a separate temp file for partial transcriptions
    let temp_dir = std::env::temp_dir();
    let wav_path = temp_dir.join("v-voice-claude-partial.wav");

    let wav_path_clone = wav_path.clone();
    tokio::task::spawn_blocking(move || {
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: WHISPER_SAMPLE_RATE,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut writer = hound::WavWriter::create(&wav_path_clone, spec)
            .map_err(|e| format!("Failed to create partial WAV: {}", e))?;

        for &sample in &audio_data {
            let s = (sample * 32767.0).clamp(-32768.0, 32767.0) as i16;
            writer
                .write_sample(s)
                .map_err(|e| format!("Failed to write WAV sample: {}", e))?;
        }
        writer
            .finalize()
            .map_err(|e| format!("Failed to finalize WAV: {}", e))?;

        Ok::<(), String>(())
    })
    .await
    .map_err(|e| format!("WAV write task error: {}", e))??;

    let cli_str = cli.to_str().unwrap();
    let model_str = model.to_str().unwrap();
    let wav_str = wav_path.to_str().unwrap();

    let mut cmd = tokio::process::Command::new(cli_str);
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        cmd.creation_flags(0x08000000); // CREATE_NO_WINDOW
    }
    cmd.arg("-m").arg(model_str);
    cmd.arg("-f").arg(wav_str);
    cmd.arg("--no-timestamps");

    if language != "auto" {
        cmd.arg("-l").arg(language);
    }

    // Build prompt: combine Vietnamese hint with context from previous chunks
    let prompt_text = if language == "vi" {
        if prompt.is_empty() {
            "Xin chào, đây là bản ghi âm tiếng Việt. Hãy chuyển đổi chính xác với dấu thanh đầy đủ.".to_string()
        } else {
            format!("Xin chào, tiếng Việt. {}", prompt)
        }
    } else {
        prompt.to_string()
    };

    if !prompt_text.is_empty() {
        cmd.arg("--prompt").arg(&prompt_text);
    }

    cmd.arg("--no-prints");

    let output = cmd
        .output()
        .await
        .map_err(|e| format!("Failed to run whisper-cli: {}", e))?;

    let _ = tokio::fs::remove_file(&wav_path).await;

    if !output.status.success() {
        // For partial, just return empty on error (don't break the UI)
        return Ok(String::new());
    }

    let text = String::from_utf8_lossy(&output.stdout);
    Ok(text.trim().to_string())
}

/// Write samples to a WAV file in memory and return the bytes
fn samples_to_wav_bytes(samples: &[f32], sample_rate: u32) -> Result<Vec<u8>, String> {
    let resampled = if sample_rate != WHISPER_SAMPLE_RATE {
        resample(samples, sample_rate, WHISPER_SAMPLE_RATE)
    } else {
        samples.to_vec()
    };

    let mut cursor = Cursor::new(Vec::new());
    {
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: WHISPER_SAMPLE_RATE,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut writer = hound::WavWriter::new(&mut cursor, spec)
            .map_err(|e| format!("Failed to create WAV writer: {}", e))?;

        for &sample in &resampled {
            let s = (sample * 32767.0).clamp(-32768.0, 32767.0) as i16;
            writer
                .write_sample(s)
                .map_err(|e| format!("WAV write error: {}", e))?;
        }
        writer
            .finalize()
            .map_err(|e| format!("WAV finalize error: {}", e))?;
    }

    Ok(cursor.into_inner())
}

/// Transcribe audio using a cloud API (OpenAI or Groq Whisper API)
pub async fn transcribe_cloud(
    samples: Vec<f32>,
    sample_rate: u32,
    language: &str,
    provider: &str,
    api_key: &str,
    prompt: &str,
) -> Result<String, String> {
    if api_key.is_empty() {
        return Err("API key not configured. Please add your API key in Settings.".to_string());
    }

    let wav_bytes = tokio::task::spawn_blocking({
        let samples = samples.clone();
        move || samples_to_wav_bytes(&samples, sample_rate)
    })
    .await
    .map_err(|e| format!("WAV task error: {}", e))??;

    // Determine endpoint and model based on provider
    let (api_url, model_name) = match provider {
        "groq" => (
            "https://api.groq.com/openai/v1/audio/transcriptions",
            "whisper-large-v3-turbo",
        ),
        _ => (
            "https://api.openai.com/v1/audio/transcriptions",
            "whisper-1",
        ),
    };

    // Build multipart form
    let file_part = reqwest::multipart::Part::bytes(wav_bytes)
        .file_name("audio.wav")
        .mime_str("audio/wav")
        .map_err(|e| format!("Multipart error: {}", e))?;

    let mut form = reqwest::multipart::Form::new()
        .part("file", file_part)
        .text("model", model_name.to_string())
        .text("response_format", "json");

    if language != "auto" {
        form = form.text("language", language.to_string());
    }

    // Build prompt: combine Vietnamese hint with context from previous chunks
    let prompt_text = if language == "vi" {
        if prompt.is_empty() {
            "Xin chào, đây là bản ghi âm tiếng Việt. Hãy chuyển đổi chính xác với dấu thanh đầy đủ.".to_string()
        } else {
            format!("Xin chào, tiếng Việt. {}", prompt)
        }
    } else {
        prompt.to_string()
    };

    if !prompt_text.is_empty() {
        form = form.text("prompt", prompt_text);
    }

    let client = reqwest::Client::new();
    let response = client
        .post(api_url)
        .header("Authorization", format!("Bearer {}", api_key))
        .multipart(form)
        .send()
        .await
        .map_err(|e| format!("Cloud API request failed: {}", e))?;

    if !response.status().is_success() {
        let status = response.status();
        let body = response.text().await.unwrap_or_default();
        return Err(format!("Cloud API error ({}): {}", status, body));
    }

    let body: serde_json::Value = response
        .json()
        .await
        .map_err(|e| format!("Failed to parse cloud response: {}", e))?;

    let text = body["text"]
        .as_str()
        .unwrap_or("")
        .trim()
        .to_string();

    Ok(text)
}
