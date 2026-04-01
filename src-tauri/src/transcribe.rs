use std::io::{Cursor, Read};
use std::path::PathBuf;

use futures_util::StreamExt;
use tauri::Emitter;

use crate::settings;

const WHISPER_CLI_ZIP_URL: &str =
    "https://github.com/ggml-org/whisper.cpp/releases/download/v1.8.3/whisper-bin-x64.zip";

const SHERPA_ONNX_PACKAGE_URL: &str =
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/v1.12.29/sherpa-onnx-v1.12.29-win-x64-shared-MD-Release-no-tts.tar.bz2";
const SHERPA_ONNX_PACKAGE_DIR: &str = "sherpa-onnx-v1.12.29-win-x64-shared-MD-Release-no-tts";

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

// ─── Zipformer / sherpa-onnx support ───

/// Path to the sherpa-onnx offline CLI
fn sherpa_cli_path() -> PathBuf {
    data_dir().join("bin").join("sherpa-onnx-offline.exe")
}

/// Directory containing Zipformer model files
fn zipformer_model_dir() -> PathBuf {
    data_dir().join("models").join("zipformer-vi")
}

/// Check if the Zipformer model and sherpa-onnx binary are available
pub fn is_zipformer_ready() -> bool {
    let dir = zipformer_model_dir();
    sherpa_cli_path().exists()
        && dir.join("encoder.int8.onnx").exists()
        && dir.join("decoder.int8.onnx").exists()
        && dir.join("joiner.int8.onnx").exists()
        && dir.join("tokens.txt").exists()
}

/// Download the Zipformer Vietnamese model and sherpa-onnx CLI.
pub async fn download_zipformer(app: tauri::AppHandle) -> Result<(), String> {
    let model_dir = zipformer_model_dir();
    let sherpa_cli = sherpa_cli_path();

    if is_zipformer_ready() {
        let _ = app.emit("download-progress", 100.0_f64);
        return Ok(());
    }

    // Create directories
    tokio::fs::create_dir_all(&model_dir)
        .await
        .map_err(|e| format!("Failed to create zipformer model dir: {}", e))?;

    let bin_dir = data_dir().join("bin");
    tokio::fs::create_dir_all(&bin_dir)
        .await
        .map_err(|e| format!("Failed to create bin dir: {}", e))?;

    let model_info = settings::zipformer_model();

    // Download model files (encoder, decoder, joiner, tokens)
    let files = [
        (&model_info.encoder_url, "encoder.int8.onnx"),
        (&model_info.decoder_url, "decoder.int8.onnx"),
        (&model_info.joiner_url, "joiner.int8.onnx"),
        (&model_info.tokens_url, "tokens.txt"),
    ];

    let total_files = files.len() + 1; // +1 for sherpa-onnx CLI
    for (i, (url, filename)) in files.iter().enumerate() {
        let dest = model_dir.join(filename);
        if !dest.exists() {
            let base_progress = (i as f64 / total_files as f64) * 100.0;
            let _ = app.emit("download-progress", base_progress);

            let bytes = download_bytes(&app, url, filename).await?;
            tokio::fs::write(&dest, &bytes)
                .await
                .map_err(|e| format!("Failed to save {}: {}", filename, e))?;
        }
    }

    // Download sherpa-onnx shared library package (contains CLI binary + DLLs)
    if !sherpa_cli.exists() {
        let base_progress = (files.len() as f64 / total_files as f64) * 100.0;
        let _ = app.emit("download-progress", base_progress);

        // Download tar.bz2 to temp
        let temp_dir = std::env::temp_dir();
        let archive_path = temp_dir.join("sherpa-onnx-package.tar.bz2");
        let extract_dir = temp_dir.join("sherpa-onnx-extract");

        let bytes = download_bytes(&app, SHERPA_ONNX_PACKAGE_URL, "sherpa-onnx").await?;
        tokio::fs::write(&archive_path, &bytes)
            .await
            .map_err(|e| format!("Failed to save sherpa-onnx archive: {}", e))?;

        // Extract using system tar (available on Windows 10+)
        let _ = tokio::fs::remove_dir_all(&extract_dir).await;
        tokio::fs::create_dir_all(&extract_dir)
            .await
            .map_err(|e| format!("Failed to create extract dir: {}", e))?;

        let mut tar_cmd = tokio::process::Command::new("tar");
        tar_cmd.arg("-xf")
            .arg(archive_path.to_str().unwrap())
            .arg("-C")
            .arg(extract_dir.to_str().unwrap());
        #[cfg(windows)]
        {
            tar_cmd.creation_flags(0x08000000);
        }
        let tar_output = tar_cmd.output().await
            .map_err(|e| format!("Failed to run tar: {}", e))?;
        if !tar_output.status.success() {
            let stderr = String::from_utf8_lossy(&tar_output.stderr);
            return Err(format!("Failed to extract sherpa-onnx package: {}", stderr));
        }

        // Copy sherpa-onnx-offline.exe from bin/ directory
        let extracted_root = extract_dir.join(SHERPA_ONNX_PACKAGE_DIR);
        let exe_src = extracted_root.join("bin").join("sherpa-onnx-offline.exe");
        if exe_src.exists() {
            tokio::fs::copy(&exe_src, &sherpa_cli)
                .await
                .map_err(|e| format!("Failed to copy sherpa-onnx-offline.exe: {}", e))?;
        } else {
            // Try sherpa-onnx.exe as fallback (streaming version)
            let streaming_src = extracted_root.join("bin").join("sherpa-onnx.exe");
            if streaming_src.exists() {
                tokio::fs::copy(&streaming_src, &sherpa_cli)
                    .await
                    .map_err(|e| format!("Failed to copy sherpa-onnx.exe: {}", e))?;
            } else {
                return Err("sherpa-onnx-offline.exe not found in extracted package".to_string());
            }
        }

        // Copy all DLLs from lib/ to bin/ (needed at runtime)
        let lib_dir = extracted_root.join("lib");
        if lib_dir.exists() {
            let mut entries = tokio::fs::read_dir(&lib_dir)
                .await
                .map_err(|e| format!("Failed to read lib dir: {}", e))?;
            while let Ok(Some(entry)) = entries.next_entry().await {
                let name = entry.file_name();
                if let Some(n) = name.to_str() {
                    if n.ends_with(".dll") {
                        let _ = tokio::fs::copy(entry.path(), bin_dir.join(&name)).await;
                    }
                }
            }
        }

        // Also copy DLLs from bin/ directory (some builds put DLLs there)
        let extracted_bin_dir = extracted_root.join("bin");
        if extracted_bin_dir.exists() {
            let mut entries = tokio::fs::read_dir(&extracted_bin_dir)
                .await
                .map_err(|e| format!("Failed to read extracted bin dir: {}", e))?;
            while let Ok(Some(entry)) = entries.next_entry().await {
                let name = entry.file_name();
                if let Some(n) = name.to_str() {
                    if n.ends_with(".dll") {
                        let _ = tokio::fs::copy(entry.path(), bin_dir.join(&name)).await;
                    }
                }
            }
        }

        // Cleanup temp files
        let _ = tokio::fs::remove_file(&archive_path).await;
        let _ = tokio::fs::remove_dir_all(&extract_dir).await;
    }

    let _ = app.emit("download-progress", 100.0_f64);
    Ok(())
}

/// Transcribe audio using the Zipformer model via sherpa-onnx CLI.
pub async fn transcribe_zipformer(
    samples: Vec<f32>,
    sample_rate: u32,
) -> Result<String, String> {
    let sherpa = sherpa_cli_path();
    let model_dir = zipformer_model_dir();

    if !is_zipformer_ready() {
        return Err("Zipformer model or sherpa-onnx not ready".to_string());
    }

    // Resample to 16kHz if needed
    let audio_data = if sample_rate != WHISPER_SAMPLE_RATE {
        resample(&samples, sample_rate, WHISPER_SAMPLE_RATE)
    } else {
        samples
    };

    // Write audio to a temporary WAV file
    let temp_dir = std::env::temp_dir();
    let wav_path = temp_dir.join("v-voice-zipformer.wav");

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
            writer.write_sample(s)
                .map_err(|e| format!("WAV write error: {}", e))?;
        }
        writer.finalize()
            .map_err(|e| format!("WAV finalize error: {}", e))?;
        Ok::<(), String>(())
    })
    .await
    .map_err(|e| format!("WAV write task error: {}", e))??;

    // Build sherpa-onnx command
    let mut cmd = tokio::process::Command::new(sherpa.to_str().unwrap());
    #[cfg(windows)]
    {
        cmd.creation_flags(0x08000000); // CREATE_NO_WINDOW
    }

    cmd.arg("--transducer-encoder").arg(model_dir.join("encoder.int8.onnx").to_str().unwrap());
    cmd.arg("--transducer-decoder").arg(model_dir.join("decoder.int8.onnx").to_str().unwrap());
    cmd.arg("--transducer-joiner").arg(model_dir.join("joiner.int8.onnx").to_str().unwrap());
    cmd.arg("--tokens").arg(model_dir.join("tokens.txt").to_str().unwrap());
    cmd.arg(wav_path.to_str().unwrap());

    let output = cmd
        .output()
        .await
        .map_err(|e| format!("Failed to run sherpa-onnx: {}", e))?;

    // Clean up temp file
    let _ = tokio::fs::remove_file(&wav_path).await;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!("sherpa-onnx failed: {}", stderr));
    }

    // Parse sherpa-onnx output — it prints the filename then the recognized text
    let raw = String::from_utf8_lossy(&output.stdout);
    // The output format is typically:
    //   /path/to/file.wav
    //   recognized text here
    // We want just the recognized text (skip lines that look like file paths)
    let text: String = raw
        .lines()
        .filter(|line| {
            let trimmed = line.trim();
            !trimmed.is_empty()
                && !trimmed.ends_with(".wav")
                && !trimmed.starts_with('/')
                && !trimmed.starts_with("----")
                && !trimmed.contains("v-voice-zipformer")
                && !trimmed.contains(":\\") // Windows absolute paths like C:\
                && !trimmed.starts_with("Duration")
                && !trimmed.starts_with("Wave duration")
                && !trimmed.starts_with("Elapsed")
                && !trimmed.starts_with("Real time factor")
                && !trimmed.starts_with("NumThreads")
                && !trimmed.starts_with("num_threads")
        })
        .collect::<Vec<&str>>()
        .join(" ");

    Ok(text.trim().to_string())
}

// ─── Granite 4.0 1B Speech support ───

/// Directory containing the downloaded Granite model
fn granite_model_dir() -> PathBuf {
    data_dir().join("models").join("granite-speech")
}

/// Path to the Granite inference server Python script
fn granite_server_script() -> PathBuf {
    // The script is bundled alongside the app binary
    // In development, it lives in the project scripts/ directory
    let exe_dir = std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(|p| p.to_path_buf()))
        .unwrap_or_else(|| PathBuf::from("."));

    // Check next to the binary first (production), then dev path
    let prod_path = exe_dir.join("scripts").join("granite_server.py");
    if prod_path.exists() {
        return prod_path;
    }

    // Dev: the script is in the project root's scripts/ dir
    let dev_path = exe_dir
        .ancestors()
        .take(5)
        .find(|p| p.join("scripts").join("granite_server.py").exists())
        .map(|p| p.join("scripts").join("granite_server.py"))
        .unwrap_or_else(|| {
            data_dir().join("scripts").join("granite_server.py")
        });
    dev_path
}

/// Check if the Granite model and Python dependencies are available
pub fn is_granite_ready() -> bool {
    let dir = granite_model_dir();
    dir.join("config.json").exists()
        && (dir.join("model.safetensors").exists()
            || dir.join("model.safetensors.index.json").exists())
}

/// Download the Granite Speech model from HuggingFace.
/// Uses `huggingface-cli download` or direct download of key files.
pub async fn download_granite(app: tauri::AppHandle) -> Result<(), String> {
    if is_granite_ready() {
        let _ = app.emit("download-progress", 100.0_f64);
        return Ok(());
    }

    let model_dir = granite_model_dir();
    tokio::fs::create_dir_all(&model_dir)
        .await
        .map_err(|e| format!("Failed to create granite model dir: {}", e))?;

    let _ = app.emit("download-progress", 5.0_f64);

    let model_info = crate::settings::granite_model();

    // Try using huggingface-cli to download the model
    // This handles large models with multiple shards properly
    let mut cmd = tokio::process::Command::new("huggingface-cli");
    #[cfg(windows)]
    {
        cmd.creation_flags(0x08000000);
    }
    cmd.arg("download")
        .arg(&model_info.model_id)
        .arg("--local-dir")
        .arg(model_dir.to_str().unwrap());

    let _ = app.emit("download-progress", 10.0_f64);

    let output = cmd
        .output()
        .await
        .map_err(|e| format!("Failed to run huggingface-cli. Make sure it's installed (pip install huggingface_hub): {}", e))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!(
            "huggingface-cli download failed: {}. \
             Install with: pip install huggingface_hub",
            stderr
        ));
    }

    let _ = app.emit("download-progress", 100.0_f64);
    Ok(())
}

/// Check if the Granite server is running and healthy
pub async fn is_granite_server_running(port: u16) -> bool {
    let url = format!("http://127.0.0.1:{}/health", port);
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(2))
        .build()
        .unwrap_or_else(|_| reqwest::Client::new());

    match client.get(&url).send().await {
        Ok(resp) => {
            if let Ok(body) = resp.json::<serde_json::Value>().await {
                body["status"].as_str() == Some("ready")
            } else {
                false
            }
        }
        Err(_) => false,
    }
}

/// Start the Granite inference server as a background process.
/// Returns Ok if the server starts successfully (or is already running).
pub async fn start_granite_server(port: u16) -> Result<(), String> {
    // Already running?
    if is_granite_server_running(port).await {
        return Ok(());
    }

    let model_dir = granite_model_dir();
    if !is_granite_ready() {
        return Err("Granite model not downloaded yet".to_string());
    }

    let script = granite_server_script();
    if !script.exists() {
        // Copy bundled script to data dir
        let dest_dir = data_dir().join("scripts");
        let dest = dest_dir.join("granite_server.py");
        if !dest.exists() {
            return Err(format!(
                "Granite server script not found at {:?}. Please ensure granite_server.py is available.",
                script
            ));
        }
    }

    let mut cmd = tokio::process::Command::new("python");
    #[cfg(windows)]
    {
        cmd.creation_flags(0x08000000); // CREATE_NO_WINDOW
    }
    cmd.arg(script.to_str().unwrap())
        .arg("--model-dir")
        .arg(model_dir.to_str().unwrap())
        .arg("--port")
        .arg(port.to_string());

    cmd.stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null());

    let _child = cmd
        .spawn()
        .map_err(|e| format!("Failed to start Granite server: {}. Make sure Python is installed.", e))?;

    // Wait for server to become ready (up to 120 seconds for model loading)
    for i in 0..240 {
        tokio::time::sleep(std::time::Duration::from_millis(500)).await;
        if is_granite_server_running(port).await {
            return Ok(());
        }
        // Log progress periodically
        if i % 10 == 0 && i > 0 {
            eprintln!("[granite] Waiting for server... ({:.0}s)", i as f64 * 0.5);
        }
    }

    Err("Granite server failed to start within 120 seconds".to_string())
}

/// Stop the Granite inference server
pub async fn stop_granite_server(port: u16) -> Result<(), String> {
    let url = format!("http://127.0.0.1:{}/shutdown", port);
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(5))
        .build()
        .unwrap_or_else(|_| reqwest::Client::new());

    let _ = client.post(&url).send().await;
    Ok(())
}

/// Transcribe audio using the local Granite inference server
pub async fn transcribe_granite(
    samples: Vec<f32>,
    sample_rate: u32,
    port: u16,
    language: &str,
) -> Result<String, String> {
    if !is_granite_ready() {
        return Err("Granite model not downloaded yet".to_string());
    }

    // Make sure the server is running
    if !is_granite_server_running(port).await {
        // Try to start it
        start_granite_server(port).await?;
    }

    // Build WAV bytes
    let wav_bytes = tokio::task::spawn_blocking({
        let samples = samples.clone();
        move || samples_to_wav_bytes(&samples, sample_rate)
    })
    .await
    .map_err(|e| format!("WAV task error: {}", e))??;

    // Send to the local server
    let url = format!("http://127.0.0.1:{}/transcribe", port);

    let file_part = reqwest::multipart::Part::bytes(wav_bytes)
        .file_name("audio.wav")
        .mime_str("audio/wav")
        .map_err(|e| format!("Multipart error: {}", e))?;

    let mut form = reqwest::multipart::Form::new()
        .part("file", file_part);

    if language != "auto" {
        form = form.text("language", language.to_string());
    }

    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(120))
        .build()
        .unwrap_or_else(|_| reqwest::Client::new());

    let response = client
        .post(&url)
        .multipart(form)
        .send()
        .await
        .map_err(|e| format!("Granite server request failed: {}", e))?;

    if !response.status().is_success() {
        let status = response.status();
        let body = response.text().await.unwrap_or_default();
        return Err(format!("Granite server error ({}): {}", status, body));
    }

    let body: serde_json::Value = response
        .json()
        .await
        .map_err(|e| format!("Failed to parse Granite response: {}", e))?;

    if let Some(error) = body["error"].as_str() {
        return Err(format!("Granite inference error: {}", error));
    }

    let text = body["text"]
        .as_str()
        .unwrap_or("")
        .trim()
        .to_string();

    Ok(text)
}
