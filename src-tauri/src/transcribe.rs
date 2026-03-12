use std::io::{Cursor, Read};
use std::path::PathBuf;

use futures_util::StreamExt;
use tauri::Emitter;

const MODEL_URL: &str =
    "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.en.bin";

const WHISPER_CLI_ZIP_URL: &str =
    "https://github.com/ggml-org/whisper.cpp/releases/download/v1.8.3/whisper-bin-x64.zip";

const WHISPER_SAMPLE_RATE: u32 = 16000;

/// Returns the base directory for all v-voice-claude data
fn data_dir() -> PathBuf {
    let base = dirs::data_dir().unwrap_or_else(|| PathBuf::from("."));
    base.join("v-voice-claude")
}

/// Path to the GGML model file
pub fn model_path() -> PathBuf {
    data_dir().join("models").join("ggml-tiny.en.bin")
}

/// Path to the whisper-cli.exe binary
fn cli_path() -> PathBuf {
    data_dir().join("bin").join("whisper-cli.exe")
}

/// Check if both the model and CLI binary are available
pub fn is_ready() -> bool {
    model_path().exists() && cli_path().exists()
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

/// Download the GGML model and whisper-cli.exe binary (if not already present).
/// Emits "download-progress" events to the frontend.
pub async fn download_model(app: tauri::AppHandle) -> Result<(), String> {
    let model = model_path();
    let cli = cli_path();

    // If both exist, nothing to do
    if model.exists() && cli.exists() {
        let _ = app.emit("download-progress", 100.0_f64);
        return Ok(());
    }

    // --- Download model (ggml-tiny.en.bin ~75MB) ---
    if !model.exists() {
        let _ = app.emit("download-progress", 0.0_f64);

        if let Some(parent) = model.parent() {
            tokio::fs::create_dir_all(parent)
                .await
                .map_err(|e| format!("Failed to create models dir: {}", e))?;
        }

        let bytes = download_bytes(&app, MODEL_URL, "model").await?;
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

            // Extract the filename from the path inside the zip
            let file_name = name.rsplit('/').next().unwrap_or(&name);

            // We want whisper-cli.exe and any .dll files it might need
            let should_extract =
                file_name == "whisper-cli.exe" || file_name.ends_with(".dll");

            if should_extract && !file.is_dir() {
                let dest = bin_dir.join(file_name);
                let mut buf = Vec::new();
                file.read_to_end(&mut buf)
                    .map_err(|e| format!("Failed to read {} from zip: {}", file_name, e))?;
                // Use std::fs since we're not in an async block for the zip reader
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

/// Transcribe audio by writing a temporary WAV file and calling whisper-cli.exe
pub async fn transcribe_audio(samples: Vec<f32>, sample_rate: u32) -> Result<String, String> {
    let model = model_path();
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
            // Convert f32 [-1.0, 1.0] to i16
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

    // Call whisper-cli.exe as a subprocess
    let output = tokio::process::Command::new(cli.to_str().unwrap())
        .arg("-m")
        .arg(model.to_str().unwrap())
        .arg("-f")
        .arg(wav_path.to_str().unwrap())
        .arg("--no-timestamps")
        .arg("-l")
        .arg("en")
        .arg("--no-prints")
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
