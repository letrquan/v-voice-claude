use std::io::{Cursor, Read};
use std::path::{Path, PathBuf};
use std::process::{Child, Stdio};
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

use futures_util::StreamExt;
use tauri::Emitter;
use tokio::io::AsyncWriteExt;

use crate::settings;

const WHISPER_CLI_ZIP_URL: &str =
    "https://github.com/ggml-org/whisper.cpp/releases/download/v1.8.3/whisper-bin-x64.zip";

const SHERPA_ONNX_PACKAGE_URL: &str =
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/v1.12.29/sherpa-onnx-v1.12.29-win-x64-shared-MD-Release-no-tts.tar.bz2";
const SHERPA_ONNX_PACKAGE_DIR: &str = "sherpa-onnx-v1.12.29-win-x64-shared-MD-Release-no-tts";

const WHISPER_SAMPLE_RATE: u32 = 16000;
const WHISPER_SERVER_RETRY_DELAY: Duration = Duration::from_secs(60);

#[derive(Default)]
struct WhisperServerState {
    model_id: Option<String>,
    child: Option<Child>,
    port: Option<u16>,
    retry_after: Option<Instant>,
    generation: u64,
}

fn whisper_server_state() -> &'static Mutex<WhisperServerState> {
    static STATE: OnceLock<Mutex<WhisperServerState>> = OnceLock::new();
    STATE.get_or_init(|| Mutex::new(WhisperServerState::default()))
}

fn whisper_server_start_lock() -> &'static tokio::sync::Mutex<()> {
    static LOCK: OnceLock<tokio::sync::Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| tokio::sync::Mutex::new(()))
}

fn whisper_http_client() -> &'static reqwest::Client {
    static CLIENT: OnceLock<reqwest::Client> = OnceLock::new();
    CLIENT.get_or_init(|| {
        reqwest::Client::builder()
            .timeout(Duration::from_secs(120))
            .build()
            .unwrap_or_else(|_| reqwest::Client::new())
    })
}

fn whisper_health_client() -> &'static reqwest::Client {
    static CLIENT: OnceLock<reqwest::Client> = OnceLock::new();
    CLIENT.get_or_init(|| {
        reqwest::Client::builder()
            .timeout(Duration::from_secs(1))
            .build()
            .unwrap_or_else(|_| reqwest::Client::new())
    })
}

fn whisper_thread_count() -> String {
    std::thread::available_parallelism()
        .map(|count| count.get().saturating_sub(1).clamp(2, 8))
        .unwrap_or(4)
        .to_string()
}

fn configure_whisper_command(cmd: &mut tokio::process::Command) {
    cmd.arg("--threads")
        .arg(whisper_thread_count())
        .arg("--split-on-word")
        .arg("--suppress-nst");
}

fn stop_whisper_server_locked(state: &mut WhisperServerState) {
    if let Some(mut child) = state.child.take() {
        let _ = child.kill();
        let _ = child.wait();
    }
    state.model_id = None;
    state.port = None;
}

pub fn stop_whisper_server() {
    let mut state = whisper_server_state()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    stop_whisper_server_locked(&mut state);
    state.retry_after = None;
    state.generation = state.generation.wrapping_add(1);
}

async fn whisper_server_healthy(port: u16) -> bool {
    whisper_health_client()
        .get(format!("http://127.0.0.1:{}/", port))
        .send()
        .await
        .is_ok()
}

async fn ensure_whisper_server(model_id: &str) -> Result<u16, String> {
    let _start_guard = whisper_server_start_lock().lock().await;

    let (existing_port, start_generation) = {
        let mut state = whisper_server_state()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());

        if let Some(retry_after) = state.retry_after {
            if retry_after > Instant::now() {
                return Err("Whisper server is temporarily unavailable".to_string());
            }
            state.retry_after = None;
        }

        let child_running = match state.child.as_mut() {
            Some(child) => matches!(child.try_wait(), Ok(None)),
            None => false,
        };

        if !child_running {
            stop_whisper_server_locked(&mut state);
        }

        let existing_port = if state.model_id.as_deref() == Some(model_id) && child_running {
            state.port
        } else {
            if child_running {
                stop_whisper_server_locked(&mut state);
            }
            None
        };
        (existing_port, state.generation)
    };

    if let Some(port) = existing_port {
        if whisper_server_healthy(port).await {
            return Ok(port);
        }
        let mut state = whisper_server_state()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        stop_whisper_server_locked(&mut state);
    }

    let server = server_path();
    let model = model_path(model_id);
    if !server.exists() || !model.exists() {
        return Err("Whisper server or model is not ready".to_string());
    }

    let listener = std::net::TcpListener::bind(("127.0.0.1", 0))
        .map_err(|e| format!("Failed to reserve Whisper server port: {}", e))?;
    let port = listener
        .local_addr()
        .map_err(|e| format!("Failed to read Whisper server port: {}", e))?
        .port();
    drop(listener);

    let mut command = std::process::Command::new(&server);
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        command.creation_flags(0x08000000); // CREATE_NO_WINDOW
    }
    command
        .arg("-m")
        .arg(&model)
        .arg("--host")
        .arg("127.0.0.1")
        .arg("--port")
        .arg(port.to_string())
        .arg("--threads")
        .arg(whisper_thread_count())
        .arg("--no-timestamps")
        .arg("--split-on-word")
        .arg("--suppress-nst")
        .arg("--no-language-probabilities")
        .stdout(Stdio::null())
        .stderr(Stdio::null());

    if let Some(bin_dir) = server.parent() {
        command.current_dir(bin_dir);
    }

    let mut child = match command.spawn() {
        Ok(child) => child,
        Err(error) => {
            let mut state = whisper_server_state()
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            if state.generation == start_generation {
                state.retry_after = Some(Instant::now() + WHISPER_SERVER_RETRY_DELAY);
            }
            return Err(format!("Failed to start whisper-server: {}", error));
        }
    };

    {
        let mut state = whisper_server_state()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if state.generation != start_generation {
            let _ = child.kill();
            let _ = child.wait();
            return Err("Whisper server startup was cancelled".to_string());
        }
        state.model_id = Some(model_id.to_string());
        state.child = Some(child);
        state.port = Some(port);
    }

    for _ in 0..300 {
        if whisper_server_healthy(port).await {
            return Ok(port);
        }

        let process_failure = {
            let mut state = whisper_server_state()
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            if state.generation != start_generation {
                Some(("Whisper server startup was cancelled".to_string(), false))
            } else {
                match state.child.as_mut() {
                    Some(child) => match child.try_wait() {
                        Ok(Some(status)) => Some((
                            format!("Whisper server exited during startup: {}", status),
                            true,
                        )),
                        Ok(None) => None,
                        Err(error) => Some((
                            format!("Failed to inspect whisper-server: {}", error),
                            true,
                        )),
                    },
                    None => Some(("Whisper server stopped during startup".to_string(), false)),
                }
            }
        };

        if let Some((error, should_retry_later)) = process_failure {
            if should_retry_later {
                let mut state = whisper_server_state()
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner());
                if state.generation == start_generation {
                    stop_whisper_server_locked(&mut state);
                    state.retry_after = Some(Instant::now() + WHISPER_SERVER_RETRY_DELAY);
                }
            }
            return Err(error);
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }

    let mut state = whisper_server_state()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    stop_whisper_server_locked(&mut state);
    state.retry_after = Some(Instant::now() + WHISPER_SERVER_RETRY_DELAY);
    Err("Whisper server failed to become ready".to_string())
}

fn valid_zipformer_tokens(path: &std::path::Path) -> bool {
    std::fs::read_to_string(path)
        .map(|contents| {
            let trimmed = contents.trim_start();
            !trimmed.starts_with('{') && contents.lines().count() > 10
        })
        .unwrap_or(false)
}

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

fn server_path() -> PathBuf {
    data_dir().join("bin").join("whisper-server.exe")
}

/// Check if the model and both Whisper execution paths are available.
pub fn is_ready(model_id: &str) -> bool {
    model_path(model_id).exists() && cli_path().exists() && server_path().exists()
}

/// The slice of the overall progress bar that a single download owns.
///
/// Multi-file downloads used to emit an absolute 0-100 per file, so the bar
/// swept back to zero for every one of them. Giving each transfer a range keeps
/// the reported progress monotonic across the whole operation.
#[derive(Clone, Copy)]
struct ProgressRange {
    start: f64,
    end: f64,
}

impl ProgressRange {
    const FULL: ProgressRange = ProgressRange {
        start: 0.0,
        end: 100.0,
    };

    fn new(start: f64, end: f64) -> Self {
        Self { start, end }
    }

    /// Map a 0.0-1.0 fraction of this transfer onto the overall bar.
    fn at(&self, fraction: f64) -> f64 {
        self.start + (self.end - self.start) * fraction.clamp(0.0, 1.0)
    }
}

fn emit_progress(app: &tauri::AppHandle, value: f64) {
    let _ = app.emit("download-progress", value);
}

/// Emit at most one event per half a percent so a multi-gigabyte download does
/// not flood the frontend with one message per chunk.
fn report_progress(
    app: &tauri::AppHandle,
    range: ProgressRange,
    downloaded: u64,
    total: u64,
    last_emitted: &mut f64,
) {
    if total == 0 {
        return;
    }
    let value = range.at(downloaded as f64 / total as f64);
    if value - *last_emitted >= 0.5 {
        *last_emitted = value;
        emit_progress(app, value);
    }
}

/// Start a download, failing on non-2xx so an error page is never mistaken for
/// the file we asked for.
async fn begin_download(url: &str, label: &str) -> Result<reqwest::Response, String> {
    reqwest::get(url)
        .await
        .map_err(|e| format!("{} download failed: {}", label, e))?
        .error_for_status()
        .map_err(|e| format!("{} download failed: {}", label, e))
}

/// Download a URL into memory, emitting progress events within `range`.
/// Only for small files — anything model-sized should use `download_to_file`.
async fn download_bytes(
    app: &tauri::AppHandle,
    url: &str,
    label: &str,
    range: ProgressRange,
) -> Result<Vec<u8>, String> {
    let response = begin_download(url, label).await?;

    let total_size = response.content_length().unwrap_or(0);
    let mut downloaded: u64 = 0;
    let mut last_emitted = -1.0_f64;
    let mut bytes = Vec::with_capacity(total_size as usize);

    let mut stream = response.bytes_stream();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|e| format!("{} download error: {}", label, e))?;
        downloaded += chunk.len() as u64;
        bytes.extend_from_slice(&chunk);
        report_progress(app, range, downloaded, total_size, &mut last_emitted);
    }

    emit_progress(app, range.at(1.0));
    Ok(bytes)
}

/// Stream a URL straight to disk.
///
/// The bytes go to a sibling `.part` file that is renamed into place only after
/// the transfer completes. An interrupted download therefore leaves no file at
/// all, rather than a truncated one that every later `exists()` check would
/// report as a ready model.
async fn download_to_file(
    app: &tauri::AppHandle,
    url: &str,
    label: &str,
    dest: &Path,
    range: ProgressRange,
) -> Result<(), String> {
    if let Some(parent) = dest.parent() {
        tokio::fs::create_dir_all(parent)
            .await
            .map_err(|e| format!("Failed to create {}: {}", parent.display(), e))?;
    }

    let file_name = dest
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| format!("Invalid destination for {}", label))?;
    let temp = dest.with_file_name(format!("{}.part", file_name));
    let _ = tokio::fs::remove_file(&temp).await;

    match download_into(app, url, label, &temp, range).await {
        Ok(()) => tokio::fs::rename(&temp, dest)
            .await
            .map_err(|e| format!("Failed to finalize {}: {}", label, e)),
        Err(error) => {
            let _ = tokio::fs::remove_file(&temp).await;
            Err(error)
        }
    }
}

async fn download_into(
    app: &tauri::AppHandle,
    url: &str,
    label: &str,
    temp: &Path,
    range: ProgressRange,
) -> Result<(), String> {
    let response = begin_download(url, label).await?;

    let total_size = response.content_length().unwrap_or(0);
    let mut downloaded: u64 = 0;
    let mut last_emitted = -1.0_f64;

    let mut file = tokio::fs::File::create(temp)
        .await
        .map_err(|e| format!("Failed to create {}: {}", temp.display(), e))?;

    let mut stream = response.bytes_stream();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|e| format!("{} download error: {}", label, e))?;
        file.write_all(&chunk)
            .await
            .map_err(|e| format!("Failed to write {}: {}", label, e))?;
        downloaded += chunk.len() as u64;
        report_progress(app, range, downloaded, total_size, &mut last_emitted);
    }

    file.flush()
        .await
        .map_err(|e| format!("Failed to flush {}: {}", label, e))?;
    file.sync_all()
        .await
        .map_err(|e| format!("Failed to sync {}: {}", label, e))?;

    emit_progress(app, range.at(1.0));
    Ok(())
}

/// Write bytes through a `.part` file so a crash mid-write cannot leave a
/// half-written executable behind.
fn write_file_atomically(dest: &Path, bytes: &[u8]) -> Result<(), String> {
    let file_name = dest
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| format!("Invalid destination: {}", dest.display()))?;
    let temp = dest.with_file_name(format!("{}.part", file_name));

    std::fs::write(&temp, bytes)
        .map_err(|e| format!("Failed to write {}: {}", file_name, e))?;
    std::fs::rename(&temp, dest).map_err(|e| {
        let _ = std::fs::remove_file(&temp);
        format!("Failed to finalize {}: {}", file_name, e)
    })
}

/// Download the specified GGML model and Whisper binaries (if not already present).
/// Emits "download-progress" events to the frontend.
pub async fn download_model(app: tauri::AppHandle, model_id: &str) -> Result<(), String> {
    let model = model_path(model_id);
    let cli = cli_path();
    let server = server_path();

    // Look up model URL from the available models list
    let model_info = settings::available_models()
        .into_iter()
        .find(|m| m.id == model_id)
        .ok_or_else(|| format!("Unknown model: {}", model_id))?;

    let needs_model = !model.exists();
    let needs_binaries = !cli.exists() || !server.exists();

    // If all runtime files exist, nothing to do.
    if !needs_model && !needs_binaries {
        emit_progress(&app, 100.0);
        return Ok(());
    }

    emit_progress(&app, 0.0);

    // The model dwarfs the ~4 MB binary zip, so it owns almost the whole bar
    // when both are needed.
    let (model_range, binaries_range) = if needs_model && needs_binaries {
        (
            ProgressRange::new(0.0, 92.0),
            ProgressRange::new(92.0, 100.0),
        )
    } else {
        (ProgressRange::FULL, ProgressRange::FULL)
    };

    // --- Download model ---
    if needs_model {
        download_to_file(&app, &model_info.url, "model", &model, model_range).await?;
    }

    // --- Download whisper-cli.exe and whisper-server.exe (~4MB zip) ---
    if needs_binaries {
        let bin_dir = data_dir().join("bin");
        tokio::fs::create_dir_all(&bin_dir)
            .await
            .map_err(|e| format!("Failed to create bin dir: {}", e))?;

        let zip_bytes =
            download_bytes(&app, WHISPER_CLI_ZIP_URL, "whisper-cli", binaries_range).await?;

        // Extract .exe and .dll files from the zip in a single pass
        let cursor = Cursor::new(zip_bytes);
        let mut archive =
            zip::ZipArchive::new(cursor).map_err(|e| format!("Failed to open zip: {}", e))?;

        let mut found_cli = cli.exists();
        let mut found_server = server.exists();
        for i in 0..archive.len() {
            let mut file = archive
                .by_index(i)
                .map_err(|e| format!("Zip entry error: {}", e))?;

            let name = file.name().to_string();
            let file_name = name.rsplit('/').next().unwrap_or(&name);

            let should_extract = file_name == "whisper-cli.exe"
                || file_name == "whisper-server.exe"
                || file_name.ends_with(".dll");

            if should_extract && !file.is_dir() {
                let dest = bin_dir.join(file_name);
                let mut buf = Vec::new();
                file.read_to_end(&mut buf)
                    .map_err(|e| format!("Failed to read {} from zip: {}", file_name, e))?;
                write_file_atomically(&dest, &buf)?;

                if file_name == "whisper-cli.exe" {
                    found_cli = true;
                } else if file_name == "whisper-server.exe" {
                    found_server = true;
                }
            }
        }

        if !found_cli {
            return Err(
                "whisper-cli.exe not found in the downloaded zip archive".to_string(),
            );
        }
        if !found_server {
            return Err(
                "whisper-server.exe not found in the downloaded zip archive".to_string(),
            );
        }
    }

    emit_progress(&app, 100.0);
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
    configure_whisper_command(&mut cmd);

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
/// Prefer the persistent server and fall back to the CLI if it cannot start.
pub async fn transcribe_partial(
    samples: Vec<f32>,
    sample_rate: u32,
    model_id: &str,
    language: &str,
    prompt: &str,
) -> Result<String, String> {
    match transcribe_partial_server(
        samples.clone(),
        sample_rate,
        model_id,
        language,
        prompt,
    )
    .await
    {
        Ok(text) => Ok(text),
        Err(error) => {
            eprintln!("[whisper-server] {}; falling back to whisper-cli", error);
            transcribe_partial_cli(samples, sample_rate, model_id, language, prompt).await
        }
    }
}

async fn transcribe_partial_server(
    samples: Vec<f32>,
    sample_rate: u32,
    model_id: &str,
    language: &str,
    prompt: &str,
) -> Result<String, String> {
    let port = ensure_whisper_server(model_id).await?;
    let wav_bytes = tokio::task::spawn_blocking(move || samples_to_wav_bytes(&samples, sample_rate))
        .await
        .map_err(|e| format!("WAV task error: {}", e))??;

    let file_part = reqwest::multipart::Part::bytes(wav_bytes)
        .file_name("audio.wav")
        .mime_str("audio/wav")
        .map_err(|e| format!("Multipart error: {}", e))?;

    let mut form = reqwest::multipart::Form::new()
        .part("file", file_part)
        .text("response_format", "json")
        .text("temperature", "0");

    if language != "auto" {
        form = form.text("language", language.to_string());
    }

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

    let response = whisper_http_client()
        .post(format!("http://127.0.0.1:{}/inference", port))
        .multipart(form)
        .send()
        .await
        .map_err(|e| format!("Whisper server request failed: {}", e))?;

    let status = response.status();
    let body = response
        .text()
        .await
        .map_err(|e| format!("Failed to read Whisper server response: {}", e))?;
    if !status.is_success() {
        return Err(format!("Whisper server error ({}): {}", status, body));
    }

    let json: serde_json::Value = serde_json::from_str(&body)
        .map_err(|e| format!("Failed to parse Whisper server response: {}", e))?;
    Ok(json["text"].as_str().unwrap_or("").trim().to_string())
}

/// CLI fallback for systems where whisper-server cannot be started.
/// Uses a separate temp file so it doesn't conflict with the final transcription.
async fn transcribe_partial_cli(
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
    configure_whisper_command(&mut cmd);

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
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!("whisper-cli failed: {}", stderr.trim()));
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
        && valid_zipformer_tokens(&dir.join("tokens.txt"))
}

/// Download the Zipformer Vietnamese model and sherpa-onnx CLI.
pub async fn download_zipformer(app: tauri::AppHandle) -> Result<(), String> {
    let model_dir = zipformer_model_dir();
    let sherpa_cli = sherpa_cli_path();

    if is_zipformer_ready() {
        emit_progress(&app, 100.0);
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

    // Split the bar into one slice per step (four model files + the sherpa CLI)
    // so progress only ever moves forward, including over files we skip.
    let total_steps = files.len() + 1;
    let slice = 100.0 / total_steps as f64;

    for (i, (url, filename)) in files.iter().enumerate() {
        let dest = model_dir.join(filename);
        let range = ProgressRange::new(i as f64 * slice, (i + 1) as f64 * slice);
        let needs_download = !dest.exists()
            || (*filename == "tokens.txt" && !valid_zipformer_tokens(&dest));
        if needs_download {
            download_to_file(&app, url, filename, &dest, range).await?;
        } else {
            emit_progress(&app, range.at(1.0));
        }
    }

    // Download sherpa-onnx shared library package (contains CLI binary + DLLs)
    if !sherpa_cli.exists() {
        // Leave the last few percent for extracting and copying the DLLs.
        let archive_range = ProgressRange::new(files.len() as f64 * slice, 96.0);

        // Download tar.bz2 to temp
        let temp_dir = std::env::temp_dir();
        let archive_path = temp_dir.join("sherpa-onnx-package.tar.bz2");
        let extract_dir = temp_dir.join("sherpa-onnx-extract");

        download_to_file(
            &app,
            SHERPA_ONNX_PACKAGE_URL,
            "sherpa-onnx",
            &archive_path,
            archive_range,
        )
        .await?;

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

    emit_progress(&app, 100.0);
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
