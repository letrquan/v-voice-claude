mod keyboard;
mod settings;
mod transcribe;

use settings::SettingsState;
use std::sync::Mutex;
use tauri::menu::{Menu, MenuItem};
use tauri::tray::TrayIconBuilder;
use tauri::{Emitter, Manager, RunEvent, WebviewUrl, WebviewWindowBuilder};

#[tauri::command]
async fn is_model_ready(state: tauri::State<'_, SettingsState>) -> Result<bool, String> {
    let model_id = state.0.lock().unwrap().model.clone();
    Ok(transcribe::is_ready(&model_id))
}

#[tauri::command]
async fn download_model(
    app: tauri::AppHandle,
    state: tauri::State<'_, SettingsState>,
) -> Result<(), String> {
    let model_id = state.0.lock().unwrap().model.clone();
    transcribe::download_model(app, &model_id).await
}

#[tauri::command]
async fn download_specific_model(
    app: tauri::AppHandle,
    model_id: String,
) -> Result<(), String> {
    transcribe::download_model(app, &model_id).await
}

#[tauri::command]
async fn transcribe(
    samples: Vec<f32>,
    sample_rate: u32,
    state: tauri::State<'_, SettingsState>,
) -> Result<String, String> {
    let settings = state.0.lock().unwrap().clone();
    if settings.transcription_mode == "cloud" {
        transcribe::transcribe_cloud(
            samples,
            sample_rate,
            &settings.language,
            &settings.cloud_provider,
            &settings.cloud_api_key,
            "",
        )
        .await
    } else if settings.local_engine == "zipformer" {
        transcribe::transcribe_zipformer(samples, sample_rate).await
    } else if settings.local_engine == "granite" {
        transcribe::transcribe_granite(
            samples,
            sample_rate,
            settings.granite_api_port,
            &settings.language,
        )
        .await
    } else {
        transcribe::transcribe_audio(samples, sample_rate, &settings.model, &settings.language)
            .await
    }
}

#[tauri::command]
async fn transcribe_streaming(
    samples: Vec<f32>,
    sample_rate: u32,
    prompt: String,
    state: tauri::State<'_, SettingsState>,
) -> Result<String, String> {
    let settings = state.0.lock().unwrap().clone();
    if settings.transcription_mode == "cloud" {
        transcribe::transcribe_cloud(
            samples,
            sample_rate,
            &settings.language,
            &settings.cloud_provider,
            &settings.cloud_api_key,
            &prompt,
        )
        .await
    } else if settings.local_engine == "zipformer" {
        transcribe::transcribe_zipformer(samples, sample_rate).await
    } else if settings.local_engine == "granite" {
        transcribe::transcribe_granite(
            samples,
            sample_rate,
            settings.granite_api_port,
            &settings.language,
        )
        .await
    } else {
        transcribe::transcribe_partial(samples, sample_rate, &settings.model, &settings.language, &prompt)
            .await
    }
}

#[tauri::command]
fn stop_whisper_server() {
    transcribe::stop_whisper_server();
}

#[tauri::command]
async fn download_zipformer_model(
    app: tauri::AppHandle,
) -> Result<(), String> {
    transcribe::download_zipformer(app).await
}

#[tauri::command]
async fn is_zipformer_model_ready() -> Result<bool, String> {
    Ok(transcribe::is_zipformer_ready())
}

#[tauri::command]
async fn download_granite_model(
    app: tauri::AppHandle,
) -> Result<(), String> {
    transcribe::download_granite(app).await
}

#[tauri::command]
async fn is_granite_model_ready() -> Result<bool, String> {
    Ok(transcribe::is_granite_ready())
}

#[tauri::command]
async fn start_granite_server(
    state: tauri::State<'_, SettingsState>,
) -> Result<(), String> {
    let port = state.0.lock().unwrap().granite_api_port;
    transcribe::start_granite_server(port).await
}

#[tauri::command]
async fn stop_granite_server(
    state: tauri::State<'_, SettingsState>,
) -> Result<(), String> {
    let port = state.0.lock().unwrap().granite_api_port;
    transcribe::stop_granite_server(port).await
}

#[tauri::command]
async fn is_granite_server_running(
    state: tauri::State<'_, SettingsState>,
) -> Result<bool, String> {
    let port = state.0.lock().unwrap().granite_api_port;
    Ok(transcribe::is_granite_server_running(port).await)
}

#[tauri::command]
async fn type_text(text: String) -> Result<(), String> {
    tokio::task::spawn_blocking(move || keyboard::type_text(&text))
        .await
        .map_err(|e| format!("Task join error: {}", e))?
}

// ─── Session log ───
// The pill has no visible console and never holds focus while the hotkey is
// held, so front-end events are mirrored to a file that can be read afterwards.

const LOG_MAX_BYTES: u64 = 2 * 1024 * 1024;

fn log_file_path() -> std::path::PathBuf {
    settings::data_dir().join("logs").join("session.log")
}

#[tauri::command]
fn log_event(line: String) {
    use std::io::Write;

    let path = log_file_path();
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }

    // Rotate rather than grow without bound; one previous file is enough to
    // cover the session before the one being reported.
    if std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0) > LOG_MAX_BYTES {
        let _ = std::fs::rename(&path, path.with_extension("log.1"));
    }

    if let Ok(mut file) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
    {
        let _ = writeln!(file, "{}", line);
    }
}

#[tauri::command]
fn get_log_path() -> String {
    log_file_path().to_string_lossy().to_string()
}

/// Open or focus the settings window
fn open_settings_window(app: &tauri::AppHandle) {
    if let Some(win) = app.get_webview_window("settings") {
        let _ = win.set_focus();
        let _ = win.show();
    } else {
        let _win = WebviewWindowBuilder::new(app, "settings", WebviewUrl::App("index.html".into()))
            .title("V Voice Settings")
            .inner_size(720.0, 640.0)
            .min_inner_size(500.0, 400.0)
            .resizable(true)
            .center()
            .decorations(true)
            .build();
    }
}

#[tauri::command]
fn open_settings(app: tauri::AppHandle) {
    open_settings_window(&app);
}

/// Logical size of the idle pill window.
const PILL_SIZE: f64 = 48.0;

/// Build a fingerprint string for a monitor: "name_widthxheight"
fn monitor_fingerprint(monitor: &tauri::Monitor) -> String {
    let size = monitor.size();
    let name = monitor.name().map(|s| s.clone()).unwrap_or_else(|| "unknown".to_string());
    format!("{}_{}×{}", name, size.width, size.height)
}

/// Find a saved pill position on any monitor that is currently attached.
///
/// Positions are saved against whichever monitor the pill was dragged onto, so
/// restoring has to search every display. Looking only at the primary meant a
/// pill parked on a second screen was written under a fingerprint that was
/// never read back, and it silently reverted to bottom-centre on every launch.
fn saved_pill_position(
    window: &tauri::WebviewWindow,
    settings: &settings::AppSettings,
) -> Option<(f64, f64)> {
    if settings.pill_positions.is_empty() {
        return None;
    }

    // Primary first, so a setup with a saved position on several displays
    // restores predictably.
    let mut candidates: Vec<tauri::Monitor> = Vec::new();
    if let Ok(Some(primary)) = window.primary_monitor() {
        candidates.push(primary);
    }
    if let Ok(monitors) = window.available_monitors() {
        for monitor in monitors {
            let fingerprint = monitor_fingerprint(&monitor);
            if !candidates
                .iter()
                .any(|known| monitor_fingerprint(known) == fingerprint)
            {
                candidates.push(monitor);
            }
        }
    }

    for monitor in &candidates {
        let Some(pos) = settings.pill_positions.get(&monitor_fingerprint(monitor)) else {
            continue;
        };

        let scale = monitor.scale_factor();
        let mon_x = monitor.position().x as f64 / scale;
        let mon_y = monitor.position().y as f64 / scale;
        let mon_w = monitor.size().width as f64 / scale;
        let mon_h = monitor.size().height as f64 / scale;

        let abs_x = mon_x + pos.x;
        let abs_y = mon_y + pos.y;

        // Reject coordinates left over from a different resolution or scale
        // factor, so the pill can never restore off-screen.
        if abs_x >= mon_x
            && abs_x <= mon_x + mon_w - PILL_SIZE
            && abs_y >= mon_y
            && abs_y <= mon_y + mon_h - PILL_SIZE
        {
            return Some((abs_x, abs_y));
        }
    }

    None
}

#[tauri::command]
fn save_pill_position(
    app: tauri::AppHandle,
    state: tauri::State<'_, SettingsState>,
    x: f64,
    y: f64,
) -> Result<(), String> {
    let win = app.get_webview_window("main")
        .ok_or("Main window not found")?;

    // Find which monitor the pill center is on
    let monitors = win.available_monitors()
        .map_err(|e| format!("Cannot list monitors: {}", e))?;

    let center_x = x + PILL_SIZE / 2.0;
    let center_y = y + PILL_SIZE / 2.0;

    let mut best_monitor: Option<&tauri::Monitor> = None;
    for mon in &monitors {
        let pos = mon.position();
        let size = mon.size();
        let scale = mon.scale_factor();
        let mx = pos.x as f64 / scale;
        let my = pos.y as f64 / scale;
        let mw = size.width as f64 / scale;
        let mh = size.height as f64 / scale;

        if center_x >= mx && center_x < mx + mw && center_y >= my && center_y < my + mh {
            best_monitor = Some(mon);
            break;
        }
    }

    // The pill can sit outside every monitor's bounds mid-drag, or after a
    // display is unplugged. Anchor it to the primary in that case so the
    // fingerprint we save under is one restore will actually look at.
    let primary = win.primary_monitor().ok().flatten();
    let monitor = best_monitor
        .or(primary.as_ref())
        .or_else(|| monitors.first());

    if let Some(mon) = monitor {
        let fp = monitor_fingerprint(mon);
        let scale = mon.scale_factor();
        let mon_x = mon.position().x as f64 / scale;
        let mon_y = mon.position().y as f64 / scale;

        // Store position relative to monitor origin
        let rel_x = x - mon_x;
        let rel_y = y - mon_y;

        let mut settings = state.0.lock().unwrap();
        settings.pill_positions.insert(
            fp,
            settings::PillPosition { x: rel_x, y: rel_y },
        );
        let settings_clone = settings.clone();
        drop(settings);

        // Persist (best effort)
        let _ = save_pill_settings(&app, &settings_clone);
    }

    Ok(())
}

/// Helper to save just settings (without emitting settings-changed event)
fn save_pill_settings(app: &tauri::AppHandle, settings: &settings::AppSettings) -> Result<(), String> {
    use tauri_plugin_store::StoreExt;
    let store = app.store("settings.json")
        .map_err(|e| format!("Store error: {}", e))?;
    let val = serde_json::to_value(settings)
        .map_err(|e| format!("Serialize error: {}", e))?;
    store.set("settings", val);
    store.save().map_err(|e| format!("Save error: {}", e))?;
    Ok(())
}

#[derive(serde::Serialize)]
struct PillPositionResult {
    x: f64,
    y: f64,
}

#[tauri::command]
fn get_pill_position(
    app: tauri::AppHandle,
    state: tauri::State<'_, SettingsState>,
) -> Option<PillPositionResult> {
    let win = app.get_webview_window("main")?;
    let settings = state.0.lock().unwrap().clone();
    let (x, y) = saved_pill_position(&win, &settings)?;
    Some(PillPositionResult { x, y })
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    // Auto-grant WebView2 microphone permission without showing a browser-style popup.
    // --use-fake-ui-for-media-stream bypasses the permission dialog but still uses the real mic.
    #[cfg(windows)]
    {
        let mut args = std::env::var("WEBVIEW2_ADDITIONAL_BROWSER_ARGUMENTS").unwrap_or_default();
        if !args.is_empty() { args.push(' '); }
        args.push_str("--autoplay-policy=no-user-gesture-required --use-fake-ui-for-media-stream");
        std::env::set_var("WEBVIEW2_ADDITIONAL_BROWSER_ARGUMENTS", &args);
    }

    let app = tauri::Builder::default()
        .plugin(tauri_plugin_global_shortcut::Builder::new().build())
        .plugin(tauri_plugin_store::Builder::new().build())
        .invoke_handler(tauri::generate_handler![
            is_model_ready,
            download_model,
            download_specific_model,
            transcribe,
            transcribe_streaming,
            stop_whisper_server,
            type_text,
            download_zipformer_model,
            is_zipformer_model_ready,
            download_granite_model,
            is_granite_model_ready,
            start_granite_server,
            stop_granite_server,
            is_granite_server_running,
            settings::get_settings,
            settings::set_settings,
            settings::get_available_models,
            settings::get_downloaded_models,
            settings::is_model_downloaded,
            settings::delete_model,
            settings::get_zipformer_model,
            settings::is_zipformer_ready,
            settings::get_granite_model,
            settings::is_granite_ready,
            open_settings,
            save_pill_position,
            get_pill_position,
            log_event,
            get_log_path,
        ])
        .setup(|app| {
            // ── Load settings into managed state ──
            let loaded = settings::load_settings(&app.handle());
            app.manage(SettingsState(Mutex::new(loaded)));

            // ── System tray ──
            let show_hide =
                MenuItem::with_id(app, "show_hide", "Show / Hide", true, None::<&str>)?;
            let settings_item =
                MenuItem::with_id(app, "settings", "Settings", true, None::<&str>)?;
            let quit = MenuItem::with_id(app, "quit", "Quit", true, None::<&str>)?;

            let menu = Menu::with_items(app, &[&show_hide, &settings_item, &quit])?;

            let _tray = TrayIconBuilder::new()
                .icon(app.default_window_icon().unwrap().clone())
                .menu(&menu)
                .show_menu_on_left_click(false)
                .on_menu_event(move |app, event| match event.id.as_ref() {
                    "show_hide" => {
                        if let Some(win) = app.get_webview_window("main") {
                            if win.is_visible().unwrap_or(false) {
                                let _ = win.hide();
                            } else {
                                let _ = win.show();
                                let _ = win.set_focus();
                            }
                        }
                    }
                    "settings" => {
                        open_settings_window(app);
                    }
                    "quit" => {
                        let _ = app.emit("app-quit", ());
                        app.exit(0);
                    }
                    _ => {}
                })
                .build(app)?;

            // ── Position main window ──
            if let Some(window) = app.get_webview_window("main") {
                let _ = window.set_always_on_top(true);

                // Try to restore a saved position from any attached monitor
                let loaded_settings = app.state::<SettingsState>().0.lock().unwrap().clone();

                if let Some((x, y)) = saved_pill_position(&window, &loaded_settings) {
                    let _ = window.set_position(tauri::Position::Logical(
                        tauri::LogicalPosition::new(x, y),
                    ));
                } else if let Ok(Some(monitor)) = window.primary_monitor() {
                    // Fallback: bottom-center of primary monitor
                    let screen_size = monitor.size();
                    let screen_pos = monitor.position();
                    let scale = monitor.scale_factor();

                    let sw = screen_size.width as f64 / scale;
                    let sh = screen_size.height as f64 / scale;
                    let sx = screen_pos.x as f64 / scale;
                    let sy = screen_pos.y as f64 / scale;

                    let x = sx + (sw / 2.0) - (PILL_SIZE / 2.0);
                    let y = sy + sh - PILL_SIZE - 24.0;

                    let _ = window.set_position(tauri::Position::Logical(
                        tauri::LogicalPosition::new(x, y),
                    ));
                }
            }

            Ok(())
        })
        .build(tauri::generate_context!())
        .expect("error while building tauri application");

    app.run(|_, event| {
        if matches!(event, RunEvent::ExitRequested { .. } | RunEvent::Exit) {
            transcribe::stop_whisper_server();
        }
    });
}
