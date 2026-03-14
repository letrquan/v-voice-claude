mod keyboard;
mod settings;
mod transcribe;

use settings::SettingsState;
use std::sync::Mutex;
use tauri::menu::{Menu, MenuItem};
use tauri::tray::TrayIconBuilder;
use tauri::{Emitter, Manager, WebviewUrl, WebviewWindowBuilder};

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
    } else {
        transcribe::transcribe_partial(samples, sample_rate, &settings.model, &settings.language, &prompt)
            .await
    }
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
async fn type_text(text: String) -> Result<(), String> {
    tokio::task::spawn_blocking(move || keyboard::type_text(&text))
        .await
        .map_err(|e| format!("Task join error: {}", e))?
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

/// Build a fingerprint string for a monitor: "name_widthxheight"
fn monitor_fingerprint(monitor: &tauri::Monitor) -> String {
    let size = monitor.size();
    let name = monitor.name().map(|s| s.clone()).unwrap_or_else(|| "unknown".to_string());
    format!("{}_{}×{}", name, size.width, size.height)
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

    let _factor = win.scale_factor().unwrap_or(1.0);
    let center_x = x + 24.0; // half of pill width (48)
    let center_y = y + 24.0;

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

    // Fallback to primary
    let monitor = best_monitor.or_else(|| {
        win.primary_monitor().ok().flatten().as_ref().and_then(|_| monitors.first())
    });

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
    let monitor = win.primary_monitor().ok()??;
    let fp = monitor_fingerprint(&monitor);
    let scale = monitor.scale_factor();
    let mon_x = monitor.position().x as f64 / scale;
    let mon_y = monitor.position().y as f64 / scale;

    let settings = state.0.lock().unwrap();
    let pos = settings.pill_positions.get(&fp)?;

    Some(PillPositionResult {
        x: mon_x + pos.x,
        y: mon_y + pos.y,
    })
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

    tauri::Builder::default()
        .plugin(tauri_plugin_global_shortcut::Builder::new().build())
        .plugin(tauri_plugin_store::Builder::new().build())
        .invoke_handler(tauri::generate_handler![
            is_model_ready,
            download_model,
            download_specific_model,
            transcribe,
            transcribe_streaming,
            type_text,
            download_zipformer_model,
            is_zipformer_model_ready,
            settings::get_settings,
            settings::set_settings,
            settings::get_available_models,
            settings::get_downloaded_models,
            settings::is_model_downloaded,
            settings::delete_model,
            settings::get_zipformer_model,
            settings::is_zipformer_ready,
            open_settings,
            save_pill_position,
            get_pill_position,
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

                // Try to restore saved position for this monitor
                let loaded_settings = app.state::<SettingsState>().0.lock().unwrap().clone();
                let mut restored = false;

                if let Ok(Some(monitor)) = window.primary_monitor() {
                    let fp = monitor_fingerprint(&monitor);
                    if let Some(pos) = loaded_settings.pill_positions.get(&fp) {
                        let scale = monitor.scale_factor();
                        let mon_x = monitor.position().x as f64 / scale;
                        let mon_y = monitor.position().y as f64 / scale;
                        let mon_w = monitor.size().width as f64 / scale;
                        let mon_h = monitor.size().height as f64 / scale;

                        let abs_x = mon_x + pos.x;
                        let abs_y = mon_y + pos.y;

                        // Validate position is still within monitor bounds
                        if abs_x >= mon_x && abs_x < mon_x + mon_w - 24.0
                            && abs_y >= mon_y && abs_y < mon_y + mon_h - 24.0
                        {
                            let _ = window.set_position(tauri::Position::Logical(
                                tauri::LogicalPosition::new(abs_x, abs_y),
                            ));
                            restored = true;
                        }
                    }

                    // Fallback: bottom-center of primary monitor
                    if !restored {
                        let screen_size = monitor.size();
                        let screen_pos = monitor.position();
                        let scale = monitor.scale_factor();

                        let sw = screen_size.width as f64 / scale;
                        let sh = screen_size.height as f64 / scale;
                        let sx = screen_pos.x as f64 / scale;
                        let sy = screen_pos.y as f64 / scale;

                        let win_w = 48.0;
                        let win_h = 48.0;
                        let x = sx + (sw / 2.0) - (win_w / 2.0);
                        let y = sy + sh - win_h - 24.0;

                        let _ = window.set_position(tauri::Position::Logical(
                            tauri::LogicalPosition::new(x, y),
                        ));
                    }
                }
            }

            Ok(())
        })
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
