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
        )
        .await
    } else {
        transcribe::transcribe_partial(samples, sample_rate, &settings.model, &settings.language)
            .await
    }
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

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
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
            settings::get_settings,
            settings::set_settings,
            settings::get_available_models,
            settings::get_downloaded_models,
            settings::is_model_downloaded,
            settings::delete_model,
            open_settings,
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

            // ── Position main window at bottom-center ──
            if let Some(window) = app.get_webview_window("main") {
                let _ = window.set_always_on_top(true);

                if let Ok(Some(monitor)) = window.primary_monitor() {
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

            Ok(())
        })
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
