mod keyboard;
mod transcribe;

use tauri::Manager;

#[tauri::command]
async fn is_model_ready() -> Result<bool, String> {
    Ok(transcribe::is_ready())
}

#[tauri::command]
async fn download_model(app: tauri::AppHandle) -> Result<(), String> {
    transcribe::download_model(app).await
}

#[tauri::command]
async fn transcribe(samples: Vec<f32>, sample_rate: u32) -> Result<String, String> {
    transcribe::transcribe_audio(samples, sample_rate).await
}

#[tauri::command]
async fn type_text(text: String) -> Result<(), String> {
    // Run in blocking thread since enigo uses OS APIs
    tokio::task::spawn_blocking(move || keyboard::type_text(&text))
        .await
        .map_err(|e| format!("Task join error: {}", e))?
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_global_shortcut::Builder::new().build())
        .invoke_handler(tauri::generate_handler![
            is_model_ready,
            download_model,
            transcribe,
            type_text,
        ])
        .setup(|app| {
            if let Some(window) = app.get_webview_window("main") {
                let _ = window.set_always_on_top(true);

                // Position at bottom-center of primary monitor
                if let Ok(Some(monitor)) = window.primary_monitor() {
                    let screen_size = monitor.size();
                    let screen_pos = monitor.position();
                    let scale = monitor.scale_factor();

                    // Work in logical pixels
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
