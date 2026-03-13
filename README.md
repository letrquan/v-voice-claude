<p align="center">
  <img src="src-tauri/icons/icon.png" width="80" height="80" alt="V Voice logo" />
</p>

<h1 align="center">V Voice</h1>

<p align="center">
  <strong>A floating voice-to-text desktop widget powered by Whisper</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/version-0.1.0-7DF9C4?style=flat-square" />
  <img src="https://img.shields.io/badge/platform-Windows-A78BFA?style=flat-square" />
  <img src="https://img.shields.io/badge/built_with-Tauri_2-FFB547?style=flat-square" />
  <img src="https://img.shields.io/badge/license-MIT-F0EEF8?style=flat-square" />
</p>

---

**V Voice** is a lightweight, always-on-top floating pill widget that converts your speech to text and types it directly into any focused application. Think of it as a system-level dictation tool — press a hotkey, speak, and your words appear wherever your cursor is.

## ✨ Features

### 🎙️ Dual Transcription Engines
- **Local Mode** — Runs [Whisper.cpp](https://github.com/ggml-org/whisper.cpp) entirely on your machine. No internet required. Choose from `tiny` to `large` models.
- **Cloud Mode** — Send audio to **OpenAI Whisper API** or **Groq** (blazing fast, free tier available) for higher accuracy without GPU requirements.

### ⚡ Real-Time Streaming
- Live partial transcriptions appear as you speak (updates every ~2 seconds)
- Final high-accuracy transcription runs when you stop talking
- Blinking cursor indicator shows streaming is active

### 🎯 Smart Voice Activity Detection (VAD)
- Automatically detects when you start and stop speaking
- Configurable silence threshold and duration
- Visual ring indicator shows silence countdown
- 1.2s warm-up grace period prevents false triggers on mic initialization

### 🇻🇳 Vietnamese Language Support
- Prioritized Vietnamese in language selector
- Smart auto-switching from English-only models to multilingual models
- Vietnamese-specific prompt engineering for accurate diacritics and tone marks

### 🖥️ Multi-Monitor Awareness
- Remembers pill position per-monitor using display fingerprinting
- Drag the pill anywhere — it restores to the exact position on restart
- Gracefully falls back to bottom-center if monitor configuration changes

### 🎨 Minimal, Beautiful UI
- Glassmorphic dark pill widget (48px circle when idle)
- Expands to show waveform visualization and live transcript
- Sound-reactive glow ring that pulses with your voice
- Fully draggable with system tray integration

---

## 🚀 Quick Start

### Install from Release

1. Download `V Voice_0.1.0_x64-setup.exe` from the [Releases](../../releases) page
2. Run the installer
3. Launch **V Voice** — the pill appears at the bottom-center of your screen

### Default Hotkeys

| Hotkey | Action |
|--------|--------|
| `Ctrl+Shift+Space` | **Hold to talk** — release to transcribe and type |
| `Ctrl+Shift+Q` | Quit the application |

### First Use

1. On first launch, V Voice downloads the `tiny.en` Whisper model (~75 MB)
2. Wait for the download progress to complete
3. Hold `Ctrl+Shift+Space`, speak, and release — your words will be typed into the focused app

---

## ⚙️ Settings

Right-click the system tray icon → **Settings**, or click the gear icon.

### Transcription Engine
Toggle between **Local** (Whisper on your machine) and **Cloud** (OpenAI / Groq API).

### Cloud Provider Setup

| Provider | Model | Speed | Get API Key |
|----------|-------|-------|-------------|
| **OpenAI** | `whisper-1` | Good | [platform.openai.com](https://platform.openai.com) |
| **Groq** | `whisper-large-v3-turbo` | Blazing fast | [console.groq.com](https://console.groq.com) (free tier) |

### Local Models

| Model | Size | Speed | Accuracy | Best For |
|-------|------|-------|----------|----------|
| `tiny.en` | 75 MB | ⚡⚡⚡ | ★★ | Quick English dictation |
| `base.en` | 142 MB | ⚡⚡⚡ | ★★★ | Everyday English use |
| `small.en` | 466 MB | ⚡⚡ | ★★★★ | Accurate English |
| `small` | 466 MB | ⚡⚡ | ★★★★ | Multilingual (recommended for Vietnamese) |
| `medium` | 1.5 GB | ⚡ | ★★★★★ | High accuracy, any language |
| `large-v3-turbo` | 1.6 GB | ⚡ | ★★★★★ | Maximum accuracy |

---

## 🛠️ Development

### Prerequisites

- [Node.js](https://nodejs.org/) ≥ 18
- [Rust](https://rustup.rs/) (stable)
- [Tauri CLI](https://v2.tauri.app/start/prerequisites/) prerequisites

### Setup

```bash
# Clone the repo
git clone https://github.com/yourusername/v-voice-claude.git
cd v-voice-claude

# Install frontend dependencies
npm install

# Run in development mode
npm run tauri dev
```

### Build Release

```bash
npm run tauri build
```

The installer will be generated at:
```
src-tauri/target/release/bundle/nsis/V Voice_0.1.0_x64-setup.exe
```

### Project Structure

```
v-voice-claude/
├── src/                    # React frontend
│   ├── App.tsx             # Main app component (state machine, VAD, streaming)
│   ├── index.css           # All styles (pill, settings, animations)
│   ├── components/
│   │   └── Waveform.tsx    # Audio waveform visualization
│   ├── hooks/
│   │   └── useAudioCapture.ts  # WebAudio mic capture + buffer snapshot
│   └── pages/
│       └── SettingsPage.tsx    # Settings UI
├── src-tauri/              # Rust backend
│   └── src/
│       ├── lib.rs          # Tauri commands, window management, pill position
│       ├── transcribe.rs   # Whisper CLI + Cloud API transcription
│       ├── settings.rs     # Settings schema, model definitions
│       └── keyboard.rs     # Simulated typing via enigo
├── public/
│   └── audio-processor.js  # AudioWorklet for PCM capture
└── package.json
```

---

## 🔧 Tech Stack

| Layer | Technology |
|-------|-----------|
| Framework | [Tauri 2](https://v2.tauri.app/) |
| Frontend | React 18 + TypeScript + Vite |
| Backend | Rust (tokio async runtime) |
| Speech-to-Text | [whisper.cpp](https://github.com/ggml-org/whisper.cpp) / OpenAI API / Groq API |
| Audio Capture | Web Audio API + AudioWorklet |
| Keyboard Simulation | [enigo](https://crates.io/crates/enigo) |
| Styling | Vanilla CSS + TailwindCSS 4 |

---

## 📝 License

MIT © V Voice

---

<p align="center">
  <sub>Built with 🎤 and ❤️</sub>
</p>
