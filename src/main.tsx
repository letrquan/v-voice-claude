import React from "react";
import ReactDOM from "react-dom/client";
import { getCurrentWindow } from "@tauri-apps/api/window";
import "@fontsource/jetbrains-mono/400.css";
import "@fontsource/jetbrains-mono/500.css";
import "@fontsource/inter/400.css";
import "@fontsource/inter/500.css";
import App from "./App";
import SettingsPage from "./pages/SettingsPage";
import "./index.css";

const windowLabel = getCurrentWindow().label;

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    {windowLabel === "settings" ? <SettingsPage /> : <App />}
  </React.StrictMode>
);
