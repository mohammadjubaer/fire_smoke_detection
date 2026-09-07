import React from "react";
import ReactDOM from "react-dom/client";
import "./index.css";
import App from "./App.tsx";

// Hide loading screen when React mounts
const loadingEl = document.getElementById('loading');
if (loadingEl) {
  loadingEl.style.display = 'none';
}

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
