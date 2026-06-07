import React from "react";
import ReactDOM from "react-dom/client";
import { App } from "./App.js";
import "@copilotkit/react-ui/styles.css";
import "./styles.css";

const container = document.getElementById("root");
if (container === null) {
  throw new Error("Root container #root not found in index.html");
}

ReactDOM.createRoot(container).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);
