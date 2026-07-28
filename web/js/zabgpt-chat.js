/**
 * zabgpt-chat.js — standalone ZabGPT chat panel.
 * Completely independent of the AR module so it works even if AR fails.
 */

import { askZabGPT } from "./api.js";

const fab   = document.getElementById("zabgpt-fab");
const panel = document.getElementById("zabgpt-panel");
const close = document.getElementById("zabgpt-close");
const input = document.getElementById("zabgpt-input");
const send  = document.getElementById("zabgpt-send");
const msgs  = document.getElementById("zabgpt-messages");

if (!fab || !panel) {
  console.warn("ZabGPT: panel elements not found");
} else {
  fab.addEventListener("click", () => {
    panel.classList.add("open");
    input.focus();
  });

  close.addEventListener("click", () => {
    panel.classList.remove("open");
  });

  send.addEventListener("click", sendMessage);

  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  });
}

function appendMsg(text, role) {
  const div = document.createElement("div");
  div.className = "zabgpt-msg " + role;
  div.textContent = text;
  msgs.appendChild(div);
  msgs.scrollTop = msgs.scrollHeight;
  return div;
}

async function sendMessage() {
  const query = input.value.trim();
  if (!query) return;

  input.value = "";
  input.blur(); // dismiss keyboard so messages stay visible
  send.disabled = true;
  appendMsg(query, "user");
  const loading = appendMsg("Thinking... (first response may take ~10s)", "loading");

  try {
    const result = await askZabGPT(query);
    loading.remove();
    appendMsg(result.answer ?? "No answer returned.", "bot");
  } catch (err) {
    loading.remove();
    appendMsg("Error: " + err.message, "bot");
  } finally {
    send.disabled = false;
    // Don't call input.focus() — on mobile it reopens the keyboard
    // and pushes messages out of view
  }
}
