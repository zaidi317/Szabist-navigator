/**
 * main-home.js — bootstraps destination.html
 * Fetches /api/locations, renders the destination list, wires navigation.
 */

import { getLocations } from "./api.js";
import { state } from "./state.js";

const FLOOR_LABEL = { 0: "Ground floor", 1: "First floor", 2: "Second floor" };
const LOCATION_EMOJI = {
  Cafe:         "🍽️",
  Courtyard:    "🌿",
  FountainArea: "⛲",
  helmetarea:   "🪖",
  SSC:          "🏛️",
  stairs:       "🚪",
};

async function init() {
  const list    = document.getElementById("dest-list");
  const banner  = document.getElementById("banner");
  const startBtn = document.getElementById("btn-start");

  if (!list) return; // not on destination page

  let selected = null;

  function showBanner(msg, type = "error") {
    banner.textContent = msg;
    banner.className = `banner show ${type}`;
  }

  try {
    const data = await getLocations();
    list.innerHTML = "";

    data.locations.forEach(loc => {
      const li = document.createElement("li");
      li.className = "dest-item";
      li.dataset.nodeId = loc.node_id;
      li.innerHTML = `
        <div class="dest-icon">${LOCATION_EMOJI[loc.node_id] ?? "📍"}</div>
        <div>
          <div class="dest-name">${loc.label}</div>
          <div class="dest-floor">${FLOOR_LABEL[loc.floor] ?? `Floor ${loc.floor}`}</div>
        </div>
      `;
      li.addEventListener("click", () => {
        list.querySelectorAll(".dest-item").forEach(el => el.classList.remove("selected"));
        li.classList.add("selected");
        selected = loc;
        if (startBtn) startBtn.disabled = false;
      });
      list.appendChild(li);
    });
  } catch (err) {
    showBanner(err.message);
  }

  if (startBtn) {
    startBtn.disabled = true;
    startBtn.addEventListener("click", () => {
      if (!selected) return;
      state.setDestination(selected.node_id, selected.label);
      // Always recalibrate at the start of a new AR session
      state.setCalibration(null, null);
      window.location.href = `calibrate.html?dest=${encodeURIComponent(selected.node_id)}`;
    });
  }
}

init();
