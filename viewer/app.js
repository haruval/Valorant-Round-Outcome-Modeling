"use strict";

const $ = (id) => document.getElementById(id);
const state = { catalog: null, round: null, time: 0, playing: false, rate: 1,
  selected: null, filter: "all", rotate: false, lastTick: 0, frameIndex: -1,
  lastLog: "", lastFeature: -2, lastRolling: "", activeWindow: null, selectedModel: "temporal_cnn", points: [], loadVersion: 0, maps: {}, mapImages: new Map() };
const colors = { attacker: "#f47d66", defender: "#61d6bd" };
const featureLabels = ["Elapsed", "ATK alive", "DEF alive", "ATK kills", "DEF kills", "Spike planted",
  "ATK loadout", "DEF loadout", "ATK abilities", "DEF abilities", "First kill known", "First kill ATK"];
const temporalLabels = ["First kill time / window", "Elimination observed", "Last elimination age / window",
  "Plant age / window", "ATK kills · last 5s", "DEF kills · last 5s", "ATK abilities · last 5s", "DEF abilities · last 5s"];
const rollingLabels = ["ATK alive / 5", "DEF alive / 5", "ATK buy loadout", "DEF buy loadout", "Spike planted",
  "Plant age / 45s", "Defusing", "Defuse checkpoint", "ATK fresh / 5", "DEF fresh / 5", "ATK age / 20s", "DEF age / 20s"];

function element(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}
function formatTime(seconds, precise = false) {
  const safe = Math.max(0, seconds);
  const minutes = String(Math.floor(safe / 60)).padStart(2, "0");
  const remainder = precise ? (safe % 60).toFixed(1).padStart(4, "0") : String(Math.floor(safe % 60)).padStart(2, "0");
  return `${minutes}:${remainder}`;
}
function readable(value) { return String(value || "Unknown").replaceAll("-", " ").replaceAll("_", " "); }
function mapName(value) { return value.charAt(0).toUpperCase() + value.slice(1); }
function reportError(error) {
  $("error").textContent = error.message || String(error);
  $("error").hidden = false;
  setPlaying(false);
}
async function getJSON(url) {
  const response = await fetch(url, { credentials: "omit", cache: "no-store" });
  if (!response.ok) throw new Error(`Could not load local replay (${response.status}). Restart the local server and reload.`);
  return response.json();
}

function setPlaying(playing) {
  state.playing = playing;
  state.lastTick = 0;
  $("play").textContent = playing ? "Ⅱ" : "▶";
  $("play").setAttribute("aria-label", playing ? "Pause replay" : "Play replay");
}
function findFrame(time) {
  const frames = state.round.frames;
  let low = 0, high = frames.length - 1;
  while (low < high) {
    const middle = Math.ceil((low + high) / 2);
    if (frames[middle].t <= time + 0.000001) low = middle;
    else high = middle - 1;
  }
  return low;
}
function seek(time) {
  if (!state.round) return;
  state.time = Math.min(Math.max(time, 0), state.round.duration);
  if (state.time >= state.round.duration) setPlaying(false);
  render();
}
function fillMaps() {
  const series = $("series-select").value;
  const match = state.catalog.series.find((item) => item.id === series);
  $("match-title").textContent = match.teams.join(" vs ");
  $("tournament-name").textContent = match.tournament || "Tournament unknown";
  $("tournament-name").title = match.stage || "";
  const date = new Date(`${match.date}T12:00:00Z`);
  $("match-date").textContent = new Intl.DateTimeFormat("en-US", { month: "long", day: "numeric", year: "numeric", timeZone: "UTC" }).format(date) + (match.timezone === "UTC" ? " (UTC)" : "");
  $("match-date").title = `Match date · ${match.timezone}`;
  const maps = new Map();
  for (const round of state.catalog.rounds) {
    if (round.series_id === series) maps.set(round.game_id, round.map);
  }
  $("map-select").replaceChildren(...Array.from(maps, ([id, name]) => {
    const option = element("option", "", mapName(name)); option.value = id; return option;
  }));
  fillRounds();
}
function fillRounds() {
  const rounds = state.catalog.rounds.filter((r) => r.series_id === $("series-select").value && r.game_id === $("map-select").value);
  $("round-list").replaceChildren(...rounds.map((round) => {
    const button = element("button", "round-item");
    button.type = "button"; button.dataset.round = round.id;
    button.setAttribute("aria-label", `${readable(round.round_id)}, ${formatTime(round.duration)}`);
    const label = element("span");
    label.append(element("i", "round-dot"), element("span", "", `R ${round.round_id.replace("round-", "").padStart(2, "0")}`));
    button.append(label, element("span", "round-duration", formatTime(round.duration)));
    button.addEventListener("click", () => loadRound(round.id).catch(reportError));
    return button;
  }));
  if (rounds.length) loadRound(rounds[0].id).catch(reportError);
}
async function loadRound(id) {
  setPlaying(false);
  const version = ++state.loadVersion;
  $("play").disabled = true; $("timeline").disabled = true;
  const round = await getJSON(`/api/round/${encodeURIComponent(id)}`);
  if (version !== state.loadVersion) return;
  state.round = round; state.time = 0; state.selected = null;
  state.lastRolling = ""; state.activeWindow = null;
  state.frameIndex = -1; state.lastLog = ""; state.lastFeature = -2;
  $("timeline").max = String(round.duration);
  $("timeline").disabled = false; $("play").disabled = false;
  $("time-total").textContent = formatTime(round.duration, true);
  $("round-title").textContent = `${mapName(round.map)} / ${readable(round.round_id)}`;
  const map = state.maps[round.map];
  if (map && !state.mapImages.has(round.map)) {
    const artwork = new Image();
    state.mapImages.set(round.map, artwork);
    artwork.onload = () => { if (state.round?.map === round.map) render(); };
    artwork.onerror = () => { if (state.round?.map === round.map) render(); };
    artwork.src = map.image;
  }
  const atk = round.teams.find((team) => team.side === "attacker");
  const def = round.teams.find((team) => team.side === "defender");
  $("atk-name").textContent = atk.name; $("def-name").textContent = def.name;
  $("score").replaceChildren(document.createTextNode(`${atk.score} `), element("span", "", ":"), document.createTextNode(` ${def.score}`));
  for (const button of $("round-list").children) {
    button.classList.toggle("active", button.dataset.round === id);
    if (button.dataset.round === id) button.setAttribute("aria-current", "true");
    else button.removeAttribute("aria-current");
  }
  const isRolling = round.ml.mode === "rolling";
  $("feature-heading").textContent = isRolling ? "State snapshot" : "Sequence features";
  $("features").replaceChildren(...(isRolling ? round.ml.rolling.display_names : round.ml.feature_names).map((name, index) => {
    const row = element("div", "feature"); row.title = name;
    row.append(element("span", "feature-name", (isRolling ? rollingLabels : featureLabels)[index] || name), element("span", "feature-value", "—"));
    return row;
  }));
  $("temporal-features").replaceChildren(...round.ml.temporal_feature_names.map((name, index) => {
    const row = element("div", "feature"); row.title = name;
    row.append(element("span", "feature-name", temporalLabels[index]), element("span", "feature-value", "—"));
    return row;
  }));
  $("fixed-summary").hidden = isRolling;
  $("rolling-evidence").hidden = !isRolling;
  $("prediction-curve").hidden = true;
  $("window-bar").max = isRolling ? 20 : round.ml.cutoff;
  $("cutoff-label").textContent = isRolling ? "20s history · every 5s" : `${round.ml.cutoff}s observation cutoff`;
  const models = (isRolling ? round.ml.run?.models : round.ml.prediction?.run.models) || {};
  if (!(state.selectedModel in models)) state.selectedModel = ["temporal_cnn", "history_logistic", "logistic_temporal"].find((id) => id in models) || Object.keys(models)[0];
  $("model-select").disabled = !Object.keys(models).length;
  $("model-select").replaceChildren(...Object.entries(models).map(([id, label]) => {
    const option = element("option", "", label); option.value = id; return option;
  }));
  if ($("model-select").disabled) $("model-select").append(element("option", "", "No predictions loaded"));
  else $("model-select").value = state.selectedModel;
  renderProvenance();
  $("error").hidden = true;
  render();
}

function renderRoster(frame) {
  $("roster").replaceChildren();
  for (const team of state.round.teams) {
    const heading = element("div", `roster-team ${team.side === "attacker" ? "attack" : "defense"}`);
    heading.append(element("span", "", team.name), element("span", "", team.side === "attacker" ? "ATTACK" : "DEFEND"));
    $("roster").append(heading);
    for (const player of frame.players.filter((p) => p.team_id === team.id)) {
      const button = element("button", `player-row ${player.side === "defender" ? "defense" : "attack"}`);
      button.classList.toggle("dead", !player.alive);
      button.classList.toggle("selected", state.selected === player.id);
      button.setAttribute("aria-pressed", String(state.selected === player.id));
      button.setAttribute("aria-label", `Inspect ${player.name}, ${player.agent}${player.alive ? "" : ", eliminated"}`);
      const identity = element("span", "player-identity");
      identity.append(element("strong", "", player.name), element("small", "", player.agent));
      const health = element("span", "player-health", player.alive ? String(player.currentHealth ?? "—") : "OUT");
      if (player.alive) health.append(element("small", "", "HP"));
      button.append(element("span", "player-number", player.alive ? String(player.number) : "×"), identity, health);
      button.addEventListener("click", () => selectPlayer(player.id));
      $("roster").append(button);
    }
  }
}
function selectPlayer(id) {
  state.selected = state.selected === id ? null : id;
  state.frameIndex = -1;
  render();
}
function renderPlayer(frame) {
  const panel = $("player-detail");
  const player = frame.players.find((p) => p.id === state.selected);
  if (!player) {
    panel.replaceChildren(element("p", "", "Select a player for details."));
    return;
  }
  panel.replaceChildren(element("h3", "", `${player.name} / ${mapName(player.agent)}`));
  const age = Math.max(0, state.time - player.position_at);
  const coordinates = player.position ? `x ${Math.round(player.position.x)} · y ${Math.round(player.position.y)} · ${age.toFixed(1)}s ago` : "Position unavailable";
  panel.append(element("div", "selection-stats", `${player.currentHealth ?? "—"} HP · ${player.currentArmor ?? "—"} armor · ${player.alive ? "Alive" : "Eliminated"}`), element("div", "selection-stats", coordinates));
  const abilities = element("div", "ability-list");
  for (const ability of Object.values(player.abilities)) {
    const status = ability.ready === true ? "ready" : ability.ready === false ? "unavailable" : "unreported";
    const chip = element("span", `ability-chip ${status}`, `${ability.ready === true ? "●" : "○"} ${readable(ability.name || ability.id)}`);
    chip.title = `Last reported readiness: ${status}`;
    abilities.append(chip);
  }
  panel.append(abilities);
}
function renderEvents() {
  const events = state.round.events.filter((e) => e.t <= state.time + 0.000001 && (state.filter === "all" || e.category === state.filter));
  const signature = `${state.round.id}/${state.filter}/${events.length}`;
  if (signature === state.lastLog) return;
  state.lastLog = signature;
  const container = $("event-log");
  if (!events.length) {
    const empty = element("div", "empty-state");
    empty.append(element("strong", "", "No events yet."), element("span", "", ""));
    container.replaceChildren(empty); return;
  }
  container.replaceChildren(...events.slice().reverse().map((event) => {
    const row = element("button", "event-row");
    const description = element("span", "event-description");
    if (event.category === "ability") description.append(element("strong", "", event.actor), document.createTextNode(` used ${readable(event.ability)}`));
    else if (event.category === "kill") description.append(element("strong", "", event.actor), document.createTextNode(` → ${event.target}`));
    else description.append(element("strong", "", event.label));
    row.append(element("time", "", formatTime(event.t)), element("span", `event-icon ${event.category}`, event.category === "kill" ? "×" : event.category === "ability" ? "✧" : "◇"), description);
    row.addEventListener("click", () => { setPlaying(false); seek(event.t); });
    return row;
  }));
  container.scrollTop = 0;
}
function currentRollingPrediction() {
  return (state.round.ml.predictions || []).filter((p) => p.at_seconds <= state.time).at(-1);
}
function renderProvenance() {
  const isRolling = state.round.ml.mode === "rolling";
  const prediction = isRolling ? currentRollingPrediction() : state.round.ml.prediction;
  const container = $("provenance"); container.replaceChildren();
  function row(label, value, shorten = false) {
    const group = element("div"), detail = element("dd", "", shorten ? `${value.slice(0, 12)}…` : value);
    detail.title = value;
    group.append(element("dt", "", label), detail); container.append(group);
  }
  row("Data SHA-256", state.round.source_sha256, true);
  if (isRolling) row("Observation", "20s history · 5s stride");
  if (!prediction) {
    $("provenance-status").textContent = state.round.ml.run ? "No prediction at or before this time." : "No prediction file loaded.";
    if (isRolling && state.round.ml.rolling.schema_id) row("Features", state.round.ml.rolling.schema_id);
    return;
  }
  const run = isRolling ? state.round.ml.run : prediction.run;
  const parts = { train: "Training · used for fitting", validation: "Validation",
    test: "Test · held-out series", unassigned: "Unassigned · not an evaluation split" };
  $("provenance-status").textContent = run.purpose === "ui_fixture" ? "UI test fixture — not model output" :
    run.mode === "development_smoke_test" ? "Development check · same series" : "Series holdout experiment";
  row("Partition", parts[prediction.partition]);
  row("Run", run.run_id, isRolling);
  row("Cutoff", `${isRolling ? prediction.at_seconds : prediction.observation_seconds}s after buy time`);
  row("Features", isRolling ? run.schema_id : `Sequence v${run.feature_schema_version} · temporal v${run.temporal_schema_version}`);
  if (isRolling && run.feature_set && run.feature_set !== "legacy") {
    row("Feature set", run.feature_set.replaceAll("_", " "));
    row("Input support", `${run.masked_channel_count} channels masked from training-only coverage`);
  }
  row("Model SHA-256", prediction.model_artifact_sha256[state.selectedModel], true);
  row("Run SHA-256", prediction.run_manifest_sha256, true);
  row("Checks", "Data and run records match");
}

function renderML() {
  const ml = state.round.ml;
  if (ml.mode === "rolling") { renderRollingML(); return; }
  $("window-bar").value = Math.min(state.time, ml.cutoff);
  $("window-progress").textContent = `${Math.floor(Math.min(state.time, ml.cutoff))} / ${ml.cutoff}s`;
  const index = Math.min(Math.floor(state.time) - 1, ml.features.length - 1);
  if (index !== state.lastFeature) {
    state.lastFeature = index;
    const values = index >= 0 ? ml.features[index] : null;
    Array.from($("features").children).forEach((row, i) => { row.lastChild.textContent = values ? values[i].toFixed(2) : "—"; });
    $("feature-time").textContent = values ? `Bin ending ${index + 1}s${state.time >= ml.cutoff ? " · frozen" : ""}` : "Waiting for first bin";
  }
  const imported = ml.prediction;
  const ready = state.time >= ml.cutoff;
  if (imported && ready) {
    const probability = imported.probabilities[state.selectedModel];
    $("prediction").replaceChildren(document.createTextNode(`${(probability * 100).toFixed(1)}`), element("small", "", " / 100%"));
    $("prediction-progress").hidden = false; $("prediction-progress").value = probability;
    $("model-status").textContent = imported.run.purpose === "ui_fixture" ? "Synthetic UI test value" : `Saved prediction · ${ml.cutoff}s cutoff`;
    $("model-caption").textContent = imported.partition === "train" ? "This round was used to fit the model." :
      imported.partition === "validation" ? "Validation round. Not a final test result." :
      imported.partition === "unassigned" ? "This round is outside the run’s evaluation partitions." : "Held-out test round.";
  } else {
    $("prediction").replaceChildren(document.createTextNode("—"), element("small", "", " / 100%"));
    $("prediction-progress").hidden = true;
    $("model-status").textContent = imported ? `Prediction available at ${imported.observation_seconds}s` : "No predictions loaded";
    $("model-caption").textContent = imported ? "Waiting for the observation window." : `Features use the first ${ml.cutoff} seconds.`;
  }
  $("model-comparison").hidden = !imported;
  if (imported) $("model-comparison").replaceChildren(...Object.entries(imported.run.models).map(([id, label]) => {
    const row = element("div", "comparison-row");
    row.append(element("span", "", label), element("strong", "", ready ? `${(imported.probabilities[id] * 100).toFixed(1)}%` : "—"));
    return row;
  }));
  Array.from($("temporal-features").children).forEach((row, i) => {
    row.lastChild.textContent = ready && ml.temporal_features.length ? ml.temporal_features[i].toFixed(2) : "—";
  });
  $("summary-time").textContent = ready && ml.temporal_features.length ? "Frozen at cutoff" : `Available at ${ml.cutoff}s`;
  if (ml.eligibility !== "eligible") {
    $("feature-time").textContent = `Not eligible at ${ml.cutoff}s`;
    $("model-caption").textContent = `${ml.eligibility}. Replay is still available.`;
  }
}

function renderRollingML() {
  const ml = state.round.ml, rolling = ml.rolling;
  const window = rolling.windows.filter((w) => w.at_seconds <= state.time).at(-1);
  const prediction = currentRollingPrediction();
  state.activeWindow = window || null;
  $("window-bar").value = Math.min(state.time, 20);
  $("window-progress").textContent = window ? `${window.window_start}–${window.at_seconds}s` : `${Math.floor(Math.min(state.time, 20))} / 20s`;
  const signature = `${window?.at_seconds}/${prediction?.at_seconds}/${state.selectedModel}`;
  if (signature !== state.lastRolling) {
    state.lastRolling = signature;
    renderProvenance();
    $("rolling-window-range").textContent = window ? `${window.window_start}–${window.at_seconds}s` : "Waiting for 20s";
    $("feature-time").textContent = window ? `Snapshot at ${window.at_seconds}s` : "Waiting for 20s";
    Array.from($("features").children).forEach((row, i) => { row.lastChild.textContent = window ? window.features[i].toFixed(2) : "—"; });
    const counts = window?.counts;
    $("rolling-counts").replaceChildren(...[
      ["Ability uses · ATK / DEF", counts ? `${counts.atk_abilities} / ${counts.def_abilities}` : "—"],
      ["Kills · ATK / DEF", counts ? `${counts.atk_kills} / ${counts.def_kills}` : "—"],
      ["Trade candidates", counts ? String(counts.trade_candidates) : "—"],
      ...(window?.movement ? [
        ["Position report pairs · ATK / DEF", `${window.movement.atk.report_pairs} / ${window.movement.def.report_pairs}`],
        ["Pairs after recon cast · ATK / DEF", `${window.movement.atk.after_recon_pairs} / ${window.movement.def.after_recon_pairs}`],
      ] : []),
    ].map(([label, value]) => { const row = element("div", "comparison-row"); row.append(element("span", "", label), element("strong", "", value)); return row; }));
    $("rolling-coverage").textContent = window ? `${window.fresh_position_fraction === null ? "No" : (100 * window.fresh_position_fraction).toFixed(0) + "%"} fresh position coverage across this window. Coordinates older than 2s are masked in model inputs.` : "Uses only observations before each prediction time.";
    $("rolling-abilities").replaceChildren(...(window?.abilities || []).map((item) => {
      const chip = element("span", "ability-chip ready", `${item.side.toUpperCase()} · ${readable(item.ability)} ×${item.uses}`);
      const tags = ml.run?.schema?.ability_taxonomy?.abilities?.[item.ability] || rolling.ability_tags?.[item.ability];
      if (tags) chip.title = `Model inputs: ${tags.join(", ").replaceAll("_", " ")}. Cast functions, not confirmed effects.`;
      return chip;
    }));
    $("rolling-cast-count").textContent = counts ? String(counts.atk_abilities + counts.def_abilities) : "—";
    $("prediction").replaceChildren(document.createTextNode(prediction ? (prediction.probabilities[state.selectedModel] * 100).toFixed(1) : "—"), element("small", "", " / 100%"));
    $("prediction-progress").hidden = !prediction;
    if (prediction) $("prediction-progress").value = prediction.probabilities[state.selectedModel];
    $("model-status").textContent = prediction ? (ml.run.purpose === "ui_fixture" ? "Synthetic UI test value" : `Saved prediction at ${prediction.at_seconds}s`) : ml.run ? "Waiting for a saved prediction" : "No predictions loaded";
    $("model-caption").textContent = prediction ? `${prediction.window_start}–${prediction.at_seconds}s observed · ${prediction.partition === "train" ? "used for fitting" : prediction.partition === "validation" ? "validation, not final test" : prediction.partition === "test" ? "held-out series" : "unassigned round"}.` : rolling.eligibility || "20 seconds of history, updated every 5 seconds.";
    $("model-comparison").hidden = !ml.run;
    if (ml.run) $("model-comparison").replaceChildren(...Object.entries(ml.run.models).map(([id, label]) => {
      const row = element("div", "comparison-row"); row.append(element("span", "", label), element("strong", "", prediction ? `${(prediction.probabilities[id] * 100).toFixed(1)}%` : "—")); return row;
    }));
  }
  $("prediction-curve").hidden = !prediction;
  if (prediction) {
    const { ctx, width, height } = fitCanvas($("prediction-curve"));
    ctx.strokeStyle = "#dce0d8"; ctx.beginPath(); ctx.moveTo(0, height / 2); ctx.lineTo(width, height / 2); ctx.stroke();
    const points = ml.predictions.filter((p) => p.at_seconds <= state.time);
    ctx.strokeStyle = colors.attacker; ctx.lineWidth = 1.7; ctx.beginPath();
    points.forEach((p, i) => {
      const x = p.at_seconds / state.round.duration * width, y = 6 + (1 - p.probabilities[state.selectedModel]) * (height - 12);
      if (i === 0) ctx.moveTo(x, y);
      else { ctx.lineTo(x, 6 + (1 - points[i - 1].probabilities[state.selectedModel]) * (height - 12)); ctx.lineTo(x, y); }
    });
    ctx.lineTo(state.time / state.round.duration * width, 6 + (1 - prediction.probabilities[state.selectedModel]) * (height - 12)); ctx.stroke();
    ctx.fillStyle = colors.attacker;
    for (const p of points) {
      ctx.beginPath();
      ctx.arc(p.at_seconds / state.round.duration * width, 6 + (1 - p.probabilities[state.selectedModel]) * (height - 12), 2.2, 0, Math.PI * 2);
      ctx.fill();
    }
    ctx.font = "8px ui-monospace, monospace"; ctx.fillStyle = "#758082"; ctx.fillText("100%", 0, 8); ctx.fillText("0%", 0, height - 1);
  }
}

function fitCanvas(canvas) {
  const width = canvas.clientWidth, height = canvas.clientHeight;
  const ratio = window.devicePixelRatio || 1;
  const targetW = Math.round(width * ratio), targetH = Math.round(height * ratio);
  if (canvas.width !== targetW || canvas.height !== targetH) { canvas.width = targetW; canvas.height = targetH; }
  const ctx = canvas.getContext("2d"); ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  ctx.clearRect(0, 0, width, height);
  return { ctx, width, height };
}
function drawMap(frame) {
  const { ctx, width, height } = fitCanvas($("minimap"));
  const bounds = state.round.bounds;
  const map = state.maps[state.round.map];
  const artwork = state.mapImages.get(state.round.map);
  const ready = artwork?.complete && artwork.naturalWidth > 0;
  const dx = bounds.max.x - bounds.min.x, dy = bounds.max.y - bounds.min.y;
  const scale = Math.min((width - 40) / dx, (height - 80) / dy);
  const size = Math.min(width - 24, height - 66);
  const mapWidth = ready ? size : dx * scale, mapHeight = ready ? size : dy * scale;
  const x0 = (width - mapWidth) / 2, y0 = (height - mapHeight) / 2;
  function normalized(x, y) {
    if (state.rotate) { x = 1 - x; y = 1 - y; }
    return [x0 + x * mapWidth, y0 + y * mapHeight];
  }
  function project(p) {
    // VALORANT world axes are swapped relative to the minimap image axes.
    return ready ? normalized(p.y * map.xMultiplier + map.xScalarToAdd,
                              p.x * map.yMultiplier + map.yScalarToAdd)
                 : normalized((p.x - bounds.min.x) / dx, 1 - (p.y - bounds.min.y) / dy);
  }
  const gradient = ctx.createRadialGradient(width * .5, height * .5, 0, width * .5, height * .5, width * .6);
  gradient.addColorStop(0, "#1c3038"); gradient.addColorStop(1, "#101b23");
  ctx.fillStyle = gradient; ctx.fillRect(0, 0, width, height);
  $("map-kind").textContent = ready ? "MINIMAP" : map ? "MAP IMAGE UNAVAILABLE" : "COORDINATE VIEW";
  if (ready) {
    ctx.save();
    ctx.translate(width / 2, height / 2);
    if (state.rotate) ctx.rotate(Math.PI);
    ctx.globalAlpha = .68;
    ctx.drawImage(artwork, -size / 2, -size / 2, size, size);
    ctx.restore();
    for (const site of map.sites) {
      const [x, y] = normalized(site.x, site.y);
      ctx.fillStyle = "#d5d9bb"; ctx.font = "bold 12px ui-monospace, monospace"; ctx.textAlign = "center";
      ctx.fillText(site.label, x, y + 4);
    }
  } else {
    ctx.lineWidth = 1; ctx.strokeStyle = "#637d7b30";
    for (let n = 0; n <= 10; n++) {
      const x = x0 + n * mapWidth / 10, y = y0 + n * mapHeight / 10;
      ctx.beginPath(); ctx.moveTo(x, y0); ctx.lineTo(x, y0 + mapHeight); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(x0, y); ctx.lineTo(x0 + mapWidth, y); ctx.stroke();
    }
  }
  if ($("trails-toggle").checked) {
    for (const player of frame.players) {
      if (state.selected && state.selected !== player.id) continue;
      ctx.strokeStyle = colors[player.side]; ctx.globalAlpha = state.selected ? .5 : .2;
      ctx.lineWidth = state.selected ? 1.6 : 1; ctx.beginPath();
      let started = false;
      for (const past of state.round.frames) {
        if (past.t > state.time) break;
        if (past.t < state.time - 10) continue;
        const previous = past.players.find((p) => p.id === player.id);
        if (!previous?.position) { started = false; continue; }
        const [x, y] = project(previous.position);
        if (!started) { ctx.moveTo(x, y); started = true; } else ctx.lineTo(x, y);
      }
      const head = player.position ? project(player.position) : null;
      if (head && started) ctx.lineTo(...head);
      ctx.stroke(); ctx.globalAlpha = 1;
    }
  }
  if ($("abilities-toggle").checked) {
    for (const event of state.round.events) {
      const age = state.time - event.t;
      if (event.category !== "ability" || !event.position || age < 0 || age > 3) continue;
      const [x, y] = project(event.position);
      ctx.globalAlpha = 1 - age / 3; ctx.strokeStyle = colors[event.side] || "#c5cba9"; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.arc(x, y, 16 + age * 6, 0, Math.PI * 2); ctx.stroke();
      ctx.font = "8px ui-monospace, monospace"; ctx.fillStyle = "#e2e9d8"; ctx.textAlign = "center";
      ctx.fillText(readable(event.ability), x, y - 27); ctx.globalAlpha = 1;
    }
  }
  if ($("kills-toggle").checked && state.activeWindow && ready) {
    for (const kill of state.activeWindow.kills) {
      if (!kill.victim_position) continue;
      const [x, y] = normalized(...kill.victim_position);
      ctx.strokeStyle = kill.side === "atk" ? colors.attacker : colors.defender;
      ctx.lineWidth = 2; ctx.beginPath(); ctx.moveTo(x - 6, y - 6); ctx.lineTo(x + 6, y + 6);
      ctx.moveTo(x - 6, y + 6); ctx.lineTo(x + 6, y - 6); ctx.stroke();
      ctx.font = "8px ui-monospace, monospace"; ctx.fillStyle = "#e2e9d8"; ctx.textAlign = "center";
      ctx.fillText(`${kill.t.toFixed(1)}s`, x, y - 10);
    }
  }
  state.points = [];
  for (const player of frame.players) {
    if (!player.position) continue;
    const [x, y] = project(player.position);
    const selected = state.selected === player.id;
    const age = Math.max(0, state.time - player.position_at);
    ctx.globalAlpha = player.alive ? (age > 10 ? .65 : 1) : .38;
    ctx.strokeStyle = colors[player.side]; ctx.fillStyle = colors[player.side];
    if (selected) { ctx.lineWidth = 1; ctx.beginPath(); ctx.arc(x, y, 15, 0, Math.PI * 2); ctx.stroke(); }
    if (player.alive) {
      ctx.shadowBlur = selected ? 12 : 5; ctx.shadowColor = `${colors[player.side]}88`;
      ctx.beginPath(); ctx.arc(x, y, selected ? 9 : 7.5, 0, Math.PI * 2); ctx.fill(); ctx.shadowBlur = 0;
      ctx.fillStyle = "#182d31"; ctx.font = "bold 8px ui-monospace, monospace"; ctx.textAlign = "center";
      ctx.fillText(String(player.number), x, y + 3);
    } else {
      ctx.lineWidth = 1.6; ctx.beginPath(); ctx.moveTo(x - 4, y - 4); ctx.lineTo(x + 4, y + 4); ctx.moveTo(x - 4, y + 4); ctx.lineTo(x + 4, y - 4); ctx.stroke();
    }
    ctx.globalAlpha = 1;
    state.points.push({ x, y, player });
  }
  const occupied = state.points.map((p) => ({ left: p.x - 9, right: p.x + 9, top: p.y - 9, bottom: p.y + 9 }));
  const overlaps = (a, b) => a.left < b.right && a.right > b.left && a.top < b.bottom && a.bottom > b.top;
  for (const point of state.points.slice().sort((a, b) => Number(b.player.id === state.selected) - Number(a.player.id === state.selected))) {
    const { x, y, player } = point;
    ctx.font = `${player.id === state.selected ? "bold " : ""}8px ui-monospace, monospace`;
    const half = ctx.measureText(player.name).width / 2 + 3;
    const candidates = [[0, 21], [0, -17], [half + 15, 3], [-half - 15, 3], [0, 34], [0, -30], [half + 15, 21], [-half - 15, -17]];
    let placement = null;
    for (const [offsetX, offsetY] of candidates) {
      const box = { left: x + offsetX - half, right: x + offsetX + half, top: y + offsetY - 8, bottom: y + offsetY + 3 };
      if (box.left >= 4 && box.right <= width - 4 && !occupied.some((other) => overlaps(box, other))) {
        placement = { x: x + offsetX, y: y + offsetY, box }; break;
      }
    }
    if (!placement) continue; // Overlapping markers remain selectable through the roster.
    occupied.push(placement.box);
    ctx.globalAlpha = player.alive ? .9 : .35;
    ctx.strokeStyle = "#76939255"; ctx.lineWidth = .5;
    ctx.beginPath(); ctx.moveTo(x, y); ctx.lineTo(placement.x, placement.y - 4); ctx.stroke();
    ctx.strokeStyle = "#13232b"; ctx.lineWidth = 3; ctx.textAlign = "center";
    ctx.strokeText(player.name, placement.x, placement.y);
    ctx.fillStyle = "#dce6df"; ctx.fillText(player.name, placement.x, placement.y); ctx.globalAlpha = 1;
  }
}
function drawTrack() {
  const { ctx, width, height } = fitCanvas($("event-track"));
  for (const event of state.round.events) {
    const x = event.t / state.round.duration * width;
    ctx.fillStyle = event.category === "kill" ? "#c16b50" : event.category === "ability" ? "#83a48b" : "#b9aa6d";
    ctx.globalAlpha = event.t <= state.time ? .9 : .27;
    ctx.fillRect(x, 2, event.category === "kill" ? 2 : 1, event.category === "kill" ? 7 : 4);
  }
  ctx.globalAlpha = 1;
  if (state.round.ml.mode === "rolling") {
    if (state.activeWindow) {
      const start = state.activeWindow.window_start / state.round.duration * width;
      const end = state.activeWindow.at_seconds / state.round.duration * width;
      ctx.fillStyle = "#dc573f22"; ctx.fillRect(start, 0, end - start, height);
    }
    ctx.strokeStyle = "#ce6046";
    for (const window of state.round.ml.rolling.windows) {
      const x = window.at_seconds / state.round.duration * width;
      ctx.beginPath(); ctx.moveTo(x, height - 5); ctx.lineTo(x, height); ctx.stroke();
    }
    return;
  }
  if (state.round.duration > state.round.ml.cutoff) {
    const x = state.round.ml.cutoff / state.round.duration * width; ctx.strokeStyle = "#ce6046"; ctx.setLineDash([2, 2]);
    ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, height - 2); ctx.stroke(); ctx.setLineDash([]);
  }
}
function render() {
  if (!state.round) return;
  const index = findFrame(state.time), frame = state.round.frames[index];
  $("timeline").value = String(state.time);
  $("time-current").textContent = formatTime(state.time, true);
  $("phase").textContent = frame.winner ? `${frame.winner.toUpperCase()} WIN` : state.time === 0 ? "LIVE START" : "IN PROGRESS";
  $("spike-state").textContent = `SPIKE ${frame.spike.toUpperCase()}`;
  $("snapshot-age").textContent = `Snapshot ${(state.time - frame.t).toFixed(1)}s ago`;
  for (const [side, prefix] of [["attacker", "atk"], ["defender", "def"]]) {
    $(prefix + "-alive").textContent = `${frame.players.filter((p) => p.side === side && p.alive).length} alive`;
  }
  if (index !== state.frameIndex) { state.frameIndex = index; renderRoster(frame); }
  renderPlayer(frame); renderEvents(); renderML(); drawMap(frame); drawTrack();
}

$("series-select").addEventListener("change", fillMaps);
$("map-select").addEventListener("change", fillRounds);
$("model-select").addEventListener("change", (event) => {
  state.selectedModel = event.target.value; renderProvenance(); renderML();
});
$("timeline").addEventListener("input", (event) => { setPlaying(false); seek(Number(event.target.value)); });
$("play").addEventListener("click", () => {
  if (!state.round) return;
  if (state.time >= state.round.duration) seek(0);
  setPlaying(!state.playing);
});
$("speed").addEventListener("change", (event) => { state.rate = Number(event.target.value); });
$("previous-event").addEventListener("click", () => {
  if (!state.round) return;
  const event = state.round.events.filter((e) => e.t < state.time - .001).at(-1);
  setPlaying(false); seek(event ? event.t : 0);
});
$("next-event").addEventListener("click", () => {
  if (!state.round) return;
  const event = state.round.events.find((e) => e.t > state.time + .001);
  setPlaying(false); seek(event ? event.t : state.round.duration);
});
for (const button of document.querySelectorAll("[data-filter]")) {
  button.addEventListener("click", () => {
    state.filter = button.dataset.filter;
    document.querySelectorAll("[data-filter]").forEach((b) => b.classList.toggle("active", b === button));
    if (state.round) renderEvents();
  });
}
for (const id of ["trails-toggle", "abilities-toggle", "kills-toggle"]) $(id).addEventListener("change", render);
$("rotate").addEventListener("click", () => { state.rotate = !state.rotate; render(); });
$("minimap").addEventListener("mousemove", (event) => {
  const rect = $("minimap").getBoundingClientRect();
  const x = event.clientX - rect.left, y = event.clientY - rect.top;
  const point = state.points.find((p) => Math.hypot(p.x - x, p.y - y) < 13);
  const tip = $("map-tooltip"); tip.hidden = !point;
  $("minimap").style.cursor = point ? "pointer" : "default";
  if (point) {
    tip.textContent = `${point.player.name} · ${point.player.agent} · ${point.player.alive ? `${point.player.currentHealth ?? "—"} HP` : "eliminated"}`;
    tip.style.left = `${Math.max(8, Math.min(x + 12, rect.width - 230))}px`;
    tip.style.top = `${Math.max(35, y - 39)}px`;
  }
});
$("minimap").addEventListener("mouseleave", () => { $("map-tooltip").hidden = true; });
$("minimap").addEventListener("click", (event) => {
  const rect = $("minimap").getBoundingClientRect();
  const point = state.points.find((p) => Math.hypot(p.x - (event.clientX - rect.left), p.y - (event.clientY - rect.top)) < 13);
  if (point) selectPlayer(point.player.id);
});
document.addEventListener("keydown", (event) => {
  if (!state.round || ["INPUT", "SELECT", "BUTTON", "SUMMARY", "A"].includes(document.activeElement.tagName)) return;
  if (event.code === "Space") { event.preventDefault(); $("play").click(); }
  if (event.code === "ArrowRight" || event.code === "ArrowLeft") {
    event.preventDefault(); setPlaying(false); seek(state.time + (event.code === "ArrowRight" ? 1 : -1));
  }
});
new ResizeObserver(() => { if (state.round) render(); }).observe($("minimap"));
document.addEventListener("visibilitychange", () => { if (document.hidden) setPlaying(false); });
function tick(now) {
  if (state.playing && state.round) {
    if (state.lastTick) seek(state.time + Math.min((now - state.lastTick) / 1000, .25) * state.rate);
    state.lastTick = now;
  }
  requestAnimationFrame(tick);
}
requestAnimationFrame(tick);

async function start() {
  [state.catalog, state.maps] = await Promise.all([getJSON("/api/catalog"), getJSON("/maps/manifest.json")]);
  const catalog = state.catalog;
  $("dataset-summary").textContent = `${catalog.rounds.length} rounds · ${catalog.map_count} maps`;
  const series = [...new Set(catalog.rounds.map((r) => r.series_id))];
  $("series-select").replaceChildren(...series.map((id) => { const match = catalog.series.find((item) => item.id === id); const option = element("option", "", match.teams.join(" vs ")); option.value = id; return option; }));
  fillMaps();
}
start().catch(reportError);
