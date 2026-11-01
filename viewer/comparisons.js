"use strict";

let entryPair = null;
let entrySignature = "";
let comparisonSignature = "";

function setupComparisons(round) {
  const choices = round.ml.comparisons || [];
  entryPair = null; entrySignature = ""; comparisonSignature = "";
  $("run-picker").hidden = !choices.length;
  $("run-comparison").hidden = !choices.length;
  $("run-select").replaceChildren(...choices.map((choice) => {
    const option = element("option", "", choice.label); option.value = choice.id; return option;
  }));
  if (choices.length) $("run-select").value = round.ml.selected_comparison;
  $("run-select").onchange = () => {
    const choice = choices.find((item) => item.id === $("run-select").value);
    if (!choice) return;
    round.ml.run = choice.run;
    round.ml.predictions = choice.predictions;
    round.ml.selected_comparison = choice.id;
    state.selectedModel = choice.architecture;
    $("model-select").replaceChildren(...Object.entries(choice.run.models).map(([id, label]) => {
      const option = element("option", "", label); option.value = id; return option;
    }));
    $("model-select").value = state.selectedModel;
    state.lastRolling = "";
    renderProvenance(); render();
  };
  $("entry-review").hidden = !round.ml.entry_support;
  $("entry-player").replaceChildren(element("option", "", "All players"), ...round.frames[0].players.map((p) => {
    const option = element("option", "", `${p.side === "attacker" ? "ATK" : "DEF"} · ${p.name}`);
    option.value = p.id; return option;
  }));
  $("entry-player").firstChild.value = "";
  for (const id of ["entry-player", "entry-moving-only"]) $(id).onchange = () => {
    entrySignature = ""; entryPair = null; render();
  };
  $("hide-predictions").onchange = () => document.body.classList.toggle("probabilities-hidden", $("hide-predictions").checked);
}

function renderComparisons(window) {
  const ml = state.round.ml;
  const signature = `${state.round.id}/${window?.at_seconds}`;
  if (comparisonSignature !== signature) {
    comparisonSignature = signature;
    $("run-comparison-values").replaceChildren(...(ml.comparisons || []).map((choice) => {
      const prediction = choice.predictions.find((p) => p.at_seconds === window?.at_seconds);
      const row = element("div", "comparison-row");
      const button = element("button", "comparison-choice", choice.label);
      button.type = "button";
      button.title = `Run ${choice.run.run_id}\nManifest ${choice.run.manifest_sha256}\nPredictions ${choice.predictions_sha256}`;
      button.onclick = () => { $("run-select").value = choice.id; $("run-select").onchange(); };
      row.append(button, element("strong", "", prediction ? `${(prediction.probabilities[choice.architecture] * 100).toFixed(1)}%` : "—"));
      return row;
    }));
  }
  const review = ml.entry_support?.windows.find((w) => w.at_seconds === window?.at_seconds);
  const reviewKey = `${signature}/${$("entry-player").value}/${$("entry-moving-only").checked}`;
  if (entrySignature === reviewKey) return;
  entrySignature = reviewKey; entryPair = null;
  $("entry-pairs").replaceChildren();
  if (!review) {
    $("entry-summary").textContent = "Waiting for a complete 20-second observation window.";
    return;
  }
  const pairs = review.pairs.filter((p) => (!$("entry-player").value || p.player_id === $("entry-player").value)
    && (!$("entry-moving-only").checked || p.displacement > 0));
  $("entry-summary").textContent = `${review.window_start}–${review.at_seconds}s · ${review.moving_pair_count} moving pairs, ${review.ally_linked_moving_pairs} near allied casts, ${review.unusable_report_count} unusable reports across both teams. Showing ${Math.min(60, pairs.length)} of ${pairs.length} filtered pairs.`;
  if (!pairs.length) $("entry-pairs").append(element("p", "muted", "No pairs match these filters. This does not establish that no movement or useful utility occurred."));
  for (const pair of pairs.slice(0, 60)) {
    const item = element("details", "entry-pair");
    const summary = element("summary", "", `${pair.side.toUpperCase()} ${pair.player} · ${pair.start_t.toFixed(2)}–${pair.t.toFixed(2)}s · ${pair.displacement.toFixed(0)} units`);
    summary.title = `Position source IDs: ${pair.source_ids.flat().join(", ")}`;
    const highlight = element("button", "text-button", "Highlight displacement on map");
    highlight.type = "button";
    highlight.onclick = () => { entryPair = pair; selectPlayer(pair.player_id); };
    item.append(summary, highlight, element("p", "muted", `Report gap ${pair.report_gap.toFixed(3)}s. A straight segment joins observations; the route is unknown.`));
    if (!pair.casts.length) item.append(element("p", "muted", "No allied or self cast in the preceding 5 seconds within this window."));
    for (const cast of pair.casts) {
      const spacing = cast.relation === "self" ? "" : cast.spacing_at_first_report === null
        ? " · support spacing unknown" : ` · spacing ${cast.spacing_at_first_report.toFixed(0)} units (position age ${cast.support_position_age.toFixed(2)}s)`;
      const line = element("p", "entry-cast", `${cast.t.toFixed(2)}s ${cast.caster}: ${readable(cast.ability)} · ${cast.relation === "self" ? "self-cast" : "ally"}, ${cast.timing === "preceding" ? "before first report" : "between reports"} · ${cast.delay_to_second_report.toFixed(2)}s before second report${spacing}`);
      line.title = `Cast source ID: ${cast.event_id}`;
      item.append(line);
    }
    $("entry-pairs").append(item);
  }
}

function drawEntryPair(ctx, project) {
  if (!entryPair || entryPair.t >= (state.activeWindow?.at_seconds || 0)
      || entryPair.start_t < state.activeWindow.window_start) return;
  const a = project({ x: entryPair.from[0], y: entryPair.from[1] });
  const b = project({ x: entryPair.to[0], y: entryPair.to[1] });
  ctx.save(); ctx.strokeStyle = "#ffe18a"; ctx.fillStyle = "#ffe18a"; ctx.lineWidth = 3;
  ctx.setLineDash([5, 4]); ctx.beginPath(); ctx.moveTo(...a); ctx.lineTo(...b); ctx.stroke(); ctx.setLineDash([]);
  for (const point of [a, b]) { ctx.beginPath(); ctx.arc(...point, 5, 0, Math.PI * 2); ctx.stroke(); }
  ctx.font = "10px ui-monospace, monospace"; ctx.textAlign = "left";
  ctx.fillText(`${entryPair.player} · ${entryPair.start_t.toFixed(2)}–${entryPair.t.toFixed(2)}s`, b[0] + 9, b[1] - 9);
  ctx.restore();
}

function setupReviewCases(catalog) {
  const cases = catalog.entry_review_cases || [];
  $("review-case-picker").hidden = !cases.length;
  $("review-case-select").replaceChildren(element("option", "", "Choose a window"), ...cases.map((item, i) => {
    const option = element("option", "", `${mapName(item.map)} R${item.round[2].replace("round-", "")} · ${item.at_seconds}s`);
    option.value = String(i); option.title = item.selection_reason; return option;
  }));
  $("review-case-select").firstChild.value = "";
  $("review-case-select").onchange = async () => {
    if ($("review-case-select").value === "") return;
    try {
      const item = cases[Number($("review-case-select").value)];
      const round = catalog.rounds.find((r) => [r.series_id, r.game_id, r.round_id].every((value, i) => value === item.round[i]));
      if (!round) throw new Error("Review round is absent from this dataset.");
      $("series-select").value = round.series_id; fillMaps();
      $("map-select").value = round.game_id; fillRounds();
      await loadRound(round.id);
      if (state.round?.id !== round.id) return;
      if (item.focus_player_id) $("entry-player").value = item.focus_player_id;
      $("hide-predictions").checked = true; $("hide-predictions").onchange();
      entrySignature = ""; seek(item.at_seconds);
    } catch (error) { reportError(error); }
  };
}
