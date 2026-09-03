const STATE_STYLES = {
  SEARCHING: { color: "text-hud-dim", ring: "ring-hud-dim/40" },
  ACQUIRED: { color: "text-hud-amber", ring: "ring-hud-amber/40" },
  TRACKING: { color: "text-hud-cyan", ring: "ring-hud-cyan/40" },
  LOCKED: { color: "text-hud-green", ring: "ring-hud-green/40" },
  APPROACHING: { color: "text-hud-green", ring: "ring-hud-green/60" },
};

function nearestSample(telemetry, t) {
  if (!telemetry || telemetry.length === 0) return null;
  // telemetry is sorted by t; binary search for closest.
  let lo = 0, hi = telemetry.length - 1;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (telemetry[mid].t < t) lo = mid + 1;
    else hi = mid;
  }
  if (lo > 0 && Math.abs(telemetry[lo - 1].t - t) < Math.abs(telemetry[lo].t - t)) {
    return telemetry[lo - 1];
  }
  return telemetry[lo];
}

function Metric({ label, value, unit }) {
  return (
    <div className="flex items-baseline justify-between border-b border-hud-grid/30 py-1">
      <span className="text-[10px] uppercase tracking-wide text-hud-dim">{label}</span>
      <span className="text-sm text-hud-white">
        {value}
        {unit && <span className="ml-0.5 text-[10px] text-hud-dim">{unit}</span>}
      </span>
    </div>
  );
}

export default function HudTelemetryPanel({ telemetry, currentTime }) {
  const sample = nearestSample(telemetry, currentTime);
  const state = sample?.state || "SEARCHING";
  const style = STATE_STYLES[state] || STATE_STYLES.SEARCHING;

  return (
    <div className={`bracket-frame rounded-md border border-hud-grid/60 bg-hud-panel p-4 ring-1 ${style.ring}`}>
      <div className="mb-3 flex items-center justify-between">
        <h2 className="font-display text-sm font-medium text-hud-white">Telemetry</h2>
        <span className={`text-xs font-medium ${style.color}`}>{state}</span>
      </div>

      {!sample ? (
        <p className="text-xs text-hud-dim">Play the output feed to see live readouts.</p>
      ) : (
        <div className="space-y-0.5">
          <Metric label="Track ID" value={sample.track_id ?? "—"} />
          <Metric label="Confidence" value={sample.confidence ?? "—"} unit="%" />
          <Metric
            label="Distance"
            value={sample.distance_m ?? "—"}
            unit={sample.distance_m !== undefined ? "m" : ""}
          />
          <Metric
            label="Altitude"
            value={sample.altitude_m ?? "—"}
            unit={sample.altitude_m !== undefined ? "m" : ""}
          />
          <Metric
            label="Alignment"
            value={sample.alignment_pct ?? "—"}
            unit={sample.alignment_pct !== undefined ? "%" : ""}
          />
          <Metric
            label="Lateral offset"
            value={sample.lateral_offset_m ?? "—"}
            unit={sample.lateral_offset_m !== undefined ? "m" : ""}
          />
          <Metric
            label="Vertical offset"
            value={sample.vertical_offset_m ?? "—"}
            unit={sample.vertical_offset_m !== undefined ? "m" : ""}
          />
          <Metric
            label="Approach angle"
            value={sample.approach_angle_deg ?? "—"}
            unit={sample.approach_angle_deg !== undefined ? "deg" : ""}
          />
          <Metric
            label="Landing confidence"
            value={sample.landing_confidence_pct ?? "—"}
            unit={sample.landing_confidence_pct !== undefined ? "%" : ""}
          />
          <Metric label="FPS" value={sample.fps ?? "—"} />
        </div>
      )}
    </div>
  );
}
