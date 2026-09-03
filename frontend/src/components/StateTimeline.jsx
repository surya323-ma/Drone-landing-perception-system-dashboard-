const STATE_COLOR = {
  SEARCHING: "#8FA0AC",
  ACQUIRED: "#EBAA3C",
  TRACKING: "#3CF0FF",
  LOCKED: "#78EB8C",
  APPROACHING: "#4CFF7A",
};

export default function StateTimeline({ telemetry, duration, currentTime, onSeek }) {
  if (!telemetry || telemetry.length === 0 || !duration) return null;

  return (
    <div className="rounded-md border border-hud-grid/60 bg-hud-panel p-4">
      <div className="mb-2 flex items-center justify-between">
        <h3 className="font-display text-xs font-medium text-hud-dim">Tracking State Timeline</h3>
        <div className="flex gap-3 text-[10px] text-hud-dim">
          {Object.entries(STATE_COLOR).map(([state, color]) => (
            <span key={state} className="flex items-center gap-1">
              <span className="h-1.5 w-1.5 rounded-full" style={{ background: color }} />
              {state}
            </span>
          ))}
        </div>
      </div>
      <div
        className="relative h-4 w-full cursor-pointer overflow-hidden rounded"
        onClick={(e) => {
          if (!onSeek) return;
          const rect = e.currentTarget.getBoundingClientRect();
          const frac = (e.clientX - rect.left) / rect.width;
          onSeek(frac * duration);
        }}
      >
        {telemetry.map((sample, i) => {
          const next = telemetry[i + 1];
          const start = (sample.t / duration) * 100;
          const end = ((next ? next.t : duration) / duration) * 100;
          return (
            <div
              key={i}
              className="absolute top-0 h-full"
              style={{
                left: `${start}%`,
                width: `${Math.max(end - start, 0.15)}%`,
                background: STATE_COLOR[sample.state] || "#8FA0AC",
                opacity: 0.85,
              }}
            />
          );
        })}
        <div
          className="pointer-events-none absolute top-0 h-full w-px bg-hud-white"
          style={{ left: `${duration ? (currentTime / duration) * 100 : 0}%` }}
        />
      </div>
    </div>
  );
}
