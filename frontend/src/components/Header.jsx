export default function Header() {
  return (
    <header className="flex items-center justify-between border-b border-hud-grid/60 px-6 py-4">
      <div className="flex items-baseline gap-3">
        <h1 className="font-display text-xl font-semibold tracking-tight text-hud-white">
          Perception Dashboard
        </h1>
        <span className="text-xs text-hud-dim">
          Drone Landing Perception System — approach &amp; landing telemetry
        </span>
      </div>
      <div className="flex items-center gap-2 text-xs text-hud-dim">
        <span className="h-1.5 w-1.5 rounded-full bg-hud-cyan pulse-glow" />
        <span>dev/creator = tubakhxn</span>
      </div>
    </header>
  );
}
