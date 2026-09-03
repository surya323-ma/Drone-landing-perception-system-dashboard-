import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
} from "recharts";

const axisStyle = { fontSize: 10, fill: "#8FA0AC" };

function ChartCard({ title, children }) {
  return (
    <div className="rounded-md border border-hud-grid/60 bg-hud-panel p-4">
      <h3 className="mb-2 font-display text-xs font-medium text-hud-dim">{title}</h3>
      <div className="h-40">
        <ResponsiveContainer width="100%" height="100%">
          {children}
        </ResponsiveContainer>
      </div>
    </div>
  );
}

const tooltipStyle = {
  contentStyle: {
    background: "#12181F",
    border: "1px solid #32465A",
    borderRadius: 6,
    fontSize: 11,
  },
  labelStyle: { color: "#8FA0AC" },
};

export default function TelemetryCharts({ telemetry }) {
  if (!telemetry || telemetry.length === 0) return null;

  // Thin the series for chart legibility/perf on long runs.
  const step = Math.max(1, Math.floor(telemetry.length / 300));
  const data = telemetry.filter((_, i) => i % step === 0);

  return (
    <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
      <ChartCard title="Confidence / Alignment (%)">
        <LineChart data={data}>
          <CartesianGrid stroke="#32465A" strokeOpacity={0.3} vertical={false} />
          <XAxis dataKey="t" tick={axisStyle} tickFormatter={(v) => `${v.toFixed(0)}s`} />
          <YAxis tick={axisStyle} domain={[0, 100]} width={30} />
          <Tooltip {...tooltipStyle} labelFormatter={(v) => `t = ${v}s`} />
          <Line type="monotone" dataKey="confidence" name="Confidence" stroke="#3CF0FF" dot={false} strokeWidth={1.5} isAnimationActive={false} />
          <Line type="monotone" dataKey="alignment_pct" name="Alignment" stroke="#EBAA3C" dot={false} strokeWidth={1.5} isAnimationActive={false} />
        </LineChart>
      </ChartCard>

      <ChartCard title="Distance / Altitude (m)">
        <LineChart data={data}>
          <CartesianGrid stroke="#32465A" strokeOpacity={0.3} vertical={false} />
          <XAxis dataKey="t" tick={axisStyle} tickFormatter={(v) => `${v.toFixed(0)}s`} />
          <YAxis tick={axisStyle} width={30} />
          <Tooltip {...tooltipStyle} labelFormatter={(v) => `t = ${v}s`} />
          <Line type="monotone" dataKey="distance_m" name="Distance" stroke="#78EB8C" dot={false} strokeWidth={1.5} isAnimationActive={false} />
          <Line type="monotone" dataKey="altitude_m" name="Altitude" stroke="#5AC8E6" dot={false} strokeWidth={1.5} isAnimationActive={false} />
        </LineChart>
      </ChartCard>
    </div>
  );
}
