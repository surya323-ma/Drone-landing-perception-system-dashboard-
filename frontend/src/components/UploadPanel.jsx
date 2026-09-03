import { useRef, useState } from "react";

const STATE_COLOR = {
  SEARCHING: "text-hud-dim",
  ACQUIRED: "text-hud-amber",
  TRACKING: "text-hud-cyan",
  LOCKED: "text-hud-green",
  APPROACHING: "text-hud-green",
};

export default function UploadPanel({ job, uploadPct, onUpload, onDemo, disabled }) {
  const fileRef = useRef(null);
  const [dragOver, setDragOver] = useState(false);

  const handleFiles = (files) => {
    if (files && files[0]) onUpload(files[0]);
  };

  const processing = job && (job.status === "queued" || job.status === "processing");

  return (
    <div className="bracket-frame rounded-md border border-hud-grid/60 bg-hud-panel p-4">
      <div className="mb-3 flex items-center justify-between">
        <h2 className="font-display text-sm font-medium text-hud-white">Input Source</h2>
        {job && (
          <span className={`text-xs ${STATE_COLOR[job.state] || "text-hud-dim"}`}>
            {job.status.toUpperCase()}
          </span>
        )}
      </div>

      <div
        onDragOver={(e) => {
          e.preventDefault();
          setDragOver(true);
        }}
        onDragLeave={() => setDragOver(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragOver(false);
          handleFiles(e.dataTransfer.files);
        }}
        className={`flex flex-col items-center justify-center gap-2 rounded border border-dashed px-4 py-6 text-center transition-colors ${
          dragOver ? "border-hud-cyan bg-hud-cyan/5" : "border-hud-grid"
        }`}
      >
        <p className="text-xs text-hud-dim">
          Drop drone footage here, or
        </p>
        <div className="flex gap-2">
          <button
            type="button"
            disabled={disabled}
            onClick={() => fileRef.current?.click()}
            className="rounded border border-hud-cyan/50 bg-hud-cyan/10 px-3 py-1.5 text-xs font-medium text-hud-cyan transition-colors hover:bg-hud-cyan/20 disabled:cursor-not-allowed disabled:opacity-40"
          >
            Upload Footage
          </button>
          <button
            type="button"
            disabled={disabled}
            onClick={onDemo}
            className="rounded border border-hud-amber/50 bg-hud-amber/10 px-3 py-1.5 text-xs font-medium text-hud-amber transition-colors hover:bg-hud-amber/20 disabled:cursor-not-allowed disabled:opacity-40"
          >
            Run Synthetic Demo
          </button>
        </div>
        <input
          ref={fileRef}
          type="file"
          accept="video/mp4,video/quicktime,video/x-msvideo,video/webm,video/x-matroska"
          className="hidden"
          onChange={(e) => handleFiles(e.target.files)}
        />
        <p className="text-[10px] text-hud-dim/70">mp4, mov, avi, mkv, webm — up to 500MB</p>
      </div>

      {uploadPct !== null && uploadPct < 100 && (
        <div className="mt-3">
          <div className="mb-1 flex justify-between text-[10px] text-hud-dim">
            <span>Uploading</span>
            <span>{uploadPct}%</span>
          </div>
          <div className="h-1 w-full overflow-hidden rounded bg-hud-grid/40">
            <div className="h-full bg-hud-amber transition-all" style={{ width: `${uploadPct}%` }} />
          </div>
        </div>
      )}

      {processing && (
        <div className="mt-3">
          <div className="mb-1 flex justify-between text-[10px] text-hud-dim">
            <span>
              Processing frame {job.frame}
              {job.total_frames ? ` / ${job.total_frames}` : ""}
            </span>
            <span className={STATE_COLOR[job.state] || "text-hud-dim"}>{job.state}</span>
          </div>
          <div className="h-1 w-full overflow-hidden rounded bg-hud-grid/40">
            <div
              className="h-full bg-hud-cyan transition-all"
              style={{ width: `${job.progress_pct || 0}%` }}
            />
          </div>
        </div>
      )}

      {job?.status === "error" && (
        <p className="mt-3 text-xs text-hud-red">Error: {job.error}</p>
      )}
    </div>
  );
}
