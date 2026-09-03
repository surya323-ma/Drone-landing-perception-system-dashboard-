import { forwardRef } from "react";

const VideoStage = forwardRef(function VideoStage({ src, onTimeUpdate }, ref) {
  if (!src) {
    return (
      <div className="flex aspect-video items-center justify-center rounded-md border border-hud-grid/60 bg-hud-panel text-xs text-hud-dim">
        No output yet — upload footage or run the demo to render a perception feed.
      </div>
    );
  }
  return (
    <div className="bracket-frame overflow-hidden rounded-md border border-hud-grid/60 bg-black">
      <video
        ref={ref}
        src={src}
        controls
        autoPlay
        muted
        loop
        onTimeUpdate={onTimeUpdate}
        className="aspect-video w-full bg-black"
      />
    </div>
  );
});

export default VideoStage;
