import { useCallback, useEffect, useRef, useState } from "react";
import Header from "./components/Header.jsx";
import UploadPanel from "./components/UploadPanel.jsx";
import VideoStage from "./components/VideoStage.jsx";
import HudTelemetryPanel from "./components/HudTelemetryPanel.jsx";
import TelemetryCharts from "./components/TelemetryCharts.jsx";
import StateTimeline from "./components/StateTimeline.jsx";
import { startUploadJob, startDemoJob, getJobStatus, getJobTelemetry, jobVideoUrl } from "./api.js";

const POLL_MS = 800;

export default function App() {
  const [jobId, setJobId] = useState(null);
  const [job, setJob] = useState(null);
  const [uploadPct, setUploadPct] = useState(null);
  const [telemetry, setTelemetry] = useState([]);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const videoRef = useRef(null);
  const pollRef = useRef(null);

  const stopPolling = () => {
    if (pollRef.current) {
      clearInterval(pollRef.current);
      pollRef.current = null;
    }
  };

  const pollTelemetry = useCallback(async (id) => {
    try {
      const data = await getJobTelemetry(id);
      setTelemetry(data.telemetry);
    } catch (_) {
      /* job may not have any samples yet */
    }
  }, []);

  useEffect(() => {
    if (!jobId) return;
    stopPolling();
    pollRef.current = setInterval(async () => {
      try {
        const status = await getJobStatus(jobId);
        setJob(status);
        await pollTelemetry(jobId);
        if (status.status === "done" || status.status === "error") {
          stopPolling();
        }
      } catch (_) {
        stopPolling();
      }
    }, POLL_MS);
    return stopPolling;
  }, [jobId, pollTelemetry]);

  const handleUpload = async (file) => {
    setUploadPct(0);
    setTelemetry([]);
    setJob(null);
    try {
      const { job_id } = await startUploadJob(file, setUploadPct);
      setJobId(job_id);
    } catch (e) {
      setJob({ status: "error", error: e.message });
    } finally {
      setUploadPct(null);
    }
  };

  const handleDemo = async () => {
    setTelemetry([]);
    setJob(null);
    try {
      const { job_id } = await startDemoJob();
      setJobId(job_id);
    } catch (e) {
      setJob({ status: "error", error: e.message });
    }
  };

  const busy = job && (job.status === "queued" || job.status === "processing");
  const videoSrc = job?.status === "done" ? jobVideoUrl(jobId) : null;

  return (
    <div className="min-h-screen text-hud-white">
      <Header />
      <main className="mx-auto max-w-6xl px-6 py-6">
        <div className="grid grid-cols-1 gap-4 lg:grid-cols-3">
          <div className="lg:col-span-2 space-y-4">
            <UploadPanel
              job={job}
              uploadPct={uploadPct}
              onUpload={handleUpload}
              onDemo={handleDemo}
              disabled={!!busy}
            />
            <VideoStage
              ref={videoRef}
              src={videoSrc}
              onTimeUpdate={(e) => setCurrentTime(e.currentTarget.currentTime)}
            />
            {videoSrc && (
              <video
                className="hidden"
                src={videoSrc}
                onLoadedMetadata={(e) => setDuration(e.currentTarget.duration)}
              />
            )}
            <StateTimeline
              telemetry={telemetry}
              duration={duration || (telemetry.at(-1)?.t ?? 0)}
              currentTime={currentTime}
              onSeek={(t) => {
                if (videoRef.current) videoRef.current.currentTime = t;
              }}
            />
            <TelemetryCharts telemetry={telemetry} />
          </div>

          <div className="space-y-4">
            <HudTelemetryPanel telemetry={telemetry} currentTime={currentTime} />
          </div>
        </div>
      </main>
    </div>
  );
}
