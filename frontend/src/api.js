const BASE = "/api";

async function asJson(res) {
  if (!res.ok) {
    let msg = res.statusText;
    try {
      const body = await res.json();
      msg = body.error || msg;
    } catch (_) {
      /* ignore */
    }
    throw new Error(msg);
  }
  return res.json();
}

export async function startUploadJob(file, onUploadProgress) {
  return new Promise((resolve, reject) => {
    const form = new FormData();
    form.append("video", file);
    const xhr = new XMLHttpRequest();
    xhr.open("POST", `${BASE}/jobs/upload`);
    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable && onUploadProgress) {
        onUploadProgress(Math.round((e.loaded / e.total) * 100));
      }
    };
    xhr.onload = () => {
      try {
        const body = JSON.parse(xhr.responseText);
        if (xhr.status >= 200 && xhr.status < 300) resolve(body);
        else reject(new Error(body.error || "Upload failed"));
      } catch (e) {
        reject(e);
      }
    };
    xhr.onerror = () => reject(new Error("Network error during upload"));
    xhr.send(form);
  });
}

export async function startDemoJob() {
  const res = await fetch(`${BASE}/jobs/demo`, { method: "POST" });
  return asJson(res);
}

export async function getJobStatus(jobId) {
  const res = await fetch(`${BASE}/jobs/${jobId}`);
  return asJson(res);
}

export async function getJobTelemetry(jobId) {
  const res = await fetch(`${BASE}/jobs/${jobId}/telemetry`);
  return asJson(res);
}

export function jobVideoUrl(jobId) {
  return `${BASE}/jobs/${jobId}/video`;
}
