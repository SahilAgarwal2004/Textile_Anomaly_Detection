import { useState, useRef, useCallback } from "react";

const API_BASE = "http://127.0.0.1:8000";

async function callAPI(imageFile) {
  const form = new FormData();
  form.append("file", imageFile);

  const res = await fetch(`${API_BASE}/predict`, {
    method: "POST",
    body: form,
  });

  if (!res.ok) {
    const err = await res.text();
    throw new Error(`Server error ${res.status}: ${err}`);
  }

  const data = await res.json();
  return {
    prediction: data.prediction === "ANOMALY" ? "anomaly" : "normal",
    ...(data.prediction === "ANOMALY" && {
      mask_url: `data:image/png;base64,${data.defect_patch_base64}`,
      label: "Defect Detected",
      explanation: data.reasoning,
    }),
  };
}

function UploadZone({ onFile, disabled }) {
  const inputRef = useRef();
  const [dragging, setDragging] = useState(false);

  const handle = (file) => {
    if (file && file.type.startsWith("image/")) onFile(file);
  };

  return (
    <div
      onClick={() => !disabled && inputRef.current.click()}
      onDragOver={(e) => { e.preventDefault(); if (!disabled) setDragging(true); }}
      onDragLeave={() => setDragging(false)}
      onDrop={(e) => { e.preventDefault(); setDragging(false); if (!disabled) handle(e.dataTransfer.files[0]); }}
      className={[
        "relative flex flex-col items-center justify-center gap-4",
        "border-2 border-dashed rounded-2xl p-16 text-center",
        "transition-all duration-200 select-none",
        disabled ? "opacity-40 cursor-not-allowed" : "cursor-pointer",
        dragging
          ? "border-lime-400 bg-lime-400/5"
          : "border-zinc-700 bg-zinc-900 hover:border-zinc-500 hover:bg-zinc-800/60",
      ].join(" ")}
    >
      <input
        ref={inputRef}
        type="file"
        accept="image/*"
        className="hidden"
        onChange={(e) => handle(e.target.files[0])}
      />
      <div className={`transition-colors duration-200 ${dragging ? "text-lime-400" : "text-zinc-600"}`}>
        <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.3">
          <path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4" />
          <polyline points="17 8 12 3 7 8" />
          <line x1="12" y1="3" x2="12" y2="15" />
        </svg>
      </div>
      <div>
        <p className="text-zinc-200 text-lg font-light tracking-wide">Drop a textile image here</p>
        <p className="text-zinc-500 text-xs mt-1 tracking-widest uppercase">or click to browse · PNG, JPG, WEBP</p>
      </div>
    </div>
  );
}

function Spinner() {
  return (
    <div className="flex flex-col items-center gap-4 py-12">
      <div className="w-10 h-10 rounded-full border-2 border-zinc-700 border-t-lime-400 animate-spin" />
      <p className="text-zinc-500 text-xs tracking-widest uppercase">Analysing fabric…</p>
    </div>
  );
}

function Badge({ isAnomaly }) {
  return (
    <div className={[
      "flex items-center gap-2.5 px-5 py-3 text-xs tracking-widest uppercase font-medium border-b",
      isAnomaly
        ? "bg-red-950/40 text-red-400 border-red-900/50"
        : "bg-emerald-950/40 text-emerald-400 border-emerald-900/50",
    ].join(" ")}>
      <span className={`w-2 h-2 rounded-full flex-shrink-0 ${isAnomaly ? "bg-red-400 animate-pulse" : "bg-emerald-400"}`} />
      {isAnomaly ? "Anomaly Detected" : "Normal Fabric"}
    </div>
  );
}

// Single image panel — fills its box completely, label pill overlaid
function ImagePanel({ src, alt, label, tall = false }) {
  return (
    <figure className={`relative w-full bg-zinc-950 overflow-hidden ${tall ? "h-96" : "h-72"}`}>
      <span className="absolute top-2.5 left-2.5 z-10 text-[10px] tracking-widest uppercase text-zinc-400 bg-black/70 backdrop-blur-sm px-2 py-0.5 rounded-full border border-zinc-700/50 pointer-events-none">
        {label}
      </span>
      <img
        src={src}
        alt={alt}
        className="absolute inset-0 w-full h-full object-contain object-center"
      />
    </figure>
  );
}

function ResultCard({ result, originalUrl }) {
  const isAnomaly = result.prediction === "anomaly";

  return (
    <div
      className={[
        "rounded-2xl overflow-hidden border",
        isAnomaly ? "border-red-900/50 bg-zinc-900" : "border-emerald-900/50 bg-zinc-900",
      ].join(" ")}
      style={{ animation: "fadeUp .35s ease both" }}
    >
      <Badge isAnomaly={isAnomaly} />

      {/* Image grid — side by side when anomaly, full-width when normal */}
      {isAnomaly ? (
        <div className="grid md:grid-cols-2 grid-cols-1 gap-px bg-zinc-800">
          <ImagePanel src={originalUrl} alt="Original textile" label="Original" />
          <ImagePanel src={result.mask_url}  alt="Defect patch"     label="Defect Patch" />
        </div>
      ) : (
        <ImagePanel src={originalUrl} alt="Original textile" label="Original" tall />
      )}

      {/* Reasoning */}
      {isAnomaly && (
        <div className="px-6 py-5 border-t border-red-900/30 space-y-2">
          <div className="flex items-center gap-2 text-red-400">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2">
              <circle cx="12" cy="12" r="10" />
              <line x1="12" y1="8" x2="12" y2="12" />
              <line x1="12" y1="16" x2="12.01" y2="16" />
            </svg>
            <span className="text-sm font-semibold tracking-wide">{result.label}</span>
          </div>
          <p className="text-zinc-400 text-sm leading-relaxed">{result.explanation}</p>
        </div>
      )}

      {!isAnomaly && (
        <p className="px-6 py-5 text-sm text-zinc-500 border-t border-emerald-900/30 leading-relaxed">
          No irregularities detected. The textile pattern falls within expected quality parameters.
        </p>
      )}
    </div>
  );
}

export default function App() {
  const [originalUrl, setOriginalUrl] = useState(null);
  const [status, setStatus] = useState("idle");
  const [result, setResult] = useState(null);
  const [errorMsg, setErrorMsg] = useState("");

  const handleFile = useCallback(async (file) => {
    setOriginalUrl(URL.createObjectURL(file));
    setResult(null);
    setErrorMsg("");
    setStatus("loading");
    try {
      const data = await callAPI(file);
      setResult(data);
      setStatus("done");
    } catch (e) {
      setErrorMsg(e.message || "Unknown error");
      setStatus("error");
    }
  }, []);

  const reset = () => {
    setStatus("idle");
    setOriginalUrl(null);
    setResult(null);
    setErrorMsg("");
  };

  return (
    <>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Fraunces:ital,wght@0,300;0,600;1,300&display=swap');
        body { font-family: 'DM Mono', monospace; }
        .serif { font-family: 'Fraunces', Georgia, serif; }
        @keyframes fadeUp {
          from { opacity: 0; transform: translateY(14px); }
          to   { opacity: 1; transform: none; }
        }
      `}</style>

      <div className="min-h-screen bg-zinc-950 text-zinc-100 flex flex-col">

        <header className="px-8 pt-10">
          <div className="max-w-3xl mx-auto">
            <div className="flex items-baseline gap-4 flex-wrap">
              <div className="flex items-center gap-2">
                <span className="text-lime-400 text-2xl">◈</span>
                <span className="serif text-2xl font-semibold tracking-tight">FabricScan</span>
              </div>
              <span className="text-zinc-600 text-xs tracking-widest uppercase">
                Zero-Shot Textile Anomaly Detection
              </span>
            </div>
            <div className="mt-6 h-px bg-zinc-800" />
          </div>
        </header>

        <main className="flex-1 px-8 py-10 max-w-3xl mx-auto w-full flex flex-col gap-6">

          <div className="flex flex-col gap-3">
            <UploadZone onFile={handleFile} disabled={status === "loading"} />
            {status !== "idle" && (
              <button
                onClick={reset}
                className="self-start text-xs text-zinc-500 border border-zinc-800 hover:border-zinc-600 hover:text-zinc-300 px-3 py-1.5 rounded-lg tracking-wide transition-colors cursor-pointer"
              >
                ↺ Upload another image
              </button>
            )}
          </div>

          {status === "loading" && <Spinner />}

          {status === "error" && (
            <div className="bg-red-950/30 border border-red-900/50 text-red-400 rounded-xl px-5 py-4 text-sm">
              <strong>Analysis failed:</strong> {errorMsg}
            </div>
          )}

          {status === "done" && result && (
            <ResultCard result={result} originalUrl={originalUrl} />
          )}
        </main>

        <footer className="text-center py-6 text-zinc-700 text-xs tracking-wide px-4">
          Connect your detection API in the{" "}
          <code className="text-zinc-600">callAPI</code> function · Results shown for illustration
        </footer>
      </div>
    </>
  );
}