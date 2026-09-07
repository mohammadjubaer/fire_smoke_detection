import { useState, useRef, useEffect, useCallback } from 'react';
import * as ort from 'onnxruntime-web';
import {
  preprocessImage,
  postprocessDetections,
  postprocessRawOutput,
  CLASS_COLORS,
  type Detection,
} from './utils/yolo';

// Configure ONNX Runtime to use CDN for WASM files
ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/';

type AppState = 'idle' | 'loading-model' | 'ready' | 'detecting' | 'error';

export default function App() {
  const [state, setState] = useState<AppState>('idle');
  const [modelName, setModelName] = useState<string>('');
  const [errorMsg, setErrorMsg] = useState<string>('');
  const [detections, setDetections] = useState<Detection[]>([]);
  const [fps, setFps] = useState<number>(0);
  const [confidence, setConfidence] = useState<number>(0.3);
  const [cameraActive, setCameraActive] = useState(false);
  const [modelInfo, setModelInfo] = useState<{ inputs: string; outputs: string; outputDims: string } | null>(null);
  const [showDebug, setShowDebug] = useState(false);

  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const overlayRef = useRef<HTMLCanvasElement>(null);
  const preprocessCanvasRef = useRef<HTMLCanvasElement>(null);
  const sessionRef = useRef<ort.InferenceSession | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const animFrameRef = useRef<number>(0);
  const lastTimeRef = useRef<number>(0);
  const frameCountRef = useRef<number>(0);
  const confidenceRef = useRef<number>(0.3);
  const outputDimsSetRef = useRef<boolean>(false);

  // Keep confidence ref in sync
  useEffect(() => {
    confidenceRef.current = confidence;
  }, [confidence]);

  // Load model from file
  const loadModel = useCallback(async (file: File) => {
    setState('loading-model');
    setErrorMsg('');
    setModelName(file.name);
    outputDimsSetRef.current = false;

    try {
      const arrayBuffer = await file.arrayBuffer();
      
      const session = await ort.InferenceSession.create(arrayBuffer, {
        executionProviders: ['wasm'],
        graphOptimizationLevel: 'all',
      });

      sessionRef.current = session;

      // Get model info
      const inputNames = session.inputNames;
      const outputNames = session.outputNames;
      const inputInfos = inputNames.map(name => {
        const info = session.inputMetadata[name as keyof typeof session.inputMetadata];
        return `${name}: [${(info as any).dimensions?.join(', ') || '?'}]`;
      }).join(', ');
      const outputInfos = outputNames.map(name => {
        const info = session.outputMetadata[name as keyof typeof session.outputMetadata];
        return `${name}: [${(info as any).dimensions?.join(', ') || '?'}]`;
      }).join(', ');

      setModelInfo({ inputs: inputInfos, outputs: outputInfos, outputDims: '' });
      setState('ready');
    } catch (err: any) {
      console.error('Model load error:', err);
      setErrorMsg(`Failed to load model: ${err.message}`);
      setState('error');
    }
  }, []);

  // Start camera
  const startCamera = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: 'environment',
          width: { ideal: 1280 },
          height: { ideal: 720 },
        },
        audio: false,
      });

      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }
      setCameraActive(true);
    } catch (err: any) {
      console.error('Camera error:', err);
      setErrorMsg(`Camera access failed: ${err.message}. Make sure you're on HTTPS or localhost.`);
      setState('error');
    }
  }, []);

  // Stop camera
  const stopCamera = useCallback(() => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
      streamRef.current = null;
    }
    if (animFrameRef.current) {
      cancelAnimationFrame(animFrameRef.current);
      animFrameRef.current = 0;
    }
    setCameraActive(false);
    setDetections([]);
    setFps(0);
  }, []);

  // Run detection loop
  const runDetection = useCallback(async () => {
    if (!sessionRef.current || !videoRef.current || !canvasRef.current || !overlayRef.current || !preprocessCanvasRef.current) {
      return;
    }

    const video = videoRef.current;
    const canvas = canvasRef.current;
    const overlay = overlayRef.current;
    const preprocessCanvas = preprocessCanvasRef.current;

    if (video.readyState < 2) {
      animFrameRef.current = requestAnimationFrame(runDetection);
      return;
    }

    const ctx = canvas.getContext('2d')!;
    const overlayCtx = overlay.getContext('2d')!;
    const preprocessCtx = preprocessCanvas.getContext('2d')!;

    const videoWidth = video.videoWidth;
    const videoHeight = video.videoHeight;

    // Set canvas sizes
    canvas.width = videoWidth;
    canvas.height = videoHeight;
    overlay.width = videoWidth;
    overlay.height = videoHeight;

    // Draw video frame
    ctx.drawImage(video, 0, 0, videoWidth, videoHeight);

    // Determine input size from model
    const inputName = sessionRef.current.inputNames[0];
    const inputDims = (sessionRef.current.inputMetadata as any)[inputName]?.dimensions || [1, 3, 640, 640];
    const inputSize = inputDims[2] || 640; // Assume square input

    // Preprocess
    preprocessCanvas.width = inputSize;
    preprocessCanvas.height = inputSize;
    const { inputData, ratio, padX, padY } = preprocessImage(
      preprocessCanvas,
      preprocessCtx,
      video,
      inputSize
    );

    // Create tensor
    const tensor = new ort.Tensor('float32', inputData, [1, 3, inputSize, inputSize]);

    try {
      // Run inference
      const feeds: Record<string, ort.Tensor> = {};
      feeds[inputName] = tensor;
      
      const results = await sessionRef.current.run(feeds);
      const outputName = sessionRef.current.outputNames[0];
      const outputTensor = results[outputName];
      const outputData = outputTensor.data as Float32Array;
      const outputDims = outputTensor.dims;

      // Update output dims info on first run
      if (!outputDimsSetRef.current) {
        outputDimsSetRef.current = true;
        setModelInfo(prev => prev ? { ...prev, outputDims: `[${Array.from(outputDims).join(', ')}]` } : null);
      }

      let newDetections: Detection[];
      const dims = Array.from(outputDims) as number[];

      // Check if output is already in [batch, num_detections, 6] format (YOLOv10)
      if (dims.length === 3 && dims[2] === 6) {
        const numDetections = dims[1];
        newDetections = postprocessDetections(
          outputData,
          numDetections,
          ratio,
          padX,
          padY,
          confidenceRef.current
        );
      } else if (dims.length === 2 && dims[1] === 6) {
        // [num_detections, 6] without batch dimension
        const numDetections = dims[0];
        newDetections = postprocessDetections(
          outputData,
          numDetections,
          ratio,
          padX,
          padY,
          confidenceRef.current
        );
      } else {
        // Raw output format (e.g., YOLOv8 style)
        newDetections = postprocessRawOutput(
          outputData,
          dims,
          ratio,
          padX,
          padY,
          confidenceRef.current
        );
      }

      setDetections(newDetections);

      // Draw detections on overlay
      overlayCtx.clearRect(0, 0, videoWidth, videoHeight);
      newDetections.forEach(det => {
        const color = CLASS_COLORS[det.label] || '#22c55e';
        const w = det.x2 - det.x1;
        const h = det.y2 - det.y1;

        // Draw bounding box
        overlayCtx.strokeStyle = color;
        overlayCtx.lineWidth = 3;
        overlayCtx.strokeRect(det.x1, det.y1, w, h);

        // Draw label background
        const label = `${det.label} ${(det.confidence * 100).toFixed(0)}%`;
        overlayCtx.font = 'bold 16px Arial';
        const textMetrics = overlayCtx.measureText(label);
        const textHeight = 20;
        const textWidth = textMetrics.width + 10;

        overlayCtx.fillStyle = color;
        overlayCtx.fillRect(det.x1, det.y1 - textHeight - 4, textWidth, textHeight + 4);

        // Draw label text
        overlayCtx.fillStyle = '#ffffff';
        overlayCtx.fillText(label, det.x1 + 5, det.y1 - 8);
      });
    } catch (err) {
      console.error('Inference error:', err);
    }

    // Calculate FPS
    frameCountRef.current++;
    const now = performance.now();
    if (now - lastTimeRef.current >= 1000) {
      setFps(frameCountRef.current);
      frameCountRef.current = 0;
      lastTimeRef.current = now;
    }

    animFrameRef.current = requestAnimationFrame(runDetection);
  }, []);

  // Start/stop detection
  const toggleDetection = useCallback(() => {
    if (state === 'detecting') {
      if (animFrameRef.current) {
        cancelAnimationFrame(animFrameRef.current);
        animFrameRef.current = 0;
      }
      setState('ready');
      setDetections([]);
      setFps(0);
      const overlay = overlayRef.current;
      if (overlay) {
        const ctx = overlay.getContext('2d');
        ctx?.clearRect(0, 0, overlay.width, overlay.height);
      }
    } else {
      setState('detecting');
      lastTimeRef.current = performance.now();
      frameCountRef.current = 0;
      runDetection();
    }
  }, [state, runDetection]);

  // Handle file upload
  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      loadModel(file);
    }
  };

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      stopCamera();
      if (sessionRef.current) {
        sessionRef.current.release();
      }
    };
  }, [stopCamera]);

  // Auto-start camera when model is ready
  useEffect(() => {
    if (state === 'ready' && !cameraActive) {
      startCamera();
    }
  }, [state, cameraActive, startCamera]);

  return (
    <div className="h-screen w-screen bg-slate-900 flex flex-col overflow-hidden">
      {/* Header */}
      <div className="bg-slate-800 px-4 py-3 flex items-center justify-between border-b border-slate-700 z-10">
        <div className="flex items-center gap-2">
          <div className="w-8 h-8 rounded-full bg-gradient-to-br from-orange-500 to-red-600 flex items-center justify-center">
            🔥
          </div>
          <div>
            <h1 className="text-white font-bold text-sm leading-tight">Fire & Smoke Detector</h1>
            <p className="text-slate-400 text-xs">YOLOv10 ONNX • Real-time</p>
          </div>
        </div>
        {state === 'detecting' && (
          <div className="flex items-center gap-2">
            <div className="w-2 h-2 rounded-full bg-green-500 animate-pulse" />
            <span className="text-green-400 text-xs font-mono">{fps} FPS</span>
          </div>
        )}
      </div>

      {/* Main content */}
      <div className="flex-1 relative overflow-hidden">
        {/* Video element (hidden, used as source for canvas) */}
        <video
          ref={videoRef}
          className="hidden"
          playsInline
          muted
        />

        {/* Main display canvas - shows video + detections */}
        <canvas
          ref={canvasRef}
          className="absolute top-0 left-0 w-full h-full object-cover"
          style={{ display: cameraActive ? 'block' : 'none' }}
        />

        {/* Overlay canvas for detection boxes */}
        <canvas
          ref={overlayRef}
          className="absolute top-0 left-0 w-full h-full object-cover pointer-events-none"
          style={{ display: cameraActive ? 'block' : 'none' }}
        />

        {/* Hidden preprocess canvas */}
        <canvas ref={preprocessCanvasRef} className="hidden" />

        {/* Idle state - Model upload */}
        {(state === 'idle' || state === 'error' || state === 'loading-model') && (
          <div className="absolute inset-0 flex items-center justify-center p-6">
            <div className="bg-slate-800 rounded-2xl p-8 max-w-sm w-full shadow-2xl border border-slate-700">
              <div className="text-center mb-6">
                <div className="w-16 h-16 mx-auto mb-4 rounded-full bg-gradient-to-br from-orange-500/20 to-red-600/20 flex items-center justify-center">
                  <span className="text-3xl">🔥</span>
                </div>
                <h2 className="text-white text-xl font-bold mb-2">Fire & Smoke Detection</h2>
                <p className="text-slate-400 text-sm">
                  Load your YOLOv10n ONNX model to start detecting fire and smoke in real-time using your camera.
                </p>
              </div>

              {state !== 'loading-model' && (
                <label className="block w-full cursor-pointer">
                  <div className="border-2 border-dashed border-slate-600 rounded-xl p-6 text-center hover:border-orange-500 transition-colors">
                    <div className="text-4xl mb-2">📁</div>
                    <p className="text-slate-300 text-sm font-medium">Tap to upload model</p>
                    <p className="text-slate-500 text-xs mt-1">.onnx file (YOLOv10n)</p>
                  </div>
                  <input
                    type="file"
                    accept=".onnx"
                    onChange={handleFileUpload}
                    className="hidden"
                  />
                </label>
              )}

              {state === 'loading-model' && (
                <div className="mt-4 flex items-center justify-center gap-2">
                  <div className="w-4 h-4 border-2 border-orange-500 border-t-transparent rounded-full animate-spin" />
                  <span className="text-orange-400 text-sm">Loading model...</span>
                </div>
              )}

              {state === 'error' && errorMsg && (
                <div className="mt-4 p-3 bg-red-900/30 border border-red-700 rounded-lg">
                  <p className="text-red-400 text-xs">{errorMsg}</p>
                </div>
              )}

              <div className="mt-6 p-3 bg-slate-700/50 rounded-lg">
                <p className="text-slate-400 text-xs leading-relaxed">
                  💡 <strong className="text-slate-300">Tips:</strong> Use HTTPS or localhost for camera access. 
                  Works best on mobile Chrome/Safari. The model runs entirely in your browser - no data is sent anywhere.
                </p>
              </div>
            </div>
          </div>
        )}

        {/* Detection results overlay */}
        {state === 'detecting' && detections.length > 0 && (
          <div className="absolute bottom-20 left-4 right-4 z-10">
            <div className="bg-slate-800/90 backdrop-blur-sm rounded-xl p-3 border border-slate-700">
              <div className="flex items-center gap-2 mb-2">
                <div className="w-2 h-2 rounded-full bg-red-500 animate-pulse" />
                <span className="text-white text-xs font-bold">
                  {detections.length} Detection{detections.length > 1 ? 's' : ''}
                </span>
              </div>
              <div className="flex flex-wrap gap-2">
                {detections.map((det, i) => (
                  <div
                    key={i}
                    className="px-2 py-1 rounded-md text-xs font-medium"
                    style={{
                      backgroundColor: (CLASS_COLORS[det.label] || '#22c55e') + '33',
                      color: CLASS_COLORS[det.label] || '#22c55e',
                      border: `1px solid ${CLASS_COLORS[det.label] || '#22c55e'}55`,
                    }}
                  >
                    {det.label === 'fire' ? '🔥' : '💨'} {det.label} {(det.confidence * 100).toFixed(0)}%
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Bottom controls */}
      {state !== 'idle' && state !== 'loading-model' && state !== 'error' && (
        <div className="bg-slate-800 border-t border-slate-700 px-4 py-3 z-10">
          {/* Model info */}
          {modelInfo && (
            <div className="mb-2">
              <div className="text-xs text-slate-500 truncate">
                📦 {modelName}
              </div>
              {showDebug && modelInfo.outputDims && (
                <div className="text-xs text-slate-600 mt-1 font-mono">
                  Output shape: {modelInfo.outputDims}
                </div>
              )}
              <button
                onClick={() => setShowDebug(!showDebug)}
                className="text-xs text-slate-600 underline mt-0.5"
              >
                {showDebug ? 'Hide' : 'Show'} debug info
              </button>
            </div>
          )}

          {/* Confidence slider */}
          <div className="flex items-center gap-3 mb-3">
            <span className="text-slate-400 text-xs whitespace-nowrap">Confidence</span>
            <input
              type="range"
              min="0.1"
              max="0.9"
              step="0.05"
              value={confidence}
              onChange={(e) => setConfidence(parseFloat(e.target.value))}
              className="flex-1 h-1 bg-slate-700 rounded-full appearance-none cursor-pointer accent-orange-500"
            />
            <span className="text-orange-400 text-xs font-mono w-8 text-right">
              {(confidence * 100).toFixed(0)}%
            </span>
          </div>

          {/* Action buttons */}
          <div className="flex gap-3">
            <button
              onClick={toggleDetection}
              className={`flex-1 py-3 rounded-xl font-bold text-sm transition-all ${
                state === 'detecting'
                  ? 'bg-red-600 text-white active:bg-red-700'
                  : 'bg-gradient-to-r from-orange-500 to-red-600 text-white active:opacity-90'
              }`}
            >
              {state === 'detecting' ? '⏹ Stop Detection' : '▶ Start Detection'}
            </button>
            <button
              onClick={() => {
                stopCamera();
                setState('idle');
                setModelName('');
                setModelInfo(null);
                outputDimsSetRef.current = false;
                if (sessionRef.current) {
                  sessionRef.current.release();
                  sessionRef.current = null;
                }
              }}
              className="px-4 py-3 rounded-xl bg-slate-700 text-slate-300 text-sm font-medium active:bg-slate-600"
            >
              🔄
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
