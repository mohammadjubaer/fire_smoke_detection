import { useState, useRef, useEffect, useCallback } from 'react';
import {
  preprocessImage,
  postprocessDetections,
  postprocessRawOutput,
  CLASS_COLORS,
  type Detection,
} from './utils/yolo';

type AppState = 'idle' | 'loading-model' | 'ready' | 'detecting' | 'error';

export default function App() {
  const [state, setState] = useState<AppState>('idle');
  const [modelName, setModelName] = useState('');
  const [errorMsg, setErrorMsg] = useState('');
  const [detections, setDetections] = useState<Detection[]>([]);
  const [fps, setFps] = useState(0);
  const [confidence, setConfidence] = useState(0.3);
  const [cameraActive, setCameraActive] = useState(false);
  const [modelInfo, setModelInfo] = useState<{ inputs: string; outputs: string; outputDims: string } | null>(null);
  const [showDebug, setShowDebug] = useState(false);
  const [isSecure, setIsSecure] = useState(true);
  const [ortLoaded, setOrtLoaded] = useState(false);

  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const overlayRef = useRef<HTMLCanvasElement>(null);
  const preprocessCanvasRef = useRef<HTMLCanvasElement>(null);
  const sessionRef = useRef<any>(null);
  const ortRef = useRef<any>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const animFrameRef = useRef<number>(0);
  const lastTimeRef = useRef<number>(0);
  const frameCountRef = useRef<number>(0);
  const confidenceRef = useRef<number>(0.3);
  const outputDimsSetRef = useRef<boolean>(false);

  useEffect(() => {
    confidenceRef.current = confidence;
  }, [confidence]);

  useEffect(() => {
    const secure = window.location.protocol === 'https:' || 
                   window.location.hostname === 'localhost' || 
                   window.location.hostname === '127.0.0.1';
    setIsSecure(secure);
    
    // Load ORT from CDN
    loadORT();
  }, []);

  const loadORT = async () => {
    try {
      console.log('Loading ONNX Runtime from CDN...');
      const script = document.createElement('script');
      script.src = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.21.0/dist/ort.min.js';
      script.onload = () => {
        console.log('ORT loaded successfully');
        const ort = (window as any).ort;
        if (ort) {
          ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.21.0/dist/';
          ort.env.logLevel = 'error';
          ortRef.current = ort;
          setOrtLoaded(true);
        } else {
          throw new Error('ORT global not found after script load');
        }
      };
      script.onerror = () => {
        throw new Error('Failed to load ONNX Runtime script from CDN');
      };
      document.head.appendChild(script);
    } catch (err: any) {
      console.error('ORT load error:', err);
      setErrorMsg(`Failed to load ONNX Runtime: ${err.message}`);
      setState('error');
    }
  };

  const loadModel = useCallback(async (file: File) => {
    if (!ortRef.current) {
      setErrorMsg('ONNX Runtime not loaded yet. Please wait...');
      setState('error');
      return;
    }

    setState('loading-model');
    setErrorMsg('');
    setModelName(file.name);
    outputDimsSetRef.current = false;

    try {
      console.log('Loading model:', file.name);
      const arrayBuffer = await file.arrayBuffer();
      
      const session = await ortRef.current.InferenceSession.create(arrayBuffer, {
        executionProviders: ['wasm'],
        graphOptimizationLevel: 'all',
      });

      sessionRef.current = session;

      const inputNames = session.inputNames;
      const outputNames = session.outputNames;
      const inputInfos = inputNames.map((name: string) => {
        const info = session.inputMetadata[name];
        return `${name}: [${info.dimensions?.join(', ') || '?'}]`;
      }).join(', ');
      const outputInfos = outputNames.map((name: string) => {
        const info = session.outputMetadata[name];
        return `${name}: [${info.dimensions?.join(', ') || '?'}]`;
      }).join(', ');

      setModelInfo({ inputs: inputInfos, outputs: outputInfos, outputDims: '' });
      setState('ready');
      console.log('Model loaded successfully');
    } catch (err: any) {
      console.error('Model load error:', err);
      setErrorMsg(`Failed to load model: ${err.message}`);
      setState('error');
    }
  }, []);

  const handleFileUpload = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      loadModel(file);
    }
  }, [loadModel]);

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

  const stopCamera = useCallback(() => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
      streamRef.current = null;
    }
    setCameraActive(false);
  }, []);

  const runDetection = useCallback(async () => {
    if (!sessionRef.current || !cameraActive) return;

    const video = videoRef.current;
    const canvas = canvasRef.current;
    const overlay = overlayRef.current;
    const preprocessCanvas = preprocessCanvasRef.current;

    if (!video || !canvas || !overlay || !preprocessCanvas || video.readyState < 2) {
      animFrameRef.current = requestAnimationFrame(runDetection);
      return;
    }

    const ctx = canvas.getContext('2d')!;
    const overlayCtx = overlay.getContext('2d')!;
    const preprocessCtx = preprocessCanvas.getContext('2d')!;

    const videoWidth = video.videoWidth;
    const videoHeight = video.videoHeight;

    canvas.width = videoWidth;
    canvas.height = videoHeight;
    overlay.width = videoWidth;
    overlay.height = videoHeight;

    ctx.drawImage(video, 0, 0, videoWidth, videoHeight);

    const inputName = sessionRef.current.inputNames[0];
    const inputDims = sessionRef.current.inputMetadata[inputName]?.dimensions || [1, 3, 640, 640];
    const inputSize = inputDims[2] || 640;

    preprocessCanvas.width = inputSize;
    preprocessCanvas.height = inputSize;
    const { inputData, ratio, padX, padY } = preprocessImage(
      preprocessCanvas,
      preprocessCtx,
      video,
      inputSize
    );

    const tensor = new ortRef.current.Tensor('float32', inputData, [1, 3, inputSize, inputSize]);

    try {
      const feeds: Record<string, any> = {};
      feeds[inputName] = tensor;
      
      const results = await sessionRef.current.run(feeds);
      const outputName = sessionRef.current.outputNames[0];
      const outputTensor = results[outputName];
      const outputData = outputTensor.data as Float32Array;
      const outputDims = outputTensor.dims;

      if (!outputDimsSetRef.current) {
        outputDimsSetRef.current = true;
        setModelInfo(prev => prev ? { ...prev, outputDims: `[${Array.from(outputDims).join(', ')}]` } : null);
      }

      let newDetections: Detection[];
      const dims: number[] = Array.from(outputDims) as number[];

      if (dims.length === 3 && dims[2] === 6) {
        const numDetections = dims[1] as number;
        newDetections = postprocessDetections(
          outputData,
          numDetections,
          ratio,
          padX,
          padY,
          confidenceRef.current
        );
      } else if (dims.length === 2 && dims[1] === 6) {
        const numDetections = dims[0] as number;
        newDetections = postprocessDetections(
          outputData,
          numDetections,
          ratio,
          padX,
          padY,
          confidenceRef.current
        );
      } else {
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

      overlayCtx.clearRect(0, 0, videoWidth, videoHeight);
      newDetections.forEach(det => {
        const color = CLASS_COLORS[det.label] || '#22c55e';
        const w = det.x2 - det.x1;
        const h = det.y2 - det.y1;

        overlayCtx.strokeStyle = color;
        overlayCtx.lineWidth = 3;
        overlayCtx.strokeRect(det.x1, det.y1, w, h);

        const label = `${det.label} ${(det.confidence * 100).toFixed(0)}%`;
        overlayCtx.font = 'bold 16px Arial';
        const textMetrics = overlayCtx.measureText(label);
        const textHeight = 20;
        const textWidth = textMetrics.width + 10;

        overlayCtx.fillStyle = color;
        overlayCtx.fillRect(det.x1, det.y1 - textHeight, textWidth, textHeight);

        overlayCtx.fillStyle = '#ffffff';
        overlayCtx.fillText(label, det.x1 + 5, det.y1 - 5);
      });

      frameCountRef.current++;
      const now = performance.now();
      if (now - lastTimeRef.current >= 1000) {
        setFps(frameCountRef.current);
        frameCountRef.current = 0;
        lastTimeRef.current = now;
      }
    } catch (err: any) {
      console.error('Detection error:', err);
      setErrorMsg(`Detection error: ${err.message}`);
    }

    animFrameRef.current = requestAnimationFrame(runDetection);
  }, [cameraActive]);

  const toggleDetection = useCallback(() => {
    if (state === 'detecting') {
      setState('ready');
      cancelAnimationFrame(animFrameRef.current);
      setDetections([]);
      setFps(0);
    } else if (state === 'ready') {
      setState('detecting');
      lastTimeRef.current = performance.now();
      frameCountRef.current = 0;
      runDetection();
    }
  }, [state, runDetection]);

  useEffect(() => {
    if (state === 'ready' && cameraActive) {
      startCamera();
    }
  }, [state, cameraActive, startCamera]);

  useEffect(() => {
    return () => {
      stopCamera();
      cancelAnimationFrame(animFrameRef.current);
      if (sessionRef.current) {
        sessionRef.current.release();
      }
    };
  }, [stopCamera]);

  if (!ortLoaded && state === 'idle') {
    return (
      <div className="h-screen w-screen bg-slate-900 flex items-center justify-center">
        <div className="text-center">
          <div className="w-12 h-12 border-4 border-orange-500 border-t-transparent rounded-full animate-spin mx-auto mb-4" />
          <p className="text-white text-lg">Loading ONNX Runtime...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="h-screen w-screen bg-slate-900 flex flex-col overflow-hidden">
      {!isSecure && (
        <div className="bg-yellow-600 text-white text-xs px-4 py-2 text-center font-medium">
          ⚠️ Camera requires HTTPS. Please access this page via HTTPS or localhost.
        </div>
      )}
      
      <div className="bg-slate-800 px-4 py-3 flex items-center justify-between border-b border-slate-700 z-10">
        <div className="flex items-center gap-2">
          <span className="text-2xl">🔥</span>
          <h1 className="text-white font-bold text-sm sm:text-base">Fire & Smoke Detector</h1>
        </div>
        {state === 'detecting' && (
          <div className="flex items-center gap-2">
            <div className="w-2 h-2 rounded-full bg-green-500 animate-pulse" />
            <span className="text-green-400 text-xs font-mono">{fps} FPS</span>
          </div>
        )}
      </div>

      <div className="flex-1 relative overflow-hidden">
        <video ref={videoRef} className="hidden" playsInline muted />

        <canvas
          ref={canvasRef}
          className="absolute top-0 left-0 w-full h-full object-cover"
          style={{ display: cameraActive ? 'block' : 'none' }}
        />

        <canvas
          ref={overlayRef}
          className="absolute top-0 left-0 w-full h-full object-cover pointer-events-none"
          style={{ display: cameraActive ? 'block' : 'none' }}
        />

        <canvas ref={preprocessCanvasRef} className="hidden" />

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
      </div>

      {(state === 'ready' || state === 'detecting') && (
        <div className="bg-slate-800 border-t border-slate-700 p-4 space-y-3 z-10">
          <div className="flex items-center justify-between text-xs">
            <span className="text-slate-400 truncate flex-1">{modelName}</span>
            <button
              onClick={() => setShowDebug(!showDebug)}
              className="text-slate-500 hover:text-slate-300 ml-2"
            >
              {showDebug ? '▼' : '▶'} Info
            </button>
          </div>

          {showDebug && modelInfo && (
            <div className="text-xs text-slate-500 space-y-1 bg-slate-900/50 rounded p-2">
              <div>Input: {modelInfo.inputs}</div>
              <div>Output: {modelInfo.outputs}</div>
              {modelInfo.outputDims && <div>Output shape: {modelInfo.outputDims}</div>}
            </div>
          )}

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
