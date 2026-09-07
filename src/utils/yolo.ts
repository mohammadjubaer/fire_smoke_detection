// YOLOv10 preprocessing and postprocessing utilities

export interface Detection {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  confidence: number;
  classId: number;
  label: string;
}

export interface PreprocessResult {
  inputData: Float32Array;
  ratio: number;
  padX: number;
  padY: number;
}

// Class labels for fire and smoke detection
export const CLASS_LABELS = ['fire', 'smoke'];

// Colors for each class
export const CLASS_COLORS: Record<string, string> = {
  fire: '#ef4444',
  smoke: '#94a3b8',
};

/**
 * Preprocess an image for YOLOv10 inference
 * - Resize with letterboxing to 640x640
 * - Normalize to [0, 1]
 * - Convert to NCHW format
 */
export function preprocessImage(
  canvas: HTMLCanvasElement,
  ctx: CanvasRenderingContext2D,
  source: HTMLVideoElement | HTMLImageElement,
  inputSize: number = 640
): PreprocessResult {
  const srcWidth = source instanceof HTMLVideoElement ? source.videoWidth : source.naturalWidth;
  const srcHeight = source instanceof HTMLVideoElement ? source.videoHeight : source.naturalHeight;

  // Calculate letterbox scaling
  const ratio = Math.min(inputSize / srcWidth, inputSize / srcHeight);
  const newWidth = Math.round(srcWidth * ratio);
  const newHeight = Math.round(srcHeight * ratio);
  const padX = (inputSize - newWidth) / 2;
  const padY = (inputSize - newHeight) / 2;

  // Fill with gray (letterbox padding)
  ctx.fillStyle = '#808080';
  ctx.fillRect(0, 0, inputSize, inputSize);

  // Draw the resized image centered
  ctx.drawImage(source, padX, padY, newWidth, newHeight);

  // Get pixel data
  const imageData = ctx.getImageData(0, 0, inputSize, inputSize);
  const pixels = imageData.data;

  // Convert HWC to NCHW and normalize to [0, 1]
  const inputData = new Float32Array(3 * inputSize * inputSize);
  for (let i = 0; i < pixels.length / 4; i++) {
    const r = pixels[i * 4] / 255.0;
    const g = pixels[i * 4 + 1] / 255.0;
    const b = pixels[i * 4 + 2] / 255.0;
    
    inputData[i] = r;                           // R channel
    inputData[inputSize * inputSize + i] = g;   // G channel
    inputData[2 * inputSize * inputSize + i] = b; // B channel
  }

  return { inputData, ratio, padX, padY };
}

/**
 * Postprocess YOLOv10 output
 * YOLOv10 output shape: [1, num_detections, 6]
 * Each detection: [x1, y1, x2, y2, confidence, class_id]
 */
export function postprocessDetections(
  output: Float32Array,
  numDetections: number,
  ratio: number,
  padX: number,
  padY: number,
  confThreshold: number = 0.3,
  iouThreshold: number = 0.45
): Detection[] {
  const detections: Detection[] = [];

  for (let i = 0; i < numDetections; i++) {
    const offset = i * 6;
    const x1 = output[offset];
    const y1 = output[offset + 1];
    const x2 = output[offset + 2];
    const y2 = output[offset + 3];
    const confidence = output[offset + 4];
    const classId = Math.round(output[offset + 5]);

    if (confidence < confThreshold) continue;

    // Remove letterbox padding and rescale to original image coordinates
    const origX1 = (x1 - padX) / ratio;
    const origY1 = (y1 - padY) / ratio;
    const origX2 = (x2 - padX) / ratio;
    const origY2 = (y2 - padY) / ratio;

    const label = CLASS_LABELS[classId] || `class_${classId}`;

    detections.push({
      x1: origX1,
      y1: origY1,
      x2: origX2,
      y2: origY2,
      confidence,
      classId,
      label,
    });
  }

  // Apply NMS
  return nonMaxSuppression(detections, iouThreshold);
}

/**
 * Postprocess for models that output raw predictions (like YOLOv8 format)
 * Output shape: [1, num_classes+4, num_detections]
 * Transposed: [1, num_detections, num_classes+4]
 */
export function postprocessRawOutput(
  output: Float32Array,
  dims: number[],
  ratio: number,
  padX: number,
  padY: number,
  confThreshold: number = 0.3,
  iouThreshold: number = 0.45
): Detection[] {
  const detections: Detection[] = [];
  
  // Determine output format
  let numClasses: number;
  let numDetections: number;
  let isTransposed = false;

  if (dims.length === 3) {
    if (dims[1] === 1 || dims[0] === 1) {
      // [1, num_classes+4, num_detections] or [num_classes+4, num_detections, 1]
      const d = dims[0] === 1 ? dims[1] : dims[0];
      const n = dims[0] === 1 ? dims[2] : dims[1];
      
      if (d < n) {
        // [1, num_classes+4, num_detections]
        numClasses = d - 4;
        numDetections = n;
      } else {
        // [1, num_detections, num_classes+4]
        numClasses = n - 4;
        numDetections = d;
        isTransposed = true;
      }
    } else if (dims[2] < dims[1]) {
      numClasses = dims[2] - 4;
      numDetections = dims[1];
      isTransposed = true;
    } else {
      numClasses = dims[1] - 4;
      numDetections = dims[2];
    }
  } else {
    numClasses = CLASS_LABELS.length;
    numDetections = output.length / (numClasses + 4);
  }

  for (let i = 0; i < numDetections; i++) {
    let x1: number, y1: number, x2: number, y2: number;
    let maxConf = 0;
    let maxClassId = 0;

    if (isTransposed) {
      // [num_detections, num_classes+4]
      const offset = i * (numClasses + 4);
      x1 = output[offset];
      y1 = output[offset + 1];
      x2 = output[offset + 2];
      y2 = output[offset + 3];
      
      for (let c = 0; c < numClasses; c++) {
        const score = output[offset + 4 + c];
        if (score > maxConf) {
          maxConf = score;
          maxClassId = c;
        }
      }
    } else {
      // [num_classes+4, num_detections]
      x1 = output[i];
      y1 = output[numDetections + i];
      x2 = output[2 * numDetections + i];
      y2 = output[3 * numDetections + i];
      
      for (let c = 0; c < numClasses; c++) {
        const score = output[(4 + c) * numDetections + i];
        if (score > maxConf) {
          maxConf = score;
          maxClassId = c;
        }
      }
    }

    if (maxConf < confThreshold) continue;

    // Convert from center format to corner format if needed
    const cx = (x1 + x2) / 2;
    const cy = (y1 + y2) / 2;
    const w = x2 - x1;
    const h = y2 - y1;

    // Check if it's center format (cx, cy, w, h) or corner format (x1, y1, x2, y2)
    let finalX1: number, finalY1: number, finalX2: number, finalY2: number;
    
    if (x1 < x2 && y1 < y2 && w > 0 && h > 0) {
      // Could be either format - assume corner format for YOLOv10
      finalX1 = x1;
      finalY1 = y1;
      finalX2 = x2;
      finalY2 = y2;
    } else {
      // Center format
      finalX1 = cx - w / 2;
      finalY1 = cy - h / 2;
      finalX2 = cx + w / 2;
      finalY2 = cy + h / 2;
    }

    // Remove letterbox padding and rescale
    const origX1 = (finalX1 - padX) / ratio;
    const origY1 = (finalY1 - padY) / ratio;
    const origX2 = (finalX2 - padX) / ratio;
    const origY2 = (finalY2 - padY) / ratio;

    const label = CLASS_LABELS[maxClassId] || `class_${maxClassId}`;

    detections.push({
      x1: origX1,
      y1: origY1,
      x2: origX2,
      y2: origY2,
      confidence: maxConf,
      classId: maxClassId,
      label,
    });
  }

  return nonMaxSuppression(detections, iouThreshold);
}

/**
 * Non-Maximum Suppression
 */
function nonMaxSuppression(detections: Detection[], iouThreshold: number): Detection[] {
  // Sort by confidence descending
  detections.sort((a, b) => b.confidence - a.confidence);
  
  const result: Detection[] = [];
  const suppressed = new Set<number>();

  for (let i = 0; i < detections.length; i++) {
    if (suppressed.has(i)) continue;
    result.push(detections[i]);

    for (let j = i + 1; j < detections.length; j++) {
      if (suppressed.has(j)) continue;
      
      // Only suppress same class
      if (detections[i].classId !== detections[j].classId) continue;

      const iou = calculateIoU(detections[i], detections[j]);
      if (iou > iouThreshold) {
        suppressed.add(j);
      }
    }
  }

  return result;
}

function calculateIoU(a: Detection, b: Detection): number {
  const x1 = Math.max(a.x1, b.x1);
  const y1 = Math.max(a.y1, b.y1);
  const x2 = Math.min(a.x2, b.x2);
  const y2 = Math.min(a.y2, b.y2);

  const intersection = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  const areaA = (a.x2 - a.x1) * (a.y2 - a.y1);
  const areaB = (b.x2 - b.x1) * (b.y2 - b.y1);
  const union = areaA + areaB - intersection;

  return union > 0 ? intersection / union : 0;
}
