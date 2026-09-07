declare module 'onnxruntime-web' {
  export const env: {
    wasm: {
      wasmPaths: string;
    };
    logLevel: string;
  };

  export class Tensor {
    constructor(type: string, data: Float32Array, dims: number[]);
    data: Float32Array;
    dims: readonly number[];
  }

  export class InferenceSession {
    static create(
      model: ArrayBuffer | Uint8Array | string,
      options?: {
        executionProviders?: string[];
        graphOptimizationLevel?: string;
      }
    ): Promise<InferenceSession>;

    inputNames: string[];
    outputNames: string[];
    inputMetadata: Record<string, { dimensions: number[]; type: string }>;
    outputMetadata: Record<string, { dimensions: number[]; type: string }>;

    run(feeds: Record<string, Tensor>): Promise<Record<string, Tensor>>;
    release(): Promise<void>;
  }
}
