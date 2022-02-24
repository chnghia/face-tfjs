// import * as tfconv from '@tensorflow/tfjs-converter';
import * as tf from '@tensorflow/tfjs-core';
import { BlazeFaceModel, NormalizedFace } from './blazeface/index';
import { FaceEmotionModel } from './faceemotion/index';
// import * as tfjsWasm from '@tensorflow/tfjs-backend-wasm';
// import '@tensorflow/tfjs-backend-webgl';
// import '@tensorflow/tfjs-backend-cpu';

// tfjsWasm.setWasmPaths(
//   `https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-wasm@${tfjsWasm.version_wasm}/dist/`);

export interface Prediction {
}

export class EmotionPipeline {
  private readonly maxFacesNumber: number;

  constructor(private readonly faceDetector: BlazeFaceModel, private readonly emotionDetector: FaceEmotionModel) {
    this.maxFacesNumber = 1;
  }

  async estimateEmotion(image: tf.Tensor3D|ImageData|HTMLVideoElement|HTMLImageElement|
      HTMLCanvasElement): Promise<Prediction> {
    const returnTensors = false;
    const flipHorizontal = true;
    const annotateBoxes = true;

    const predictions = await this.faceDetector.estimateFaces(image, returnTensors, flipHorizontal, annotateBoxes);

    const result: Prediction = {
    };

    if (predictions.length > 0) {
      for (let i = 0; i < predictions.length; i++) {
        if (returnTensors) {
          predictions[i].topLeft = predictions[i].topLeft.arraySync();
          predictions[i].bottomRight = predictions[i].bottomRight.arraySync();
          if (annotateBoxes) {
            predictions[i].landmarks = predictions[i].landmarks.arraySync();
          }
        }

        const start = predictions[i].topLeft;
        const end = predictions[i].bottomRight;
        const size = [end[0] - start[0], end[1] - start[1]];
        // console.log(size);
      }
    }

    return predictions;
  }
}