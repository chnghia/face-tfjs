// import * as tfconv from '@tensorflow/tfjs-converter';
import * as tf from '@tensorflow/tfjs-core';
// import { Tensor1D } from '@tensorflow/tfjs-core';
// import { Tensor3D } from '@tensorflow/tfjs-core';
import { BlazeFaceModel } from './blazeface/index';
import { FaceEmotionModel } from './faceemotion/index';
// import * as tfjsWasm from '@tensorflow/tfjs-backend-wasm';
// import '@tensorflow/tfjs-backend-webgl';
// import '@tensorflow/tfjs-backend-cpu';

// tfjsWasm.setWasmPaths(
//   `https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-wasm@${tfjsWasm.version_wasm}/dist/`);

export interface Prediction {
  // face?: NormalizedFace,
  // emotions?: number[],
}

export interface PositiveActive {

}

const NOR_POINTS = [
  [-0.2826086956521739, -0.32608695652173914], // neutral
  [-0.8768115942028986, 0.19202898550724637], // happy
  [0.8659420289855072, -0.36231884057971014], // sad
  [0.8369565217391305, 0.5942028985507246], // angry
  [-0.057971014492753624, 0.9710144927536232] // surprised
];

const OLD_MAX = 1;
const OLD_MIN = -1;
const NEW_MAX = 1;
const NEW_MIN = 0;

function getInputTensorDimensions(input: tf.Tensor3D|ImageData|HTMLVideoElement|
                                  HTMLImageElement|
                                  HTMLCanvasElement): [number, number] {
  return input instanceof tf.Tensor ? [input.shape[0], input.shape[1]] :
                                      [input.height, input.width];
}

function normRange(v: number): number {
  const oldRange = (OLD_MAX - OLD_MIN);
  const newRange = (NEW_MAX - NEW_MIN);
  return (((v - OLD_MIN) * newRange) / oldRange) + NEW_MIN;
}

function norm(v: number[]): number {
  return Math.sqrt(v[0] ** 2 + v[1] ** 2);
}

function dotProduct(v1: number[], v2: number[]): number {
  return v1[0]*v2[0] + v1[1]*v2[1];
}

export class EmotionPipeline {
  // private readonly maxFacesNumber: number;
  private normalizationConstant: number;
  private inputMin: number;

  constructor(
    private readonly faceDetector: BlazeFaceModel,
    private readonly emotionDetector: FaceEmotionModel
  ) {
    // this.maxFacesNumber = 1;
    this.normalizationConstant = 0.449;
    this.inputMin = 0.226;
  }

  async estimateEmotion(
    input: tf.Tensor3D | ImageData | HTMLVideoElement | HTMLImageElement | HTMLCanvasElement
  ): Promise<Prediction> {

    const [, width] = getInputTensorDimensions(input);
    const image = tf.tidy(() => {
      if (!(input instanceof tf.Tensor)) {
        input = tf.browser.fromPixels(input);
      }
      return tf.expandDims(tf.cast((input as tf.Tensor), 'float32'), 0);
    });
    
    const returnTensors = false;
    const flipHorizontal = true;
    const annotateBoxes = true;

    const predictions = await this.faceDetector.infer(image as tf.Tensor4D, width, returnTensors, flipHorizontal, annotateBoxes);

    let results = [];

    if (predictions.length > 0) {
      for (let i = 0; i < predictions.length; i++) {
        const start = predictions[i].topLeft as [number, number];
        const end = predictions[i].bottomRight as [number, number];
        const h = image.shape[1];
        const w = image.shape[2];

        const boxes = [[
          start[1] / h, start[0] / w, end[1] / h, end[0] / w
        ]];

        let faceImage = tf.image.cropAndResize(image as tf.Tensor4D, boxes, [0], [60, 60]);

        // Normalize the image from [0, 255] to [inputMin, inputMax].
        const normalized = tf.div(
          tf.sub(
            tf.div(
              tf.cast(faceImage, 'float32'),
              255),
            this.normalizationConstant),
          this.inputMin);

        const logits = await this.emotionDetector.predict(normalized as tf.Tensor4D);
        // tf.image.
        const softmax = tf.softmax(logits);
        const values = await softmax.dataSync();
        softmax.dispose();

        const result: Prediction = {
          face: predictions[i],
          emotions: values,
        };
        results.push(result);
        // return result;
      }
    }

    image.dispose();

    return results;
  }

  findEmotionIndex(emotions: number[]): number {
    const max_value = Math.max(...emotions);
    const max_index = emotions.indexOf(max_value);
    return max_index
  }

  estimatePoint(emotions: number[]): number[] {
    const max_index = this.findEmotionIndex(emotions);
    const nor_point = NOR_POINTS[max_index];
    return [nor_point[0] * emotions[max_index], nor_point[1] * emotions[max_index]];
  }

  estimatePositive(emotions: number[]): number {
    return normRange(this.estimatePoint(emotions)[0]);
  }

  estimateActive(emotions: number[]): number {
    return normRange(this.estimatePoint(emotions)[1]);
  }

  estimateVibe(emotions: number[]): number {
    const point = this.estimatePoint(emotions);
    const axis = [-1, 1];
    const res = dotProduct(point, axis) / (norm(axis)*0.7071);
    if (res < -1) {
      return normRange(-1.0);
    } else {
      return normRange(res);
    }
  }
}