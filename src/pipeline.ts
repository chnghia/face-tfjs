// import * as tfconv from '@tensorflow/tfjs-converter';
import * as tf from '@tensorflow/tfjs-core';
import { Tensor3D } from '@tensorflow/tfjs-core';
import { BlazeFaceModel, NormalizedFace } from './blazeface/index';
import { FaceEmotionModel } from './faceemotion/index';
// import * as tfjsWasm from '@tensorflow/tfjs-backend-wasm';
// import '@tensorflow/tfjs-backend-webgl';
// import '@tensorflow/tfjs-backend-cpu';

// tfjsWasm.setWasmPaths(
//   `https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-wasm@${tfjsWasm.version_wasm}/dist/`);

export interface Prediction {
}

function getInputTensorDimensions(input: tf.Tensor3D|ImageData|HTMLVideoElement|
                                  HTMLImageElement|
                                  HTMLCanvasElement): [number, number] {
  return input instanceof tf.Tensor ? [input.shape[0], input.shape[1]] :
                                      [input.height, input.width];
}
export class EmotionPipeline {
  private readonly maxFacesNumber: number;
  private normalizationConstant: number;
  private inputMin: number;

  constructor(
    private readonly faceDetector: BlazeFaceModel,
    private readonly emotionDetector: FaceEmotionModel
  ) {
    this.maxFacesNumber = 1;
    this.normalizationConstant = 0.449;
    this.inputMin = -0.226;
  }

  async estimateEmotion(input: tf.Tensor3D | ImageData | HTMLVideoElement | HTMLImageElement | HTMLCanvasElement): Promise<Prediction> {

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

    console.log(predictions);

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
        // console.log(start);
        // console.log(end);
        // const size: number[] = [60, 60];

        const h = image.shape[1];
        const w = image.shape[2];

        const boxes = [[
          start[1] / h, start[0] / w, end[1] / h, end[0] / w
        ]];

        let faceImage = tf.image.cropAndResize(image as tf.Tensor4D, boxes, [0], [60, 60]);

        // Normalize the image from [0, 255] to [inputMin, inputMax].
        const normalized: tf.Tensor3D = tf.add(
          tf.mul(tf.cast(faceImage, 'float32'), this.normalizationConstant),
          this.inputMin);

        const emotions = await this.emotionDetector.infer(normalized);
        // tf.image.

        console.log(emotions.arraySync()[0]);
      }
    }

    image.dispose();

    return predictions;
  }
}