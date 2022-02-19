import * as tfconv from '@tensorflow/tfjs-converter';
import * as tf from '@tensorflow/tfjs-core';

export class FaceEmotionModel {
  private faceEmotionModel: tfconv.GraphModel;
  private width: number;
  private height: number;

  constructor(
    model: tfconv.GraphModel
  ) {
    this.faceEmotionModel = model;
    this.width = 60;
    this.height = 60;
  }

  async estimateEmotions(
    input: tf.Tensor3D | ImageData | HTMLVideoElement | HTMLImageElement |
      HTMLCanvasElement,
    returnTensors = false, flipHorizontal = false,
    annotateBoxes = true): Promise<any> {
    
    const inputImage = tf.tidy(() => {
      if (!(input instanceof tf.Tensor)) {
        input = tf.browser.fromPixels(input);
      }
      return tf.expandDims(tf.cast((input as tf.Tensor), 'float32'), 0);
    });
    console.log(`inputImage: ${inputImage}`);
    const resizedImage = tf.image.resizeBilinear(inputImage as tf.Tensor4D,
        [this.width, this.height]);
    const normalizedImage = tf.mul(tf.sub(tf.div(resizedImage, 255), 0.5), 2);
    console.log(`normalizedImage: ${normalizedImage}`);
    const batchedPrediction = this.faceEmotionModel.predict(normalizedImage);
    
    const prediction = tf.squeeze((batchedPrediction as tf.Tensor3D));
    console.log(prediction);
    
    if (returnTensors) {
      return {};
    }
    return {};
  }

  dispose(): void {
    if (this.faceEmotionModel != null) {
      this.faceEmotionModel.dispose();
    }
  }
}