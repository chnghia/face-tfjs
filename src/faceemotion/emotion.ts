import * as tfconv from '@tensorflow/tfjs-converter';
import * as tf from '@tensorflow/tfjs-core';

export class FaceEmotionModel {
  private faceEmotionModel: tfconv.GraphModel;

  constructor(
    model: tfconv.GraphModel
  ) {
    this.faceEmotionModel = model;
  }

  async estimateFaces(
    input: tf.Tensor3D | ImageData | HTMLVideoElement | HTMLImageElement |
      HTMLCanvasElement,
    returnTensors = false, flipHorizontal = false,
    annotateBoxes = true): Promise<any> {
  }

  dispose(): void {
    if (this.faceEmotionModel != null) {
      this.faceEmotionModel.dispose();
    }
  }
}