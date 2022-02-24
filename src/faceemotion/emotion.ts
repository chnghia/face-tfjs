import * as tfconv from '@tensorflow/tfjs-converter';
import * as tf from '@tensorflow/tfjs-core';

const IMAGE_SIZE = 60;

export const IMAGENET_CLASSES: { [classId: number]: string } = {
  0: "neutral", 1: "happy", 2: "sad", 3: "angry", 4: "surprised",
};

export class FaceEmotionModel {
  private model: tfconv.GraphModel;
  // private width: number;
  // private height: number;
  private normalizationConstant: number;
  private inputMin: number;
  private modelUrl?: string | tf.io.IOHandler;

  constructor(
    model: tfconv.GraphModel
  ) {
    this.model = model;
    // this.width = 60;
    // this.height = 60;
    this.normalizationConstant = 0.449;
    this.inputMin = -0.226;
  }

  async load() {
    this.model = await tfconv.loadGraphModel(this.modelUrl);
    

    // Warmup the model.
    const result = tf.tidy(
      () => this.model.predict(tf.zeros(
        [1, IMAGE_SIZE, IMAGE_SIZE, 3]))) as tf.Tensor;
    await result.data();
    result.dispose();
  }

  infer(
    img: tf.Tensor3D | ImageData | HTMLVideoElement | HTMLImageElement | HTMLCanvasElement,
    embedding = false): tf.Tensor {
    return tf.tidy(() => {
      if (!(img instanceof tf.Tensor)) {
        img = tf.browser.fromPixels(img);
      }

      // Normalize the image from [0, 255] to [inputMin, inputMax].
      const normalized: tf.Tensor3D = tf.add(
        tf.mul(tf.cast(img, 'float32'), this.normalizationConstant),
        this.inputMin);
      
      // Resize the image to
      let resized = normalized;
      if (img.shape[0] !== IMAGE_SIZE || img.shape[1] !== IMAGE_SIZE) {
        const alignCorners = true;
        resized = tf.image.resizeBilinear(
          normalized, [IMAGE_SIZE, IMAGE_SIZE], alignCorners);
      }

      // Reshape so we can pass it to predict.
      const batched = tf.reshape(resized, [-1, IMAGE_SIZE, IMAGE_SIZE, 3]);

      let result: tf.Tensor2D;

      if (embedding) {
        // const embeddingName = EMBEDDING_NODES[this.version];
        // const internal =
        //     this.model.execute(batched, embeddingName) as tf.Tensor4D;
        // result = tf.squeeze(internal, [1, 2]);
      } else {
        const logits1001 = this.model.predict(batched) as tf.Tensor2D;
        // Remove the very first logit (background noise).
        // console.log(logits1001.dataSync());
        result = logits1001;
      }

      return result;
    });
  }

  async classify(img: tf.Tensor3D | ImageData | HTMLImageElement | HTMLCanvasElement |
    HTMLVideoElement,
    topk = 3): Promise<Array<{ className: string, probability: number }>> {
    const logits = this.infer(img) as tf.Tensor2D;

    const classes = await getTopKClasses(logits, topk);

    logits.dispose();

    return classes;
  }
}

async function getTopKClasses(logits: tf.Tensor2D, topK: number):
    Promise<Array<{className: string, probability: number}>> {
  const softmax = tf.softmax(logits);
  const values = await softmax.data();
  softmax.dispose();

  const valuesAndIndices = [];
  for (let i = 0; i < values.length; i++) {
    valuesAndIndices.push({value: values[i], index: i});
  }
  valuesAndIndices.sort((a, b) => {
    return b.value - a.value;
  });
  const topkValues = new Float32Array(topK);
  const topkIndices = new Int32Array(topK);
  for (let i = 0; i < topK; i++) {
    topkValues[i] = valuesAndIndices[i].value;
    topkIndices[i] = valuesAndIndices[i].index;
  }

  const topClassesAndProbs = [];
  for (let i = 0; i < topkIndices.length; i++) {
    topClassesAndProbs.push({
      className: IMAGENET_CLASSES[topkIndices[i]],
      probability: topkValues[i]
    });
  }
  return topClassesAndProbs;
}
