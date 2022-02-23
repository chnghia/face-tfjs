import * as tf from '@tensorflow/tfjs-core';
import * as tfconv from '@tensorflow/tfjs-converter';
import { FaceEmotionModel } from './emotion';

const FACEEMOTION_MODEL_URL =
  'https://storage.googleapis.com/mal-public/emotion/web_model2';

interface FaceEmotionConfig {
  modelUrl?: string | tf.io.IOHandler;
}

export async function load({
  modelUrl
}: FaceEmotionConfig = {}): Promise<FaceEmotionModel> {
  let emotionModel : tfconv.GraphModel;
  if (modelUrl != null) {
    emotionModel = await tfconv.loadGraphModel(modelUrl);
  } else {
    emotionModel = await tfconv.loadGraphModel(FACEEMOTION_MODEL_URL, {
      fromTFHub: true,
    });
  }

  // Warmup the model.
  const result = tf.tidy(
                      () => emotionModel.predict(tf.zeros(
                          [1, 60, 60, 3]))) as tf.Tensor;
  await result.data();
  result.dispose();

  const model = new FaceEmotionModel(
    emotionModel
  );
  return model;
}

export {FaceEmotionModel} from './emotion';