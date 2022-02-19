import * as tf from '@tensorflow/tfjs-core';
import * as tfconv from '@tensorflow/tfjs-converter';
import { FaceEmotionModel } from './emotion';

const FACEEMOTION_MODEL_URL =
  'https://storage.googleapis.com/mal-public/emotion/web_model';

interface FaceEmotionConfig {
  modelUrl?: string | tf.io.IOHandler;
}

export async function loadFaceEmotion({
  modelUrl
}: FaceEmotionConfig = {}): Promise<FaceEmotionModel> {
  let emotion;
  if (modelUrl != null) {
    emotion = await tfconv.loadGraphModel(modelUrl);
  } else {
    emotion = await tfconv.loadGraphModel(FACEEMOTION_MODEL_URL, {
      fromTFHub: true,
    });
  }

  const model = new FaceEmotionModel(
    emotion
  );
  return model;
}

export {FaceEmotionModel} from './emotion';