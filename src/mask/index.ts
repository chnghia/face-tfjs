import * as tf from '@tensorflow/tfjs-core';
import * as tfconv from '@tensorflow/tfjs-converter';
import { FaceMaskModel } from './mask';

const FACEMASK_MODEL_URL = 'https://storage.googleapis.com/mal-public/mask/web_model9/model.json'

interface FaceMaskConfig {
    modelUrl?: string | tf.io.IOHandler;
}

export async function loadMask({
    modelUrl
  }: FaceMaskConfig = {}): Promise<FaceMaskModel> {
    let maskModel : tfconv.GraphModel;
    if (modelUrl != null) {
      maskModel = await tfconv.loadGraphModel(modelUrl);
    } else {
      maskModel = await tfconv.loadGraphModel(FACEMASK_MODEL_URL);
    }
    //console.log(maskModel);
  
    // Warmup the model.
    const result = tf.tidy(() => maskModel.predict(tf.zeros([1, 224, 224, 3]))) as tf.Tensor;
    await result.data();
    //console.log(result);
    result.dispose();
  
    const model = new FaceMaskModel(
      maskModel
    );
    return model;
  }
export {FaceMaskModel} from './mask';
