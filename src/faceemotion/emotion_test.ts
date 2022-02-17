import * as tf from '@tensorflow/tfjs-core';
// tslint:disable-next-line: no-imports-from-dist
import {describeWithFlags, NODE_ENVS} from '@tensorflow/tfjs-core/dist/jasmine_util';

import {FaceEmotionModel} from './emotion';
import * as faceemotion from './index';
import {stubbedImageVals} from './test_util';

describeWithFlags('FaceEmotion', NODE_ENVS, () => {
  let model: FaceEmotionModel;
  beforeAll(async () => {
    model = await faceemotion.loadFaceEmotion({ modelUrl: 'file:///../../weights/web_model/model.json'});
  });

  it('estimateFaces does not leak memory', async () => {
    const input: tf.Tensor3D = tf.zeros([128, 128, 3]);
    const beforeTensors = tf.memory().numTensors;
    await model.estimateFaces(input);

    expect(tf.memory().numTensors).toEqual(beforeTensors);
  });

  it('estimateFaces returns objects with expected properties', async () => {
    // Stubbed image contains a single face.
    const input: tf.Tensor3D = tf.tensor3d(stubbedImageVals, [128, 128, 3]);
    const result = await model.estimateFaces(input);

    const face = result[0];

    expect(face.topLeft).toBeDefined();
    expect(face.bottomRight).toBeDefined();
    expect(face.landmarks).toBeDefined();
    expect(face.probability).toBeDefined();
  });
});
