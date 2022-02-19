import * as tf from '@tensorflow/tfjs-core';
// tslint:disable-next-line: no-imports-from-dist
import {describeWithFlags, NODE_ENVS} from '@tensorflow/tfjs-core/dist/jasmine_util';

import {FaceEmotionModel} from './emotion';
import * as faceemotion from './index';
import {stubbedImageVals} from './test_util';

describeWithFlags('FaceEmotion', NODE_ENVS, () => {
  let model: FaceEmotionModel;
  beforeAll(async () => {
    model = await faceemotion.loadFaceEmotion();
  });

  it('estimateEmotions does not leak memory', async () => {
    const input: tf.Tensor3D = tf.zeros([60, 60, 3]);
    const beforeTensors = tf.memory().numTensors;
    await model.estimateEmotions(input);

    expect(tf.memory().numTensors).toEqual(beforeTensors);
  });

  it('estimateEmotions returns objects with expected properties', async () => {
    // Stubbed image contains a single face.
    const input: tf.Tensor3D = tf.tensor3d(stubbedImageVals.slice(0, 10800), [60, 60, 3]);
    const result = await model.estimateEmotions(input);

    const face = result[0];
    // expect(face.topLeft).toBeDefined();
    // expect(face.bottomRight).toBeDefined();
    // expect(face.landmarks).toBeDefined();
    expect(face.probability).toBeDefined();
  });
});
