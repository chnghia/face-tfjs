import * as tf from '@tensorflow/tfjs-core';
// tslint:disable-next-line: no-imports-from-dist
import {describeWithFlags, NODE_ENVS} from '@tensorflow/tfjs-core/dist/jasmine_util';

import {FaceEmotionModel} from './emotion';
import * as faceemotion from './index';
import {stubbedImageVals} from './test_util';

describeWithFlags('FaceEmotion', NODE_ENVS, () => {
  let model: FaceEmotionModel;
  beforeAll(async () => {
    model = await faceemotion.load();
  });

  it('infer input and output shape', async () => {
    const input: tf.Tensor3D = tf.zeros([60, 60, 3]);
    // const beforeTensors = tf.memory().numTensors;
    const predicted = await model.infer(input);
    // console.log(predicted.shape);
    expect(predicted.shape).toEqual([1, 5]);
  });

  it('infer returns objects with expected properties', async () => {
    // Stubbed image contains a single face.
    const input: tf.Tensor3D = tf.tensor3d(stubbedImageVals.slice(0, 10800), [60, 60, 3]);
    const result = await model.infer(input);

    const face = result;
    // console.log(face[0]);
    // expect(face.topLeft).toBeDefined();
    // expect(face.bottomRight).toBeDefined();
    // expect(face.landmarks).toBeDefined();
    expect(face).toBeDefined();
  });

  it('model infer doen\'t leak', async () => {
    const x: tf.Tensor3D = tf.zeros([60, 60, 3]);
    const numTensorsBefore = tf.memory().numTensors;
    model.infer(x);

    expect(tf.memory().numTensors).toBe(numTensorsBefore + 1);
  });
});
