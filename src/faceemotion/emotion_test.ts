import * as tf from '@tensorflow/tfjs-core';
// tslint:disable-next-line: no-imports-from-dist
import {describeWithFlags, NODE_ENVS} from '@tensorflow/tfjs-core/dist/jasmine_util';

import {FaceEmotionModel} from './emotion';
import * as faceemotion from './index';
import {stubbedImageVals} from './test_util';

describeWithFlags('FaceEmotion', NODE_ENVS, () => {
  let model: FaceEmotionModel;
  
  beforeAll(async () => {
    // originalTimeout = jasmine.DEFAULT_TIMEOUT_INTERVAL;
    // jasmine.DEFAULT_TIMEOUT_INTERVAL = 10000000;
    model = await faceemotion.loadFaceEmotions();
  });

  // beforeEach(function () {
  //     originalTimeout = jasmine.DEFAULT_TIMEOUT_INTERVAL;
  //     jasmine.DEFAULT_TIMEOUT_INTERVAL = 10000000;
  // });

  // afterEach(function() {
  //   jasmine.DEFAULT_TIMEOUT_INTERVAL = originalTimeout;
  // });

  // let originalTimeout: number;

  it('infer input and output shape', async () => {
    
    const input: tf.Tensor3D = tf.zeros([60, 60, 3]);
    // const beforeTensors = tf.memory().numTensors;
    // console.log(model);
    const predicted = await model.infer(input);
    // console.log(predicted.shape);
    expect(predicted.shape).toEqual([1, 5]);
  });

  it('infer returns objects with expected properties', async () => {
    // Stubbed image contains a single face.
    const input: tf.Tensor3D = tf.tensor3d(stubbedImageVals.slice(0, 10800), [60, 60, 3]);
    const result = await model.infer(input);

    const face = result;
    expect(face).toBeDefined();
  });

  it('model infer doen\'t leak', async () => {
    const x: tf.Tensor3D = tf.zeros([60, 60, 3]);
    const numTensorsBefore = tf.memory().numTensors;
    await model.infer(x);

    expect(tf.memory().numTensors).toBe(numTensorsBefore + 1);
  });
});
