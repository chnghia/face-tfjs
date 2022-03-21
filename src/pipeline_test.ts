import * as tf from '@tensorflow/tfjs-core';
// tslint:disable-next-line: no-imports-from-dist
import { describeWithFlags, NODE_ENVS } from '@tensorflow/tfjs-core/dist/jasmine_util';
import * as facetfjs from './index';
import { FaceEmotionModel, BlazeFaceModel, EmotionPipeline } from './index';
import { stubbedImageVals } from './blazeface/test_util';
import { Prediction } from './pipeline';

describeWithFlags('Pipeline', NODE_ENVS, () => {
  let modelEmotion: FaceEmotionModel;
  let modelFace: BlazeFaceModel;
  let pipe: EmotionPipeline;
  // let originalTimeout: number;
  
  // beforeEach(function () {
  //     originalTimeout = jasmine.DEFAULT_TIMEOUT_INTERVAL;
  //     jasmine.DEFAULT_TIMEOUT_INTERVAL = 10000000;
  // });

  // afterEach(function() {
  //   jasmine.DEFAULT_TIMEOUT_INTERVAL = originalTimeout;
  // });

  beforeAll(async () => {
    // originalTimeout = jasmine.DEFAULT_TIMEOUT_INTERVAL;
    jasmine.DEFAULT_TIMEOUT_INTERVAL = 10000000;
    modelEmotion = await facetfjs.loadFaceEmotions();
    modelFace = await facetfjs.loadBlazeFace();
    pipe = new EmotionPipeline(modelFace, modelEmotion);
  });

  it('infer input and output shape', async () => {
    

    const input: tf.Tensor3D = tf.zeros([128, 128, 3]);
    // const beforeTensors = tf.memory().numTensors;
    const predicted = await pipe.estimateEmotion(input);
    // console.log(predicted.shape);
    expect(predicted).toBeDefined();
  });

  it('estimateFaces returns objects with expected properties', async () => {
    // Stubbed image contains a single face.
    const input: tf.Tensor3D = tf.tensor3d(stubbedImageVals, [128, 128, 3]);
    const result = await pipe.estimateEmotion(input);

    const faces = result as Array<Prediction>;

    // console.log(faces);
    expect(faces).toBeDefined();
    // expect(faces[0].emotions).toBeDefined();
  });

  it('estimatePoint returns objects with expected properties', async () => {
    // Stubbed image contains a single face.
    // const input: tf.Tensor3D = tf.tensor3d(stubbedImageVals, [128, 128, 3]);
    // const result = await pipe.estimateEmotion(input);
    let emotions = [0.85045341, 0.14814392, 0.00043213543, 0.00056787342, 0.00040270519];

    const point = pipe.estimatePoint(emotions);
    const pos = pipe.estimatePositive(emotions);
    const act = pipe.estimateActive(emotions);
    console.log(pos)

    expect(point).toBeDefined();
    expect(pos).toBeDefined();
    expect(act).toBeDefined();
    // expect(faces.emotions).toBeDefined();
  });

  it('estimateVibe returns objects with expected properties', async () => {
    let emotions = [0.85045341, 0.14814392, 0.00043213543, 0.00056787342, 0.00040270519];
    
    const vibe = pipe.estimateVibe(emotions);
    console.log(vibe);

    expect(vibe).toBeDefined();
  });
});
