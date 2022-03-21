/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

import * as facetfjs from '@tensorflow-models/face-tfjs';
import * as tf from '@tensorflow/tfjs-core';
import * as tfjsWasm from '@tensorflow/tfjs-backend-wasm';
import '@tensorflow/tfjs-backend-webgl';
import '@tensorflow/tfjs-backend-cpu';

tfjsWasm.setWasmPaths(
  `https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-wasm@${tfjsWasm.version_wasm}/dist/`);

// const stats = new Stats();
// stats.showPanel(0);
// document.body.prepend(stats.domElement);

let modelFace;
let modelEmotion;
let ctx;
let videoWidth;
let videoHeight;
let video;
let canvas;
let valueNeutral;
let valueHappy;
let valueSad;
let valueAngry;
let valueSuprised;
let valuePositve;
let valueActive;
let valueVibe;

const state = {
  backend: 'wasm',
};

// const gui = new dat.GUI();
// gui.add(state, 'backend', ['wasm', 'webgl', 'cpu'])
//     .onChange(async (backend) => {
//       await tf.setBackend(backend);
//     });

async function setupCamera() {
  video = document.getElementById('video');

  const stream = await navigator.mediaDevices.getUserMedia({
    'audio': false,
    'video': {facingMode: 'user'},
  });
  video.srcObject = stream;

  return new Promise((resolve) => {
    video.onloadedmetadata = () => {
      resolve(video);
    };
  });
}

const renderPrediction = async () => {
  // stats.begin();

  const returnTensors = false;
  // const flipHorizontal = true;
  const annotateBoxes = true;
  // const predictions = await modelFace.estimateFaces(
  //   video, returnTensors, flipHorizontal, annotateBoxes);
  const predictions = await pipeline.estimateEmotion(video);

  if (predictions.length > 0) {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    // console.log(predictions);

    for (let i = 0; i < predictions.length; i++) {
      let face = predictions[0].face;
      let emotions = predictions[0].emotions;
      if (returnTensors) {
        face.topLeft = face.topLeft.arraySync();
        face.bottomRight = face.bottomRight.arraySync();
        if (annotateBoxes) {
          face.landmarks = face.landmarks.arraySync();
        }
      }

      const start = face.topLeft;
      const end = face.bottomRight;
      const size = [end[0] - start[0], end[1] - start[1]];

      // const croppedInput = cutBoxFromImageAndResize(
      //   box, rotatedImage, [this.meshWidth, this.meshHeight]);

      ctx.strokeStyle = 'rgba(0, 255, 0, 1.0)';
      ctx.strokeRect(start[0], start[1], size[0], size[1]);

      valueNeutral.style.width = `${emotions[0].toFixed(2) * 100}%`;
      valueHappy.style.width = `${emotions[1].toFixed(2) * 100}%`;
      valueSad.style.width = `${emotions[2].toFixed(2) * 100}%`;
      valueAngry.style.width = `${emotions[3].toFixed(2) * 100}%`;
      valueSurprised.style.width = `${emotions[4].toFixed(2) * 100}%`;
      
      // console.log('Positive: ', pipeline.estimatePositive(emotions));
      // console.log('Active: ', pipeline.estimateActive(emotions));
      // console.log('Vibe: ', pipeline.estimateVibe(emotions));
      valuePositive.style.width = `${pipeline.estimatePositive(emotions).toFixed(2)*100}%`;
      valueActive.style.width = `${pipeline.estimateActive(emotions).toFixed(2)*100}%`;
      valueVibe.style.width = `${pipeline.estimateVibe(emotions).toFixed(2)*100}%`;


      // if (annotateBoxes) {
        // const landmarks = face.landmarks;

        // ctx.fillStyle = 'rgba(0, 255, 0, 1.0)';
        // for (let j = 0; j < landmarks.length; j++) {
        //   const x = landmarks[j][0];
        //   const y = landmarks[j][1];
        //   ctx.fillRect(x, y, 5, 5);
        // }

        // ctx.font = '16px Georgia';
        // ctx.fillText(`Neutral: ${emotions[0].toFixed(2)}`, start[0], landmarks[3][1]  - 45);
        // ctx.fillText(`Happy: ${emotions[1].toFixed(2)}`, start[0], landmarks[3][1]    - 25);
        // ctx.fillText(`Sad: ${emotions[2].toFixed(2)}`, start[0], landmarks[3][1]      - 5);
        // ctx.fillText(`Angry: ${emotions[3].toFixed(2)}`, start[0], landmarks[3][1]    + 15);
        // ctx.fillText(`Surprised: ${emotions[4].toFixed(2)}`, start[0], landmarks[3][1]+ 35);
      // }
    }
  }

  // stats.end();

  requestAnimationFrame(renderPrediction);
};

const setupPage = async () => {
  await tf.setBackend(state.backend);
  await setupCamera();
  video.play();

  videoWidth = video.videoWidth;
  videoHeight = video.videoHeight;
  video.width = videoWidth;
  video.height = videoHeight;

  canvas = document.getElementById('output');
  canvas.width = videoWidth;
  canvas.height = videoHeight;
  ctx = canvas.getContext('2d');
  ctx.fillStyle = 'rgba(255, 0, 0, 0.5)';

  modelFace = await facetfjs.loadBlazeFace();
  modelEmotion = await facetfjs.loadFaceEmotions();
  pipeline = new facetfjs.EmotionPipeline(modelFace, modelEmotion);

  valueNeutral = document.getElementById('value_neutral');
  valueHappy = document.getElementById('value_happy');
  valueSad = document.getElementById('value_sad');
  valueAngry = document.getElementById('value_angry');
  valueSurprised = document.getElementById('value_suprised');

  valuePositive = document.getElementById('value_positive');
  valueActive = document.getElementById('value_active');
  valueVibe = document.getElementById('value_vibe');

  renderPrediction();
};

setupPage();
