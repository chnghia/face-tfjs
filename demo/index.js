/* eslint-disable max-len */
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
// import { math } from '@tensorflow/tfjs-core';

tfjsWasm.setWasmPaths(
  `https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-wasm@${tfjsWasm.version_wasm}/dist/`);

const stats = new Stats();
stats.showPanel(0);
document.getElementById('stats').appendChild(stats.domElement);
// document.body.prepend(stats.domElement);

let modelFace;
let modelEmotion;
let ctxOutput;
// let webcamWidth;
// let webcamHeight;
let webcam;
let canvas;
let valueNeutral;
let valueHappy;
let valueSad;
let valueAngry;
let valueSuprised;
let valuePositive;
let valueActive;
let valueVibe;
// let imgWidth;
// let imgHeight;
let viewerType;
let img;
let videoClip;
// let imgVideo;
let canvasV;

let dataEmotionsNeutral = [];
let dataEmotionsAngry = [];
let dataEmotionsHappy = [];
let dataEmotionsSurprised = [];
let dataEmotionsSad = [];
let dataEmotionsPositive = [];
let dataEmotionsActive = [];
let dataEmotionsVibe = [];

let numberCells;
let numberCell;
let myChart;
// eslint-disable-next-line one-var
let beginTime = Date.now(), prevTime = beginTime;
const maxFrameChart = 50;
const timeFrameCapture = 400;
const tensionValue = 0.4;

const colorAngry = 'rgba(255, 77, 79, 1)';
const colorNeutral = 'rgba(186, 231, 255, 1)';

const datasetAngry = {
  label: 'ANGRY',
  data: dataEmotionsAngry,
  fill: false,
  tension: tensionValue,
  backgroundColor: [colorAngry],
  borderColor: [colorAngry],
  borderWidth: 2,
};
const datasetNeutral = {
  label: 'NEUTRAL',
  data: dataEmotionsNeutral,
  fill: false,
  tension: tensionValue,
  backgroundColor: [colorNeutral],
  borderColor: [colorNeutral],
  borderWidth: 2,
};
const datasetHappy = {
  label: 'HAPPY',
  data: dataEmotionsHappy,
  fill: false,
  tension: tensionValue,
  backgroundColor: [
    'rgba(24, 144, 255, 1)',
  ],
  borderColor: [
    'rgba(24, 144, 255, 1)',
  ],
  borderWidth: 2,
};
const datasetSad = {
  label: 'SAD',
  data: dataEmotionsSad,
  fill: false,
  tension: tensionValue,
  backgroundColor: [
    'rgba(89, 126, 247, 1)',
  ],
  borderColor: [
    'rgba(89, 126, 247, 1)',
  ],
  borderWidth: 2,
};
const datasetSurprised = {
  label: 'SURPRISE',
  data: dataEmotionsSurprised,
  fill: false,
  tension: tensionValue,
  backgroundColor: [
    'rgba(255, 197, 61, 1)',
  ],
  borderColor: [
    'rgba(255, 197, 61, 1)',
  ],
  borderWidth: 2,
};

const datasetPositive = {
  label: 'POSITIVE',
  data: dataEmotionsPositive,
  fill: false,
  tension: tensionValue,
  backgroundColor: [
    'rgba(67, 67, 67, 1)',
  ],
  borderColor: [
    'rgba(67, 67, 61, 1)',
  ],
  borderWidth: 2,
};
const datasetActive = {
  label: 'ACTIVE',
  data: dataEmotionsActive,
  fill: false,
  tension: tensionValue,
  backgroundColor: [
    'rgba(115, 209, 61, 1)',
  ],
  borderColor: [
    'rgba(115, 209, 61, 1)',
  ],
  borderWidth: 2,
};
const datasetVibe = {
  label: 'VIBE',
  data: dataEmotionsVibe,
  fill: false,
  tension: tensionValue,
  backgroundColor: [
    'rgba(247, 89, 171, 1)',
  ],
  borderColor: [
    'rgba(247, 89, 171, 1)',
  ],
  borderWidth: 2,
};

const state = {
  backend: 'wasm',
};

// const gui = new dat.GUI();
// gui.add(state, 'backend', ['wasm', 'webgl', 'cpu'])
//     .onChange(async (backend) => {
//       await tf.setBackend(backend);
//     });

async function setupCamera() {
  webcam = document.getElementById('webcam');

  const stream = await navigator.mediaDevices.getUserMedia({
    'audio': false,
    // 'video': {facingMode: 'user'},
    'video': {width: 520, height: 390},
  });
  webcam.srcObject = stream;

  return new Promise((resolve) => {
    webcam.onloadedmetadata = () => {
      resolve(webcam);
    };
  });
}

async function setupVideoClip() {
  videoClip = document.getElementById('videoClip');
  // videoClip.play();

  // const stream = await navigator.mediaDevices.getUserMedia({
  //   'audio': false,
  //   // 'video': {facingMode: 'user'},
  //   'video': {width: 520, height: 390},
  // });
  // webcam.srcObject = stream;

  // return new Promise((resolve) => {
  //   videoClip.onloadedmetadata = () => {
  //     resolve(videoClip);
  //   };
  // });
}

function stopCamera() {
  stream = webcam.srcObject;
  stream.getTracks().forEach(function(track) {
    track.stop();
  });
}

async function setupImg() {
  img = document.getElementById('img');

  const fileTag = document.getElementById('imageInput');
  fileTag.addEventListener('change', function() {
    loadImg(this);
  });
  const preview = document.getElementById('img');
  function loadImg(input) {
    if (input.files && input.files[0]) {
      const reader = new FileReader();
      reader.onload = function(e) {
        preview.setAttribute('src', e.target.result);
      };
      reader.readAsDataURL(input.files[0]);
      document.getElementById('dragFile').style.display = 'none';
      document.getElementById('btn_close_img').style.display = 'block';
    }
  }
}
async function closeImg() {
  document.getElementById('img').src = '';
  document.getElementById('imageInput').value = '';
  document.getElementById('dragFile').style.display = 'block';
  document.getElementById('btn_close_img').style.display = 'none';
}

// async function setupVideoClip(input) {
//   videoClip = document.getElementById('videoClip');
//   source = document.getElementById('source');
//   const file = input.files[0];
//   const reader = new FileReader();
//   reader.onload = function(e) {
//     const src = e.target.result;
//     source.setAttribute('src', src);
//     videoClip.load();
//     videoClip.play();
//   };
// }

function emotionValue(emotion, toFixed=2) {
  return emotion.toFixed(toFixed);
}

function emotionValuePercent(emotion, toFixed=2) {
  return (emotion * 100).toFixed(toFixed);
}

function displayEmotionValues(emotions) {
  valueNeutral.style.width = `${emotionValuePercent(emotions[0])}%`;
  valueNeutralLabel.textContent = `${emotionValuePercent(emotions[0])}%`;
  valueHappy.style.width = `${emotionValuePercent(emotions[1])}%`;
  valueHappyLabel.textContent = `${emotionValuePercent(emotions[1])}%`;
  valueSad.style.width = `${emotionValuePercent(emotions[2])}%`;
  valueSadLabel.textContent = `${emotionValuePercent(emotions[2])}%`;
  valueAngry.style.width = `${emotionValuePercent(emotions[3])}%`;
  valueAngryLabel.textContent = `${emotionValuePercent(emotions[3])}%`;
  valueSuprised.style.width = `${emotionValuePercent(emotions[4])}%`;
  valueSuprisedLabel.textContent = `${emotionValuePercent(emotions[4])}%`;

  valuePositive.style.width = `${emotionValuePercent(pipeline.estimatePositive(emotions))}%`;
  valuePositiveLabel.textContent = `${emotionValuePercent(pipeline.estimatePositive(emotions))}%`;
  valueActive.style.width = `${emotionValuePercent(pipeline.estimateActive(emotions))}%`;
  valueActiveLabel.textContent = `${emotionValuePercent(pipeline.estimateActive(emotions))}%`;
  valueVibe.style.width = `${emotionValuePercent(pipeline.estimateVibe(emotions))}%`;
  valueVibeLabel.textContent = `${emotionValuePercent(pipeline.estimateVibe(emotions))}%`;
}

function displayBoundingBox(face, newWidth, newHeight, oldWidth, oldHeight, returnTensors = false) {
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

  ctxOutput.strokeStyle = 'rgba(0, 255, 0, 1.0)';
  const ratioWidth = newWidth / oldWidth;
  const ratioHeight= newHeight / oldHeight;

  const newX = convertRange(start[0], newWidth, oldWidth);
  const newY = convertRange(start[1], newHeight, oldHeight);
  // console.log('offsetWidth:' + newWidth);
  // console.log('offsetHeight:' + oldHeight);
  // console.log('width:' + oldWidth);
  // console.log('height:' + oldHeight);
  // console.log('topleft:' + face.topLeft);
  // console.log('bottomRight:' + face.bottomRight);
  // console.log('width:' + size[0]);
  // console.log('height:' + size[1]);
  // console.log('x: ' + start[0]);
  // console.log('y: ' + start[1]);
  // console.log('new x: ' + newX);
  // console.log('new y: ' + newY);
  ctxOutput.strokeRect(newX, newY, size[0]*ratioWidth, size[1]*ratioHeight);
}

function pushEmotionsToChart(emotions) {
  const emotionsNeutral = [emotionValue(emotions[0], 3)];
  dataEmotionsNeutral.push(emotionsNeutral);

  const emotionsHappy = [emotionValue(emotions[1], 3)];
  dataEmotionsHappy.push(emotionsHappy);

  const emotionsSad = [emotionValue(emotions[2], 3)];
  dataEmotionsSad.push(emotionsSad);

  const emotionsAngry = [emotionValue(emotions[3], 3)];
  dataEmotionsAngry.push(emotionsAngry);

  const emotionsSurprise = [emotionValue(emotions[4], 3)];
  dataEmotionsSurprised.push(emotionsSurprise);

  const emotionPositive = [emotionValue(pipeline.estimatePositive(emotions), 3)];
  dataEmotionsPositive.push(emotionPositive);

  const emotionActive = [emotionValue(pipeline.estimateActive(emotions), 3)];
  dataEmotionsActive.push(emotionActive);

  const emotionVibe = [emotionValue(pipeline.estimateVibe(emotions), 3)];
  dataEmotionsVibe.push(emotionVibe);

  if (dataEmotionsAngry.length >= maxFrameChart) {
    numberCell += 1;
    numberCells.push(numberCell);
  }

  myChart.update();
  myChart.data.labels = numberCells;
  myChart.data.datasets[0].data = dataEmotionsAngry;
  myChart.data.datasets[1].data = dataEmotionsNeutral;
  myChart.data.datasets[2].data = dataEmotionsHappy;
  myChart.data.datasets[3].data = dataEmotionsSad;
  myChart.data.datasets[4].data = dataEmotionsSurprised;
  myChart.data.datasets[5].data = dataEmotionsPositive;
  myChart.data.datasets[6].data = dataEmotionsActive;
  myChart.data.datasets[7].data = dataEmotionsVibe;

  if (dataEmotionsAngry.length >= maxFrameChart+1) {
    dataEmotionsAngry.shift();
    dataEmotionsNeutral.shift();
    dataEmotionsHappy.shift();
    dataEmotionsSad.shift();
    dataEmotionsSurprised.shift();
    dataEmotionsPositive.shift();
    dataEmotionsActive.shift();
    dataEmotionsVibe.shift();
    numberCells.shift();
    myChart.update();
  }
}

function convertRange(value, newMax, oldMax) {
  // const oldMax = oldRange;
  const oldMin = 0;
  // const newMax = newRange;
  const newMin = 0;
  const oldRange = (oldMax - oldMin);
  const newRange = (newMax - newMin);
  const newValue = (((value - oldMin) * newRange) / oldRange) + newMin;
  return newValue;
}

const renderPrediction = async () => {
  stats.begin();

  let time = Date.now();

  if ( time >= prevTime + timeFrameCapture ) {
    prevTime = time;
    let viewer;
    // eslint-disable-next-line one-var
    let oldWidth, oldHeight, newWidth, newHeight;
    const checkImgEmpty = document.getElementById('img');
    const src = checkImgEmpty.getAttribute('src');
    if(src == ''){
      if (viewerType == 'video') {
        oldWidth = videoClip.videoWidth;
        oldHeight = videoClip.videoHeight;
        newWidth = videoClip.offsetWidth;
        newHeight = videoClip.offsetHeight;
  
        canvas.width = videoClip.offsetWidth;
        canvas.height = videoClip.offsetHeight;
        // scale the canvas accordingly
        canvasV.width = videoClip.videoWidth;
        canvasV.height = videoClip.videoHeight;
        // draw the video at that frame
        canvasV.getContext('2d')
          .drawImage(videoClip, 0, 0, canvasV.width, canvasV.height);
  
        viewer = canvasV;
      } else {
        viewer = webcam;
        oldWidth = webcam.videoWidth;
        oldHeight = webcam.videoHeight;
        newWidth = webcam.offsetWidth;
        newHeight = webcam.offsetHeight;
        canvas.width = viewer.offsetWidth;
        canvas.height = viewer.offsetHeight;
      }
    }else{
      if (viewerType == 'img') {
        viewer = img;
        canvas.width = img.offsetWidth;
        canvas.height = img.offsetHeight;
        oldWidth = img.width;
        oldHeight = img.height;
        newWidth = img.offsetWidth;
        newHeight = img.offsetHeight;
      } else if (viewerType == 'video') {
        oldWidth = videoClip.videoWidth;
        oldHeight = videoClip.videoHeight;
        newWidth = videoClip.offsetWidth;
        newHeight = videoClip.offsetHeight;
  
        canvas.width = videoClip.offsetWidth;
        canvas.height = videoClip.offsetHeight;
        // scale the canvas accordingly
        canvasV.width = videoClip.videoWidth;
        canvasV.height = videoClip.videoHeight;
        // draw the video at that frame
        canvasV.getContext('2d')
          .drawImage(videoClip, 0, 0, canvasV.width, canvasV.height);
  
        viewer = canvasV;
      } else {
        viewer = webcam;
        oldWidth = webcam.videoWidth;
        oldHeight = webcam.videoHeight;
        newWidth = webcam.offsetWidth;
        newHeight = webcam.offsetHeight;
        canvas.width = viewer.offsetWidth;
        canvas.height = viewer.offsetHeight;
      }
    }
    const predictions = await pipeline.estimateEmotion(viewer);
    ctxOutput.clearRect(0, 0, canvas.width, canvas.height);
    // console.log('predictions: ' + predictions.length);

    if (predictions.length > 0) {
      // console.log(predictions);
      for (let i = 0; i < predictions.length; i++) {
        let face = predictions[0].face;
        let emotions = predictions[0].emotions;
        // console.log(emotions);

        pushEmotionsToChart(emotions);
        displayBoundingBox(face, newWidth, newHeight, oldWidth, oldHeight);
        displayEmotionValues(emotions);
      }
    }
  }
  stats.end();
  requestAnimationFrame(renderPrediction);
};

async function showImg() {
  // myChart.reset();

  document.getElementById('dropdown_button').innerHTML = 'Picture<svg class="w-4 h-4 ml-2" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7"></path></svg>';

  document.getElementById('uploadImg').style.display = 'block';
  document.getElementById('webcam').style.display = 'none';
  document.getElementById('movie').style.display = 'none';

  setupOutputValue();
  stopCamera();
  await setupImg();

  imgWidth = img.width;
  imgHeight = img.height;

  viewerType = 'img';
  // renderPredictionImg();
}

function setupOutputValue() {
  valueNeutralLabel.innerHTML = '%';
  valueHappyLabel.innerHTML = '%';
  valueSadLabel.innerHTML = '%';
  valueAngryLabel.innerHTML = '%';
  valueSuprisedLabelinnerHTML = '%';
  valuePositiveLabel.innerHTML = '%';
  valueActiveLabel.innerHTML = '%';
  valueVibeLabel.innerHTML = '%';

  valueNeutral.style.width = '0%';
  valueHappy.style.width = '0%';
  valueSad.style.width = '0%';
  valueAngry.style.width = '0%';
  valueSuprised.style.width = '0%';
  valuePositive.style.width = '0%';
  valueActive.style.width = '0%';
  valueVibe.style.width = '0%';
};

async function showVideoClip() {
  // myChart.reset();
  viewerType = 'video';

  document.getElementById('dropdown_button').innerHTML = 'Video<svg class="w-4 h-4 ml-2" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7"></path></svg>';

  document.getElementById('movie').style.display = 'block';
  document.getElementById('webcam').style.display = 'none';
  document.getElementById('uploadImg').style.display = 'none';

  await setupVideoClip();
  stopCamera();

  videoClip = document.getElementById('videoClip');
  videoClip.play();
}

async function showWebcam() {
  // myChart.reset();

  document.getElementById('dropdown_button').innerHTML = 'Webcam<svg class="w-4 h-4 ml-2" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7"></path></svg>';

  document.getElementById('webcam').style.display = 'block';
  document.getElementById('uploadImg').style.display = 'none';
  document.getElementById('movie').style.display = 'none';
  document.getElementById('output').style.display = 'block';
  setupOutputValue();
  await setupCamera();
  webcam.play();

  viewerType = 'webcam';
}

async function showEmotions() {
  document.getElementById('block_positive').style.display = 'none',
  document.getElementById('block_active').style.display = 'none',
  document.getElementById('block_vibe').style.display = 'none',

  document.getElementById('block_neutral').style.display = 'flex',
  document.getElementById('block_angry').style.display = 'flex',
  document.getElementById('block_happy').style.display = 'flex',
  document.getElementById('block_sad').style.display = 'flex',
  document.getElementById('block_supprise').style.display = 'flex',

  document.getElementById('btn_emotions').style.color = 'rgba(24, 144, 255, 1)';
  document.getElementById('btn_emotions').style.borderColor = 'rgba(24, 144, 255, 1)';

  document.getElementById('btn_positive-active').style.color = 'rgb(17 24 39)';
  document.getElementById('btn_positive-active').style.borderColor = 'rgb(243 244 246)';

  document.getElementById('btn_vibes').style.color = 'rgb(17 24 39)';
  document.getElementById('btn_vibes').style.borderColor = 'rgb(243 244 246)';

  // document.getElementById('btn_all').style.color = 'rgb(17 24 39)';
  // document.getElementById('btn_all').style.borderColor = 'rgb(243 244 246)';

  document.getElementById('label_neutral').style.display = 'flex';
  document.getElementById('bar_neutral').style.display = 'flex';
  document.getElementById('label_sad').style.display = 'flex';
  document.getElementById('bar_sad').style.display = 'flex';
  document.getElementById('label_happy').style.display = 'flex';
  document.getElementById('bar_happy').style.display = 'flex';
  document.getElementById('label_angry').style.display = 'flex';
  document.getElementById('bar_angry').style.display = 'flex';
  document.getElementById('label_surprised').style.display = 'flex';
  document.getElementById('bar_surprised').style.display = 'flex';

  document.getElementById('border_positive-active').style.borderStyle = 'none';

  document.getElementById('label_positive').style.display = 'none';
  document.getElementById('bar_positive').style.display = 'none';
  document.getElementById('label_active').style.display = 'none';
  document.getElementById('bar_active').style.display = 'none';


  document.getElementById('border_vibes').style.borderStyle = 'none';
  document.getElementById('label_vibes').style.display = 'none';
  document.getElementById('bar_vibes').style.display = 'none';
}

async function showPositiveActive() {
  document.getElementById('block_positive').style.display = 'flex';
  document.getElementById('block_active').style.display = 'flex';
  document.getElementById('block_positive').style.paddingLeft = '3em';
  // document.getElementById('block_active').style.paddingLeft = '1em',
  document.getElementById('block_vibe').style.display = 'none',

  document.getElementById('block_neutral').style.display = 'none',
  document.getElementById('block_angry').style.display = 'none',
  document.getElementById('block_happy').style.display = 'none',
  document.getElementById('block_sad').style.display = 'none',
  document.getElementById('block_supprise').style.display = 'none',

  document.getElementById('btn_positive-active').style.color = 'rgba(24, 144, 255, 1)';
  document.getElementById('btn_positive-active').style.borderColor = 'rgba(24, 144, 255, 1)';

  document.getElementById('btn_emotions').style.color = 'rgb(17 24 39)';
  document.getElementById('btn_emotions').style.borderColor = 'rgb(243 244 246)';

  document.getElementById('btn_vibes').style.color = 'rgb(17 24 39)';
  document.getElementById('btn_vibes').style.borderColor = 'rgb(243 244 246)';

  // document.getElementById('btn_all').style.color = 'rgb(17 24 39)';
  // document.getElementById('btn_all').style.borderColor = 'rgb(243 244 246)';

  document.getElementById('border_positive-active').style.borderStyle = 'none';

  document.getElementById('label_neutral').style.display = 'none';
  document.getElementById('bar_neutral').style.display = 'none';
  document.getElementById('label_sad').style.display = 'none';
  document.getElementById('bar_sad').style.display = 'none';
  document.getElementById('label_happy').style.display = 'none';
  document.getElementById('bar_happy').style.display = 'none';
  document.getElementById('label_angry').style.display = 'none';
  document.getElementById('bar_angry').style.display = 'none';
  document.getElementById('label_surprised').style.display = 'none';
  document.getElementById('bar_surprised').style.display = 'none';

  document.getElementById('label_positive').style.display = 'flex';
  document.getElementById('bar_positive').style.display = 'flex';
  document.getElementById('label_active').style.display = 'flex';
  document.getElementById('bar_active').style.display = 'flex';

  document.getElementById('border_vibes').style.borderStyle = 'none';
  document.getElementById('label_vibes').style.display = 'none';
  document.getElementById('bar_vibes').style.display = 'none';
}

async function showVibes() {
  document.getElementById('block_positive').style.display = 'none',
  document.getElementById('block_active').style.display = 'none',

  document.getElementById('block_vibe').style.display = 'flex',

  document.getElementById('block_neutral').style.display = 'none',
  document.getElementById('block_angry').style.display = 'none',
  document.getElementById('block_happy').style.display = 'none',
  document.getElementById('block_sad').style.display = 'none',
  document.getElementById('block_supprise').style.display = 'none',

  document.getElementById('btn_vibes').style.color = 'rgba(24, 144, 255, 1)';
  document.getElementById('btn_vibes').style.borderColor = 'rgba(24, 144, 255, 1)';

  document.getElementById('btn_emotions').style.color = 'rgb(17 24 39)';
  document.getElementById('btn_emotions').style.borderColor = 'rgb(243 244 246)';

  document.getElementById('btn_positive-active').style.color = 'rgb(17 24 39)';
  document.getElementById('btn_positive-active').style.borderColor = 'rgb(243 244 246)';

  // document.getElementById('btn_all').style.color = 'rgb(17 24 39)';
  // document.getElementById('btn_all').style.borderColor = 'rgb(243 244 246)';

  document.getElementById('border_positive-active').style.borderStyle = 'none';

  document.getElementById('label_neutral').style.display = 'none';
  document.getElementById('bar_neutral').style.display = 'none';
  document.getElementById('label_sad').style.display = 'none';
  document.getElementById('bar_sad').style.display = 'none';
  document.getElementById('label_happy').style.display = 'none';
  document.getElementById('bar_happy').style.display = 'none';
  document.getElementById('label_angry').style.display = 'none';
  document.getElementById('bar_angry').style.display = 'none';
  document.getElementById('label_surprised').style.display = 'none';
  document.getElementById('bar_surprised').style.display = 'none';

  document.getElementById('label_positive').style.display = 'none';
  document.getElementById('bar_positive').style.display = 'none';
  document.getElementById('label_active').style.display = 'none';
  document.getElementById('bar_active').style.display = 'none';

  document.getElementById('border_vibes').style.borderStyle = 'none';

  document.getElementById('label_vibes').style.display = 'flex';
  document.getElementById('bar_vibes').style.display = 'flex';
}

const setupMyChartEmotions = async () => {
  numberCell = maxFrameChart;
  numberCells = Array.from({length: maxFrameChart}, (_, i) => i + 1);
  dataEmotionsNeutral = Array.from({length: maxFrameChart}, (_, i) => 0);
  dataEmotionsAngry = Array.from({length: maxFrameChart}, (_, i) => 0);
  dataEmotionsHappy = Array.from({length: maxFrameChart}, (_, i) => 0);
  dataEmotionsSad = Array.from({length: maxFrameChart}, (_, i) => 0);
  dataEmotionsSurprised = Array.from({length: maxFrameChart}, (_, i) => 0);
  dataEmotionsPositive = Array.from({length: maxFrameChart}, (_, i) => 0);
  dataEmotionsActive = Array.from({length: maxFrameChart}, (_, i) => 0);
  dataEmotionsVibe = Array.from({length: maxFrameChart}, (_, i) => 0);
  const ctxChart = document.getElementById('myChart').getContext('2d');
  myChart = new Chart(ctxChart, {
        backgroundColor: 'white',
        type: 'line',
        data: {
            labels: numberCells,
            datasets: [
              datasetAngry,
              datasetNeutral,
              datasetHappy,
              datasetSad,
              datasetSurprised,
              datasetPositive,
              datasetActive,
              datasetVibe,
            ],
        },
        options: {
            animations: {
              duration: timeFrameCapture * 1.5,
              easing: 'linear',
            },
            scales: {
              yAxes: [{
                ticks: {
                  beginAtZero: true,
                  min: 0,
                  max: 1,
                  },
                }],
                xAxes: [{
                  gridLines: {
                    display: false,
                  },
                  position: 'bottom',
                  ticks: {
                    beginAtZero: true,
                    display: false,
                  },
                }],
            },
            legend: {
              display: false,
            },
            tooltips: {
              enabled: false,
            },
            hover: {
              mode: null,
            },
            elements: {
              point: {
                radius: 0,
              },
            },
        },
  });
};

function hideDataEmotions() {
  myChart.getDatasetMeta(5).hidden = true;
  myChart.getDatasetMeta(6).hidden = true;
  myChart.getDatasetMeta(7).hidden = true;

  myChart.getDatasetMeta(0).hidden = false;
  myChart.getDatasetMeta(1).hidden = false;
  myChart.getDatasetMeta(2).hidden = false;
  myChart.getDatasetMeta(3).hidden = false;
  myChart.getDatasetMeta(4).hidden = false;
}

function hideDataPositive() {
  myChart.getDatasetMeta(0).hidden = true;
  myChart.getDatasetMeta(1).hidden = true;
  myChart.getDatasetMeta(2).hidden = true;
  myChart.getDatasetMeta(3).hidden = true;
  myChart.getDatasetMeta(4).hidden = true;
  myChart.getDatasetMeta(7).hidden = true;
  myChart.getDatasetMeta(5).hidden = false;
  myChart.getDatasetMeta(6).hidden = false;
}

function hideDataVibe() {
  myChart.getDatasetMeta(0).hidden = true;
  myChart.getDatasetMeta(1).hidden = true;
  myChart.getDatasetMeta(2).hidden = true;
  myChart.getDatasetMeta(3).hidden = true;
  myChart.getDatasetMeta(4).hidden = true;

  myChart.getDatasetMeta(5).hidden = true;
  myChart.getDatasetMeta(6).hidden = true;

  myChart.getDatasetMeta(7).hidden = false;
}

const setupModel = async () => {
  modelFace = await facetfjs.loadBlazeFace();
  modelEmotion = await facetfjs.loadFaceEmotions();
  pipeline = new facetfjs.EmotionPipeline(modelFace, modelEmotion);
};

function setupEmotionElements() {
  valueNeutral = document.getElementById('value_neutral');
  valueHappy = document.getElementById('value_happy');
  valueSad = document.getElementById('value_sad');
  valueAngry = document.getElementById('value_angry');
  valueSuprised = document.getElementById('value_suprised');

  valuePositive = document.getElementById('value_positive');
  valueActive = document.getElementById('value_active');
  valueVibe = document.getElementById('value_vibe');

  valueNeutralLabel = document.getElementById('value_neutral_label');
  valueHappyLabel = document.getElementById('value_happy_label');
  valueSadLabel = document.getElementById('value_sad_label');
  valueAngryLabel = document.getElementById('value_angry_label');
  valueSuprisedLabel = document.getElementById('value_suprised_label');

  valuePositiveLabel = document.getElementById('value_positive_label');
  valueActiveLabel = document.getElementById('value_active_label');
  valueVibeLabel = document.getElementById('value_vibe_label');
}

function rememberNoPopup() {
  localStorage.setItem('remember', true);
}

const setupPage = async () => {
  await tf.setBackend(state.backend);
  document.getElementById('remember_no_popup').addEventListener('click', rememberNoPopup);

  const remember = localStorage.getItem('remember');
  if (remember !== undefined && remember) {
  } else {
    toggleModal('authorized_camera');
  }

  await setupCamera();
  webcam.play();

  viewerType = 'webcam';

  videoWidth = webcam.videoWidth;
  videoHeight = webcam.videoHeight;
  webcam.width = videoWidth;
  webcam.height = videoHeight;

  canvasV = document.createElement('canvas');
  // imgVideo = document.createElement('img');

  canvas = document.getElementById('output');
  canvas.width = videoWidth;
  canvas.height = videoHeight;
  ctxOutput = canvas.getContext('2d');
  ctxOutput.fillStyle = 'rgba(255, 0, 0, 0.5)';

  await setupModel();

  setupEmotionElements();
  setupMyChartEmotions();
  renderPrediction();

  document.getElementById('btnPicture').addEventListener('click', showImg);
  document.getElementById('btnWebcam').addEventListener('click', showWebcam);
  document.getElementById('btnVideoClip').addEventListener('click', showVideoClip);

  document.getElementById('btn_emotions').addEventListener('click', showEmotions);
  document.getElementById('btn_positive-active').addEventListener('click', showPositiveActive);
  document.getElementById('btn_vibes').addEventListener('click', showVibes);
  // document.getElementById('btn_all').addEventListener('click', showAll);
  document.getElementById('btn_emotions').addEventListener('click', hideDataEmotions);
  document.getElementById('btn_positive-active').addEventListener('click', hideDataPositive);
  document.getElementById('btn_vibes').addEventListener('click', hideDataVibe);
  document.getElementById('btn_close_img').addEventListener('click',closeImg);
  document.getElementById('btn_emotions').click();

};

setupPage();
