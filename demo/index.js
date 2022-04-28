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
import { math } from '@tensorflow/tfjs-core';

tfjsWasm.setWasmPaths(
  `https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-wasm@${tfjsWasm.version_wasm}/dist/`);

const stats = new Stats();
stats.showPanel(0);
document.getElementById('stats').appendChild(stats.domElement);
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
let valuePositive;
let valueActive;
let valueVibe;
let imgWidth;
let imgHeight;
let img;
let videoClip;
let videoClipWidth;
let videoClipHeight;
let emotionsAngry;
let emotionsHappy;
let emotionsSad;
let emotionsNeutral;
let emotionsSurprise;
let emotionPositive;
let emotionActive;
let emotionVibe;
var dataEmotionsNeutral = [];
var dataEmotionsAngry = [];
var dataEmotionHappy = [];
var dataEmotionSurprised = []; 
var dataEmotionSad = [];
var dataEmtionsPositive = [];
var dataEmtionActive = [];
var dataEmtionVibe = [];

var dataEmotionsNeutral_Img = [];
var dataEmotionsAngry_Img= [];
var dataEmotionHappy_Img = [];
var dataEmotionSurprised_Img = []; 
var dataEmotionSad_Img = [];
var dataEmtionsPositive_Img = [];
var dataEmtionActive_Img = [];
var dataEmtionVibe_Img = [];

var dataEmotionsNeutral_Clip = [];
var dataEmotionsAngry_Clip= [];
var dataEmotionHappy_Clip = [];
var dataEmotionSurprised_Clip = []; 
var dataEmotionSad_Clip = [];
var dataEmtionsPositive_Clip= [];
var dataEmtionActive_Clip = [];
var dataEmtionVibe_Clip = [];

var dataEmotionsNeutral_Webcam = [];
var dataEmotionsAngry_Webcam= [];
var dataEmotionHappy_Webcam = [];
var dataEmotionSurprised_Webcam = []; 
var dataEmotionSad_Webcam = [];
var dataEmtionsPositive_Webcam = [];
var dataEmtionActive_Webcam = [];
var dataEmtionVibe_Webcam = [];

let numberCells;
let myChart;

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
    }
  }
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

const renderPrediction = async () => {
  stats.begin();

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

      emotionsNeutral = [emotions[0].toFixed(3)];
      dataEmotionsNeutral.push(emotionsNeutral);
      // localStorage.setItem('dataNeutral', dataEmotionsNeutral);
      

      emotionsHappy = [emotions[1].toFixed(3)];
      dataEmotionHappy.push(emotionsHappy);
      // localStorage.setItem('dataHappy', dataEmotionHappy);

      emotionsSad = [emotions[2].toFixed(3)];
      dataEmotionSad.push(emotionsSad);
      // localStorage.setItem('dataSad', dataEmotionSad);

      emotionsAngry = [emotions[3].toFixed(3)];
      dataEmotionsAngry.push(emotionsAngry);
      // localStorage.setItem('dataAngry', dataEmotionsAngry);

      emotionsSurprise = [emotions[4].toFixed(3)];
      dataEmotionSurprised.push(emotionsSurprise);
      // localStorage.setItem('dataSurprise', dataEmotionSurprised);

      emotionPositive = [(pipeline.estimatePositive(emotions).toFixed(3))];
      dataEmtionsPositive.push(emotionPositive);

      emotionActive = [(pipeline.estimateActive(emotions).toFixed(3))];
      dataEmtionActive.push(emotionActive);

      emotionVibe = [(pipeline.estimateVibe(emotions).toFixed(3))];
      dataEmtionVibe.push(emotionVibe);

      if (dataEmotionsAngry.length >= 500) {
        
        dataEmotionsAngry.shift();
        dataEmotionsNeutral.shift();
        dataEmotionHappy.shift();
        dataEmotionSad.shift();
        dataEmotionSurprised.shift();
        dataEmtionsPositive.shift();
        dataEmtionActive.shift();
        dataEmtionVibe.shift();

        numberCells.push(numberCells.length);
        numberCells.shift();
      }

      myChart.data.labels = numberCells;
      myChart.data.datasets[0].data = dataEmotionsAngry;
      myChart.data.datasets[1].data = dataEmotionsNeutral;
      myChart.data.datasets[2].data = dataEmotionHappy;
      myChart.data.datasets[3].data = dataEmotionSad;
      myChart.data.datasets[4].data = dataEmotionSurprised;
      myChart.data.datasets[5].data = dataEmtionsPositive;
      myChart.data.datasets[6].data = dataEmtionActive;
      myChart.data.datasets[7].data = dataEmtionVibe;
      myChart.update();


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

      valueNeutral.style.width = `${(emotions[0] * 100).toFixed(2)}%`;
      valueNeutralLabel.textContent = `${(emotions[0] * 100).toFixed(2)}%`;
      valueHappy.style.width = `${(emotions[1] * 100).toFixed(2)}%`;
      valueHappyLabel.textContent = `${(emotions[1] * 100).toFixed(2)}%`;
      valueSad.style.width = `${(emotions[2] * 100).toFixed(2)}%`;
      valueSadLabel.textContent = `${(emotions[2] * 100).toFixed(2)}%`;
      valueAngry.style.width = `${(emotions[3] * 100).toFixed(2)}%`;
      valueAngryLabel.textContent = `${(emotions[3] * 100).toFixed(2)}%`;
      valueSuprised.style.width = `${(emotions[4] * 100).toFixed(2)}%`;
      valueSuprisedLabel.textContent = `${(emotions[4] * 100).toFixed(2)}%`;

      // console.log('Positive: ', pipeline.estimatePositive(emotions));
      // console.log('Active: ', pipeline.estimateActive(emotions));
      // console.log('Vibe: ', pipeline.estimateVibe(emotions));
      valuePositive.style.width = `${(pipeline.estimatePositive(emotions) * 100).toFixed(2)}%`;
      valuePositiveLabel.textContent = `${(pipeline.estimatePositive(emotions) * 100).toFixed(2)}%`;
      valueActive.style.width = `${(pipeline.estimateActive(emotions) * 100).toFixed(2)}%`;
      valueActiveLabel.textContent = `${(pipeline.estimateActive(emotions) * 100).toFixed(2)}%`;
      valueVibe.style.width = `${(pipeline.estimateVibe(emotions) * 100).toFixed(2)}%`;
      valueVibeLabel.textContent = `${(pipeline.estimateVibe(emotions) * 100).toFixed(2)}%`;


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
      emotionsNeutral = `${(emotions[0]).toFixed(3)}`;
      // console.log('emotionNeutral: ', emotionsNeutral);
      // dataNeutral = `${(emotions[0]).toFixed(3)}`;
      // const combine = emotionsNeutral.concat(dataNeutral);
      // console.log('Combine data: ' ,combine);
      // dataNeutral = [`${(emotions[0]).toFixed(3)}`];
      emotionsHappy = `${(emotions[1]).toFixed(3)}`;
      // console.log('happy: ', emotionsHappy);
      emotionsSad = `${(emotions[2]).toFixed(3)}`
      emotionsAngry = `${(emotions[3]).toFixed(3)}`;
      // console.log('angry: ',emotionsAngry)
      emotionsSurprise = `${(emotions[4]).toFixed(3)}`

      // arrayEmotions = Array.from(emotions);
      // console.log(arrayEmotions);
    }
  }
  stats.end();
  requestAnimationFrame(renderPrediction);
};

const renderPredictionImg = async () => {
  stats.begin();
  const returnTensors = false;
  // const flipHorizontal = true;
  const annotateBoxes = true;
  // const predictions = await modelFace.estimateFaces(
  //   video, returnTensors, flipHorizontal, annotateBoxes);
  const predictions = await pipeline.estimateEmotion(img);

  if (predictions.length > 0) {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    // console.log(predictions);

    for (let i = 0; i < predictions.length; i++) {
      let face = predictions[0].face;
      let emotions = predictions[0].emotions;

      emotionsNeutral= [emotions[0].toFixed(3)];
      dataEmotionsNeutral_Img.push(emotionsNeutral);
      // localStorage.setItem('dataNeutral', dataEmotionsNeutral);

      emotionsHappy = [emotions[1].toFixed(3)];
      dataEmotionHappy_Img.push(emotionsHappy);

      //console.log(emotionsAngry);
      // localStorage.setItem('dataHappy', dataEmotionHappy);

      emotionsSad = [emotions[2].toFixed(3)];
      dataEmotionSad_Img.push(emotionsSad);
      // localStorage.setItem('dataSad', dataEmotionSad);

      emotionsAngry = [emotions[3].toFixed(3)];
      dataEmotionsAngry_Img.push(emotionsAngry);
      // localStorage.setItem('dataAngry', dataEmotionsAngry);

      emotionsSurprise = [emotions[4].toFixed(3)];
      dataEmotionSurprised_Img.push(emotionsSurprise);

      emotionPositive = [(pipeline.estimatePositive(emotions).toFixed(3))];
      dataEmtionsPositive_Img.push(emotionPositive);

      emotionActive = [(pipeline.estimateActive(emotions).toFixed(3))];
      dataEmtionActive_Img.push(emotionActive);

      emotionVibe = [(pipeline.estimateVibe(emotions).toFixed(3))];
      dataEmtionVibe_Img.push(emotionVibe);

      if (dataEmotionsAngry_Img.length >= 500) {
        dataEmotionsAngry_Img.shift();
        dataEmotionsNeutral_Img.shift();
        dataEmotionHappy_Img.shift();
        console.log(emotionsNeutral);
        dataEmotionSad_Img.shift();
        dataEmotionSurprised_Img.shift();
        dataEmtionsPositive_Img.shift();
        dataEmtionActive_Img.shift();
        dataEmtionVibe_Img.shift();
        numberCells.push(numberCells.length);
        numberCells.shift();
      }

      myChart.data.labels = numberCells;
      myChart.data.datasets[0].data = dataEmotionsAngry_Img;
      myChart.data.datasets[1].data = dataEmotionsNeutral_Img;
      myChart.data.datasets[2].data = dataEmotionHappy_Img;
      myChart.data.datasets[3].data = dataEmotionSad_Img;
      myChart.data.datasets[4].data = dataEmotionSurprised_Img;
      myChart.data.datasets[5].data = dataEmtionsPositive_Img;
      myChart.data.datasets[6].data = dataEmtionActive_Img;
      myChart.data.datasets[7].data = dataEmtionVibe_Img;
      myChart.update();

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

      valueNeutral.style.width = `${(emotions[0] * 100).toFixed(2)}%`;
      valueNeutralLabel.textContent = `${(emotions[0] * 100).toFixed(2)}%`;
      valueHappy.style.width = `${(emotions[1] * 100).toFixed(2)}%`;
      valueHappyLabel.textContent = `${(emotions[1] * 100).toFixed(2)}%`;
      valueSad.style.width = `${(emotions[2] * 100).toFixed(2)}%`;
      valueSadLabel.textContent = `${(emotions[2] * 100).toFixed(2)}%`;
      valueAngry.style.width = `${(emotions[3] * 100).toFixed(2)}%`;
      valueAngryLabel.textContent = `${(emotions[3] * 100).toFixed(2)}%`;
      valueSuprised.style.width = `${(emotions[4] * 100).toFixed(2)}%`;
      valueSuprisedLabel.textContent = `${(emotions[4] * 100).toFixed(2)}%`;

      // console.log('Positive: ', pipeline.estimatePositive(emotions));
      // console.log('Active: ', pipeline.estimateActive(emotions));
      // console.log('Vibe: ', pipeline.estimateVibe(emotions));
      valuePositive.style.width = `${(pipeline.estimatePositive(emotions) * 100).toFixed(2)}%`;
      valuePositiveLabel.textContent = `${(pipeline.estimatePositive(emotions) * 100).toFixed(2)}%`;
      valueActive.style.width = `${(pipeline.estimateActive(emotions) * 100).toFixed(2)}%`;
      valueActiveLabel.textContent = `${(pipeline.estimateActive(emotions) * 100).toFixed(2)}%`;
      valueVibe.style.width = `${(pipeline.estimateVibe(emotions) * 100).toFixed(2)}%`;
      valueVibeLabel.textContent = `${(pipeline.estimateVibe(emotions) * 100).toFixed(2)}%`;


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
  stats.end();

  requestAnimationFrame(renderPredictionImg);
};

const renderPredictionVideoClip = async () => {
  stats.begin();
  const returnTensors = false;
  // const flipHorizontal = true;
  const annotateBoxes = true;
  // const predictions = await modelFace.estimateFaces(
  //   video, returnTensors, flipHorizontal, annotateBoxes);
  const predictions = await pipeline.estimateEmotion(videoClip);

  if (predictions.length > 0) {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    // console.log(predictions);

    for (let i = 0; i < predictions.length; i++) {
      let face = predictions[0].face;
      let emotions = predictions[0].emotions;

      emotionsNeutral = [emotions[0].toFixed(3)];
      dataEmotionsNeutral_Clip.push(emotionsNeutral);
      // localStorage.setItem('dataNeutral', dataEmotionsNeutral);

      emotionsHappy = [emotions[1].toFixed(3)];
      dataEmotionHappy_Clip.push(emotionsHappy);
      // localStorage.setItem('dataHappy', dataEmotionHappy);

      emotionsSad = [emotions[2].toFixed(3)];
      dataEmotionSad_Clip.push(emotionsSad);
      // localStorage.setItem('dataSad', dataEmotionSad);

      emotionsAngry = [emotions[3].toFixed(3)];
      dataEmotionsAngry_Clip.push(emotionsAngry);
      // localStorage.setItem('dataAngry', dataEmotionsAngry);

      emotionsSurprise = [emotions[4].toFixed(3)];
      dataEmotionSurprised_Clip.push(emotionsSurprise);

      emotionPositive = [(pipeline.estimatePositive(emotions).toFixed(3))];
      dataEmtionsPositive_Clip.push(emotionPositive);

      emotionActive = [(pipeline.estimateActive(emotions).toFixed(3))];
      dataEmtionActive_Clip.push(emotionActive);

      emotionVibe = [(pipeline.estimateVibe(emotions).toFixed(3))];
      dataEmtionVibe_Clip.push(emotionVibe);

      if (dataEmotionsAngry_Clip.length >= 500) {
        dataEmotionsAngry_Clip.shift();
        dataEmotionsNeutral_Clip.shift();
        dataEmotionHappy_Clip.shift();
        dataEmotionSad_Clip.shift();
        dataEmotionSurprised_Clip.shift();
        dataEmtionsPositive_Clip.shift();
        dataEmtionActive_Clip.shift();
        dataEmtionVibe_Clip.shift();
        numberCells.push(numberCells.length);
        numberCells.shift();
      }

      myChart.data.labels = numberCells;
      myChart.data.datasets[0].data = dataEmotionsAngry_Clip;
      myChart.data.datasets[1].data = dataEmotionsNeutral_Clip;
      myChart.data.datasets[2].data = dataEmotionHappy_Clip;
      myChart.data.datasets[3].data = dataEmotionSad_Clip;
      myChart.data.datasets[4].data = dataEmotionSurprised_Clip;
      myChart.data.datasets[5].data = dataEmtionsPositive_Clip;
      myChart.data.datasets[6].data = dataEmtionActive_Clip;
      myChart.data.datasets[7].data = dataEmtionVibe_Clip;
      myChart.update();

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

      valueNeutral.style.width = `${(emotions[0] * 100).toFixed(2)}%`;
      valueNeutralLabel.textContent = `${(emotions[0] * 100).toFixed(2)}%`;
      valueHappy.style.width = `${(emotions[1] * 100).toFixed(2)}%`;
      valueHappyLabel.textContent = `${(emotions[1] * 100).toFixed(2)}%`;
      valueSad.style.width = `${(emotions[2] * 100).toFixed(2)}%`;
      valueSadLabel.textContent = `${(emotions[2] * 100).toFixed(2)}%`;
      valueAngry.style.width = `${(emotions[3] * 100).toFixed(2)}%`;
      valueAngryLabel.textContent = `${(emotions[3] * 100).toFixed(2)}%`;
      valueSuprised.style.width = `${(emotions[4] * 100).toFixed(2)}%`;
      valueSuprisedLabel.textContent = `${(emotions[4] * 100).toFixed(2)}%`;

      // console.log('Positive: ', pipeline.estimatePositive(emotions));
      // console.log('Active: ', pipeline.estimateActive(emotions));
      // console.log('Vibe: ', pipeline.estimateVibe(emotions));
      valuePositive.style.width = `${(pipeline.estimatePositive(emotions) * 100).toFixed(2)}%`;
      valuePositiveLabel.textContent = `${(pipeline.estimatePositive(emotions) * 100).toFixed(2)}%`;
      valueActive.style.width = `${(pipeline.estimateActive(emotions) * 100).toFixed(2)}%`;
      valueActiveLabel.textContent = `${(pipeline.estimateActive(emotions) * 100).toFixed(2)}%`;
      valueVibe.style.width = `${(pipeline.estimateVibe(emotions) * 100).toFixed(2)}%`;
      valueVibeLabel.textContent = `${(pipeline.estimateVibe(emotions) * 100).toFixed(2)}%`;


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
  stats.end();

  requestAnimationFrame(renderPredictionVideoClip);
};

async function showImg() {
  myChart.reset();
  myChartEmotions_Img();
  document.getElementById('uploadImg').style.display = 'block';
  document.getElementById('video').style.display = 'none';
  document.getElementById('movie').style.display = 'none';

  document.getElementById('value_neutral_label').innerHTML = '%';
  document.getElementById('value_happy_label').innerHTML = '%';
  document.getElementById('value_sad_label').innerHTML = '%';
  document.getElementById('value_suprised_label').innerHTML = '%';
  document.getElementById('value_positive_label').innerHTML = '%';
  document.getElementById('value_active_label').innerHTML = '%';
  document.getElementById('value_vibe_label').innerHTML = '%';

  document.getElementById('value_neutral').style.width = '0%';
  document.getElementById('value_happy').style.width = '0%';
  document.getElementById('value_sad').style.width = '0%';
  document.getElementById('value_angry_label').innerHTML = '%';
  document.getElementById('value_angry').style.width = '0%';
  document.getElementById('value_suprised').style.width = '0%';
  document.getElementById('value_positive').style.width = '0%';
  document.getElementById('value_active').style.width = '0%';
  document.getElementById('value_vibe').style.width = '0%';

  stream = video.srcObject;
  stream.getTracks().forEach(function(track) {
    track.stop();
  });
  await tf.setBackend(state.backend);
  await setupImg();

  imgWidth = img.width;
  imgHeight = img.height;

  canvas = document.getElementById('output');
  canvas.width = imgWidth;
  canvas.height = imgHeight;
  ctx = canvas.getContext('2d');
  ctx.fillStyle = 'rgba(255, 0, 0, 0.5)';

  modelFace = await facetfjs.loadBlazeFace();
  modelEmotion = await facetfjs.loadFaceEmotions();
  pipeline = new facetfjs.EmotionPipeline(modelFace, modelEmotion);

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
  renderPredictionImg();
}

async function showVideoClip() {
  myChart.reset();
  document.getElementById('movie').style.display = 'block';
  document.getElementById('video').style.display = 'none';
  document.getElementById('uploadImg').style.display = 'none';

  stream = video.srcObject;
  stream.getTracks().forEach(function(track) {
    track.stop();
  });

  document.getElementById('value_neutral_label').innerHTML = '%';
  document.getElementById('value_happy_label').innerHTML = '%';
  document.getElementById('value_sad_label').innerHTML = '%';
  document.getElementById('value_angry_label').innerHTML = '%';
  document.getElementById('value_suprised_label').innerHTML = '%';
  document.getElementById('value_positive_label').innerHTML = '%';
  document.getElementById('value_active_label').innerHTML = '%';
  document.getElementById('value_vibe_label').innerHTML = '%';

  document.getElementById('value_neutral').style.width = '0%';
  document.getElementById('value_happy').style.width = '0%';
  document.getElementById('value_sad').style.width = '0%';
  document.getElementById('value_angry').style.width = '0%';
  document.getElementById('value_suprised').style.width = '0%';
  document.getElementById('value_positive').style.width = '0%';
  document.getElementById('value_active').style.width = '0%';
  document.getElementById('value_vibe').style.width = '0%';

  await tf.setBackend(state.backend);
  videoClip = document.getElementById('videoClip');
  videoClip.play();
  myChartEmotions_Clip();

  videoClipWidth = videoClip.width;
  videoClipHeight = videoClip.height;

  canvas = document.getElementById('output');
  canvas.width = videoClipWidth;
  canvas.height = videoClipHeight;
  ctx = canvas.getContext('2d');
  ctx.fillStyle = 'rgba(255, 0, 0, 0.5)';

  modelFace = await facetfjs.loadBlazeFace();
  modelEmotion = await facetfjs.loadFaceEmotions();
  pipeline = new facetfjs.EmotionPipeline(modelFace, modelEmotion);

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

  renderPredictionVideoClip();
}

const renderPrediction_Webcam = async () => {
  stats.begin();

  const returnTensors  = false;
  const annotateBoxes = true;

  const predictions = await pipeline.estimateEmotion(video);
  if (predictions.length > 0) {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    // console.log(predictions);
    for (let i = 0; i < predictions.length; i++) {
      let face = predictions[0].face;
      let emotions = predictions[0].emotions;

      emotionsNeutral = [emotions[0].toFixed(3)];
      dataEmotionsNeutral_Webcam.push(emotionsNeutral);
      // localStorage.setItem('dataNeutral', dataEmotionsNeutral);
      

      emotionsHappy = [emotions[1].toFixed(3)];
      dataEmotionHappy_Webcam.push(emotionsHappy);
      // localStorage.setItem('dataHappy', dataEmotionHappy);

      emotionsSad = [emotions[2].toFixed(3)];
      dataEmotionSad_Webcam.push(emotionsSad);
      // localStorage.setItem('dataSad', dataEmotionSad);

      emotionsAngry = [emotions[3].toFixed(3)];
      dataEmotionsAngry_Webcam.push(emotionsAngry);
      // localStorage.setItem('dataAngry', dataEmotionsAngry);

      emotionsSurprise = [emotions[4].toFixed(3)];
      dataEmotionSurprised_Webcam.push(emotionsSurprise);
      // localStorage.setItem('dataSurprise', dataEmotionSurprised);

      emotionPositive = [(pipeline.estimatePositive(emotions).toFixed(3))];
      dataEmtionsPositive_Webcam.push(emotionPositive);

      emotionActive = [(pipeline.estimateActive(emotions).toFixed(3))];
      dataEmtionActive_Webcam.push(emotionActive);

      emotionVibe = [(pipeline.estimateVibe(emotions).toFixed(3))];
      dataEmtionVibe_Webcam.push(emotionVibe);

      if (dataEmotionsAngry_Webcam.length >= 500) {
        
        dataEmotionsAngry_Webcam.shift();
        dataEmotionsNeutral_Webcam.shift();
        dataEmotionHappy_Webcam.shift();
        dataEmotionSad_Webcam.shift();
        dataEmotionSurprised_Webcam.shift();
        dataEmotionSurprised_Webcam.shift();
        dataEmtionsPositive_Webcam.shift();
        dataEmtionActive_Webcam.shift();
        dataEmtionVibe_Webcam.shift();

        numberCells.push(numberCells.length);
        numberCells.shift();
      }

      myChart.data.labels = numberCells;
      myChart.data.datasets[0].data = dataEmotionsAngry_Webcam;
      myChart.data.datasets[1].data = dataEmotionsNeutral_Webcam;
      myChart.data.datasets[2].data = dataEmotionHappy_Webcam;
      myChart.data.datasets[3].data = dataEmotionSad_Webcam;
      myChart.data.datasets[4].data = dataEmotionSurprised_Webcam;
      myChart.data.datasets[5].data = dataEmtionsPositive_Webcam;
      myChart.data.datasets[6].data = dataEmtionActive_Webcam;
      myChart.data.datasets[7].data = dataEmtionVibe_Webcam;
      myChart.update();

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

      valueNeutral.style.width = `${(emotions[0] * 100).toFixed(2)}%`;
      valueNeutralLabel.textContent = `${(emotions[0] * 100).toFixed(2)}%`;
      valueHappy.style.width = `${(emotions[1] * 100).toFixed(2)}%`;
      valueHappyLabel.textContent = `${(emotions[1] * 100).toFixed(2)}%`;
      valueSad.style.width = `${(emotions[2] * 100).toFixed(2)}%`;
      valueSadLabel.textContent = `${(emotions[2] * 100).toFixed(2)}%`;
      valueAngry.style.width = `${(emotions[3] * 100).toFixed(2)}%`;
      valueAngryLabel.textContent = `${(emotions[3] * 100).toFixed(2)}%`;
      valueSuprised.style.width = `${(emotions[4] * 100).toFixed(2)}%`;
      valueSuprisedLabel.textContent = `${(emotions[4] * 100).toFixed(2)}%`;

      // console.log('Positive: ', pipeline.estimatePositive(emotions));
      // console.log('Active: ', pipeline.estimateActive(emotions));
      // console.log('Vibe: ', pipeline.estimateVibe(emotions));
      valuePositive.style.width = `${(pipeline.estimatePositive(emotions) * 100).toFixed(2)}%`;
      valuePositiveLabel.textContent = `${(pipeline.estimatePositive(emotions) * 100).toFixed(2)}%`;
      valueActive.style.width = `${(pipeline.estimateActive(emotions) * 100).toFixed(2)}%`;
      valueActiveLabel.textContent = `${(pipeline.estimateActive(emotions) * 100).toFixed(2)}%`;
      valueVibe.style.width = `${(pipeline.estimateVibe(emotions) * 100).toFixed(2)}%`;
      valueVibeLabel.textContent = `${(pipeline.estimateVibe(emotions) * 100).toFixed(2)}%`;


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
      emotionsNeutral = `${(emotions[0]).toFixed(3)}`;
      // console.log('emotionNeutral: ', emotionsNeutral);
      // dataNeutral = `${(emotions[0]).toFixed(3)}`;
      // const combine = emotionsNeutral.concat(dataNeutral);
      // console.log('Combine data: ' ,combine);
      // dataNeutral = [`${(emotions[0]).toFixed(3)}`];
      emotionsHappy = `${(emotions[1]).toFixed(3)}`;
      // console.log('happy: ', emotionsHappy);
      emotionsSad = `${(emotions[2]).toFixed(3)}`
      emotionsAngry = `${(emotions[3]).toFixed(3)}`;
      // console.log('angry: ',emotionsAngry)
      emotionsSurprise = `${(emotions[4]).toFixed(3)}`

      // arrayEmotions = Array.from(emotions);
      // console.log(arrayEmotions);
    }
  }
  stats.end();
  requestAnimationFrame(renderPrediction_Webcam);
}

async function showWebcam() {
  await myChart.destroy();
  
  document.getElementById('video').style.display = 'block';
  document.getElementById('uploadImg').style.display = 'none';
  document.getElementById('movie').style.display = 'none';
  await tf.setBackend(state.backend);
  await setupCamera();
  myChartEmotions_Webcam();
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

  renderPrediction_Webcam();
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

  document.getElementById('btn_all').style.color = 'rgb(17 24 39)';
  document.getElementById('btn_all').style.borderColor = 'rgb(243 244 246)';

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

  document.getElementById('block_positive').style.display = 'flex',
  document.getElementById('block_active').style.display = 'flex',
  document.getElementById('block_positive').style.paddingLeft = '1em',
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

  document.getElementById('btn_all').style.color = 'rgb(17 24 39)';
  document.getElementById('btn_all').style.borderColor = 'rgb(243 244 246)';

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
  document.getElementById('block_vibe').style.paddingLeft = '5em',

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

  document.getElementById('btn_all').style.color = 'rgb(17 24 39)';
  document.getElementById('btn_all').style.borderColor = 'rgb(243 244 246)';

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

async function showAll() {

  document.getElementById('block_positive').style.display = 'flex',
  document.getElementById('block_active').style.display = 'flex',
  document.getElementById('block_vibe').style.display = 'flex',

  document.getElementById('block_vibe').style.paddingLeft = '0px',
  document.getElementById('block_positive').style.paddingLeft = '0px',
  document.getElementById('block_active').style.paddingLeft = 'px',


  document.getElementById('block_neutral').style.display = 'flex',
  document.getElementById('block_angry').style.display = 'flex',
  document.getElementById('block_happy').style.display = 'flex',
  document.getElementById('block_sad').style.display = 'flex',
  document.getElementById('block_supprise').style.display = 'flex',

  document.getElementById('btn_all').style.color = 'rgba(24, 144, 255, 1)';
  document.getElementById('btn_all').style.borderColor = 'rgba(24, 144, 255, 1)';

  document.getElementById('btn_emotions').style.color = 'rgb(17 24 39)';
  document.getElementById('btn_emotions').style.borderColor = 'rgb(243 244 246)';

  document.getElementById('btn_positive-active').style.color = 'rgb(17 24 39)';
  document.getElementById('btn_positive-active').style.borderColor = 'rgb(243 244 246)';

  document.getElementById('btn_vibes').style.color = 'rgb(17 24 39)';
  document.getElementById('btn_vibes').style.borderColor = 'rgb(243 244 246)';

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

  document.getElementById('border_positive-active').style.borderStyle = 'solid';

  document.getElementById('label_positive').style.display = 'flex';
  document.getElementById('bar_positive').style.display = 'flex';
  document.getElementById('label_active').style.display = 'flex';
  document.getElementById('bar_active').style.display = 'flex';

  document.getElementById('border_vibes').style.borderStyle = 'solid';

  document.getElementById('label_vibes').style.display = 'flex';
  document.getElementById('bar_vibes').style.display = 'flex';

  myChart.getDatasetMeta(0).hidden = false;
  myChart.getDatasetMeta(1).hidden = false;
  myChart.getDatasetMeta(2).hidden = false;
  myChart.getDatasetMeta(3).hidden = false;
  myChart.getDatasetMeta(4).hidden = false;

  myChart.getDatasetMeta(5).hidden = false;
  myChart.getDatasetMeta(6).hidden = false;

  myChart.getDatasetMeta(7).hidden = false;
}


const myChartEmotions = async() => {
  // const numberCells = ['0', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '1.0'];
    numberCells = Array.from({length: 500}, (_, i) => i + 1);
    // debugger;
    const ctx = document.getElementById('myChart');
    // const valueEmotions_Neutral = localStorage.getItem('dataNeutral');
    // const valueEmotions_Angry = localStorage.getItem('dataAngry');
    // const valueEmotions_Sad = localStorage.getItem('dataSad');
    // const valueEmotions_Surprised = localStorage.getItem('dataSurprise');
    // const valueEmotions_Happy = localStorage.getItem('dataHappy');
      myChart = new Chart(ctx, {
          type: 'line',
          data: {
              labels: numberCells,
              datasets: [
                {
                  label: 'ANGRY',
                  data: [], //valueEmotions_Angry.split(','),
                  fill: false,
                  tension: 0.4,
                  backgroundColor: [
                      'rgba(255, 77, 79, 1)',
                  ],
                  borderColor: [
                      'rgba(255, 77, 79, 1)',
                  ],
                  borderWidth: 2,
            },
            {
              label: 'NEUTRAL',
              data: [], //valueEmotions_Neutral.split(','),
              fill: false,
              tension: 0.4,
              backgroundColor: [
                'rgba(186, 231, 255, 1)',
              ],
              borderColor: [
                'rgba(186, 231, 255, 1)',
              ],
              borderWidth: 2,
            },
            {
              label: 'HAPPY',
              data: [], //valueEmotions_Happy.split(','),
              fill: false,
              backgroundColor: [
                'rgba(24, 144, 255, 1)',
              ],
              borderColor: [
                'rgba(24, 144, 255, 1)',
              ],
              borderWidth: 2,
            },
            {
              label: 'SAD',
              data: [], //valueEmotions_Sad.split(','),
              fill: false,
              backgroundColor: [
                'rgba(89, 126, 247, 1)',
              ],
              borderColor: [
                'rgba(89, 126, 247, 1)',
              ],
              borderWidth: 2,
            },
            {
              label: 'SURPRISE',
                data: [],//valueEmotions_Surprised.split(','),
                fill: false,
                backgroundColor: [
                  'rgba(255, 197, 61, 1)',
                ],
                borderColor: [
                  'rgba(255, 197, 61, 1)',
                ],
                borderWidth: 2,
              },
              {
                label: 'POSITIVE',
                  data: [],//valueEmotions_Surprised.split(','),
                  fill: false,
                  backgroundColor: [
                    'rgba(67, 67, 67, 1)',
                  ],
                  borderColor: [
                    'rgba(67, 67, 61, 1)',
                  ],
                  borderWidth: 2,
                },
                {
                  label: 'ACTIVE',
                    data: [],//valueEmotions_Surprised.split(','),
                    fill: false,
                    backgroundColor: [
                      'rgba(115, 209, 61, 1)',
                    ],
                    borderColor: [
                      'rgba(115, 209, 61, 1)',
                    ],
                    borderWidth: 2,
                  },
                  {
                    label: 'VIBE',
                      data: [],//valueEmotions_Surprised.split(','),
                      fill: false,
                      backgroundColor: [
                        'rgba(247, 89, 171, 1)',
                      ],
                      borderColor: [
                        'rgba(247, 89, 171, 1)',
                      ],
                      borderWidth: 2,
                    },
            ],
          },
          options: {
              scales: {
                yAxes: [{
                  ticks: {
                    beginAtZero: true,
                    min: 0,
                    max: 1,
                    },
                  }],
                xAxes: [{
                  ticks: {
                    beginAtZero: true,
                    // min: 0,
                    // max: 1,
                  },
                }],
              },
              backgroundColor: 'pink',
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
}

const myChartEmotions_Img = async() => {
  // const numberCells = ['0', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '1.0'];
    numberCells = Array.from({length: 500}, (_, i) => i + 1);
    // debugger;
    const ctx = document.getElementById('myChart').getContext('2d');
    // const valueEmotions_Neutral = localStorage.getItem('dataNeutral');
    // const valueEmotions_Angry = localStorage.getItem('dataAngry');
    // const valueEmotions_Sad = localStorage.getItem('dataSad');
    // const valueEmotions_Surprised = localStorage.getItem('dataSurprise');
    // const valueEmotions_Happy = localStorage.getItem('dataHappy');
      myChart = new Chart(ctx, {
          type: 'line',
          data: {
              labels: numberCells,
              datasets: [
                {
                  label: 'ANGRY',
                  data: [], //valueEmotions_Angry.split(','),
                  fill: false,
                  tension: 0.4,
                  backgroundColor: [
                      'rgba(255, 77, 79, 1)',
                  ],
                  borderColor: [
                      'rgba(255, 77, 79, 1)',
                  ],
                  borderWidth: 2,
            },
            {
              label: 'NEUTRAL',
              data: [], //valueEmotions_Neutral.split(','),
              fill: false,
              tension: 0.4,
              backgroundColor: [
                'rgba(186, 231, 255, 1)',
              ],
              borderColor: [
                'rgba(186, 231, 255, 1)',
              ],
              borderWidth: 2,
            },
            {
              label: 'HAPPY',
              data: [], //valueEmotions_Happy.split(','),
              fill: false,
              backgroundColor: [
                'rgba(24, 144, 255, 1)',
              ],
              borderColor: [
                'rgba(24, 144, 255, 1)',
              ],
              borderWidth: 2,
            },
            {
              label: 'SAD',
              data: [], //valueEmotions_Sad.split(','),
              fill: false,
              backgroundColor: [
                'rgba(89, 126, 247, 1)',
              ],
              borderColor: [
                'rgba(89, 126, 247, 1)',
              ],
              borderWidth: 2,
            },
            {
              label: 'SURPRISE',
                data: [],//valueEmotions_Surprised.split(','),
                fill: false,
                backgroundColor: [
                  'rgba(255, 197, 61, 1)',
                ],
                borderColor: [
                  'rgba(255, 197, 61, 1)',
                ],
                borderWidth: 2,
              },
              {
                label: 'POSITIVE',
                  data: [],//valueEmotions_Surprised.split(','),
                  fill: false,
                  backgroundColor: [
                    'rgba(67, 67, 67, 1)',
                  ],
                  borderColor: [
                    'rgba(67, 67, 61, 1)',
                  ],
                  borderWidth: 2,
                },
                {
                  label: 'ACTIVE',
                    data: [],//valueEmotions_Surprised.split(','),
                    fill: false,
                    backgroundColor: [
                      'rgba(115, 209, 61, 1)',
                    ],
                    borderColor: [
                      'rgba(115, 209, 61, 1)',
                    ],
                    borderWidth: 2,
                  },
                  {
                    label: 'VIBE',
                      data: [],//valueEmotions_Surprised.split(','),
                      fill: false,
                      backgroundColor: [
                        'rgba(247, 89, 171, 1)',
                      ],
                      borderColor: [
                        'rgba(247, 89, 171, 1)',
                      ],
                      borderWidth: 2,
                    },
            ],
          },
          options: {
              interaction: {
                intersect: false,
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
                  ticks: {
                    beginAtZero: true,
                    // min: 0,
                    // max: 1,
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
}

const myChartEmotions_Clip = async() => {
  // const numberCells = ['0', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '1.0'];
    numberCells = Array.from({length: 500}, (_, i) => i + 1);
    // debugger;
    const ctx = document.getElementById('myChart').getContext('2d');
    // const valueEmotions_Neutral = localStorage.getItem('dataNeutral');
    // const valueEmotions_Angry = localStorage.getItem('dataAngry');
    // const valueEmotions_Sad = localStorage.getItem('dataSad');
    // const valueEmotions_Surprised = localStorage.getItem('dataSurprise');
    // const valueEmotions_Happy = localStorage.getItem('dataHappy');
      myChart = new Chart(ctx, {
          type: 'line',
          data: {
              labels: numberCells,
              datasets: [
                {
                  label: 'ANGRY',
                  data: [], //valueEmotions_Angry.split(','),
                  fill: false,
                  tension: 0.4,
                  backgroundColor: [
                      'rgba(255, 77, 79, 1)',
                  ],
                  borderColor: [
                      'rgba(255, 77, 79, 1)',
                  ],
                  borderWidth: 2,
            },
            {
              label: 'NEUTRAL',
              data: [], //valueEmotions_Neutral.split(','),
              fill: false,
              tension: 0.4,
              backgroundColor: [
                'rgba(186, 231, 255, 1)',
              ],
              borderColor: [
                'rgba(186, 231, 255, 1)',
              ],
              borderWidth: 2,
            },
            {
              label: 'HAPPY',
              data: [], //valueEmotions_Happy.split(','),
              fill: false,
              backgroundColor: [
                'rgba(24, 144, 255, 1)',
              ],
              borderColor: [
                'rgba(24, 144, 255, 1)',
              ],
              borderWidth: 2,
            },
            {
              label: 'SAD',
              data: [], //valueEmotions_Sad.split(','),
              fill: false,
              backgroundColor: [
                'rgba(89, 126, 247, 1)',
              ],
              borderColor: [
                'rgba(89, 126, 247, 1)',
              ],
              borderWidth: 2,
            },
            {
              label: 'SURPRISE',
                data: [],//valueEmotions_Surprised.split(','),
                fill: false,
                backgroundColor: [
                  'rgba(255, 197, 61, 1)',
                ],
                borderColor: [
                  'rgba(255, 197, 61, 1)',
                ],
                borderWidth: 2,
              },
              {
                label: 'POSITIVE',
                  data: [],//valueEmotions_Surprised.split(','),
                  fill: false,
                  backgroundColor: [
                    'rgba(67, 67, 67, 1)',
                  ],
                  borderColor: [
                    'rgba(67, 67, 61, 1)',
                  ],
                  borderWidth: 2,
                },
                {
                  label: 'ACTIVE',
                    data: [],//valueEmotions_Surprised.split(','),
                    fill: false,
                    backgroundColor: [
                      'rgba(115, 209, 61, 1)',
                    ],
                    borderColor: [
                      'rgba(115, 209, 61, 1)',
                    ],
                    borderWidth: 2,
                  },
                  {
                    label: 'VIBE',
                      data: [],//valueEmotions_Surprised.split(','),
                      fill: false,
                      backgroundColor: [
                        'rgba(247, 89, 171, 1)',
                      ],
                      borderColor: [
                        'rgba(247, 89, 171, 1)',
                      ],
                      borderWidth: 2,
                    },
            ],
          },
          options: {
              interaction: {
                intersect: false,
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
                  ticks: {
                    beginAtZero: true,
                    // min: 0,
                    // max: 1,
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
}

const myChartEmotions_Webcam = async() => {
  // const numberCells = ['0', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '1.0'];
    numberCells = Array.from({length: 500}, (_, i) => i + 1);
    // debugger;
    const ctx = document.getElementById('myChart').getContext('2d');
    // const valueEmotions_Neutral = localStorage.getItem('dataNeutral');
    // const valueEmotions_Angry = localStorage.getItem('dataAngry');
    // const valueEmotions_Sad = localStorage.getItem('dataSad');
    // const valueEmotions_Surprised = localStorage.getItem('dataSurprise');
    // const valueEmotions_Happy = localStorage.getItem('dataHappy');
      myChart = new Chart(ctx, {
          type: 'line',
          data: {
              labels: numberCells,
              datasets: [
                {
                  label: 'ANGRY',
                  data: [], //valueEmotions_Angry.split(','),
                  fill: false,
                  tension: 0.4,
                  backgroundColor: [
                      'rgba(255, 77, 79, 1)',
                  ],
                  borderColor: [
                      'rgba(255, 77, 79, 1)',
                  ],
                  borderWidth: 2,
            },
            {
              label: 'NEUTRAL',
              data: [], //valueEmotions_Neutral.split(','),
              fill: false,
              tension: 0.4,
              backgroundColor: [
                'rgba(186, 231, 255, 1)',
              ],
              borderColor: [
                'rgba(186, 231, 255, 1)',
              ],
              borderWidth: 2,
            },
            {
              label: 'HAPPY',
              data: [], //valueEmotions_Happy.split(','),
              fill: false,
              backgroundColor: [
                'rgba(24, 144, 255, 1)',
              ],
              borderColor: [
                'rgba(24, 144, 255, 1)',
              ],
              borderWidth: 2,
            },
            {
              label: 'SAD',
              data: [], //valueEmotions_Sad.split(','),
              fill: false,
              backgroundColor: [
                'rgba(89, 126, 247, 1)',
              ],
              borderColor: [
                'rgba(89, 126, 247, 1)',
              ],
              borderWidth: 2,
            },
            {
              label: 'SURPRISE',
                data: [],//valueEmotions_Surprised.split(','),
                fill: false,
                backgroundColor: [
                  'rgba(255, 197, 61, 1)',
                ],
                borderColor: [
                  'rgba(255, 197, 61, 1)',
                ],
                borderWidth: 2,
              },
              {
                label: 'POSITIVE',
                  data: [],//valueEmotions_Surprised.split(','),
                  fill: false,
                  backgroundColor: [
                    'rgba(67, 67, 67, 1)',
                  ],
                  borderColor: [
                    'rgba(67, 67, 61, 1)',
                  ],
                  borderWidth: 2,
                },
                {
                  label: 'ACTIVE',
                    data: [],//valueEmotions_Surprised.split(','),
                    fill: false,
                    backgroundColor: [
                      'rgba(115, 209, 61, 1)',
                    ],
                    borderColor: [
                      'rgba(115, 209, 61, 1)',
                    ],
                    borderWidth: 2,
                  },
                  {
                    label: 'VIBE',
                      data: [],//valueEmotions_Surprised.split(','),
                      fill: false,
                      backgroundColor: [
                        'rgba(247, 89, 171, 1)',
                      ],
                      borderColor: [
                        'rgba(247, 89, 171, 1)',
                      ],
                      borderWidth: 2,
                    },
            ],
          },
          options: {
              interaction: {
                intersect: false,
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
                  ticks: {
                    beginAtZero: true,
                    // min: 0,
                    // max: 1,
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
}

// function hideData() {
//   myChart.data.datasets.forEach(function(ds) {
//     console.log(ds);
//     ds.hidden = !ds.hidden;
//     //ds._meta[0].hidden = !ds._meta[0].hidden
//     //ds._meta[0].dataset.hidden = !ds._meta[0].dataset.hidden
//   });
//   myChart.update();
// }

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

function hideDataVibe () {
  myChart.getDatasetMeta(0).hidden = true;
  myChart.getDatasetMeta(1).hidden = true;
  myChart.getDatasetMeta(2).hidden = true;
  myChart.getDatasetMeta(3).hidden = true;
  myChart.getDatasetMeta(4).hidden = true;

  myChart.getDatasetMeta(5).hidden = true;
  myChart.getDatasetMeta(6).hidden = true;

  myChart.getDatasetMeta(7).hidden = false;
}
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

  myChartEmotions();
  renderPrediction();
  //chartEmotions();
  // myChartEmotions();
  document.getElementById('btnPicture').addEventListener('click', showImg);
  document.getElementById('btnWebcam').addEventListener('click', showWebcam);
  document.getElementById('btnVideoClip').addEventListener('click', showVideoClip);
  document.getElementById('btn_emotions').addEventListener('click', showEmotions);
  document.getElementById('btn_positive-active').addEventListener('click', showPositiveActive);
  document.getElementById('btn_vibes').addEventListener('click', showVibes);
  document.getElementById('btn_all').addEventListener('click', showAll);
  document.getElementById('btn_emotions').addEventListener('click', hideDataEmotions);
  document.getElementById('btn_positive-active').addEventListener('click', hideDataPositive);
  document.getElementById('btn_vibes').addEventListener('click', hideDataVibe);

}
setupPage();
