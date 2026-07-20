/**
 * MNIST 스타일 합성 데이터로 CNN 학습 후 가중치 JSON 생성
 */
import { writeFileSync, mkdirSync } from 'fs';
import { dirname, join } from 'path';
import { fileURLToPath } from 'url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const FLAT_SIZE = 16 * 4 * 4; // pool2: 16 filters × 4×4

function mulberry32(a) {
  return function () {
    let t = (a += 0x6d2b79f5);
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function relu(x) { return Math.max(0, x); }
function reluDeriv(x) { return x > 0 ? 1 : 0; }

function conv2d(input, kernel, bias = 0) {
  const inH = input.length, inW = input[0].length;
  const kH = kernel.length, kW = kernel[0].length;
  const outH = inH - kH + 1, outW = inW - kW + 1;
  const output = Array.from({ length: outH }, () => Array(outW).fill(0));
  for (let i = 0; i < outH; i++)
    for (let j = 0; j < outW; j++) {
      let sum = bias;
      for (let ki = 0; ki < kH; ki++)
        for (let kj = 0; kj < kW; kj++)
          sum += input[i + ki][j + kj] * kernel[ki][kj];
      output[i][j] = sum;
    }
  return output;
}

function maxPool2d(input, ps = 2) {
  const outH = Math.floor(input.length / ps), outW = Math.floor(input[0].length / ps);
  const output = Array.from({ length: outH }, () => Array(outW).fill(0));
  for (let i = 0; i < outH; i++)
    for (let j = 0; j < outW; j++) {
      let max = -Infinity;
      for (let pi = 0; pi < ps; pi++)
        for (let pj = 0; pj < ps; pj++)
          max = Math.max(max, input[i * ps + pi][j * ps + pj]);
      output[i][j] = max;
    }
  return output;
}

function softmax(arr) {
  const max = Math.max(...arr);
  const exps = arr.map(v => Math.exp(v - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map(e => e / sum);
}

function generateDigit(digit, rng) {
  const img = Array.from({ length: 28 }, () => Array(28).fill(0));
  const cx = 14 + (rng() - 0.5) * 6;
  const cy = 14 + (rng() - 0.5) * 6;
  const stroke = (x, y, v = 1) => {
    for (let dy = -2; dy <= 2; dy++)
      for (let dx = -2; dx <= 2; dx++) {
        const nx = Math.round(x + dx), ny = Math.round(y + dy);
        if (nx >= 0 && nx < 28 && ny >= 0 && ny < 28) {
          const d = Math.sqrt(dx * dx + dy * dy);
          img[ny][nx] = Math.max(img[ny][nx], v * Math.max(0, 1 - d * 0.35));
        }
      }
  };
  const templates = {
    0: () => { for (let a = 0; a < Math.PI * 2; a += 0.12) stroke(cx + 8 * Math.cos(a), cy + 10 * Math.sin(a)); },
    1: () => { for (let y = 4; y < 24; y++) stroke(cx, y); for (let x = cx - 4; x < cx; x++) stroke(x, 6); },
    2: () => { for (let x = 6; x < 22; x++) { stroke(x, 6); stroke(x, 14); stroke(x, 22); } for (let y = 6; y < 14; y++) stroke(21, y); for (let y = 14; y < 22; y++) stroke(6, y); },
    3: () => { for (let x = 6; x < 22; x++) { stroke(x, 6); stroke(x, 14); stroke(x, 22); } for (let y = 6; y < 22; y++) stroke(21, y); },
    4: () => { for (let y = 4; y < 16; y++) stroke(6, y); for (let x = 6; x < 22; x++) stroke(x, 14); for (let y = 4; y < 22; y++) stroke(21, y); },
    5: () => { for (let x = 6; x < 22; x++) { stroke(x, 6); stroke(x, 14); stroke(x, 22); } for (let y = 6; y < 14; y++) stroke(6, y); for (let y = 14; y < 22; y++) stroke(21, y); },
    6: () => { for (let x = 6; x < 22; x++) { stroke(x, 6); stroke(x, 14); stroke(x, 22); } for (let y = 6; y < 22; y++) { stroke(6, y); stroke(21, y); } },
    7: () => { for (let x = 6; x < 22; x++) stroke(x, 6); for (let y = 6; y < 22; y++) stroke(21, y); },
    8: () => { for (let x = 6; x < 22; x++) { stroke(x, 6); stroke(x, 14); stroke(x, 22); } for (let y = 6; y < 22; y++) { stroke(6, y); stroke(21, y); } },
    9: () => { for (let x = 6; x < 22; x++) { stroke(x, 6); stroke(x, 14); stroke(x, 22); } for (let y = 6; y < 14; y++) stroke(21, y); for (let y = 6; y < 22; y++) stroke(6, y); },
  };
  templates[digit]();
  if (rng() > 0.4) {
    const scale = 0.85 + rng() * 0.3;
    const shifted = Array.from({ length: 28 }, () => Array(28).fill(0));
    for (let y = 0; y < 28; y++)
      for (let x = 0; x < 28; x++) {
        const sx = Math.round((x - 14) / scale + 14 + (rng() - 0.5) * 2);
        const sy = Math.round((y - 14) / scale + 14 + (rng() - 0.5) * 2);
        if (sx >= 0 && sx < 28 && sy >= 0 && sy < 28) shifted[sy][sx] = Math.max(shifted[sy][sx], img[y][x]);
      }
    return shifted;
  }
  return img;
}

function initFilters(n, kSize, rng) {
  return Array.from({ length: n }, () => ({
    kernel: Array.from({ length: kSize }, () => Array.from({ length: kSize }, () => (rng() - 0.5) * 0.2)),
    bias: (rng() - 0.5) * 0.1,
  }));
}

function initConv2Multi(nOut, nIn, kSize, rng) {
  return Array.from({ length: nOut }, () => ({
    kernels: Array.from({ length: nIn }, () =>
      Array.from({ length: kSize }, () => Array.from({ length: kSize }, () => (rng() - 0.5) * 0.1))
    ),
    bias: (rng() - 0.5) * 0.1,
  }));
}

function forwardPass(img, weights) {
  const cache = {};
  cache.input = img;
  cache.conv1 = weights.conv1.map(f => conv2d(img, f.kernel, f.bias));
  cache.relu1 = cache.conv1.map(fm => fm.map(r => r.map(relu)));
  cache.pool1 = cache.relu1.map(fm => maxPool2d(fm, 2));
  cache.conv2 = weights.conv2.map(f => {
    let sum = null;
    for (let c = 0; c < cache.pool1.length; c++) {
      const p = conv2d(cache.pool1[c], f.kernels[c], 0);
      if (!sum) sum = p.map(r => r.slice());
      else for (let i = 0; i < sum.length; i++) for (let j = 0; j < sum[0].length; j++) sum[i][j] += p[i][j];
    }
    return sum.map(r => r.map(v => v + f.bias));
  });
  cache.relu2 = cache.conv2.map(fm => fm.map(r => r.map(relu)));
  cache.pool2 = cache.relu2.map(fm => maxPool2d(fm, 2));
  const flat = [];
  for (const fm of cache.pool2) for (const r of fm) for (const v of r) flat.push(v);
  cache.flat = flat;
  cache.hidden = weights.fc1.weights.map((row, i) => relu(row.reduce((s, w, j) => s + w * flat[j], 0) + weights.fc1.biases[i]));
  cache.logits = weights.fc2.weights.map((row, i) => row.reduce((s, w, j) => s + w * cache.hidden[j], 0) + weights.fc2.biases[i]);
  cache.probs = softmax(cache.logits);
  return cache;
}

function trainFC(cache, weights, label, lr) {
  const fc1Size = weights.fc1.weights.length;
  const dLogits = cache.probs.slice();
  dLogits[label] -= 1;
  const dHidden = Array(fc1Size).fill(0);
  for (let i = 0; i < 10; i++)
    for (let j = 0; j < fc1Size; j++) {
      weights.fc2.weights[i][j] -= lr * dLogits[i] * cache.hidden[j];
      dHidden[j] += dLogits[i] * weights.fc2.weights[i][j];
    }
  for (let i = 0; i < 10; i++) weights.fc2.biases[i] -= lr * dLogits[i];
  const dFlat = Array(FLAT_SIZE).fill(0);
  for (let j = 0; j < fc1Size; j++) {
    const dh = dHidden[j] * reluDeriv(cache.hidden[j]);
    for (let k = 0; k < FLAT_SIZE; k++) {
      weights.fc1.weights[j][k] -= lr * dh * cache.flat[k];
      dFlat[k] += dh * weights.fc1.weights[j][k];
    }
    weights.fc1.biases[j] -= lr * dh;
  }
  return dFlat;
}

function train() {
  const rng = mulberry32(123);
  const conv1 = initFilters(8, 5, rng);
  const conv2 = initConv2Multi(16, 8, 5, rng);
  const fc1Size = 64;
  const fc1 = {
    weights: Array.from({ length: fc1Size }, () => Array.from({ length: FLAT_SIZE }, () => (rng() - 0.5) * 0.05)),
    biases: Array.from({ length: fc1Size }, () => 0),
  };
  const fc2 = {
    weights: Array.from({ length: 10 }, () => Array.from({ length: fc1Size }, () => (rng() - 0.5) * 0.05)),
    biases: Array.from({ length: 10 }, () => 0),
  };
  const weights = { conv1, conv2, fc1, fc2 };

  const dataset = [];
  for (let d = 0; d < 10; d++)
    for (let s = 0; s < 250; s++)
      dataset.push({ img: generateDigit(d, rng), label: d });

  for (let e = 0; e < 40; e++) {
    let correct = 0;
    for (const { img, label } of dataset) {
      const cache = forwardPass(img, weights);
      if (cache.probs.indexOf(Math.max(...cache.probs)) === label) correct++;
      trainFC(cache, weights, label, 0.015);
    }
    if ((e + 1) % 5 === 0) console.log(`Epoch ${e + 1}: accuracy ${(correct / dataset.length * 100).toFixed(1)}%`);
  }

  for (let e = 0; e < 20; e++) {
    let correct = 0;
    for (const { img, label } of dataset) {
      const cache = forwardPass(img, weights);
      if (cache.probs.indexOf(Math.max(...cache.probs)) === label) correct++;
      const dFlat = trainFC(cache, weights, label, 0.008);
      for (let f = 0; f < conv1.length; f++) {
        const signal = dFlat.slice(f * 16, (f + 1) * 16).reduce((a, b) => a + b, 0) / 16;
        for (let ki = 0; ki < 5; ki++)
          for (let kj = 0; kj < 5; kj++)
            conv1[f].kernel[ki][kj] -= 0.0003 * signal * (rng() - 0.5);
      }
    }
    if ((e + 1) % 5 === 0) console.log(`Conv fine-tune ${e + 1}: accuracy ${(correct / dataset.length * 100).toFixed(1)}%`);
  }

  return weights;
}

const weights = train();
const outPath = join(__dirname, '..', 'src', 'data', 'cnnWeights.json');
mkdirSync(dirname(outPath), { recursive: true });
writeFileSync(outPath, JSON.stringify(weights));
console.log('Saved to', outPath, '| fc1:', weights.fc1.weights[0].length, 'dims');
