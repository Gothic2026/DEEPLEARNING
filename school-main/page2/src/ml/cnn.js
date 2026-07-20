/**
 * CNN + DNN - 합성곱, 풀링, 완전연결층 정확 구현
 */

function relu(x) {
  return Math.max(0, x);
}

function softmax(arr) {
  const max = Math.max(...arr);
  const exps = arr.map((v) => Math.exp(v - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((e) => e / sum);
}

export function conv2d(input, kernel, bias = 0) {
  const inH = input.length;
  const inW = input[0].length;
  const kH = kernel.length;
  const kW = kernel[0].length;
  const outH = inH - kH + 1;
  const outW = inW - kW + 1;
  const output = Array.from({ length: outH }, () => Array(outW).fill(0));
  for (let i = 0; i < outH; i++) {
    for (let j = 0; j < outW; j++) {
      let sum = bias;
      for (let ki = 0; ki < kH; ki++) {
        for (let kj = 0; kj < kW; kj++) {
          sum += input[i + ki][j + kj] * kernel[ki][kj];
        }
      }
      output[i][j] = sum;
    }
  }
  return output;
}

export function conv2dMultiChannel(input, filters) {
  return filters.map((f) => conv2d(input, f.kernel, f.bias));
}

export function maxPool2d(input, poolSize = 2) {
  const inH = input.length;
  const inW = input[0].length;
  const outH = Math.floor(inH / poolSize);
  const outW = Math.floor(inW / poolSize);
  const output = Array.from({ length: outH }, () => Array(outW).fill(0));
  for (let i = 0; i < outH; i++) {
    for (let j = 0; j < outW; j++) {
      let maxVal = -Infinity;
      for (let pi = 0; pi < poolSize; pi++) {
        for (let pj = 0; pj < poolSize; pj++) {
          maxVal = Math.max(maxVal, input[i * poolSize + pi][j * poolSize + pj]);
        }
      }
      output[i][j] = maxVal;
    }
  }
  return output;
}

export function applyActivation(featureMaps, fn = relu) {
  return featureMaps.map((fm) => fm.map((row) => row.map(fn)));
}

export function flatten(featureMaps) {
  const flat = [];
  for (const fm of featureMaps) {
    for (const row of fm) {
      for (const v of row) flat.push(v);
    }
  }
  return flat;
}

export function dense(input, weights, biases) {
  return weights.map((row, i) => row.reduce((s, w, j) => s + w * input[j], 0) + biases[i]);
}

export class DigitCNN {
  constructor(weights) {
    this.w = weights;
  }

  forward(image28x28, captureSteps = false) {
    const steps = [];
    let x = image28x28;

    if (captureSteps) steps.push({ type: 'input', data: x.map((r) => r.slice()), label: '입력 (28×28)' });

    // Conv1: 8 filters 5x5
    let conv1 = conv2dMultiChannel(x, this.w.conv1);
    if (captureSteps) steps.push({ type: 'conv', data: conv1, label: '합성곱 1 (8개 필터, 5×5)' });

    conv1 = applyActivation(conv1);
    if (captureSteps) steps.push({ type: 'relu', data: conv1, label: 'ReLU 활성화' });

    let pool1 = conv1.map((fm) => maxPool2d(fm, 2));
    if (captureSteps) steps.push({ type: 'pool', data: pool1, label: '최대 풀링 1 (2×2)' });

    // Conv2: 16 filters 5x5 on each... simplified: apply to each channel separately and stack
    // Our architecture: conv2 takes 8-channel input as separate convolutions per filter set
    let conv2 = [];
    for (let f = 0; f < this.w.conv2.length; f++) {
      let sum = null;
      for (let c = 0; c < pool1.length; c++) {
        const partial = conv2d(pool1[c], this.w.conv2[f].kernels[c], 0);
        if (!sum) sum = partial;
        else {
          for (let i = 0; i < sum.length; i++)
            for (let j = 0; j < sum[0].length; j++) sum[i][j] += partial[i][j];
        }
      }
      conv2.push(sum.map((row) => row.map((v) => v + this.w.conv2[f].bias)));
    }
    if (captureSteps) steps.push({ type: 'conv', data: conv2, label: '합성곱 2 (16개 필터, 5×5)' });

    conv2 = applyActivation(conv2);
    if (captureSteps) steps.push({ type: 'relu', data: conv2, label: 'ReLU 활성화' });

    let pool2 = conv2.map((fm) => maxPool2d(fm, 2));
    if (captureSteps) steps.push({ type: 'pool', data: pool2, label: '최대 풀링 2 (2×2)' });

    const flat = flatten(pool2);
    if (captureSteps) steps.push({ type: 'flatten', data: flat, label: `평탄화 (${flat.length}차원)` });

    let hidden = dense(flat, this.w.fc1.weights, this.w.fc1.biases);
    hidden = hidden.map(relu);
    if (captureSteps) steps.push({ type: 'dense', data: hidden, label: '완전연결층 1 (64)' });

    let logits = dense(hidden, this.w.fc2.weights, this.w.fc2.biases);
    const probs = softmax(logits);
    if (captureSteps) steps.push({ type: 'output', data: probs, label: '출력 (0~9 확률)' });

    return { probabilities: probs, steps, logits, hidden };
  }

  predict(image28x28) {
    return this.forward(image28x28, false);
  }
}

export function centerImage(grid) {
  let minX = 28, maxX = 0, minY = 28, maxY = 0;
  for (let y = 0; y < 28; y++)
    for (let x = 0; x < 28; x++)
      if (grid[y][x] > 0.05) {
        minX = Math.min(minX, x);
        maxX = Math.max(maxX, x);
        minY = Math.min(minY, y);
        maxY = Math.max(maxY, y);
      }
  if (minX > maxX) return grid;
  const bw = maxX - minX + 1;
  const bh = maxY - minY + 1;
  const size = Math.max(bw, bh);
  const scale = 20 / size;
  const cx = (minX + maxX) / 2;
  const cy = (minY + maxY) / 2;
  const centered = Array.from({ length: 28 }, () => Array(28).fill(0));
  for (let y = 0; y < 28; y++)
    for (let x = 0; x < 28; x++) {
      const sx = Math.round((x - 14) / scale + cx);
      const sy = Math.round((y - 14) / scale + cy);
      if (sx >= 0 && sx < 28 && sy >= 0 && sy < 28) centered[y][x] = grid[sy][sx];
    }
  return centered;
}

export function normalizeDrawing(raw28x28) {
  const centered = centerImage(raw28x28);
  let maxVal = 0;
  for (const row of centered)
    for (const v of row) maxVal = Math.max(maxVal, v);
  if (maxVal === 0) return centered;
  const scale = 1 / maxVal;
  return centered.map((row) => row.map((v) => v * scale));
}

export function resizeTo28(canvasData, srcW, srcH) {
  const offscreen = document.createElement('canvas');
  offscreen.width = 28;
  offscreen.height = 28;
  const ctx = offscreen.getContext('2d');
  const temp = document.createElement('canvas');
  temp.width = srcW;
  temp.height = srcH;
  const tctx = temp.getContext('2d');
  const imgData = tctx.createImageData(srcW, srcH);
  for (let i = 0; i < canvasData.length; i++) {
    const v = Math.floor(canvasData[i] * 255);
    imgData.data[i * 4] = v;
    imgData.data[i * 4 + 1] = v;
    imgData.data[i * 4 + 2] = v;
    imgData.data[i * 4 + 3] = 255;
  }
  tctx.putImageData(imgData, 0, 0);
  ctx.fillStyle = '#000';
  ctx.fillRect(0, 0, 28, 28);
  ctx.drawImage(temp, 0, 0, 28, 28);
  const data = ctx.getImageData(0, 0, 28, 28).data;
  const grid = Array.from({ length: 28 }, () => Array(28).fill(0));
  for (let y = 0; y < 28; y++) {
    for (let x = 0; x < 28; x++) {
      grid[y][x] = data[(y * 28 + x) * 4] / 255;
    }
  }
  return normalizeDrawing(grid);
}

export function renderFeatureMapsToCanvas(featureMaps, canvas, maxMaps = 8, highlightIdx = -1) {
  if (!canvas) return;
  const parent = canvas.parentElement;
  const cw = parent?.clientWidth || canvas.width;
  const ch = parent?.clientHeight || canvas.height;
  const ctx = canvas.getContext('2d');
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  canvas.width = Math.floor(cw * dpr);
  canvas.height = Math.floor(ch * dpr);
  canvas.style.width = '100%';
  canvas.style.height = '100%';
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

  const maps = featureMaps.slice(0, maxMaps);
  const n = maps.length;
  const cols = Math.ceil(Math.sqrt(n));
  const rows = Math.ceil(n / cols);
  const cellW = Math.floor(cw / cols);
  const cellH = Math.floor(ch / rows);

  ctx.fillStyle = '#0a0a12';
  ctx.fillRect(0, 0, cw, ch);

  maps.forEach((fm, idx) => {
    const col = idx % cols;
    const row = Math.floor(idx / cols);
    const h = fm.length;
    const w = fm[0].length;
    let maxV = 0.001;
    for (const r of fm) for (const v of r) maxV = Math.max(maxV, Math.abs(v));
    const img = ctx.createImageData(w, h);
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const v = Math.floor((fm[y][x] / maxV) * 255);
        const i = (y * w + x) * 4;
        img.data[i] = v;
        img.data[i + 1] = v * 0.8;
        img.data[i + 2] = v * 0.5;
        img.data[i + 3] = 255;
      }
    }
    const off = document.createElement('canvas');
    off.width = w;
    off.height = h;
    off.getContext('2d').putImageData(img, 0, 0);
    const dx = col * cellW + 2;
    const dy = row * cellH + 2;
    const dw = cellW - 4;
    const dh = cellH - 4;
    ctx.drawImage(off, dx, dy, dw, dh);
    if (idx === highlightIdx) {
      ctx.strokeStyle = '#58a6ff';
      ctx.lineWidth = 2;
      ctx.strokeRect(dx, dy, dw, dh);
    }
  });
}

export function renderConvAnimation(input, kernel, canvas, step = 0) {
  if (!canvas) return;
  const parent = canvas.parentElement;
  const W = parent?.clientWidth || canvas.width;
  const H = parent?.clientHeight || canvas.height;
  const ctx = canvas.getContext('2d');
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  canvas.width = Math.floor(W * dpr);
  canvas.height = Math.floor(H * dpr);
  canvas.style.width = '100%';
  canvas.style.height = '100%';
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

  ctx.fillStyle = '#0a0a12';
  ctx.fillRect(0, 0, W, H);

  const inH = input.length;
  const inW = input[0].length;
  const kH = kernel.length;
  const kW = kernel[0].length;
  const outH = inH - kH + 1;
  const outW = inW - kW + 1;
  const totalPos = outH * outW;
  const pos = step % totalPos;
  const pi = Math.floor(pos / outW);
  const pj = pos % outW;

  const scale = Math.min((W - 60) / inW, (H - 30) / inH);
  const ox = 20;
  const oy = 10;

  let maxIn = 0.001;
  for (const r of input) for (const v of r) maxIn = Math.max(maxIn, v);
  for (let y = 0; y < inH; y++)
    for (let x = 0; x < inW; x++) {
      const v = input[y][x] / maxIn;
      ctx.fillStyle = `rgb(${Math.floor(v * 200)},${Math.floor(v * 200)},${Math.floor(v * 220)})`;
      ctx.fillRect(ox + x * scale, oy + y * scale, scale, scale);
    }

  ctx.strokeStyle = '#58a6ff';
  ctx.lineWidth = 2;
  ctx.strokeRect(ox + pj * scale, oy + pi * scale, kW * scale, kH * scale);

  let sum = 0;
  for (let ki = 0; ki < kH; ki++)
    for (let kj = 0; kj < kW; kj++) {
      sum += input[pi + ki][pj + kj] * kernel[ki][kj];
      const v = kernel[ki][kj];
      ctx.fillStyle = v >= 0 ? 'rgba(88,166,255,0.7)' : 'rgba(248,81,73,0.7)';
      ctx.fillRect(ox + (pj + kj) * scale + 1, oy + (pi + ki) * scale + 1, scale - 2, scale - 2);
    }

  ctx.fillStyle = '#e6edf3';
  ctx.font = '11px monospace';
  ctx.fillText(`합성곱 위치 (${pi},${pj}) → Σ=${sum.toFixed(3)}`, ox, H - 8);
}

export function renderVectorBars(values, canvas, labels) {
  if (!canvas) return;
  const parent = canvas.parentElement;
  const w = parent?.clientWidth || canvas.width;
  const h = parent?.clientHeight || canvas.height;
  const ctx = canvas.getContext('2d');
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  canvas.width = Math.floor(w * dpr);
  canvas.height = Math.floor(h * dpr);
  canvas.style.width = '100%';
  canvas.style.height = '100%';
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.fillStyle = '#0a0a12';
  ctx.fillRect(0, 0, w, h);
  const barW = w / values.length - 3;
  const maxV = Math.max(...values, 0.001);
  values.forEach((v, i) => {
    const barH = (v / maxV) * (h - 22);
    const x = i * (barW + 3) + 1;
    ctx.fillStyle = `hsl(${200 + v * 60}, 70%, 55%)`;
    ctx.fillRect(x, h - barH - 16, barW, barH);
    ctx.fillStyle = '#aaa';
    ctx.font = '9px monospace';
    ctx.textAlign = 'center';
    ctx.fillText(labels ? labels[i] : i, x + barW / 2, h - 3);
  });
}
