/**
 * MFCC 및 FFT 기반 음성 특징 추출 (클라이언트 구현)
 */

export function computeFFTReal(signal) {
  const N = signal.length;
  const spectrum = new Float32Array(N / 2);
  for (let k = 0; k < N / 2; k++) {
    let re = 0;
    let im = 0;
    for (let n = 0; n < N; n++) {
      const angle = (-2 * Math.PI * k * n) / N;
      re += signal[n] * Math.cos(angle);
      im += signal[n] * Math.sin(angle);
    }
    spectrum[k] = Math.sqrt(re * re + im * im) / N;
  }
  return spectrum;
}

export function hzToMel(hz) {
  return 2595 * Math.log10(1 + hz / 700);
}

export function melToHz(mel) {
  return 700 * (Math.pow(10, mel / 2595) - 1);
}

export function createMelFilterbank(numFilters, fftSize, sampleRate) {
  const lowMel = hzToMel(0);
  const highMel = hzToMel(sampleRate / 2);
  const melPoints = Array.from({ length: numFilters + 2 }, (_, i) =>
    melToHz(lowMel + ((highMel - lowMel) * i) / (numFilters + 1))
  );
  const bin = melPoints.map((hz) => Math.floor(((fftSize + 1) * hz) / sampleRate));
  const filters = [];
  for (let m = 1; m <= numFilters; m++) {
    const filter = new Float32Array(fftSize / 2);
    for (let k = bin[m - 1]; k < bin[m]; k++) {
      if (k >= 0 && k < filter.length) filter[k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1]);
    }
    for (let k = bin[m]; k < bin[m + 1]; k++) {
      if (k >= 0 && k < filter.length) filter[k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m]);
    }
    filters.push(filter);
  }
  return filters;
}

const DCT_MATRIX_CACHE = {};

function dctMatrix(n) {
  if (DCT_MATRIX_CACHE[n]) return DCT_MATRIX_CACHE[n];
  const mat = Array.from({ length: n }, (_, i) =>
    Array.from({ length: n }, (_, j) => Math.cos(((Math.PI * i) / n) * (j + 0.5)))
  );
  DCT_MATRIX_CACHE[n] = mat;
  return mat;
}

export function computeMFCC(signal, sampleRate = 16000, numCoeffs = 13, numFilters = 26) {
  const frameSize = 512;
  const padded = signal.length >= frameSize ? signal.slice(0, frameSize) : [...signal, ...Array(frameSize - signal.length).fill(0)];
  const windowed = padded.map((v, i) => v * (0.54 - 0.46 * Math.cos((2 * Math.PI * i) / (frameSize - 1))));
  const spectrum = computeFFTReal(windowed);
  const melFilters = createMelFilterbank(numFilters, frameSize, sampleRate);
  const melEnergies = melFilters.map((f) => {
    let e = 0;
    for (let k = 0; k < f.length; k++) e += f[k] * spectrum[k] * spectrum[k];
    return Math.log(Math.max(e, 1e-10));
  });
  const dct = dctMatrix(numCoeffs);
  const mfcc = dct.map((row) => row.reduce((s, c, j) => s + c * melEnergies[j], 0));
  return mfcc;
}

// 한국어/영어 음소 매핑 (MFCC 기반 휴리스틱)
const PHONEME_CENTROIDS = {
  ㅏ: [12, -5, 3, 2, -1, 0, 1, 0, -1, 0, 0, 0, 0],
  ㅓ: [8, -3, 5, 1, -2, 1, 0, 0, 0, 0, 0, 0, 0],
  ㅗ: [10, -4, 2, 3, -1, 1, 0, 0, 0, 0, 0, 0, 0],
  ㅜ: [9, -2, 4, 2, -1, 0, 0, 0, 0, 0, 0, 0, 0],
  ㅣ: [6, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
  ㄱ: [-2, 8, -3, 2, -1, 0, 0, 0, 0, 0, 0, 0, 0],
  ㄴ: [-1, 6, -2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
  ㄷ: [-3, 7, -2, 2, -1, 0, 0, 0, 0, 0, 0, 0, 0],
  ㄹ: [-1, 5, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
  ㅁ: [0, 4, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
  ㅂ: [-2, 6, -2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
  ㅅ: [-4, 9, -4, 3, -1, 0, 0, 0, 0, 0, 0, 0, 0],
  ㅇ: [1, 3, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
  ㅈ: [-3, 8, -3, 2, -1, 0, 0, 0, 0, 0, 0, 0, 0],
  ㅎ: [-5, 10, -5, 3, -2, 0, 0, 0, 0, 0, 0, 0, 0],
  a: [11, -4, 2, 2, -1, 0, 0, 0, 0, 0, 0, 0, 0],
  e: [7, -2, 3, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
  i: [5, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
  o: [9, -3, 2, 2, -1, 0, 0, 0, 0, 0, 0, 0, 0],
  u: [8, -2, 3, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
  s: [-4, 8, -3, 2, -1, 0, 0, 0, 0, 0, 0, 0, 0],
  t: [-3, 7, -2, 2, -1, 0, 0, 0, 0, 0, 0, 0, 0],
  n: [-1, 5, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
  r: [0, 4, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
  l: [1, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
  m: [0, 4, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
  h: [-5, 9, -4, 3, -1, 0, 0, 0, 0, 0, 0, 0, 0],
  SIL: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
};

export function mfccToPhoneme(mfcc) {
  let best = 'SIL';
  let bestDist = Infinity;
  for (const [phoneme, centroid] of Object.entries(PHONEME_CENTROIDS)) {
    let dist = 0;
    for (let i = 0; i < Math.min(mfcc.length, centroid.length); i++) {
      dist += (mfcc[i] - centroid[i]) ** 2;
    }
    if (dist < bestDist) {
      bestDist = dist;
      best = phoneme;
    }
  }
  const confidence = Math.max(0, 1 - bestDist / 200);
  return { phoneme: best, confidence };
}

export function phonemesToSentence(phonemes, transcript) {
  if (transcript) return transcript;
  return phonemes.filter((p) => p !== 'SIL').join('');
}

export function renderSpectrum(spectrum, canvas) {
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
  const bars = Math.min(spectrum.length, w);
  const step = Math.max(1, Math.floor(spectrum.length / bars));
  const maxV = Math.max(...spectrum, 0.001);
  for (let i = 0; i < bars; i++) {
    const v = spectrum[i * step] / maxV;
    ctx.fillStyle = `hsl(${200 + v * 80}, 70%, 50%)`;
    ctx.fillRect(i, h - v * h, Math.max(1, w / bars), v * h);
  }
}

export function renderMFCC(mfcc, canvas) {
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
  const barW = w / mfcc.length - 1;
  const maxV = Math.max(...mfcc.map(Math.abs), 1);
  mfcc.forEach((v, i) => {
    const norm = v / maxV;
    const barH = Math.abs(norm) * (h / 2 - 6);
    const x = i * (barW + 1);
    ctx.fillStyle = v >= 0 ? '#4fc3f7' : '#ef5350';
    if (v >= 0) ctx.fillRect(x, h / 2 - barH, barW, barH);
    else ctx.fillRect(x, h / 2, barW, barH);
  });
  ctx.strokeStyle = '#444';
  ctx.beginPath();
  ctx.moveTo(0, h / 2);
  ctx.lineTo(w, h / 2);
  ctx.stroke();
}
