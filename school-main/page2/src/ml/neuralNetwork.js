/**
 * 다층 퍼셉트론 - 순전파 / 역전파 정확 구현
 * 활성화: sigmoid, 손실: MSE (XOR용)
 */
export class MLP {
  constructor(layerSizes, seed = 42) {
    this.layerSizes = layerSizes;
    this.rng = mulberry32(seed);
    this.weights = [];
    this.biases = [];
    for (let l = 1; l < layerSizes.length; l++) {
      const inSize = layerSizes[l - 1];
      const outSize = layerSizes[l];
      const scale = Math.sqrt(2 / inSize);
      this.weights.push(
        Array.from({ length: outSize }, () =>
          Array.from({ length: inSize }, () => (this.rng() * 2 - 1) * scale)
        )
      );
      this.biases.push(Array.from({ length: outSize }, () => (this.rng() * 2 - 1) * 0.1));
    }
    this.activations = [];
    this.zValues = [];
    this.lastDeltas = null;
    this.lastWeightUpdates = null;
    this.lastBiasUpdates = null;
  }

  sigmoid(x) {
    if (x > 20) return 1;
    if (x < -20) return 0;
    return 1 / (1 + Math.exp(-x));
  }

  sigmoidDerivativeFromActivation(a) {
    return a * (1 - a);
  }

  forward(inputs) {
    this.activations = [inputs.slice()];
    this.zValues = [null];
    for (let l = 0; l < this.weights.length; l++) {
      const W = this.weights[l];
      const b = this.biases[l];
      const prev = this.activations[l];
      const z = W.map((row, i) => row.reduce((s, w, j) => s + w * prev[j], 0) + b[i]);
      const a = z.map((v) => this.sigmoid(v));
      this.zValues.push(z);
      this.activations.push(a);
    }
    return this.activations[this.activations.length - 1];
  }

  backward(target, lr = 0.5) {
    const L = this.weights.length;
    const deltas = [];
    const output = this.activations[L];
    const outputDelta = output.map((a, i) => (a - target[i]) * this.sigmoidDerivativeFromActivation(a));
    deltas[L] = outputDelta;

    for (let l = L - 1; l >= 1; l--) {
      deltas[l] = this.activations[l].map((a, j) => {
        let sum = 0;
        for (let k = 0; k < this.weights[l].length; k++) {
          sum += this.weights[l][k][j] * deltas[l + 1][k];
        }
        return sum * this.sigmoidDerivativeFromActivation(a);
      });
    }

    const weightUpdates = this.weights.map((W, l) =>
      W.map((row, i) => row.map((_, j) => lr * deltas[l + 1][i] * this.activations[l][j]))
    );
    const biasUpdates = this.biases.map((b, l) => b.map((_, i) => lr * deltas[l + 1][i]));

    for (let l = 0; l < this.weights.length; l++) {
      for (let i = 0; i < this.weights[l].length; i++) {
        for (let j = 0; j < this.weights[l][i].length; j++) {
          this.weights[l][i][j] -= weightUpdates[l][i][j];
        }
        this.biases[l][i] -= biasUpdates[l][i];
      }
    }

    this.lastDeltas = deltas;
    this.lastWeightUpdates = weightUpdates;
    this.lastBiasUpdates = biasUpdates;
    return { weightUpdates, biasUpdates, deltas };
  }

  trainStep(inputs, target, lr = 0.5) {
    this.forward(inputs);
    return this.backward(target, lr);
  }

  train(dataset, epochs = 10000, lr = 0.5, tolerance = 0.04) {
    for (let e = 0; e < epochs; e++) {
      let maxError = 0;
      for (const { input, output } of dataset) {
        const out = this.forward(input);
        this.backward(output, lr);
        maxError = Math.max(maxError, Math.abs(out[0] - output[0]));
      }
      if (maxError < tolerance) return { epochs: e + 1, converged: true, maxError };
    }
    const last = dataset.map(({ input, output }) => {
      const o = this.forward(input);
      return Math.abs(o[0] - output[0]);
    });
    return { epochs, converged: false, maxError: Math.max(...last) };
  }

  predict(inputs) {
    return this.forward(inputs)[0];
  }

  cloneState() {
    return {
      weights: this.weights.map((l) => l.map((r) => r.slice())),
      biases: this.biases.map((b) => b.slice()),
    };
  }

  loadState(state) {
    this.weights = state.weights.map((l) => l.map((r) => r.slice()));
    this.biases = state.biases.map((b) => b.slice());
  }
}

export const XOR_DATASET = [
  { input: [0, 0], output: [0], label: '0 XOR 0 = 0' },
  { input: [0, 1], output: [1], label: '0 XOR 1 = 1' },
  { input: [1, 0], output: [1], label: '1 XOR 0 = 1' },
  { input: [1, 1], output: [0], label: '1 XOR 1 = 0' },
];

export function computeDecisionGrid(mlp, resolution = 40) {
  const grid = [];
  for (let yi = 0; yi <= resolution; yi++) {
    const row = [];
    for (let xi = 0; xi <= resolution; xi++) {
      const x = xi / resolution;
      const y = yi / resolution;
      row.push(mlp.predict([x, y]));
    }
    grid.push(row);
  }
  return grid;
}

function mulberry32(a) {
  return function () {
    let t = (a += 0x6d2b79f5);
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
