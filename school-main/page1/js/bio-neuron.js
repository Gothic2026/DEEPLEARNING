/** Biological neuron network simulation */

const NEURONS = [
  { id: 0, x: 80, y: 210, threshold: 0.4, inputs: [], fired: false, receiving: false, potential: 0 },
  { id: 1, x: 220, y: 120, threshold: 0.5, inputs: [0], fired: false, receiving: false, potential: 0 },
  { id: 2, x: 220, y: 300, threshold: 0.5, inputs: [0], fired: false, receiving: false, potential: 0 },
  { id: 3, x: 400, y: 180, threshold: 0.55, inputs: [1, 2], fired: false, receiving: false, potential: 0 },
  { id: 4, x: 560, y: 210, threshold: 0.5, inputs: [3], fired: false, receiving: false, potential: 0 },
];

const SYNAPSES = [
  { from: 0, to: 1, weight: 0.6 },
  { from: 0, to: 2, weight: 0.6 },
  { from: 1, to: 3, weight: 0.55 },
  { from: 2, to: 3, weight: 0.55 },
  { from: 3, to: 4, weight: 0.7 },
];

let pulses = [];
let animating = false;
let rafId = null;

export function initBioNeuron(canvas) {
  const stimulus = document.getElementById('bioStimulus');
  const stimulusVal = document.getElementById('bioStimulusVal');
  const fireBtn = document.getElementById('bioFireBtn');
  const resetBtn = document.getElementById('bioResetBtn');

  stimulus.addEventListener('input', () => {
    stimulusVal.textContent = stimulus.value;
  });

  fireBtn.addEventListener('click', () => {
    if (animating) return;
    const strength = parseInt(stimulus.value, 10) / 100;
    fireStimulus(strength);
  });

  resetBtn.addEventListener('click', reset);

  function loop() {
    draw(canvas);
    rafId = requestAnimationFrame(loop);
  }
  loop();

  return () => {
    if (rafId) cancelAnimationFrame(rafId);
  };
}

function reset() {
  NEURONS.forEach(n => {
    n.fired = false;
    n.receiving = false;
    n.potential = 0;
  });
  pulses = [];
  animating = false;
}

function fireStimulus(strength) {
  reset();
  animating = true;
  NEURONS[0].receiving = true;
  NEURONS[0].potential = strength;

  setTimeout(() => {
    NEURONS[0].receiving = false;
    if (strength >= NEURONS[0].threshold) {
      NEURONS[0].fired = true;
      spawnPulses(0, strength);
    } else {
      animating = false;
    }
  }, 400);
}

function spawnPulses(fromId, signalStrength) {
  SYNAPSES.filter(s => s.from === fromId).forEach((syn, i) => {
    setTimeout(() => {
      const from = NEURONS[syn.from];
      const to = NEURONS[syn.to];
      pulses.push({
        x: from.x,
        y: from.y,
        tx: to.x,
        ty: to.y,
        t: 0,
        strength: signalStrength * syn.weight,
        targetId: syn.to,
        done: false,
      });
    }, i * 120);
  });
}

function onPulseArrive(pulse) {
  const neuron = NEURONS[pulse.targetId];
  neuron.receiving = true;
  neuron.potential += pulse.strength;

  setTimeout(() => {
    neuron.receiving = false;
    if (neuron.potential >= neuron.threshold) {
      neuron.fired = true;
      if (pulse.targetId < NEURONS.length - 1) {
        spawnPulses(pulse.targetId, neuron.potential);
      } else {
        setTimeout(() => { animating = false; }, 600);
      }
    } else if (pulse.targetId === NEURONS.length - 1 || !hasPendingPulses()) {
      setTimeout(() => { animating = false; }, 800);
    }
  }, 300);
}

function hasPendingPulses() {
  return pulses.some(p => !p.done);
}

function draw(canvas) {
  const ctx = canvas.getContext('2d');
  const w = canvas.width;
  const h = canvas.height;

  ctx.fillStyle = '#121820';
  ctx.fillRect(0, 0, w, h);

  // Synapses
  SYNAPSES.forEach(syn => {
    const from = NEURONS[syn.from];
    const to = NEURONS[syn.to];
    const active = from.fired || pulses.some(p => p.targetId === syn.to && !p.done);
    ctx.strokeStyle = active ? '#4dabf7' : '#2d3f56';
    ctx.lineWidth = active ? 2.5 : 1.5;
    ctx.beginPath();
    ctx.moveTo(from.x, from.y);
    ctx.lineTo(to.x, to.y);
    ctx.stroke();

    const mx = (from.x + to.x) / 2;
    const my = (from.y + to.y) / 2;
    ctx.fillStyle = '#8b9cb3';
    ctx.font = '10px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText(`w=${syn.weight}`, mx, my - 6);
  });

  // Pulses
  pulses = pulses.filter(p => {
    if (p.done) return false;
    p.t += 0.025;
    if (p.t >= 1) {
      p.done = true;
      onPulseArrive(p);
      return false;
    }
    const px = p.x + (p.tx - p.x) * ease(p.t);
    const py = p.y + (p.ty - p.y) * ease(p.t);

    ctx.beginPath();
    ctx.arc(px, py, 6, 0, Math.PI * 2);
    ctx.fillStyle = '#ffd43b';
    ctx.fill();
    ctx.beginPath();
    ctx.arc(px, py, 10, 0, Math.PI * 2);
    ctx.fillStyle = 'rgba(255, 212, 59, 0.3)';
    ctx.fill();
    return true;
  });

  // Neurons
  NEURONS.forEach((n, i) => {
    let color = '#2d3f56';
    let glow = 'rgba(45, 63, 86, 0.3)';
    if (n.fired) {
      color = '#ff922b';
      glow = 'rgba(255, 146, 43, 0.5)';
    } else if (n.receiving) {
      color = '#ffd43b';
      glow = 'rgba(255, 212, 59, 0.4)';
    }

    ctx.beginPath();
    ctx.arc(n.x, n.y, 28, 0, Math.PI * 2);
    ctx.fillStyle = glow;
    ctx.fill();

    ctx.beginPath();
    ctx.arc(n.x, n.y, 22, 0, Math.PI * 2);
    ctx.fillStyle = color;
    ctx.fill();
    ctx.strokeStyle = n.fired ? '#fff' : '#3d5270';
    ctx.lineWidth = 2;
    ctx.stroke();

    // Dendrites hint for input neuron
    if (i === 0) {
      ctx.strokeStyle = '#3d5270';
      ctx.lineWidth = 1.5;
      for (let d = -2; d <= 2; d++) {
        ctx.beginPath();
        ctx.moveTo(n.x - 30, n.y + d * 8);
        ctx.lineTo(n.x - 50, n.y + d * 14);
        ctx.stroke();
      }
      ctx.fillStyle = '#8b9cb3';
      ctx.font = '11px sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText('자극 입력', n.x - 40, n.y + 50);
    }

    // Threshold bar
    const barW = 40;
    const barH = 5;
    const bx = n.x - barW / 2;
    const by = n.y + 32;
    ctx.fillStyle = '#1e2a3a';
    ctx.fillRect(bx, by, barW, barH);
    const fillW = Math.min(n.potential / n.threshold, 1) * barW;
    ctx.fillStyle = n.potential >= n.threshold ? '#51cf66' : '#4dabf7';
    ctx.fillRect(bx, by, fillW, barH);
    ctx.strokeStyle = '#ffd43b';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(bx + barW * 0.7, by - 2);
    ctx.lineTo(bx + barW * 0.7, by + barH + 2);
    ctx.stroke();

    ctx.fillStyle = '#8b9cb3';
    ctx.font = '10px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText(`역치 ${n.threshold}`, n.x, by + 14);
    ctx.fillStyle = '#e8edf4';
    ctx.font = 'bold 12px sans-serif';
    ctx.fillText(`N${i}`, n.x, n.y + 4);
  });

  // Labels
  ctx.fillStyle = '#8b9cb3';
  ctx.font = '13px Malgun Gothic, sans-serif';
  ctx.textAlign = 'left';
  ctx.fillText('입력층', 50, 30);
  ctx.fillText('은닉층 (병렬)', 200, 30);
  ctx.fillText('통합', 390, 30);
  ctx.fillText('출력', 545, 30);
}

function ease(t) {
  return t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2;
}
