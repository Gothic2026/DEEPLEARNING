import {
  drawGrid, drawDecisionLine, fillHalfPlane, toCanvas, step,
} from './utils.js';

let points = [
  { x: 0.2, y: 0.8, label: 1 },
  { x: 0.8, y: 0.7, label: 1 },
  { x: 0.3, y: 0.2, label: 0 },
  { x: 0.7, y: 0.3, label: 0 },
];

let a = 1, b = 1, bias = -1.5;
let rafId = null;

export function initPerceptron(canvas) {
  const weightA = document.getElementById('weightA');
  const weightB = document.getElementById('weightB');
  const weightW = document.getElementById('weightW');

  [weightA, weightB, weightW].forEach(el => {
    el.addEventListener('input', () => {
      a = parseFloat(weightA.value);
      b = parseFloat(weightB.value);
      bias = parseFloat(weightW.value);
      updateFormula();
    });
  });

  canvas.addEventListener('click', (e) => {
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    const mx = (e.clientX - rect.left) * scaleX;
    const my = (e.clientY - rect.top) * scaleY;
    const padding = 50;
    const plotW = canvas.width - padding * 2;
    const plotH = canvas.height - padding * 2;
    const x = (mx - padding) / plotW;
    const y = (canvas.height - padding - my) / plotH;
    if (x >= 0 && x <= 1 && y >= 0 && y <= 1) {
      const p = a * x + b * y + bias;
      points.push({ x, y, label: step(p) });
      if (points.length > 12) points.shift();
    }
  });

  updateFormula();

  function loop() {
    draw(canvas);
    rafId = requestAnimationFrame(loop);
  }
  loop();

  return () => { if (rafId) cancelAnimationFrame(rafId); };
}

function updateFormula() {
  document.getElementById('fA').textContent = a.toFixed(1);
  document.getElementById('fB').textContent = b.toFixed(1);
  document.getElementById('fW').textContent = bias >= 0 ? `+${bias.toFixed(1)}` : bias.toFixed(1);
  document.getElementById('weightAVal').textContent = a.toFixed(1);
  document.getElementById('weightBVal').textContent = b.toFixed(1);
  document.getElementById('weightWVal').textContent = bias.toFixed(1);
}

function draw(canvas) {
  const ctx = canvas.getContext('2d');
  const w = canvas.width;
  const h = canvas.height;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const { padding } = drawGrid(ctx, w, h);

  fillHalfPlane(ctx, a, b, bias, true, w, h, padding);
  fillHalfPlane(ctx, a, b, bias, false, w, h, padding);
  drawDecisionLine(ctx, a, b, w, bias, w, h, padding, true);

  points.forEach(pt => {
    const p = a * pt.x + b * pt.y + bias;
    const active = p > 0;
    const { cx, cy } = toCanvas(pt.x, pt.y, w, h, padding);

    ctx.beginPath();
    ctx.arc(cx, cy, 14, 0, Math.PI * 2);
    ctx.fillStyle = active ? '#69db7c' : '#ff8787';
    ctx.fill();
    ctx.strokeStyle = '#fff';
    ctx.lineWidth = 1.5;
    ctx.stroke();

    ctx.fillStyle = '#0f1419';
    ctx.font = 'bold 10px sans-serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(active ? 'T' : 'F', cx, cy);
  });

  // Hover info at center of boundary
  const midX = 0.5;
  const midY = (-a * midX - bias) / (b || 0.001);
  if (midY >= 0 && midY <= 1) {
    const { cx, cy } = toCanvas(midX, midY, w, h, padding);
    ctx.fillStyle = '#4dabf7';
    ctx.font = '11px sans-serif';
    ctx.textAlign = 'left';
    ctx.fillText('p = 0 (경계)', cx + 8, cy);
  }

  const sampleP = a * 0.5 + b * 0.5 + bias;
  document.getElementById('activationResult').textContent =
    sampleP > 0 ? 'T (활성)' : 'F (비활성)';
}

