/** Shared canvas drawing utilities */

export function setupCanvas(canvas) {
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  const w = canvas.width;
  const h = canvas.height;
  canvas.style.width = `${rect.width || w}px`;
  canvas.style.height = `${rect.height || h}px`;
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { ctx, w, h };
}

export function drawGrid(ctx, w, h, padding = 50) {
  ctx.strokeStyle = '#1e2a3a';
  ctx.lineWidth = 1;
  const plotW = w - padding * 2;
  const plotH = h - padding * 2;

  for (let i = 0; i <= 4; i++) {
    const x = padding + (plotW * i) / 4;
    const y = padding + (plotH * i) / 4;
    ctx.beginPath();
    ctx.moveTo(x, padding);
    ctx.lineTo(x, h - padding);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(padding, y);
    ctx.lineTo(w - padding, y);
    ctx.stroke();
  }

  ctx.strokeStyle = '#3d5270';
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.moveTo(padding, h - padding);
  ctx.lineTo(w - padding, h - padding);
  ctx.moveTo(padding, padding);
  ctx.lineTo(padding, h - padding);
  ctx.stroke();

  ctx.fillStyle = '#8b9cb3';
  ctx.font = '12px Segoe UI, Malgun Gothic, sans-serif';
  ctx.fillText('x₁', w - padding + 8, h - padding + 4);
  ctx.fillText('x₂', padding - 4, padding - 8);

  return { padding, plotW, plotH };
}

/** Map logic coords (0-1) to canvas */
export function toCanvas(x, y, w, h, padding) {
  const plotW = w - padding * 2;
  const plotH = h - padding * 2;
  return {
    cx: padding + x * plotW,
    cy: h - padding - y * plotH,
  };
}

export function drawDecisionLine(ctx, a, b, w, bias, canvasW, canvasH, padding, blink = true) {
  const plotW = canvasW - padding * 2;
  const plotH = canvasH - padding * 2;

  // ax + by + w = 0  =>  y = (-a/b)*x - w/b  in logic space
  // In canvas: need two points on boundary
  let x1 = 0, y1, x2 = 1, y2;
  if (Math.abs(b) > 0.001) {
    y1 = (-a * x1 - bias) / b;
    y2 = (-a * x2 - bias) / b;
  } else if (Math.abs(a) > 0.001) {
    x1 = x2 = -bias / a;
    y1 = 0;
    y2 = 1;
  } else {
    return;
  }

  const p1 = toCanvas(x1, y1, canvasW, canvasH, padding);
  const p2 = toCanvas(x2, y2, canvasW, canvasH, padding);

  const alpha = blink ? 0.5 + 0.5 * Math.sin(Date.now() / 300) : 1;
  ctx.strokeStyle = `rgba(77, 171, 247, ${alpha})`;
  ctx.lineWidth = 2;
  ctx.setLineDash([6, 4]);
  ctx.beginPath();
  ctx.moveTo(p1.cx, p1.cy);
  ctx.lineTo(p2.cx, p2.cy);
  ctx.stroke();
  ctx.setLineDash([]);
}

export function fillHalfPlane(ctx, a, b, bias, positive, canvasW, canvasH, padding) {
  const plotW = canvasW - padding * 2;
  const plotH = canvasH - padding * 2;
  const steps = 40;
  for (let i = 0; i < steps; i++) {
    for (let j = 0; j < steps; j++) {
      const x = i / (steps - 1);
      const y = j / (steps - 1);
      const p = a * x + b * y + bias;
      const isPos = p > 0;
      if (isPos === positive) {
        const { cx, cy } = toCanvas(x, y, canvasW, canvasH, padding);
        ctx.fillStyle = positive
          ? 'rgba(105, 219, 124, 0.08)'
          : 'rgba(255, 135, 135, 0.08)';
        ctx.fillRect(cx - plotW / steps / 2, cy - plotH / steps / 2, plotW / steps, plotH / steps);
      }
    }
  }
}

export function drawLogicPoints(ctx, points, canvasW, canvasH, padding, highlightIdx = -1) {
  points.forEach((pt, i) => {
    const { cx, cy } = toCanvas(pt.x, pt.y, canvasW, canvasH, padding);
    const isOne = pt.label === 1;
    const r = i === highlightIdx ? 16 : 12;

    ctx.beginPath();
    ctx.arc(cx, cy, r + 4, 0, Math.PI * 2);
    ctx.fillStyle = isOne ? 'rgba(105, 219, 124, 0.2)' : 'rgba(255, 135, 135, 0.2)';
    ctx.fill();

    ctx.beginPath();
    ctx.arc(cx, cy, r, 0, Math.PI * 2);
    ctx.fillStyle = isOne ? '#69db7c' : '#ff8787';
    ctx.fill();
    ctx.strokeStyle = i === highlightIdx ? '#ffd43b' : '#fff';
    ctx.lineWidth = i === highlightIdx ? 3 : 1.5;
    ctx.stroke();

    ctx.fillStyle = '#0f1419';
    ctx.font = 'bold 11px sans-serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(`(${pt.x},${pt.y})`, cx, cy);
    ctx.fillStyle = '#8b9cb3';
    ctx.font = '10px sans-serif';
    ctx.fillText(`→ ${pt.label}`, cx, cy + r + 12);
  });
}

export const LOGIC_POINTS = [
  { x: 0, y: 0 },
  { x: 0, y: 1 },
  { x: 1, y: 0 },
  { x: 1, y: 1 },
];

export function gateLabels(gate) {
  const fns = {
    AND: (x, y) => (x && y ? 1 : 0),
    OR: (x, y) => (x || y ? 1 : 0),
    XOR: (x, y) => (x !== y ? 1 : 0),
  };
  return LOGIC_POINTS.map(p => ({
    ...p,
    label: fns[gate](p.x, p.y),
  }));
}

export function step(x) {
  return x > 0 ? 1 : 0;
}

export function perceptron(a, b, bias, x, y) {
  return step(a * x + b * y + bias);
}
