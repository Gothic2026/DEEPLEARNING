import { useState, useRef, useEffect, useCallback } from 'react';
import { MLP, XOR_DATASET, computeDecisionGrid } from '../ml/neuralNetwork';
import { useFitCanvas } from '../hooks/useFitCanvas';

const LAYER_SIZES = [2, 4, 4, 1];
const LAYER_LABELS = ['입력층', '은닉1', '은닉2', '출력'];

function NeuralNetViz({ mlp, animSignal, updateFlash, selectedInput, output, edgeDelta }) {
  const canvasRef = useRef(null);

  const draw = useCallback(({ w: W, h: H, ctx }) => {
    ctx.clearRect(0, 0, W, H);
    const nodeR = Math.min(28, H / 14);
    const spacing = Math.min(72, (H - 80) / 4);
    const positions = [];
    const layerX = LAYER_SIZES.map((_, li) => 28 + li * ((W - 56) / (LAYER_SIZES.length - 1)));

    for (let li = 0; li < LAYER_SIZES.length; li++) {
      const n = LAYER_SIZES[li];
      const col = [];
      for (let ni = 0; ni < n; ni++)
        col.push({ x: layerX[li], y: H / 2 + (ni - (n - 1) / 2) * spacing });
      positions.push(col);
    }

    for (let li = 0; li < LAYER_SIZES.length - 1; li++) {
      const Wl = mlp.weights[li];
      for (let j = 0; j < positions[li + 1].length; j++)
        for (let i = 0; i < positions[li].length; i++) {
          const wt = Wl[j][i];
          const alpha = Math.min(1, Math.abs(wt) * 2.5);
          ctx.strokeStyle = wt >= 0 ? `rgba(88,166,255,${alpha})` : `rgba(248,81,73,${alpha})`;
          ctx.lineWidth = Math.abs(wt) * 2 + 0.4;
          ctx.beginPath();
          ctx.moveTo(positions[li][i].x, positions[li][i].y);
          ctx.lineTo(positions[li + 1][j].x, positions[li + 1][j].y);
          ctx.stroke();
          if (edgeDelta?.l === li && edgeDelta.i === i && edgeDelta.j === j) {
            const mx = (positions[li][i].x + positions[li + 1][j].x) / 2;
            const my = (positions[li][i].y + positions[li + 1][j].y) / 2;
            ctx.fillStyle = '#e3b341';
            ctx.font = 'bold 16px monospace';
            ctx.textAlign = 'center';
            ctx.fillText(`Δw=${edgeDelta.dw.toFixed(2)}`, mx, my - 3);
          }
        }
    }

    for (let li = 0; li < LAYER_SIZES.length; li++) {
      const acts = li === 0 ? selectedInput : (mlp.activations[li] || []);
      const biases = li > 0 ? mlp.biases[li - 1] : null;
      for (let ni = 0; ni < positions[li].length; ni++) {
        const { x, y } = positions[li][ni];
        const act = acts[ni] ?? 0;
        const intensity = Math.floor(act * 180 + 40);
        ctx.beginPath();
        ctx.arc(x, y, nodeR, 0, Math.PI * 2);
        ctx.fillStyle = `rgb(${intensity * 0.3}, ${intensity * 0.6}, ${intensity})`;
        ctx.fill();
        ctx.strokeStyle = updateFlash?.layer === li && updateFlash?.node === ni ? '#e3b341' : '#30363d';
        ctx.lineWidth = updateFlash?.layer === li && updateFlash?.node === ni ? 2 : 1.5;
        ctx.stroke();
        ctx.fillStyle = '#fff';
        ctx.font = `${Math.max(8, nodeR * 0.55)}px monospace`;
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(act.toFixed(2), x, y - 2);
        if (biases) {
          ctx.font = `${Math.max(7, nodeR * 0.45)}px monospace`;
          ctx.fillStyle = '#8b949e';
          ctx.fillText(`b=${biases[ni].toFixed(1)}`, x, y + nodeR * 0.55);
        }
      }
    }

    ctx.fillStyle = '#8b949e';
    ctx.font = '9px sans-serif';
    ctx.textAlign = 'center';
    LAYER_LABELS.forEach((lb, li) => ctx.fillText(lb, layerX[li], 12));

    if (animSignal) {
      const { fromLayer, fromNode, toLayer, toNode, progress, phase } = animSignal;
      const from = positions[fromLayer][fromNode];
      const to = positions[toLayer][toNode];
      ctx.beginPath();
      ctx.arc(from.x + (to.x - from.x) * progress, from.y + (to.y - from.y) * progress, 12, 0, Math.PI * 2);
      ctx.fillStyle = phase === 'backward' ? '#f85149' : '#e3b341';
      ctx.fill();
    }

    if (output !== null) {
      ctx.fillStyle = output > 0.5 ? '#3fb950' : '#f85149';
      ctx.font = 'bold 30px sans-serif';
      ctx.textAlign = 'right';
      ctx.fillText(`σ(z)=${output.toFixed(3)}`, W - 6, H - 16);
    }
  }, [mlp, animSignal, updateFlash, selectedInput, output, edgeDelta]);

  useFitCanvas(canvasRef, draw, [mlp, animSignal, updateFlash, selectedInput, output, edgeDelta]);

  return (
    <div className="canvas-wrap canvas-wrap-md">
      <canvas ref={canvasRef} />
    </div>
  );
}

function DecisionGraph({ mlp, version }) {
  const canvasRef = useRef(null);

  const draw = useCallback(({ w: W, h: H, ctx }) => {
    const grid = computeDecisionGrid(mlp, 40);
    const res = grid.length - 1;
    for (let yi = 0; yi <= res; yi++)
      for (let xi = 0; xi <= res; xi++) {
        const v = grid[res - yi][xi];
        ctx.fillStyle = `rgb(${Math.floor(v * 60)},${Math.floor(v * 100 + 30)},${Math.floor((1 - v) * 180 + 40)})`;
        ctx.fillRect((xi / res) * W, (yi / res) * H, W / res + 1, H / res + 1);
      }
    [[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0]].forEach(([x, y, t]) => {
      ctx.beginPath();
      ctx.arc(x * W, (1 - y) * H, 5, 0, Math.PI * 2);
      ctx.fillStyle = t ? '#3fb950' : '#f85149';
      ctx.fill();
      ctx.strokeStyle = '#fff';
      ctx.lineWidth = 1.5;
      ctx.stroke();
    });
    ctx.fillStyle = '#8b949e';
    ctx.font = '9px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('x₁', W / 2, H - 2);
  }, [mlp, version]);

  useFitCanvas(canvasRef, draw, [mlp, version]);

  return (
    <div className="canvas-wrap canvas-wrap-sm">
      <canvas ref={canvasRef} />
    </div>
  );
}

export default function XORTab() {
  const [mlp] = useState(() => new MLP(LAYER_SIZES, 42));
  const [version, setVersion] = useState(0);
  const [selectedIdx, setSelectedIdx] = useState(0);
  const [epochInfo, setEpochInfo] = useState('');
  const [animSignal, setAnimSignal] = useState(null);
  const [updateFlash, setUpdateFlash] = useState(null);
  const [edgeDelta, setEdgeDelta] = useState(null);
  const [phaseLabel, setPhaseLabel] = useState('');
  const [isTraining, setIsTraining] = useState(false);
  const [output, setOutput] = useState(null);

  const selected = XOR_DATASET[selectedIdx];

  const animateForward = useCallback(async (inputs) => {
    setPhaseLabel('순전파');
    mlp.forward(inputs);
    setOutput(mlp.predict(inputs));
    setVersion((v) => v + 1);
    const edges = [];
    for (let li = 0; li < LAYER_SIZES.length - 1; li++)
      for (let j = 0; j < LAYER_SIZES[li + 1]; j++)
        for (let i = 0; i < LAYER_SIZES[li]; i++)
          edges.push({ fromLayer: li, fromNode: i, toLayer: li + 1, toNode: j });
    for (const edge of edges) {
      for (let p = 0; p <= 1; p += 0.15) {
        setAnimSignal({ ...edge, progress: p, phase: 'forward' });
        await new Promise((r) => setTimeout(r, 10));
      }
    }
    setAnimSignal(null);
  }, [mlp]);

  const animateBackward = useCallback(async (inputs, target) => {
    setPhaseLabel('역전파');
    const updates = mlp.trainStep(inputs, target, 1.2);
    setVersion((v) => v + 1);
    for (let li = mlp.weights.length - 1; li >= 0; li--)
      for (let j = 0; j < mlp.weights[li].length; j++)
        for (let i = 0; i < mlp.weights[li][j].length; i++) {
          setEdgeDelta({ l: li, i, j, dw: updates.weightUpdates[li][j][i] });
          setUpdateFlash({ layer: li + 1, node: j, db: updates.biasUpdates[li][j] });
          await new Promise((r) => setTimeout(r, 50));
        }
    setEdgeDelta(null);
    setUpdateFlash(null);
  }, [mlp]);

  const handleSelect = async (idx) => {
    setSelectedIdx(idx);
    await animateForward(XOR_DATASET[idx].input);
  };

  const handleTrain = async () => {
    setIsTraining(true);
    setPhaseLabel('학습 중...');
    let converged = false;
    for (let e = 0; e < 15000; e++) {
      for (const { input, output: tgt } of XOR_DATASET) mlp.trainStep(input, tgt, 1.2);
      const err = XOR_DATASET.reduce((s, { input, output: tgt }) => s + Math.abs(mlp.predict(input) - tgt[0]), 0);
      if (e % 500 === 0) {
        const acc = XOR_DATASET.filter(({ input, output: tgt }) => Math.abs(mlp.predict(input) - tgt[0]) < 0.5).length;
        setEpochInfo(`에포크 ${e + 1} · 정확도 ${acc}/4`);
        setVersion((v) => v + 1);
      }
      if (err < 0.08) {
        setEpochInfo(`${e + 1} 에포크 수렴 — XOR 100%`);
        converged = true;
        break;
      }
    }
    if (!converged) setEpochInfo('학습 완료');
    setVersion((v) => v + 1);
    await animateForward(selected.input);
    await animateBackward(selected.input, XOR_DATASET[selectedIdx].output);
    setPhaseLabel('');
    setIsTraining(false);
  };

  useEffect(() => {
    mlp.forward(selected.input);
    setOutput(mlp.predict(selected.input));
  }, []);

  const accuracy = XOR_DATASET.filter(({ input, output: tgt }) => Math.abs(mlp.predict(input) - tgt[0]) < 0.5).length;

  return (
    <div className="tab-panel">
      <div className="panel-row">
        <div className="panel-col" style={{ flex: '0 0 180px' }}>
          <div className="panel-box" style={{ height: '100%' }}>
            <div className="panel-title">XOR 데이터</div>
            <table className="data-table">
              <thead><tr><th>x₁</th><th>x₂</th><th>y</th></tr></thead>
              <tbody>
                {XOR_DATASET.map((d, i) => (
                  <tr key={i} className={selectedIdx === i ? 'selected' : ''} onClick={() => !isTraining && handleSelect(i)}>
                    <td>{d.input[0]}</td><td>{d.input[1]}</td><td>{d.output[0]}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            <div className="btn-row">
              <button className="btn primary" onClick={handleTrain} disabled={isTraining}>
                {isTraining ? '학습 중' : '학습'}
              </button>
              <span className={`badge ${accuracy === 4 ? 'green' : 'orange'}`}>{accuracy}/4</span>
            </div>
            {epochInfo && <p className="info-text" style={{ marginTop: 3 }}>{epochInfo}</p>}
            {phaseLabel && <span className="step-label">{phaseLabel}</span>}
          </div>
        </div>
        <div className="panel-box" style={{ flex: 1 }}>
          <div className="panel-title">MLP (2→4→4→1) 순전파 / 역전파</div>
          <NeuralNetViz mlp={mlp} animSignal={animSignal} updateFlash={updateFlash}
            selectedInput={selected.input} output={output} edgeDelta={edgeDelta} />
        </div>
      </div>
      <div className="panel-box" style={{ flex: '0 0 auto', maxWidth: '300px', minHeight: '300px' }}>
        <div className="panel-title">결정 경계 [0,1]²</div>
        <DecisionGraph mlp={mlp} version={version} />
      </div>
    </div>
  );
}
