import { useState, useRef, useCallback } from 'react';
import { FruitClassifier, VOCAB } from '../ml/nlpModel';
import { getNLPWeights } from '../data/initModels';
import { useFitCanvas } from '../hooks/useFitCanvas';

function NLPNetworkViz({ result }) {
  const canvasRef = useRef(null);

  const draw = useCallback(({ w: W, h: H, ctx }) => {
    ctx.clearRect(0, 0, W, H);
    if (!result) {
      ctx.fillStyle = '#666';
      ctx.font = '10px sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText('문장을 입력하세요', W / 2, H / 2);
      return;
    }
    const layers = [
      { label: '임베딩', nodes: 4, acts: result.hidden?.slice(0, 4) || [0, 0, 0, 0] },
      { label: '은닉', nodes: 5, acts: result.hidden?.slice(0, 5) || Array(5).fill(0) },
      { label: '출력', nodes: 2, acts: [result.apple, result.orange] },
    ];
    const layerX = [W * 0.15, W * 0.5, W * 0.85];
    const nodeR = Math.min(40, H / 8);
    const spacing = Math.min(84, (H - 30) / 5);

    layers.forEach((layer, li) => {
      ctx.fillStyle = '#8b949e';
      ctx.font = '18px sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText(layer.label, layerX[li], 24);
      for (let ni = 0; ni < layer.nodes; ni++) {
        const y = H / 2 + (ni - (layer.nodes - 1) / 2) * spacing;
        const act = layer.acts[ni] ?? 0;
        const intensity = Math.floor(Math.min(1, act) * 180 + 40);
        if (li < layers.length - 1) {
          for (let nj = 0; nj < layers[li + 1].nodes; nj++) {
            const ny = H / 2 + (nj - (layers[li + 1].nodes - 1) / 2) * spacing;
            ctx.strokeStyle = 'rgba(88,166,255,0.12)';
            ctx.beginPath();
            ctx.moveTo(layerX[li] + nodeR, y);
            ctx.lineTo(layerX[li + 1] - nodeR, ny);
            ctx.stroke();
          }
        }
        ctx.beginPath();
        ctx.arc(layerX[li], y, nodeR, 0, Math.PI * 2);
        ctx.fillStyle = li === 2
          ? (ni === 0 ? `rgba(248,81,73,${intensity / 255})` : `rgba(227,179,65,${intensity / 255})`)
          : `rgb(${intensity * 0.3},${intensity * 0.5},${intensity})`;
        ctx.fill();
        ctx.strokeStyle = '#30363d';
        ctx.stroke();
        ctx.fillStyle = '#fff';
        ctx.font = '21px monospace';
        ctx.fillText(typeof act === 'number' ? act.toFixed(2) : act, layerX[li], y + 2);
      }
    });
  }, [result]);

  useFitCanvas(canvasRef, draw, [result]);

  return (
    <div className="canvas-wrap canvas-wrap-md">
      <canvas ref={canvasRef} />
    </div>
  );
}

const APPLE_SVG = `data:image/svg+xml,${encodeURIComponent(`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><rect fill="#1a1a2e" width="100" height="100"/><ellipse cx="50" cy="58" rx="30" ry="32" fill="#e74c3c"/><path d="M50 26 Q55 18 60 22" stroke="#27ae60" stroke-width="3" fill="none"/></svg>`)}`;
const ORANGE_SVG = `data:image/svg+xml,${encodeURIComponent(`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><rect fill="#1a1a2e" width="100" height="100"/><circle cx="50" cy="55" r="32" fill="#f39c12"/><path d="M50 23 Q54 15 58 20" stroke="#27ae60" stroke-width="3" fill="none"/></svg>`)}`;

export default function NLPTab() {
  const [classifier] = useState(() => new FruitClassifier(getNLPWeights()));
  const [text, setText] = useState('');
  const [result, setResult] = useState(null);

  const analyze = (input) => {
    if (!input.trim()) { setResult(null); return; }
    setResult(classifier.forward(input));
  };

  const examples = ['빨간 사과를 주세요', '오렌지 please', 'I love orange', '달콤한 사과'];

  return (
    <div className="tab-panel">
      <div className="panel-row" style={{ flex: '0 0 auto' }}>
        <div className="panel-box fruit-panel">
          <div className="panel-title">사과</div>
          <img src={APPLE_SVG} alt="사과" className="fruit-img" />
          {result && <span className={`badge ${result.prediction === 'apple' ? 'green' : ''}`}>{(result.apple * 100).toFixed(0)}%</span>}
        </div>
        <div className="panel-box" style={{ flex: 1 }}>
          <div className="panel-title">문장 입력</div>
          <input className="input-field" value={text}
            onChange={(e) => { setText(e.target.value); analyze(e.target.value); }}
            placeholder="사과 또는 오렌지를 지칭하는 문장" />
          <div className="btn-row">
            {examples.map((ex) => (
              <button key={ex} className="btn" onClick={() => { setText(ex); analyze(ex); }}>{ex}</button>
            ))}
          </div>
          {result && (
            <span className="step-label" style={{ marginTop: 4, display: 'inline-block' }}>
              {result.prediction === 'apple' ? '사과 🍎' : '오렌지 🍊'}
              {' '}({(result.apple * 100).toFixed(0)}% / {(result.orange * 100).toFixed(0)}%)
            </span>
          )}
        </div>
        <div className="panel-box fruit-panel">
          <div className="panel-title">오렌지</div>
          <img src={ORANGE_SVG} alt="오렌지" className="fruit-img" />
          {result && <span className={`badge ${result.prediction === 'orange' ? 'orange' : ''}`}>{(result.orange * 100).toFixed(0)}%</span>}
        </div>
      </div>

      <div className="panel-box" style={{ flex: 1 }}>
        <div className="panel-title">NLP 신경망 (임베딩 → 은닉 → 출력)</div>
        <NLPNetworkViz result={result} />
      </div>
    </div>
  );
}
