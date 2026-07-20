import { useState, useRef, useEffect, useCallback } from 'react';
import {
  computeFFTReal, computeMFCC, mfccToPhoneme,
  renderSpectrum, renderMFCC, phonemesToSentence,
} from '../ml/audio';
import { useFitCanvas } from '../hooks/useFitCanvas';

const PIPELINE = ['마이크', 'FFT', 'MFCC', '음소', '문장'];

export default function SpeechTab() {
  const spectrumRef = useRef(null);
  const mfccRef = useRef(null);
  const spectrumDataRef = useRef(null);
  const mfccDataRef = useRef(null);
  const [isListening, setIsListening] = useState(false);
  const [transcript, setTranscript] = useState('');
  const [phonemes, setPhonemes] = useState([]);
  const [currentPhoneme, setCurrentPhoneme] = useState('-');
  const [phonemeConf, setPhonemeConf] = useState(0);
  const [sentence, setSentence] = useState('');
  const [pipelineStep, setPipelineStep] = useState(-1);
  const recognitionRef = useRef(null);
  const audioCtxRef = useRef(null);
  const analyserRef = useRef(null);
  const streamRef = useRef(null);
  const animRef = useRef(null);
  const phonemeHistoryRef = useRef([]);

  const drawSpectrum = useCallback(({ ctx }) => {
    if (spectrumDataRef.current) renderSpectrum(spectrumDataRef.current, spectrumRef.current);
  }, []);
  const drawMfcc = useCallback(({ ctx }) => {
    if (mfccDataRef.current) renderMFCC(mfccDataRef.current, mfccRef.current);
  }, []);

  useFitCanvas(spectrumRef, drawSpectrum, [drawSpectrum]);
  useFitCanvas(mfccRef, drawMfcc, [drawMfcc]);

  const processAudioFrame = useCallback((timeDomainData, sampleRate) => {
    const spectrum = computeFFTReal(Array.from(timeDomainData));
    spectrumDataRef.current = spectrum;
    renderSpectrum(spectrum, spectrumRef.current);
    setPipelineStep(1);

    const mfcc = computeMFCC(Array.from(timeDomainData), sampleRate);
    mfccDataRef.current = mfcc;
    renderMFCC(mfcc, mfccRef.current);
    setPipelineStep(2);

    const { phoneme, confidence } = mfccToPhoneme(mfcc);
    setCurrentPhoneme(phoneme);
    setPhonemeConf(confidence);
    setPipelineStep(3);

    if (phoneme !== 'SIL' && confidence > 0.25) {
      const last = phonemeHistoryRef.current[phonemeHistoryRef.current.length - 1];
      if (last !== phoneme) {
        phonemeHistoryRef.current.push(phoneme);
        setPhonemes([...phonemeHistoryRef.current.slice(-10)]);
      }
    }
  }, []);

  const startListening = async () => {
    setIsListening(true);
    setTranscript('');
    setPhonemes([]);
    setSentence('');
    setPipelineStep(0);
    phonemeHistoryRef.current = [];
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;
      const audioCtx = new AudioContext({ sampleRate: 16000 });
      audioCtxRef.current = audioCtx;
      const source = audioCtx.createMediaStreamSource(stream);
      const analyser = audioCtx.createAnalyser();
      analyser.fftSize = 512;
      analyserRef.current = analyser;
      source.connect(analyser);
      const buffer = new Float32Array(analyser.fftSize);
      const loop = () => {
        if (!analyserRef.current) return;
        analyser.getFloatTimeDomainData(buffer);
        processAudioFrame(buffer, audioCtx.sampleRate);
        animRef.current = requestAnimationFrame(loop);
      };
      loop();
      const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
      if (SpeechRecognition) {
        const rec = new SpeechRecognition();
        rec.lang = 'ko-KR';
        rec.continuous = true;
        rec.interimResults = true;
        rec.onresult = (e) => {
          let text = '';
          for (let i = 0; i < e.results.length; i++) text += e.results[i][0].transcript;
          setTranscript(text);
          setSentence(text);
          setPipelineStep(4);
        };
        rec.onend = () => { if (recognitionRef.current) rec.start(); };
        rec.start();
        recognitionRef.current = rec;
      }
    } catch (err) {
      setSentence('마이크 접근 불가');
      setIsListening(false);
    }
  };

  const stopListening = () => {
    setIsListening(false);
    if (recognitionRef.current) { recognitionRef.current.stop(); recognitionRef.current = null; }
    if (animRef.current) cancelAnimationFrame(animRef.current);
    if (streamRef.current) { streamRef.current.getTracks().forEach((t) => t.stop()); streamRef.current = null; }
    if (audioCtxRef.current) { audioCtxRef.current.close(); audioCtxRef.current = null; }
    analyserRef.current = null;
    setPipelineStep(4);
    if (!transcript && phonemes.length) setSentence(phonemesToSentence(phonemes));
  };

  useEffect(() => () => {
    if (animRef.current) cancelAnimationFrame(animRef.current);
    if (recognitionRef.current) recognitionRef.current.stop();
    if (streamRef.current) streamRef.current.getTracks().forEach((t) => t.stop());
    if (audioCtxRef.current) audioCtxRef.current.close();
  }, []);

  return (
    <div className="tab-panel">
      <div className="panel-row">
        <div className="panel-col" style={{ flex: 1 }}>
          <div className="panel-box">
            <div className="panel-title">① FFT 스펙트럼</div>
            <div className="canvas-wrap canvas-wrap-sm">
              <canvas ref={spectrumRef} />
            </div>
          </div>
          <div className="panel-box">
            <div className="panel-title">② MFCC 특징</div>
            <div className="canvas-wrap canvas-wrap-sm">
              <canvas ref={mfccRef} />
            </div>
          </div>
          <div className="panel-row" style={{ flex: '0 0 auto' }}>
            <div className="panel-box" style={{ flex: 1 }}>
              <div className="panel-title">③ 음소 추정</div>
              <div className="phoneme-row">
                <span className="metric metric-sm">{currentPhoneme}</span>
                <span className="badge blue">{(phonemeConf * 100).toFixed(0)}%</span>
              </div>
              <div className="phoneme-tags">
                {phonemes.map((p, i) => <span key={i} className="highlight-box">{p}</span>)}
                {!phonemes.length && <span className="placeholder-text">음소 시퀀스 대기...</span>}
              </div>
            </div>
            <div className="panel-box" style={{ flex: 1 }}>
              <div className="panel-title">④ 문장 예측</div>
              <div className="highlight-box">
                {sentence || transcript || '음성 인식 버튼을 누르고 말씀해 주세요'}
              </div>
            </div>
          </div>
        </div>

        <div className="panel-col speech-side">
          <div className="panel-box" style={{ height: '100%' }}>
            <button className={`btn primary ${isListening ? 'active-rec' : ''}`}
              style={{ padding: '12px 18px' }}
              onClick={isListening ? stopListening : startListening}>
              {isListening ? '■ 중지' : '● 음성 인식'}
            </button>
            <span className="step-label" style={{ textAlign: 'center' }}>
              {isListening ? PIPELINE[pipelineStep] || '처리 중' : '대기 중'}
            </span>
            {PIPELINE.map((s, i) => (
              <div key={s} className={`pipeline-chip ${pipelineStep >= i && isListening ? 'active' : ''}`}>{s}</div>
            ))}
          </div>
        </div>
      </div>

      <div className="panel-box" style={{ flex: '0 0 auto' }}>
        <div className="pipeline-row">
          {['마이크', 'FFT', 'MFCC', '음소', '문장'].map((s, i) => (
            <span key={s}>
              <span className={`pipeline-chip ${pipelineStep >= i && isListening ? 'active' : ''}`}>{s}</span>
              {i < 4 && <span style={{ color: '#444', margin: '0 2px' }}>→</span>}
            </span>
          ))}
        </div>
      </div>
    </div>
  );
}
