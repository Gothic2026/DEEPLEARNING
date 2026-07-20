export default function IntroTab() {
  return (
    <div className="tab-panel">
      <div className="intro-grid">
        <div className="intro-card">
          <h3>머신러닝과 딥러닝</h3>
          <p>기존 머신러닝은 전문적인 지식을 활용해 데이터의 특징을 추출하고 이를 디자인되었습니다. 이 과정은 사람의 시행착오와 노력이 필요합니다. 
            그래서 만들어진 딥러닝은 인간의 뇌 구조에서 영감을 받아 설계되어 복잡한 데이터 특성을 스스로 학습하고 추출합니다. 딥 러닝은 다층 신경망으로 구동되는 머신 러닝의 하위 집합이라고 할 수 있습니다.
            다층 신경망은 인공 신경망 중에서도, 최소 4개의 계층으로 모델을 훈련하는 것을 말합니다.</p>
        </div>
        <div className="intro-card">
          <h3>컴퓨터 비전 (CNN)</h3>
          <p>컴퓨팅 비전은 기계에 이미지 및 동영상과 같은 시각적 입력을 처리, 분석, 해석할 수 있는 능력을 부여하는 기능을 CNN모델을 사용하여 구현합니다. CNN모델은 합성곱 → ReLU → 풀링의 과정을 반복하여 특징 추출 후 DNN 모델로 분류합니다</p>
        </div>
        <div className="intro-card">
          <h3>음성 인식</h3>
          <p>음성 인식은 프로그램이 사람의 음성을 문자 형식으로 처리할 수 있도록 하는 기능 오디오 및 음성 신호의 문법, 구문, 구조 및 구성을 통합하여 인간의 말을 이해하고 처리합니다. FFT → MFCC → 음소 인식 → 문장 조합의 과정으로 처리됩니다.</p>
        </div>
        <div className="intro-card">
          <h3>자연어 처리 (NLP)</h3>
          <p>자연어 처리는 머신 러닝을 사용하여 컴퓨터가 인간의 언어를 이해하고 소통하도록 돕는 기능입니다. 서로 다른 언어 부분 사이의 관계 등을 통해 학습합니다.</p>
        </div>
      </div>
      <div className="panel-box" style={{ flexShrink: 0 }}>
        <p className="info-text">상단 탭에서 각 과정을 <strong>실제 알고리즘</strong>으로 체험하세요. 백엔드 불필요.</p>
      </div>
    </div>
  );
}
