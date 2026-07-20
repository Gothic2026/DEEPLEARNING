import { useState } from 'react';
import IntroTab from './tabs/IntroTab';
import XORTab from './tabs/XORTab';
import CNNTab from './tabs/CNNTab';
import SpeechTab from './tabs/SpeechTab';
import NLPTab from './tabs/NLPTab';

const TABS = [
  { id: 'intro', label: '딥러닝 개요', component: IntroTab },
  { id: 'xor', label: '순전파·역전파 (XOR)', component: XORTab },
  { id: 'cnn', label: 'CNN · 컴퓨터비전', component: CNNTab },
  { id: 'speech', label: '음성인식', component: SpeechTab },
  { id: 'nlp', label: '자연어처리', component: NLPTab },
];

export default function App() {
  const [activeTab, setActiveTab] = useState('intro');
  const ActiveComponent = TABS.find((t) => t.id === activeTab).component;

  return (
    <div className="app">
      <header className="header">
        <h1 className="header-title">딥러닝 시각화 학습</h1>
        <nav className="tabs">
          {TABS.map((tab) => (
            <button
              key={tab.id}
              className={`tab-btn ${activeTab === tab.id ? 'active' : ''}`}
              onClick={() => setActiveTab(tab.id)}
            >
              {tab.label}
            </button>
          ))}
        </nav>
      </header>
      <main className="content">
        <ActiveComponent />
      </main>
    </div>
  );
}
