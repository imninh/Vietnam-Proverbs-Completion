import React, { useState, useEffect } from 'react';
import './App.css';
import PredictionForm from './components/PredictionForm';
import ResultCard from './components/ResultCard';
import LoadingSpinner from './components/LoadingSpinner';
import { getPredictions, type ModelResult } from './services/api';

// Sample folk poetry data - Sửa kiểu dữ liệu cho đúng với ModelResult
const sampleFolkPoetry = {
  daily: "Nhiễu điều phủ lấy giá gương, Người trong một nước phải thương nhau cùng.",
  modelResults: [
    {
      model_name: "N-Gram Model",
      model_type: "n_gram" as const, // Thêm 'as const' để TypeScript hiểu đây là literal type
      results: [
        { text: "I do not covet vast fields or ponds, But treasure the scholar's brush and inkstone.", confidence: 0.95, source: "generated" },
        { text: "How far is the road to Lang Son? A mountain apart and three fields away.", confidence: 0.92, source: "completed" },
        { text: "Tell the upstream folks if you go there, Send down young jackfruit, send up flying fish.", confidence: 0.93, source: "predicted" }
      ]
    },
    {
      model_name: "Retrieval Model",
      model_type: "retrieval" as const,
      results: [
        { text: "Who built Truoi Mountain so high? Who dug Gianh River so deep?", confidence: 0.91, source: "database", metadata: { similarity: 0.89 } },
        { text: "The wind blows to the field for crabs, To the river for fish, to the pond for shrimp.", confidence: 0.96, source: "corpus", metadata: { similarity: 0.92 } },
        { text: "Can Thơ has white rice and clear water, Those who go there never wish to leave.", confidence: 0.94, source: "collection", metadata: { similarity: 0.90 } }
      ]
    },
    {
      model_name: "Ensemble Model",
      model_type: "ensemble" as const,
      results: [
        { text: "I do not covet vast fields or ponds, Who built Truoi Mountain so high?", confidence: 0.97, source: "hybrid", metadata: { combined_score: 0.95 } },
        { text: "How far is the road to Lang Son? The wind blows to the field for crabs.", confidence: 0.94, source: "fusion", metadata: { combined_score: 0.93 } },
        { text: "Tell the upstream folks if you go there, Can Thơ has white rice and clear water.", confidence: 0.95, source: "combination", metadata: { combined_score: 0.94 } }
      ]
    }
  ] as ModelResult[] // Cast to ModelResult[]
};

// Model type definition
type ModelType = 'n_gram' | 'retrieval' | 'ensemble' | 'all';

const App: React.FC = () => {
  const [modelResults, setModelResults] = useState<ModelResult[]>([]);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string>('');
  const [dailyPoetry] = useState<string>(sampleFolkPoetry.daily);
  const [inputText, setInputText] = useState<string>('');
  const [selectedModel, setSelectedModel] = useState<ModelType>('all');

  useEffect(() => {
    setModelResults(sampleFolkPoetry.modelResults);
  }, []);

  const handleSubmit = async (inputText: string): Promise<void> => {
    if (!inputText.trim()) {
      setError('Please enter Vietnamese text');
      return;
    }

    setLoading(true);
    setError('');
    setInputText(inputText);

    try {
      const data = await getPredictions(inputText);
      setModelResults(data.model_results || []);
    } catch (err) {
      setError('Unable to connect to server. Please try again later.');
      console.error('API Error:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleReset = (): void => {
    setModelResults(sampleFolkPoetry.modelResults);
    setError('');
    setInputText('');
    setSelectedModel('all');
  };

  // Lọc kết quả theo model đã chọn
  const getFilteredResults = (): ModelResult[] => {
    if (selectedModel === 'all') {
      return modelResults;
    }
    return modelResults.filter(model => model.model_type === selectedModel);
  };

  // Tính tổng số kết quả
  const getTotalResultsCount = (): number => {
    const filtered = getFilteredResults();
    return filtered.reduce((total, model) => total + (model.results?.length || 0), 0);
  };

  // Lấy tên model để hiển thị
  const getModelDisplayName = (modelType: ModelType): string => {
    switch (modelType) {
      case 'n_gram': return 'N-Gram';
      case 'retrieval': return 'Retrieval';
      case 'ensemble': return 'Ensemble';
      case 'all': return 'All Models';
      default: return modelType;
    }
  };

  return (
    <div className="app">
      <header className="header">
        <div className="container">
          <div className="header-content">
            <div className="logo-title">
              <div className="logo-icon">
                <i className="fas fa-bamboo"></i>
              </div>
              <div>
                <h1>Vietnamese Folk Poetry & Proverbs</h1>
                <p className="subtitle">Discover Cultural Heritage Through NLP Models</p>
              </div>
            </div>
            <p className="project-tagline">
              Explore Vietnamese folk poetry using N-Gram, Retrieval, and Ensemble models.
            </p>
          </div>
        </div>
      </header>

      <main className="main-content">
        <div className="container">
          {/* Daily Highlight */}
          <div className="daily-highlight">
            <div className="highlight-content">
              <h2>
                <i className="fas fa-star"></i>
                Verse of the Day
              </h2>
              <p>{dailyPoetry}</p>
              <div className="highlight-meta">
                <span><i className="fas fa-history"></i> Traditional</span>
                <span><i className="fas fa-heart"></i> Popular</span>
                <span><i className="fas fa-map-marker-alt"></i> Vietnam</span>
              </div>
            </div>
          </div>

          <div className="app-layout">
            {/* Left Panel */}
            <div className="left-panel">
              <div className="info-card">
                <div className="card-header">
                  <i className="fas fa-brain"></i>
                  <h2>NLP Models</h2>
                </div>
                <p>
                  Three specialized models for Vietnamese folk poetry analysis:
                </p>
                <div className="model-descriptions">
                  <div className="model-tag ngram-tag">
                    <i className="fas fa-chart-line"></i>
                    <span>N-Gram: Text generation</span>
                  </div>
                  <div className="model-tag retrieval-tag">
                    <i className="fas fa-search"></i>
                    <span>Retrieval: Similarity search</span>
                  </div>
                  <div className="model-tag ensemble-tag">
                    <i className="fas fa-robot"></i>
                    <span>Ensemble: Combined results</span>
                  </div>
                </div>
              </div>

              <PredictionForm
                onSubmit={handleSubmit}
                onReset={handleReset}
                disabled={loading}
              />

              {error && (
                <div className="error-message">
                  <i className="fas fa-exclamation-circle"></i> {error}
                </div>
              )}
            </div>

            {/* Right Panel */}
            <div className="right-panel">
              {loading ? (
                <div className="loading-container">
                  <LoadingSpinner />
                </div>
              ) : modelResults.length > 0 ? (
                <div className="results-section">
                  <div className="results-header">
                    <div>
                      <h2>
                        <i className="fas fa-scroll"></i>
                        {getModelDisplayName(selectedModel)} Results
                      </h2>
                      <p className="results-count">
                        {getTotalResultsCount()} verses
                        {inputText && ` for: "${inputText}"`}
                      </p>
                    </div>
                    <div className="model-selector">
                      <div className="model-filter">
                        <button 
                          className={`model-filter-btn ${selectedModel === 'all' ? 'active' : ''}`}
                          onClick={() => setSelectedModel('all')}
                        >
                          All
                        </button>
                        <button 
                          className={`model-filter-btn ngram ${selectedModel === 'n_gram' ? 'active' : ''}`}
                          onClick={() => setSelectedModel('n_gram')}
                        >
                          N-Gram
                        </button>
                        <button 
                          className={`model-filter-btn retrieval ${selectedModel === 'retrieval' ? 'active' : ''}`}
                          onClick={() => setSelectedModel('retrieval')}
                        >
                          Retrieval
                        </button>
                        <button 
                          className={`model-filter-btn ensemble ${selectedModel === 'ensemble' ? 'active' : ''}`}
                          onClick={() => setSelectedModel('ensemble')}
                        >
                          Ensemble
                        </button>
                      </div>
                      <div className="results-controls">
                        <button className="icon-btn" title="Download">
                          <i className="fas fa-download"></i>
                        </button>
                      </div>
                    </div>
                  </div>

                  <div className="results-container">
                    {getFilteredResults().map((modelResult, modelIndex) => (
                      <div key={modelIndex} className="model-results-group">
                        <div className="model-header">
                          <h3 className="model-title">
                            <i className={`fas ${modelResult.model_type === 'n_gram' ? 'fa-chart-line' : 
                                          modelResult.model_type === 'retrieval' ? 'fa-search' : 
                                          'fa-robot'}`}></i>
                            {modelResult.model_name}
                          </h3>
                          <span className="model-type-badge">{modelResult.model_type}</span>
                          {modelResult.metrics?.inference_time && (
                            <span className="model-metric">
                              <i className="fas fa-clock"></i> {modelResult.metrics.inference_time.toFixed(2)}s
                            </span>
                          )}
                        </div>
                        
                        <div className="model-results">
                          {modelResult.results.map((result, resultIndex) => (
                            <ResultCard
                              key={`${modelIndex}-${resultIndex}`}
                              result={result}
                              index={resultIndex + 1}
                              modelType={modelResult.model_type}
                            />
                          ))}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              ) : (
                <div className="placeholder-section">
                  <div className="placeholder-content">
                    <i className="fas fa-feather-alt"></i>
                    <h3>No Results Yet</h3>
                    <p>Enter Vietnamese text to see predictions from all three models.</p>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      </main>

      <footer className="footer">
        <div className="container">
          <div className="footer-content">
            <div className="footer-left">
              <p>Vietnamese Folk Poetry & Proverbs • NLP Analysis System</p>
              <p className="copyright">N-Gram, Retrieval, and Ensemble Models</p>
            </div>
            <div className="footer-right">
              <a
                href="https://github.com/imninh/Cadao-Tucngu-NLP.git"
                target="_blank"
                rel="noopener noreferrer"
                className="github-link"
              >
                <i className="fab fa-github"></i>
                <span>Source Code</span>
              </a>
              <p className="copyright">© {new Date().getFullYear()} NLP Research Group</p>
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
};

export default App;
