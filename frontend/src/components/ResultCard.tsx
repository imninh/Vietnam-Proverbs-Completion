import React from 'react'
import './ResultCard.css'

interface ResultData {
  text: string;
  confidence: number;
  source?: string;
  category?: string;
  metadata?: Record<string, any>;
}

interface ResultCardProps {
  result: ResultData;
  index: number;
  modelType?: 'n_gram' | 'retrieval' | 'ensemble';
}

const ResultCard: React.FC<ResultCardProps> = ({ result, index, modelType }) => {
  const confidencePercent = (result.confidence * 100).toFixed(0);
  
  const getConfidenceLevel = (confidence: number): string => {
    if (confidence >= 0.8) return 'high';
    if (confidence >= 0.5) return 'medium';
    return 'low';
  };

  const getModelColor = (): string => {
    switch (modelType) {
      case 'n_gram': return '#8b4513'; // Brown
      case 'retrieval': return '#2e8b57'; // Green
      case 'ensemble': return '#b22222'; // Red
      default: return '#8b4513'; // Default brown
    }
  };

  const confidenceLevel = getConfidenceLevel(result.confidence);
  const modelColor = getModelColor();

  return (
    <div className="result-card" style={{ borderLeft: `4px solid ${modelColor}` }}>
      <div className="card-header">
        <span className="card-number">#{index}</span>
        <div className="card-title">
          <h3>
            {modelType === 'n_gram' ? 'Generated' : 
             modelType === 'retrieval' ? 'Retrieved' : 
             modelType === 'ensemble' ? 'Ensemble' : 'Folk'} Verse
          </h3>
          <span className={`confidence-badge ${confidenceLevel}`}>
            {confidencePercent}%
          </span>
        </div>
      </div>
      
      <div className="card-body">
        <p className="verse-text">"{result.text}"</p>
        
        <div className="card-meta">
          {result.source && (
            <div className="meta-item">
              <span className="meta-label">Source:</span>
              <span className="meta-value">{result.source}</span>
            </div>
          )}
          
          {result.category && (
            <div className="meta-item">
              <span className="meta-label">Category:</span>
              <span className="meta-value">{result.category}</span>
            </div>
          )}

          {result.metadata && (
            <div className="meta-item">
              <span className="meta-label">Metadata:</span>
              <span className="meta-value">
                {JSON.stringify(result.metadata)}
              </span>
            </div>
          )}
          
          <div className="confidence-bar">
            <div 
              className={`confidence-fill ${confidenceLevel}`}
              style={{ 
                width: `${confidencePercent}%`,
                backgroundColor: modelColor 
              }}
            ></div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ResultCard;
