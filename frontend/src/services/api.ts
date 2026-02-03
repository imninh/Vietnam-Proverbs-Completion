const API_BASE_URL = 'http://localhost:5000/api' // URL của backend Flask

// ==== TYPES ====

export interface PredictionItem {
  text: string
  confidence: number
  score?: number
  source?: string
  metadata?: Record<string, any>
}

export interface ModelResult {
  model_name: string
  model_type: 'n_gram' | 'retrieval' | 'ensemble'
  results: PredictionItem[]
  metrics?: {
    inference_time?: number
    total_results?: number
  }
}

export interface PredictionResponse {
  input_text: string
  timestamp: string
  model_results: ModelResult[]
}

// ==== REAL API CALL ====

/**
 * Send text input to multiple models for prediction
 * @param inputText Vietnamese text input
 * @returns Promise with prediction results from all models
 */
export const getPredictions = async (
  inputText: string
): Promise<PredictionResponse> => {
  try {
    const response = await fetch(`${API_BASE_URL}/predict`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({ 
        text: inputText,
        models: ['n_gram', 'retrieval', 'ensemble'] // Yêu cầu cả 3 model
      })
    })

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`)
    }

    const data: PredictionResponse = await response.json()
    return data
  } catch (error) {
    console.error('Failed to fetch predictions:', error)
    // Fallback to mock data if API fails
    return getMockPredictions(inputText)
  }
}

// ==== MOCK API (FOR TESTING) ====

/**
 * Mock API response for testing (when backend is not available)
 */
export const getMockPredictions = async (
  inputText: string
): Promise<PredictionResponse> => {
  // Simulate API delay
  await new Promise<void>((resolve) => setTimeout(resolve, 800))

  const mockResults: ModelResult[] = [
    {
      model_name: "N-Gram Model",
      model_type: "n_gram",
      results: [
        { 
          text: `${inputText} đẹp như tranh vẽ.`, 
          confidence: 0.85,
          source: "n_gram_generation"
        },
        { 
          text: `${inputText} trong như ngọc.`, 
          confidence: 0.78,
          source: "n_gram_completion"
        },
        { 
          text: `${inputText} ngọt như mía lùi.`, 
          confidence: 0.72,
          source: "n_gram_prediction"
        }
      ],
      metrics: {
        inference_time: 0.15,
        total_results: 3
      }
    },
    {
      model_name: "Retrieval Model",
      model_type: "retrieval",
      results: [
        { 
          text: "Ai ơi bưng bát cơm đầy, Dẻo thơm một hạt đắng cay muôn phần.", 
          confidence: 0.92,
          source: "ca_dao_collection",
          metadata: { similarity: 0.89 }
        },
        { 
          text: "Chim khôn kêu tiếng rảnh rang, Người khôn nói tiếng dịu dàng dễ nghe.", 
          confidence: 0.88,
          source: "tuc_ngu_database",
          metadata: { similarity: 0.85 }
        },
        { 
          text: "Lời nói không mất tiền mua, Lựa lời mà nói cho vừa lòng nhau.", 
          confidence: 0.85,
          source: "folk_poetry_corpus",
          metadata: { similarity: 0.82 }
        }
      ],
      metrics: {
        inference_time: 0.08,
        total_results: 3
      }
    },
    {
      model_name: "Ensemble Model",
      model_type: "ensemble",
      results: [
        { 
          text: `${inputText} đẹp như tranh vẽ, Chim khôn kêu tiếng rảnh rang.`, 
          confidence: 0.95,
          source: "hybrid_generation",
          metadata: { combined_score: 0.91 }
        },
        { 
          text: `Ai ơi nhớ mãi ${inputText}, Lời nói không mất tiền mua.`, 
          confidence: 0.89,
          source: "ensemble_fusion",
          metadata: { combined_score: 0.87 }
        },
        { 
          text: `${inputText} trong như ngọc, Người khôn nói tiếng dịu dàng.`, 
          confidence: 0.86,
          source: "model_combination",
          metadata: { combined_score: 0.83 }
        }
      ],
      metrics: {
        inference_time: 0.25,
        total_results: 3
      }
    }
  ]

  return {
    input_text: inputText,
    timestamp: new Date().toISOString(),
    model_results: mockResults
  }
}
