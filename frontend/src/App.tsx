import { useState, type SetStateAction } from 'react'
import { motion, AnimatePresence } from 'framer-motion'

function App() {
  const [input, setInput] = useState('')
  const [candidates, setCandidates] = useState([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [warning, setWarning] = useState('')
  const [metadata, setMetadata] = useState(null)
  const [selectedModel, setSelectedModel] = useState('retrieval')

  const API_URL = 'http://localhost:5000'

  const examples = [
    { text: 'ăn quả nhớ', desc: 'Ca dao về lòng biết ơn' },
    { text: 'có công mài sắt', desc: 'Tục ngữ về sự chăm chỉ' },
    { text: 'gần mực', desc: 'Ảnh hưởng môi trường' },
    { text: 'học thầy không', desc: 'Ca dao về việc học' },
    { text: 'lửa thử', desc: 'Tục ngữ về thử thách' },
  ]

  const models = [
    { id: 'retrieval', name: 'Retrieval', description: 'Tìm kiếm ngữ nghĩa (TF-IDF)', color: 'from-purple-500 to-pink-500' },
    { id: 'ngram', name: 'N-gram', description: 'Mô hình ngôn ngữ (Trigram)', color: 'from-blue-500 to-cyan-500' },
    { id: 'ensemble', name: 'Ensemble', description: 'Kết hợp cả hai phương pháp', color: 'from-green-500 to-emerald-500' }
  ]

  const handleSubmit = async (e: { preventDefault: () => void }) => {
    if (e) e.preventDefault()
    
    setError('')
    setWarning('')
    setCandidates([])
    setMetadata(null)
    
    if (!input.trim()) {
      setError('Vui lòng nhập phần đầu câu ca dao')
      return
    }

    setLoading(true)

    try {
      const response = await fetch(`${API_URL}/api/complete`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          input: input,
          top_k: 3,
          model: selectedModel
        })
      })

      const data = await response.json()

      if (!response.ok) {
        setError(data.error || 'Có lỗi xảy ra')
        if (data.suggestion) {
          setError(prev => prev + '\n💡 ' + data.suggestion)
        }
      } else {
        setCandidates(data.candidates)
        setMetadata(data.metadata)
        
        if (data.warning) {
          setWarning(data.warning)
        }
      }
    } catch (err) {
      setError('⚠️ Không thể kết nối với server. Hãy chắc chắn backend đang chạy ở http://localhost:5000')
    } finally {
      setLoading(false)
    }
  }

  const handleExampleClick = (text: SetStateAction<string>) => {
    setInput(text)
    setError('')
    setWarning('')
    setCandidates([])
  }

  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 0.8) return 'bg-gradient-to-r from-green-400 to-emerald-500'
    if (confidence >= 0.6) return 'bg-gradient-to-r from-yellow-400 to-amber-500'
    if (confidence >= 0.4) return 'bg-gradient-to-r from-orange-400 to-red-500'
    return 'bg-gradient-to-r from-red-400 to-pink-500'
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-purple-50 p-4 sm:p-8">
      <div className="max-w-6xl mx-auto">
        
        {/* Header */}
        <motion.div 
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
          className="text-center mb-12"
        >
          <div className="inline-block p-2 bg-gradient-to-r from-purple-500 to-pink-500 rounded-2xl mb-4">
            <h1 className="text-5xl sm:text-6xl font-bold text-white px-8 py-4 rounded-xl">
              🎭 CaDao AI
            </h1>
          </div>
          <p className="text-2xl text-slate-700 mb-2 font-medium">
            Hoàn Thiện Ca Dao Tục Ngữ Việt Nam
          </p>
          <p className="text-slate-500 max-w-2xl mx-auto">
            Sử dụng AI để khám phá và hoàn thiện những câu ca dao, tục ngữ của dân tộc
          </p>
        </motion.div>

        {/* Main Content */}
        <div className="grid lg:grid-cols-3 gap-8">
          
          {/* Left Panel - Input & Models */}
          <div className="lg:col-span-2 space-y-8">
            
            {/* Model Selection */}
            <motion.div 
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.5, delay: 0.1 }}
              className="bg-white/80 backdrop-blur-sm rounded-2xl shadow-xl p-6 border border-slate-200"
            >
              <h2 className="text-2xl font-bold text-slate-800 mb-4 flex items-center gap-2">
                <span className="p-2 bg-gradient-to-r from-purple-500 to-pink-500 rounded-lg text-white">🤖</span>
                Chọn phương pháp AI
              </h2>
              <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
                {models.map((model) => (
                  <motion.button
                    key={model.id}
                    whileHover={{ scale: 1.05 }}
                    whileTap={{ scale: 0.95 }}
                    onClick={() => setSelectedModel(model.id)}
                    className={`p-4 rounded-xl transition-all duration-300 ${
                      selectedModel === model.id
                        ? `bg-gradient-to-br ${model.color} text-white shadow-lg transform scale-105`
                        : 'bg-white border-2 border-slate-200 hover:border-slate-300 text-slate-700'
                    }`}
                  >
                    <div className="text-lg font-semibold mb-1">{model.name}</div>
                    <div className={`text-sm ${selectedModel === model.id ? 'text-white/90' : 'text-slate-500'}`}>
                      {model.description}
                    </div>
                  </motion.button>
                ))}
              </div>
            </motion.div>

            {/* Input Form */}
            <motion.div 
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.5, delay: 0.2 }}
              className="bg-white/80 backdrop-blur-sm rounded-2xl shadow-xl p-6 border border-slate-200"
            >
              <h2 className="text-2xl font-bold text-slate-800 mb-4 flex items-center gap-2">
                <span className="p-2 bg-gradient-to-r from-blue-500 to-cyan-500 rounded-lg text-white">✍️</span>
                Nhập câu của bạn
              </h2>
              <form onSubmit={handleSubmit} className="space-y-6">
                <div className="relative">
                  <input
                    type="text"
                    placeholder="Nhập phần đầu câu... (VD: ăn quả nhớ)"
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    disabled={loading}
                    className="w-full px-6 py-4 text-xl border-2 border-slate-300 rounded-xl focus:outline-none focus:border-blue-500 focus:ring-4 focus:ring-blue-200 transition-all duration-300 disabled:bg-slate-100 placeholder-slate-400"
                  />
                  <div className="absolute right-3 top-1/2 transform -translate-y-1/2 text-slate-400">
                    {input.length > 0 && `${input.length} ký tự`}
                  </div>
                </div>
                
                <motion.button
                  type="submit"
                  disabled={loading}
                  whileHover={{ scale: 1.02 }}
                  whileTap={{ scale: 0.98 }}
                  className={`w-full py-4 rounded-xl font-bold text-white text-xl shadow-lg transition-all ${
                    loading 
                      ? 'bg-slate-400 cursor-not-allowed' 
                      : 'bg-gradient-to-r from-blue-500 to-purple-600 hover:shadow-2xl hover:from-blue-600 hover:to-purple-700'
                  }`}
                >
                  {loading ? (
                    <span className="flex items-center justify-center gap-3">
                      <svg className="animate-spin h-6 w-6 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                      </svg>
                      Đang xử lý...
                    </span>
                  ) : (
                    '✨ Hoàn Thiện Câu'
                  )}
                </motion.button>
              </form>
            </motion.div>

            {/* Examples */}
            <motion.div 
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.5, delay: 0.3 }}
              className="bg-white/80 backdrop-blur-sm rounded-2xl shadow-xl p-6 border border-slate-200"
            >
              <h3 className="text-xl font-bold text-slate-800 mb-4 flex items-center gap-2">
                <span className="p-2 bg-gradient-to-r from-amber-500 to-orange-500 rounded-lg text-white">💡</span>
                Ví dụ nhanh
              </h3>
              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
                {examples.map((example, index) => (
                  <motion.button
                    key={index}
                    whileHover={{ scale: 1.05, y: -2 }}
                    whileTap={{ scale: 0.95 }}
                    onClick={() => handleExampleClick(example.text)}
                    className="p-4 bg-gradient-to-br from-white to-slate-50 border-2 border-slate-200 rounded-xl hover:border-blue-300 hover:shadow-md transition-all duration-300 group text-left"
                  >
                    <div className="font-semibold text-slate-800 mb-2 group-hover:text-blue-600 transition-colors">
                      "{example.text}"
                    </div>
                    <div className="text-sm text-slate-600 flex items-center gap-2">
                      <span className="text-blue-500">📖</span>
                      {example.desc}
                    </div>
                  </motion.button>
                ))}
              </div>
            </motion.div>
          </div>

          {/* Right Panel - Results */}
          <div className="lg:col-span-1">
            <AnimatePresence>
              {/* Warning */}
              {warning && (
                <motion.div
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -20 }}
                  className="mb-6"
                >
                  <div className="bg-gradient-to-r from-amber-50 to-yellow-50 border-l-4 border-amber-500 p-4 rounded-xl shadow-lg">
                    <div className="flex items-start gap-3">
                      <span className="text-2xl">⚠️</span>
                      <p className="text-amber-800 font-medium">{warning}</p>
                    </div>
                  </div>
                </motion.div>
              )}

              {/* Error */}
              {error && (
                <motion.div
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -20 }}
                  className="mb-6"
                >
                  <div className="bg-gradient-to-r from-red-50 to-pink-50 border-l-4 border-red-500 p-4 rounded-xl shadow-lg">
                    <div className="flex items-start gap-3">
                      <span className="text-2xl">❌</span>
                      <p className="text-red-800 font-medium whitespace-pre-line">{error}</p>
                    </div>
                  </div>
                </motion.div>
              )}

              {/* Results */}
              {candidates.length > 0 && (
                <motion.div
                  initial={{ opacity: 0, scale: 0.9 }}
                  animate={{ opacity: 1, scale: 1 }}
                  className="bg-gradient-to-br from-white to-blue-50 rounded-2xl shadow-2xl p-6 border border-blue-100"
                >
                  <div className="flex items-center justify-between mb-6 pb-4 border-b border-blue-200">
                    <div>
                      <h2 className="text-2xl font-bold text-slate-800 flex items-center gap-2">
                        <span className="p-2 bg-gradient-to-r from-green-500 to-emerald-500 rounded-lg text-white">📊</span>
                        Kết quả
                      </h2>
                      {metadata?.is_ambiguous && (
                        <span className="inline-block mt-2 px-3 py-1 bg-gradient-to-r from-amber-100 to-orange-100 text-amber-800 text-xs font-semibold rounded-full">
                          ⚠️ Có nhiều đáp án
                        </span>
                      )}
                    </div>
                    <div className="text-right">
                      <div className="text-xs text-slate-500 mb-1">Model sử dụng</div>
                      <div className="px-3 py-1 bg-gradient-to-r from-purple-100 to-pink-100 text-purple-700 font-semibold rounded-full">
                        {models.find(m => m.id === metadata?.model_used)?.name || 'Retrieval'}
                      </div>
                    </div>
                  </div>

                  <div className="space-y-4">
                    {candidates.map((candidate, index) => (
                      <motion.div
                        key={index}
                        initial={{ opacity: 0, x: 20 }}
                        animate={{ opacity: 1, x: 0 }}
                        transition={{ delay: index * 0.1 }}
                        className="bg-white p-4 rounded-xl shadow-lg border border-slate-200 hover:shadow-xl transition-shadow"
                      >
                        <div className="flex items-center justify-between mb-3">
                          <div className="flex items-center gap-2">
                            <span className={`px-3 py-1 rounded-full text-white font-bold ${getConfidenceColor(candidate.confidence)}`}>
                              #{index + 1}
                            </span>
                            <span className="text-xs text-slate-500 bg-slate-100 px-2 py-1 rounded">
                              {candidate.model}
                            </span>
                          </div>
                          <div className="text-right">
                            <div className="text-xs text-slate-500">Độ tin cậy</div>
                            <div className="text-2xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
                              {(candidate.confidence * 100).toFixed(0)}%
                            </div>
                          </div>
                        </div>

                        <p className="text-lg text-slate-800 font-medium mb-3 leading-relaxed">
                          {candidate.text}
                        </p>

                        {/* Confidence bar */}
                        <div className="relative">
                          <div className="absolute right-0 top-0 text-xs font-semibold text-slate-600">
                            {candidate.confidence.toFixed(2)}
                          </div>
                          <div className="mt-2 bg-slate-200 rounded-full h-3 overflow-hidden">
                            <motion.div
                              initial={{ width: 0 }}
                              animate={{ width: `${candidate.confidence * 100}%` }}
                              transition={{ duration: 1, delay: index * 0.2 }}
                              className={`h-3 rounded-full ${getConfidenceColor(candidate.confidence)}`}
                            />
                          </div>
                        </div>

                        {candidate.similarity && (
                          <div className="mt-2 text-xs text-slate-500">
                            Độ tương đồng: <span className="font-semibold">{(candidate.similarity * 100).toFixed(1)}%</span>
                          </div>
                        )}
                      </motion.div>
                    ))}
                  </div>

                  <div className="mt-6 pt-4 border-t border-blue-200">
                    <p className="text-sm text-slate-600 italic">
                      💡 <strong>Giải thích:</strong> Kết quả được sắp xếp theo độ tin cậy, giá trị càng cao càng phù hợp với câu đầy đủ.
                    </p>
                  </div>
                </motion.div>
              )}

              {/* Empty State */}
              {!candidates.length && !loading && !error && !warning && (
                <motion.div
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  className="bg-gradient-to-br from-white to-slate-50 rounded-2xl shadow-xl p-8 text-center border border-slate-200"
                >
                  <div className="text-6xl mb-4">📝</div>
                  <h3 className="text-xl font-bold text-slate-700 mb-2">Chưa có kết quả</h3>
                  <p className="text-slate-500">
                    Nhập một câu ca dao hoặc chọn ví dụ để bắt đầu
                  </p>
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        </div>

        {/* Footer */}
        <motion.div 
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.5, delay: 0.8 }}
          className="mt-12 text-center"
        >
          <div className="bg-gradient-to-r from-white/50 to-slate-100/50 backdrop-blur-sm rounded-2xl p-6 shadow-lg border border-slate-200">
            <div className="flex flex-col sm:flex-row items-center justify-between gap-4">
              <div className="text-left">
                <div className="text-slate-800 font-bold text-lg">CaDao AI</div>
                <div className="text-slate-600">Dự án NLP - Bảo tồn văn hóa Việt</div>
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold bg-gradient-to-r from-purple-600 to-pink-600 bg-clip-text text-transparent">
                  {metadata?.total_found || '1,492'}
                </div>
                <div className="text-sm text-slate-500">Câu ca dao trong dataset</div>
              </div>
              <div className="text-right">
                <div className="text-slate-700 font-medium">Made with ❤️</div>
                <div className="text-xs text-slate-500">by NLP Research Team</div>
              </div>
            </div>
          </div>
        </motion.div>
      </div>
    </div>
  )
}

export default App