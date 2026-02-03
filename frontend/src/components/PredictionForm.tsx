import React, { useState, useCallback, memo } from 'react'
import type { FormEvent, ChangeEvent } from 'react'
import './PredictionForm.css'

interface PredictionFormProps {
  onSubmit: (inputText: string) => void
  onReset: () => void
  disabled: boolean
  maxLength?: number
}

const PredictionForm: React.FC<PredictionFormProps> = memo(({
  onSubmit,
  onReset,
  disabled,
  maxLength = 300
}) => {
  const [inputText, setInputText] = useState<string>('')

  const handleSubmit = useCallback((e: FormEvent<HTMLFormElement>): void => {
    e.preventDefault()
    if (inputText.trim() && !disabled) {
      onSubmit(inputText)
    }
  }, [inputText, disabled, onSubmit])

  const handleReset = useCallback((): void => {
    setInputText('')
    onReset()
  }, [onReset])

  const handleChange = useCallback((e: ChangeEvent<HTMLTextAreaElement>): void => {
    const value = e.target.value
    if (value.length <= maxLength) {
      setInputText(value)
    }
  }, [maxLength])

  const handleClear = useCallback((): void => {
    setInputText('')
  }, [])

  const isNearLimit = inputText.length > maxLength * 0.9
  const isAtLimit = inputText.length >= maxLength

  return (
    <div className="prediction-form">
      <header className="form-header">
        <h2 className="form-title">
          <i className="fas fa-feather-alt"></i>
          Input Text
        </h2>
        <p className="form-description">
          Enter Vietnamese text to find related folk poetry
        </p>
      </header>

      <form onSubmit={handleSubmit} className="form-content" noValidate>
        <div className="form-group">
          <label htmlFor="text-input" className="form-label">
            Vietnamese Text
          </label>
          <textarea
            id="text-input"
            className={`form-textarea ${isAtLimit ? 'form-textarea--limit' : ''}`}
            value={inputText}
            onChange={handleChange}
            placeholder="Type here..."
            rows={3}
            disabled={disabled}
            maxLength={maxLength}
          />
          
          <div className="form-meta">
            <button
              type="button"
              className="btn-clear"
              onClick={handleClear}
              disabled={!inputText || disabled}
            >
              <i className="fas fa-times"></i> Clear
            </button>
            <span className={`char-count ${isNearLimit ? 'char-count--warning' : ''}`}>
              {inputText.length}/{maxLength}
            </span>
          </div>
        </div>

        <div className="form-actions">
          <button
            type="button"
            className="btn btn--secondary"
            onClick={handleReset}
            disabled={disabled}
          >
            <i className="fas fa-redo-alt"></i>
            Reset
          </button>

          <button
            type="submit"
            className="btn btn--primary"
            disabled={disabled || !inputText.trim()}
          >
            {disabled ? (
              <>
                <i className="fas fa-spinner fa-spin"></i>
                Searching...
              </>
            ) : (
              <>
                <i className="fas fa-search"></i>
                Search
              </>
            )}
          </button>
        </div>
      </form>
    </div>
  )
})

PredictionForm.displayName = 'PredictionForm'

export default PredictionForm