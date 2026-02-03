import React from 'react'
import ReactDOM from 'react-dom/client'
import './index.css'
import App from './App'

// Import font cho giao diện truyền thống
const link = document.createElement('link')
link.href = 'https://fonts.googleapis.com/css2?family=Merriweather:wght@400;700&family=Noto+Serif:wght@400;600&display=swap'
link.rel = 'stylesheet'
document.head.appendChild(link)

// Import Font Awesome cho icon
const fontAwesome = document.createElement('link')
fontAwesome.href = 'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css'
fontAwesome.rel = 'stylesheet'
document.head.appendChild(fontAwesome)

const rootElement = document.getElementById('root')

if (!rootElement) {
  throw new Error('Root element with id "root" not found')
}

ReactDOM.createRoot(rootElement).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
)