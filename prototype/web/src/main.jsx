import React from 'react'
import { createRoot } from 'react-dom/client'
import '@fontsource-variable/space-grotesk'
import '@fontsource-variable/inter'
import '@fontsource-variable/jetbrains-mono'
import './styles.css'
import './tools.css'
import App from './App.jsx'

createRoot(document.getElementById('root')).render(<App />)
