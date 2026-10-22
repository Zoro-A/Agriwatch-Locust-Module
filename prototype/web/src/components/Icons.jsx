import React from 'react'

export const Mark = ({ size = 40 }) => (
  <svg width={size} height={size} viewBox="0 0 40 40" className="brand-mark" aria-hidden="true">
    <rect width="40" height="40" rx="9" fill="#1f3342" stroke="rgba(233,238,245,.2)" />
    <rect x="7" y="7" width="8" height="8" fill="#1f5fbf" />
    <rect x="16" y="7" width="8" height="8" fill="#1f5fbf" />
    <rect x="25" y="7" width="8" height="8" fill="#14202b" fillOpacity=".25" />
    <rect x="7" y="16" width="8" height="8" fill="#14202b" fillOpacity=".25" />
    <rect x="16" y="16" width="8" height="8" fill="#1f5fbf" />
    <rect x="25" y="16" width="8" height="8" fill="#f08a24" />
    <rect x="7" y="25" width="8" height="8" fill="#14202b" fillOpacity=".25" />
    <rect x="16" y="25" width="8" height="8" fill="#14202b" fillOpacity=".25" />
    <rect x="25" y="25" width="8" height="8" fill="#1f5fbf" fillOpacity=".7" />
  </svg>
)

export const Info = (p) => (
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" {...p}>
    <circle cx="12" cy="12" r="9.5" /><path d="M12 11v6M12 7.5v.01" />
  </svg>
)
export const Tick = (p) => (
  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#16803c" strokeWidth="2.4" strokeLinecap="round" strokeLinejoin="round" {...p}><path d="M4 12.5l5 5L20 6.5" /></svg>
)
export const Cross = (p) => (
  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#194fa0" strokeWidth="2.4" strokeLinecap="round" {...p}><path d="M6 6l12 12M18 6L6 18" /></svg>
)
export const Dot = (p) => (
  <svg width="18" height="18" viewBox="0 0 24 24" {...p}><circle cx="12" cy="12" r="4.5" fill="#5b6877" /></svg>
)
export const Play = () => (<svg width="14" height="14" viewBox="0 0 14 14" aria-hidden="true"><path d="M3 1.5v11l9-5.5z" fill="currentColor" /></svg>)
export const Pause = () => (<svg width="14" height="14" viewBox="0 0 14 14" aria-hidden="true"><path d="M3 1.5h3v11H3zM8 1.5h3v11H8z" fill="currentColor" /></svg>)
