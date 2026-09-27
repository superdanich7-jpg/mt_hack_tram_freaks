import { formatDateTime, formatNumber } from '../lib/format'
import type { LayoutMode } from '../types'
import './Header.css'

interface HeaderProps {
  layoutMode: LayoutMode
  onLayoutModeChange: (mode: LayoutMode) => void
  onExport: () => void
  exportDisabled: boolean
  exportRowCount: number
  generatedAt: string
  /** Что является источником данных: бэкенд или расчётный профиль. */
  dataSourceLabel: string
}

const LAYOUT_OPTIONS: { id: LayoutMode; label: string }[] = [
  { id: 'split', label: 'Сплит-режим' },
  { id: 'map', label: 'Только карта' },
  { id: 'charts', label: 'Только графики' },
]

function LayoutIcon({ mode }: { mode: LayoutMode }) {
  if (mode === 'map') {
    return (
      <svg viewBox="0 0 16 16" aria-hidden="true" width="15" height="15">
        <rect x="1.5" y="2.5" width="13" height="11" rx="2" fill="currentColor" opacity="0.85" />
      </svg>
    )
  }
  if (mode === 'charts') {
    return (
      <svg viewBox="0 0 16 16" aria-hidden="true" width="15" height="15">
        <rect x="1.5" y="9" width="3" height="4.5" rx="1" fill="currentColor" opacity="0.85" />
        <rect x="6.5" y="5.5" width="3" height="8" rx="1" fill="currentColor" opacity="0.85" />
        <rect x="11.5" y="2.5" width="3" height="11" rx="1" fill="currentColor" opacity="0.85" />
      </svg>
    )
  }
  return (
    <svg viewBox="0 0 16 16" aria-hidden="true" width="15" height="15">
      <rect x="1.5" y="2.5" width="6" height="11" rx="1.5" fill="currentColor" opacity="0.85" />
      <rect x="8.5" y="2.5" width="6" height="5" rx="1.5" fill="currentColor" opacity="0.85" />
      <rect x="8.5" y="8.5" width="6" height="5" rx="1.5" fill="currentColor" opacity="0.55" />
    </svg>
  )
}

export default function Header({
  layoutMode,
  onLayoutModeChange,
  onExport,
  exportDisabled,
  exportRowCount,
  generatedAt,
  dataSourceLabel,
}: HeaderProps) {
  return (
    <header className="app-header">
      <div className="app-header__brand">
        <img
          className="app-header__logo"
          src="/logo.png"
          alt="Логотип проекта «Московский трамвай»"
          height={36}
        />
        <div>
          <h1 className="app-header__title">Прогноз пассажиропотока трамваев Москвы</h1>
          <p className="app-header__subtitle">
            Данные: {dataSourceLabel} · обновлено {formatDateTime(generatedAt)}
          </p>
        </div>
      </div>

      <div className="app-header__actions">
        <div className="layout-switch" role="group" aria-label="Режим экрана">
          {LAYOUT_OPTIONS.map((option) => (
            <button
              key={option.id}
              type="button"
              title={option.label}
              aria-pressed={layoutMode === option.id}
              className="layout-switch__button"
              onClick={() => onLayoutModeChange(option.id)}
            >
              <LayoutIcon mode={option.id} />
              <span>{option.label}</span>
            </button>
          ))}
        </div>

        <button
          type="button"
          className="export-button"
          onClick={onExport}
          disabled={exportDisabled}
          title="Скачать текущие отображаемые данные в CSV"
        >
          <svg viewBox="0 0 16 16" width="15" height="15" aria-hidden="true">
            <path
              d="M8 1.5v8m0 0 3-3m-3 3-3-3M2.5 11v2a1.5 1.5 0 0 0 1.5 1.5h8A1.5 1.5 0 0 0 13.5 13v-2"
              fill="none"
              stroke="currentColor"
              strokeWidth="1.5"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          </svg>
          Экспорт в CSV
          <span className="export-button__count">
            {exportDisabled ? 'нет данных' : `${formatNumber(exportRowCount)} стр.`}
          </span>
        </button>
      </div>
    </header>
  )
}
