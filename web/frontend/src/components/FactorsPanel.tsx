import { EVENT_OPTIONS, WEATHER_OPTIONS } from '../data/forecast'
import { formatNumber1 } from '../lib/format'
import type { AppliedFactor, EventId, ExternalFactors, WeatherId } from '../types'
import './FactorsPanel.css'

interface FactorsPanelProps {
  factors: ExternalFactors
  applied: AppliedFactor[]
  volumeMultiplier: number
  trafficSmoothing: number
  onChange: (patch: Partial<ExternalFactors>) => void
  onReset: () => void
}

/**
 * Панель «Внешние факторы»: переключатель выходного дня, селектор погоды и ползунок пробок.
 * Любое изменение сразу пересчитывает мок-прогноз (см. src/data/forecast.ts).
 */
export default function FactorsPanel({
  factors,
  applied,
  volumeMultiplier,
  trafficSmoothing,
  onChange,
  onReset,
}: FactorsPanelProps) {
  return (
    <section className="factors-panel" aria-label="Внешние факторы">
      <div className="factors-panel__title">
        <h2>Внешние факторы</h2>
        <p>Корректировки прогноза применяются мгновенно</p>
      </div>

      <label className="switch">
        <input
          type="checkbox"
          className="switch__input"
          checked={factors.holiday}
          onChange={(event) => onChange({ holiday: event.target.checked })}
        />
        <span className="switch__track" aria-hidden="true">
          <span className="switch__thumb" />
        </span>
        <span className="switch__text">Выходной / праздничный день</span>
      </label>

      <label className="factor-field">
        <span className="factor-field__label">Погода</span>
        <span className="factor-field__control">
          <select
            value={factors.weather}
            onChange={(event) => onChange({ weather: event.target.value as WeatherId })}
          >
            {WEATHER_OPTIONS.map((option) => (
              <option key={option.id} value={option.id}>
                {option.label} (×{formatNumber1(option.coefficient)})
              </option>
            ))}
          </select>
        </span>
      </label>

      <label className="factor-field">
        <span className="factor-field__label">События</span>
        <span className="factor-field__control">
          <select
            value={factors.event}
            onChange={(event) => onChange({ event: event.target.value as EventId })}
          >
            {EVENT_OPTIONS.map((option) => (
              <option key={option.id} value={option.id}>
                {option.label} (×{formatNumber1(option.coefficient)})
              </option>
            ))}
          </select>
        </span>
      </label>

      <label className="factor-field factor-field--slider">
        <span className="factor-field__label">
          Уровень пробок: <b>{factors.traffic}</b>/10
        </span>
        <input
          type="range"
          min={0}
          max={10}
          step={1}
          value={factors.traffic}
          onChange={(event) => onChange({ traffic: Number(event.target.value) })}
        />
      </label>

      <div className="factors-panel__chips">
        <span className="factor-chip factor-chip--accent">
          Объём ×{formatNumber1(volumeMultiplier)}
        </span>
        <span className="factor-chip">
          Сглаживание пиков {Math.round(trafficSmoothing * 100)} %
        </span>
        {applied.map((factor) => (
          <span key={factor.id} className="factor-chip" title={factor.detail}>
            {factor.id === 'traffic'
              ? `${factor.label} → ${factor.detail}`
              : `${factor.label}: ×${formatNumber1(factor.coefficient)}`}
          </span>
        ))}
        <button type="button" className="factors-panel__reset" onClick={onReset}>
          Сбросить факторы
        </button>
      </div>
    </section>
  )
}
