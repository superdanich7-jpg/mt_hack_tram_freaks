import type { ForecastDataset, StopRankingRow } from '../types'
import { formatNumber, formatPercent, formatSignedPercent } from '../lib/format'
import { loadColor } from '../lib/palette'
import './StatsPanel.css'

interface StatsPanelProps {
  dataset: ForecastDataset
  ranking: StopRankingRow[]
  selectedStopId: string
  onSelectStop: (stopId: string) => void
}

export default function StatsPanel({
  dataset,
  ranking,
  selectedStopId,
  onSelectStop,
}: StatsPanelProps) {
  const { summary, meta } = dataset
  const maxShare = ranking[0]?.share ?? 0

  return (
    <section className="stats-panel">
      <div className="stats-panel__kpis">
        <article className="kpi-card kpi-card--accent">
          <span className="kpi-card__label">Прогноз за интервал</span>
          <span className="kpi-card__value">{formatNumber(summary.intervalTotal)}</span>
          <span className={`kpi-card__hint ${summary.deltaPercent >= 0 ? 'is-up' : 'is-down'}`}>
            {formatSignedPercent(summary.deltaPercent)} к прошлому периоду
          </span>
        </article>

        <article className="kpi-card">
          <span className="kpi-card__label">Прогноз за сутки</span>
          <span className="kpi-card__value">{formatNumber(summary.dayTotal)}</span>
          <span className="kpi-card__hint">пассажиров на остановках выборки</span>
        </article>

        <article className="kpi-card">
          <span className="kpi-card__label">Пиковый час суток</span>
          <span className="kpi-card__value">{summary.dayPeak?.label ?? '—'}</span>
          <span className="kpi-card__hint">
            {formatNumber(summary.dayPeak?.value ?? 0)} пасс./ч · в интервале{' '}
            {summary.peak?.label ?? '—'}
          </span>
        </article>

        <article className="kpi-card">
          <span className="kpi-card__label">Точность модели</span>
          <span className="kpi-card__value">{formatPercent(summary.accuracy)}</span>
          <span className="kpi-card__hint">{meta.horizonLabel} · MAPE мок-модели</span>
        </article>
      </div>

      <div className="stats-panel__ranking">
        <div className="stats-panel__ranking-head">
          <h2>Топ остановок</h2>
          <span>суточный прогноз · клик переключает остановку</span>
        </div>

        {ranking.length === 0 ? (
          <p className="stats-panel__empty">Нет данных — выберите маршрут с остановками</p>
        ) : (
          <ul className="ranking-list">
            {ranking.slice(0, 6).map((row, index) => {
              const isActive = row.stop.id === selectedStopId
              const color = loadColor(row.avgLoad)
              return (
                <li key={row.stop.id}>
                  <button
                    type="button"
                    className={`ranking-item ${isActive ? 'is-active' : ''}`}
                    onClick={() => onSelectStop(row.stop.id)}
                  >
                    <span className="ranking-item__index">{index + 1}</span>
                    <span className="ranking-item__main">
                      <span className="ranking-item__name">{row.stop.name}</span>
                      <span className="ranking-item__bar">
                        <span
                          style={{
                            width: `${maxShare > 0 ? (row.share / maxShare) * 100 : 0}%`,
                            background: color,
                          }}
                        />
                      </span>
                    </span>
                    <span className="ranking-item__values">
                      <b>{formatNumber(row.dailyForecast)}</b>
                      <em style={{ color }}>{formatPercent(row.avgLoad)}</em>
                    </span>
                  </button>
                </li>
              )
            })}
          </ul>
        )}
      </div>
    </section>
  )
}
