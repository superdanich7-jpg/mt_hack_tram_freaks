import { useState } from 'react'
import {
  Area,
  CartesianGrid,
  ComposedChart,
  Legend,
  Line,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import type { TooltipContentProps } from 'recharts'
import { formatCompact, formatNumber, formatPercent, formatSignedPercent } from '../lib/format'
import { CHART_COLORS, loadColor } from '../lib/palette'
import type { ForecastDataset, HourlyPoint } from '../types'
import './ChartPanel.css'

interface ChartPanelProps {
  dataset: ForecastDataset
}

/** Достаёт исходную точку данных из payload тултипа recharts. */
function pickPoint(payload: TooltipContentProps['payload']): HourlyPoint | null {
  for (const entry of payload) {
    const datum: unknown = (entry as { payload?: unknown }).payload
    if (datum && typeof datum === 'object' && 'forecast' in datum && 'load' in datum) {
      return datum as HourlyPoint
    }
  }
  return null
}

function ForecastTooltip({ active, payload }: TooltipContentProps) {
  const point = active && payload ? pickPoint(payload) : null
  if (!point) return null

  return (
    <div className="chart-tooltip">
      <p className="chart-tooltip__hour">{point.label}</p>
      <div className="chart-tooltip__row">
        <span>Прогноз</span>
        <b style={{ color: CHART_COLORS.forecast }}>{formatNumber(point.forecast)} пасс./ч</b>
      </div>
      <div className="chart-tooltip__row">
        <span>Доверительный интервал</span>
        <b>
          {formatNumber(point.low)} – {formatNumber(point.high)}
        </b>
      </div>
      <div className="chart-tooltip__row">
        <span>Факт (прошлый период)</span>
        <b style={{ color: CHART_COLORS.actual }}>{formatNumber(point.actual)}</b>
      </div>
      <div className="chart-tooltip__row">
        <span>Заполняемость</span>
        <b style={{ color: loadColor(point.load) }}>{formatPercent(point.load)}</b>
      </div>
    </div>
  )
}

export default function ChartPanel({ dataset }: ChartPanelProps) {
  const [showTable, setShowTable] = useState(false)
  const { meta, points, summary } = dataset
  const peakLabel = summary.peak?.label ?? null

  return (
    <section className="chart-panel">
      <header className="chart-panel__header">
        <div className="chart-panel__heading">
          <h2>{meta.scopeLabel}</h2>
          <p>{meta.routeLabel}</p>
        </div>
        <div className="chart-panel__badges">
          <span className="panel-badge panel-badge--accent">{meta.horizonLabel}</span>
          <span className="panel-badge">{meta.intervalLabel}</span>
          <button
            type="button"
            className="panel-toggle"
            aria-pressed={showTable}
            onClick={() => setShowTable((value) => !value)}
          >
            {showTable ? 'Скрыть таблицу' : 'Таблица данных'}
          </button>
        </div>
      </header>

      <div className="chart-panel__kpis">
        <div className="mini-kpi">
          <span className="mini-kpi__label">Прогноз за интервал</span>
          <span className="mini-kpi__value">{formatNumber(summary.intervalTotal)}</span>
          <span className={`mini-kpi__delta ${summary.deltaPercent >= 0 ? 'is-up' : 'is-down'}`}>
            {formatSignedPercent(summary.deltaPercent)} к прошлому периоду
          </span>
        </div>
        <div className="mini-kpi">
          <span className="mini-kpi__label">Пиковый час</span>
          <span className="mini-kpi__value">{summary.peak?.label ?? '—'}</span>
          <span className="mini-kpi__delta">
            {formatNumber(summary.peak?.value ?? 0)} пасс./ч
          </span>
        </div>
        <div className="mini-kpi">
          <span className="mini-kpi__label">Заполняемость (средняя)</span>
          <span className="mini-kpi__value">{formatPercent(summary.avgLoad)}</span>
          <span className="mini-kpi__delta">за сутки {formatNumber(summary.dayTotal)} пасс.</span>
        </div>
      </div>

      <div className="chart-panel__body">
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart data={points} margin={{ top: 10, right: 6, left: 0, bottom: 0 }}>
            <defs>
              <linearGradient id="forecastFill" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor={CHART_COLORS.forecast} stopOpacity={0.4} />
                <stop offset="100%" stopColor={CHART_COLORS.forecast} stopOpacity={0.02} />
              </linearGradient>
            </defs>
            <CartesianGrid stroke={CHART_COLORS.grid} strokeDasharray="3 3" vertical={false} />
            <XAxis
              dataKey="label"
              tickLine={false}
              axisLine={{ stroke: CHART_COLORS.grid }}
              tick={{ fill: CHART_COLORS.axis, fontSize: 11 }}
              tickMargin={8}
              minTickGap={10}
            />
            <YAxis
              yAxisId="volume"
              width={56}
              tickLine={false}
              axisLine={false}
              tick={{ fill: CHART_COLORS.axis, fontSize: 11 }}
              tickFormatter={formatCompact}
            />
            <YAxis
              yAxisId="load"
              orientation="right"
              width={46}
              domain={[0, 120]}
              tickLine={false}
              axisLine={false}
              tick={{ fill: CHART_COLORS.load, fontSize: 11 }}
              tickFormatter={(value: number) => `${value}%`}
            />
            <Tooltip
              content={ForecastTooltip}
              cursor={{ stroke: '#94a3b8', strokeDasharray: '4 4' }}
            />
            <Legend wrapperStyle={{ fontSize: 12, paddingTop: 6 }} iconType="plainline" />
            <Area
              yAxisId="volume"
              type="monotone"
              dataKey="forecast"
              name="Прогноз"
              stroke={CHART_COLORS.forecast}
              strokeWidth={2.4}
              fill="url(#forecastFill)"
              activeDot={{ r: 4 }}
              dot={false}
            />
            <Line
              yAxisId="volume"
              type="monotone"
              dataKey="actual"
              name="Факт (прошлый период)"
              stroke={CHART_COLORS.actual}
              strokeWidth={1.8}
              strokeDasharray="5 4"
              dot={false}
            />
            <Line
              yAxisId="load"
              type="monotone"
              dataKey="load"
              name="Заполняемость, %"
              stroke={CHART_COLORS.load}
              strokeWidth={1.5}
              dot={false}
            />
            {peakLabel && (
              <ReferenceLine
                yAxisId="volume"
                x={peakLabel}
                stroke={CHART_COLORS.selected}
                strokeDasharray="4 4"
                label={{ value: 'пик', fill: CHART_COLORS.selected, fontSize: 11, position: 'insideTopRight' }}
              />
            )}
          </ComposedChart>
        </ResponsiveContainer>
      </div>

      {showTable && (
        <div className="chart-panel__table-wrap">
          <table className="data-table">
            <thead>
              <tr>
                <th>Час</th>
                <th>Прогноз, пасс./ч</th>
                <th>Интервал</th>
                <th>Факт, пасс./ч</th>
                <th>Заполняемость</th>
              </tr>
            </thead>
            <tbody>
              {points.map((point) => (
                <tr key={point.hour}>
                  <td>{point.label}</td>
                  <td className="is-numeric">{formatNumber(point.forecast)}</td>
                  <td className="is-numeric">
                    {formatNumber(point.low)}–{formatNumber(point.high)}
                  </td>
                  <td className="is-numeric">{formatNumber(point.actual)}</td>
                  <td className="is-numeric" style={{ color: loadColor(point.load) }}>
                    {formatPercent(point.load)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </section>
  )
}
