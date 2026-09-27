import { ALL_STOPS_VALUE } from '../data/routes'
import { HORIZONS, TIME_INTERVALS } from '../data/forecast'
import type { FiltersState, HorizonId, IntervalId, TramStop } from '../types'
import { formatPercent } from '../lib/format'
import './FiltersBar.css'

interface FiltersBarProps {
  filters: FiltersState
  routes: { id: string; number: string; name: string; color: string }[]
  availableStops: TramStop[]
  selectedStop?: TramStop
  pointsCount: number
  accuracy: number
  onChange: (patch: Partial<FiltersState>) => void
  onReset: () => void
}

export default function FiltersBar({
  filters,
  routes,
  availableStops,
  selectedStop,
  pointsCount,
  accuracy,
  onChange,
  onReset,
}: FiltersBarProps) {
  return (
    <section className="filters-bar" aria-label="Фильтры прогноза">
      <label className="filter-field">
        <span className="filter-field__label">Маршрут</span>
        <span className="filter-field__control">
          <select
            value={filters.routeId}
            onChange={(event) => onChange({ routeId: event.target.value })}
          >
            <option value="all">Все маршруты ({routes.length})</option>
            {routes.map((route) => (
              <option key={route.id} value={route.id}>
                Маршрут {route.number} · {route.name}
              </option>
            ))}
          </select>
        </span>
      </label>

      <label className="filter-field">
        <span className="filter-field__label">Остановка</span>
        <span className="filter-field__control">
          <select
            value={filters.stopId}
            onChange={(event) => onChange({ stopId: event.target.value })}
          >
            <option value={ALL_STOPS_VALUE}>
              Все остановки маршрута ({availableStops.length})
            </option>
            {availableStops.map((stop) => (
              <option key={stop.id} value={stop.id}>
                {stop.name} · {stop.district}
              </option>
            ))}
          </select>
        </span>
      </label>

      <label className="filter-field">
        <span className="filter-field__label">Интервал времени</span>
        <span className="filter-field__control">
          <select
            value={filters.intervalId}
            onChange={(event) => onChange({ intervalId: event.target.value as IntervalId })}
          >
            {TIME_INTERVALS.map((interval) => (
              <option key={interval.id} value={interval.id}>
                {interval.label}
              </option>
            ))}
          </select>
        </span>
      </label>

      <label className="filter-field">
        <span className="filter-field__label">Горизонт прогноза</span>
        <span className="filter-field__control">
          <select
            value={filters.horizonId}
            onChange={(event) => onChange({ horizonId: event.target.value as HorizonId })}
          >
            {HORIZONS.map((horizon) => (
              <option key={horizon.id} value={horizon.id}>
                {horizon.label}
              </option>
            ))}
          </select>
        </span>
      </label>

      <div className="filters-bar__meta">
        <span className="filter-chip">
          <span className="filter-chip__dot" aria-hidden="true" />
          {selectedStop ? selectedStop.name : 'Все остановки маршрута'}
        </span>
        <span className="filter-chip filter-chip--muted">{pointsCount} точки/точек</span>
        <span className="filter-chip filter-chip--muted">Точность: {formatPercent(accuracy)}</span>
        <button type="button" className="filters-bar__reset" onClick={onReset}>
          Сбросить фильтры
        </button>
      </div>
    </section>
  )
}
