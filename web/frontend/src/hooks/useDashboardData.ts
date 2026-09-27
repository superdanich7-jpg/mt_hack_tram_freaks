import { useMemo } from 'react'
import {
  buildForecast,
  buildStopRanking,
  DEFAULT_EXTERNAL_FACTORS,
  DEFAULT_HORIZON_ID,
  DEFAULT_INTERVAL_ID,
} from '../data/forecast'
import { ALL_ROUTES_VALUE, ALL_STOPS_VALUE } from '../data/routes'
import type { ExternalFactors, FiltersState } from '../types'

export const INITIAL_FILTERS: FiltersState = {
  routeId: ALL_ROUTES_VALUE,
  stopId: ALL_STOPS_VALUE,
  intervalId: DEFAULT_INTERVAL_ID,
  horizonId: DEFAULT_HORIZON_ID,
}

export const INITIAL_FACTORS: ExternalFactors = { ...DEFAULT_EXTERNAL_FACTORS }

/**
 * Единая точка расчёта мок-данных дашборда: фильтры + корректирующие коэффициенты.
 * Момент формирования фиксируется один раз — иначе экспорт и «Обновлено» менялись бы на каждый рендер.
 */
export function useDashboardData(filters: FiltersState, factors: ExternalFactors) {
  const generatedAt = useMemo(() => new Date().toISOString(), [])
  const { holiday, weather, event, traffic } = factors

  const dataset = useMemo(
    () =>
      buildForecast({
        routeId: filters.routeId,
        stopId: filters.stopId,
        intervalId: filters.intervalId,
        horizonId: filters.horizonId,
        factors: { holiday, weather, event, traffic },
        generatedAt,
      }),
    [
      filters.routeId,
      filters.stopId,
      filters.intervalId,
      filters.horizonId,
      holiday,
      weather,
      event,
      traffic,
      generatedAt,
    ],
  )

  const ranking = useMemo(
    () => buildStopRanking(filters.routeId, filters.horizonId, { holiday, weather, event, traffic }),
    [filters.routeId, filters.horizonId, holiday, weather, event, traffic],
  )

  return { dataset, ranking, generatedAt }
}
