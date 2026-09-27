import { useCallback, useEffect, useMemo, useState } from 'react'
import {
  buildForecast,
  buildStopRanking,
  DEFAULT_EXTERNAL_FACTORS,
  DEFAULT_HORIZON_ID,
  DEFAULT_INTERVAL_ID,
  recordsToHourlyArray,
} from '../data/forecast'
import type { RoutePredictions } from '../data/forecast'
import { ALL_ROUTES_VALUE, ALL_STOPS_VALUE, ROUTES } from '../data/routes'
import { DEFAULT_FORECAST_DATE, fetchForecast, fetchRoutes } from '../lib/api'
import type {
  ExternalFactors,
  FiltersState,
  ForecastDataset,
  ForecastSource,
  StopRankingRow,
} from '../types'

export const INITIAL_FILTERS: FiltersState = {
  routeId: ALL_ROUTES_VALUE,
  stopId: ALL_STOPS_VALUE,
  intervalId: DEFAULT_INTERVAL_ID,
  horizonId: DEFAULT_HORIZON_ID,
  date: DEFAULT_FORECAST_DATE,
}

export const INITIAL_FACTORS: ExternalFactors = { ...DEFAULT_EXTERNAL_FACTORS }

/** Локальный справочник маршрутов: используется, пока /routes недоступен. */
export const LOCAL_ROUTE_IDS: string[] = ROUTES.map((route) => route.id)

/** Состояние дашборда: набор данных, рейтинг остановок и статус связи с бэкендом. */
export interface DashboardData {
  dataset: ForecastDataset
  ranking: StopRankingRow[]
  generatedAt: string
  /** Маршрут, по которому реально построен набор данных (подменяется на «все» для неизвестного маршрута). */
  routeId: string
  /** Остановка набора данных: сбрасывается, если маршрут подменён всей сетью. */
  stopId: string
  /** Дата прогноза, по которой построен набор данных (параметр `date` запроса /forecast). */
  date: string
  /** Маршруты, доступные в фильтре (пересечение /routes и локального справочника). */
  routeIds: string[]
  /** Ошибка загрузки /routes ('' — ошибок нет). */
  routesError: string
  /** Источник текущего набора данных: реальный бэкенд или мок-генератор. */
  source: ForecastSource
  isLoading: boolean
  /** Ошибка последнего запроса /forecast ('' — ошибок нет). */
  error: string
  reload: () => void
}

/** Человекочитаемое сообщение об ошибке HTTP-запроса. */
function describeError(error: unknown): string {
  if (error instanceof Error) return error.message
  return String(error)
}

/** Суммарный суточный ряд и ряды по маршрутам (для раскладки агрегата по остановкам). */
interface LoadedSeries {
  hours: number[]
  byRoute: RoutePredictions
}

/** Загружает суточные прогнозы по маршрутам: агрегат по сети + ряды по каждому маршруту. */
async function loadSeries(
  routes: string[],
  date: string,
  signal: AbortSignal,
): Promise<LoadedSeries> {
  const responses = await Promise.all(routes.map((route) => fetchForecast(route, date, signal)))
  // Если хотя бы по одному маршруту бэкенд не отдал сутки целиком — агрегат считаем недостоверным.
  if (responses.some((records) => records.length < 24)) {
    throw new Error(`/forecast: неполные данные по маршрутам на ${date}`)
  }
  const hours = new Array<number>(24).fill(0)
  const byRoute: RoutePredictions = {}
  routes.forEach((route, index) => {
    const routeHours = recordsToHourlyArray(responses[index])
    byRoute[route] = routeHours
    routeHours.forEach((value, hour) => {
      hours[hour] += value
    })
  })
  return { hours, byRoute }
}

/**
 * Единая точка расчёта данных дашборда.
 *
 * Фильтры и коэффициенты «Внешних факторов» применяются поверх реального прогноза,
 * полученного от FastAPI (`/routes` + `/forecast?route=&date=`, Basic-авторизация).
 * Если бэкенд недоступен или данных по маршруту и дате нет — используется
 * детерминированный генератор `src/data/forecast.ts`, чтобы дашборд не «падал» в пустоту.
 */
export function useDashboardData(filters: FiltersState, factors: ExternalFactors): DashboardData {
  const { routeId, stopId, intervalId, horizonId, date } = filters
  const { holiday, weather, event, traffic } = factors

  const [routeIds, setRouteIds] = useState<string[]>(LOCAL_ROUTE_IDS)

  /** Эффективный маршрут: если бэкенд не отдал прогноз по выбранному маршруту — работаем со всей сетью. */
  const effectiveRouteId =
    routeId === ALL_ROUTES_VALUE || LOCAL_ROUTE_IDS.includes(routeId) ? routeId : ALL_ROUTES_VALUE
  const effectiveStopId = effectiveRouteId === routeId ? stopId : ALL_STOPS_VALUE
  // Ключ запрошенного ряда: смена ключа автоматически переводит дашборд в состояние загрузки.
  const seriesKey = `${effectiveRouteId}|${date}|${routeIds.join(',')}`
  const hasSeries = effectiveRouteId !== ALL_ROUTES_VALUE || routeIds.length > 0

  const [routesError, setRoutesError] = useState('')
  const [series, setSeries] = useState<{
    key: string
    hours: number[] | null
    byRoute: RoutePredictions
    error: string
  }>({ key: '', hours: null, byRoute: {}, error: '' })
  const [generatedAt, setGeneratedAt] = useState(() => new Date().toISOString())
  const [reloadNonce, setReloadNonce] = useState(0)

  const reload = useCallback(() => setReloadNonce((nonce) => nonce + 1), [])

  // 1. Список маршрутов из бэкенда: ограничиваем фильтр теми маршрутами, по которым есть прогноз.
  useEffect(() => {
    const controller = new AbortController()
    let active = true

    fetchRoutes(controller.signal)
      .then((numbers) => {
        if (!active) return
        const known = numbers.map(String).filter((id) => LOCAL_ROUTE_IDS.includes(id))
        // Незнакомые маршруты пропускаем: для них нет остановок и координат на карте.
        if (known.length > 0) {
          setRouteIds((prev) => (prev.join(',') === known.join(',') ? prev : known))
        }
        setRoutesError('')
      })
      .catch((cause: unknown) => {
        if (!active) return
        setRoutesError(describeError(cause))
      })

    return () => {
      active = false
      controller.abort()
    }
  }, [reloadNonce])

  // 2. Прогноз на сутки: конкретный маршрут — один запрос, «Все маршруты» — сумма по сети.
  useEffect(() => {
    if (!hasSeries) return

    const targetRoutes = effectiveRouteId === ALL_ROUTES_VALUE ? routeIds : [effectiveRouteId]
    const controller = new AbortController()
    let active = true

    loadSeries(targetRoutes, date, controller.signal)
      .then((loaded) => {
        if (!active) return
        const total = loaded.hours.reduce((sum, value) => sum + value, 0)
        if (total > 0) {
          setSeries({ key: seriesKey, hours: loaded.hours, byRoute: loaded.byRoute, error: '' })
          setGeneratedAt(new Date().toISOString())
        } else {
          setSeries({
            key: seriesKey,
            hours: null,
            byRoute: {},
            error: `Бэкенд вернул нулевой прогноз на ${date} — показан расчётный профиль`,
          })
        }
      })
      .catch((cause: unknown) => {
        if (!active) return
        setSeries({ key: seriesKey, hours: null, byRoute: {}, error: describeError(cause) })
      })

    return () => {
      active = false
      controller.abort()
    }
  }, [seriesKey, hasSeries, effectiveRouteId, routeIds, date, reloadNonce])

  // Данные текущего запроса: пока ряд не получен, дашборд считает расчётный профиль.
  const isCurrent = series.key === seriesKey
  const hourlyPredictions = isCurrent ? series.hours : null
  const routePredictions = isCurrent ? series.byRoute : undefined
  const isLoading = hasSeries && !isCurrent
  const error = isCurrent ? series.error : ''
  const source: ForecastSource = hourlyPredictions ? 'backend' : 'mock'

  const dataset = useMemo(
    () =>
      buildForecast({
        routeId: effectiveRouteId,
        stopId: effectiveStopId,
        intervalId,
        horizonId,
        factors: { holiday, weather, event, traffic },
        generatedAt,
        realHourlyPredictions: hourlyPredictions ?? undefined,
        realRoutePredictions: routePredictions,
      }),
    [
      effectiveRouteId,
      effectiveStopId,
      intervalId,
      horizonId,
      holiday,
      weather,
      event,
      traffic,
      generatedAt,
      hourlyPredictions,
      routePredictions,
    ],
  )

  const ranking = useMemo(
    () =>
      buildStopRanking(
        effectiveRouteId,
        horizonId,
        { holiday, weather, event, traffic },
        hourlyPredictions ?? undefined,
        routePredictions,
      ),
    [effectiveRouteId, horizonId, holiday, weather, event, traffic, hourlyPredictions, routePredictions],
  )

  return {
    dataset,
    ranking,
    generatedAt,
    /** Маршрут, по которому реально построен набор данных (может отличаться от выбранного в фильтре). */
    routeId: effectiveRouteId,
    /** Остановка набора данных: сбрасывается при подмене маршрута. */
    stopId: effectiveStopId,
    date,
    routeIds,
    routesError,
    source,
    isLoading,
    error,
    reload,
  }
}
