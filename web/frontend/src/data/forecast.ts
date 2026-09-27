import type {
  AppliedFactor,
  EventId,
  EventOption,
  ExternalFactors,
  ForecastDataset,
  Horizon,
  HorizonId,
  HourlyPoint,
  IntervalId,
  StopRankingRow,
  TimeInterval,
  TramStop,
  WeatherId,
  WeatherOption,
} from '../types'
import { ALL_ROUTES_VALUE, ALL_STOPS_VALUE, formatRouteLabel, getStop, getStopsForRoute } from './routes'
import type { ForecastRecord } from '../lib/api'


/**
 * Относительный профиль пассажиропотока трамвая по часам суток (мок-модель).
 * Значения нормируются на сумму, поэтому итог всегда равен суточному пассажиропотоку остановки.
 */
const HOUR_WEIGHTS = [
  0.05, 0.035, 0.02, 0.02, 0.05, 0.2, 0.5, 0.95, 1.45, 1.1, 0.85, 0.8,
  0.82, 0.78, 0.8, 0.92, 1.25, 1.5, 1.3, 0.95, 0.7, 0.45, 0.22, 0.1,
]
const WEIGHT_SUM = HOUR_WEIGHTS.reduce((sum, value) => sum + value, 0)

/** Горизонты прогноза, доступные в фильтре. */
export const HORIZONS: Horizon[] = [
  { id: 'day', label: 'День (D+1)', shortLabel: 'День', growth: 0.02, accuracy: 95.8, bandWidth: 0.06 },
  { id: 'month', label: 'Месяц (M+1)', shortLabel: 'Месяц', growth: 0.06, accuracy: 92.1, bandWidth: 0.11 },
  { id: 'year', label: 'Год (Y+1)', shortLabel: 'Год', growth: 0.11, accuracy: 87.4, bandWidth: 0.17 },
]

/** Интервалы времени внутри суток. Ночь «переезжает» через полночь, поэтому часы идут не по порядку. */
export const TIME_INTERVALS: TimeInterval[] = [
  { id: 'all-day', label: 'Весь день (00:00–24:00)', hours: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23] },
  { id: 'morning', label: 'Утро (05:00–11:00)', hours: [5, 6, 7, 8, 9, 10] },
  { id: 'daytime', label: 'День (11:00–16:00)', hours: [11, 12, 13, 14, 15] },
  { id: 'evening', label: 'Вечер (16:00–22:00)', hours: [16, 17, 18, 19, 20, 21] },
  { id: 'night', label: 'Ночь (22:00–05:00)', hours: [22, 23, 0, 1, 2, 3, 4] },
]

export const DEFAULT_HORIZON_ID: HorizonId = 'day'
export const DEFAULT_INTERVAL_ID: IntervalId = 'all-day'

/** Погодные сценарии панели «Внешние факторы». Плохая погода увеличивает поток на трамвае. */
export const WEATHER_OPTIONS: WeatherOption[] = [
  { id: 'clear', label: 'Ясно', coefficient: 1, note: 'базовый объём' },
  { id: 'rain', label: 'Дождь', coefficient: 1.1, note: 'пересадки с пеших маршрутов (+10 %)' },
  { id: 'snow', label: 'Снегопад', coefficient: 1.2, note: 'отказ от личных авто в пользу трамвая (+20 %)' },
]

/** События на маршруте («Внешние факторы»). */
export const EVENT_OPTIONS: EventOption[] = [
  { id: 'none', label: 'Штатно', coefficient: 1, note: 'стандартный режим движения' },
  { id: 'roadwork', label: 'Ремонт соседних дорог', coefficient: 1.15, note: 'автомобилисты пересаживаются на трамвай (+15 %)' },
  { id: 'major-event', label: 'Массовое мероприятие', coefficient: 1.3, note: 'локальный всплеск пассажиропотока (+30 %)' },
]

/** Коэффициент выходного/праздничного дня — снижает базовый объём. */
export const HOLIDAY_VOLUME_FACTOR = 0.62

/** Максимальная доля сглаживания пиков при уровне пробок 10. */
const MAX_TRAFFIC_SMOOTHING = 0.45

/**
 * Запас провозной способности к пиковому часу базового спроса:
 * подвижной состав подбирается под пик с 15 % резерва.
 */
const CAPACITY_HEADROOM = 1.15

/** Значения по умолчанию: будний день, ясно, штатно, невысокие пробки. */
export const DEFAULT_EXTERNAL_FACTORS: ExternalFactors = {
  holiday: false,
  weather: 'clear',
  event: 'none',
  traffic: 3,
}

export function getWeatherOption(weatherId: WeatherId): WeatherOption {
  return WEATHER_OPTIONS.find((option) => option.id === weatherId) ?? WEATHER_OPTIONS[0]
}

export function getEventOption(eventId: EventId): EventOption {
  return EVENT_OPTIONS.find((option) => option.id === eventId) ?? EVENT_OPTIONS[0]
}

/** Нормализованный уровень пробок 0..10. */
export function clampTraffic(level: number): number {
  return Math.min(10, Math.max(0, Math.round(level)))
}

/** Итоговый множитель объёма: выходной × погода × события. */
export function externalVolumeMultiplier(factors: ExternalFactors): number {
  const holidayFactor = factors.holiday ? HOLIDAY_VOLUME_FACTOR : 1
  const weatherFactor = getWeatherOption(factors.weather).coefficient
  const eventFactor = getEventOption(factors.event).coefficient
  return holidayFactor * weatherFactor * eventFactor
}

/** Доля сглаживания суточного профиля от уровня пробок: 0..MAX_TRAFFIC_SMOOTHING. */
export function trafficSmoothingRatio(factors: ExternalFactors): number {
  return (clampTraffic(factors.traffic) / 10) * MAX_TRAFFIC_SMOOTHING
}

/** Описание применённых коэффициентов — для чипов в UI, колонок в CSV и футера. */
export function describeExternalFactors(factors: ExternalFactors): AppliedFactor[] {
  const weather = getWeatherOption(factors.weather)
  const event = getEventOption(factors.event)
  const smoothing = trafficSmoothingRatio(factors)
  return [
    {
      id: 'holiday',
      label: factors.holiday ? 'Выходной / праздничный день' : 'Будний день',
      coefficient: factors.holiday ? HOLIDAY_VOLUME_FACTOR : 1,
      detail: factors.holiday ? 'объём снижен' : 'объём без изменений',
    },
    {
      id: 'weather',
      label: `Погода: ${weather.label}`,
      coefficient: weather.coefficient,
      detail: weather.note,
    },
    {
      id: 'event',
      label: `Событие: ${event.label}`,
      coefficient: event.coefficient,
      detail: event.note,
    },
    {
      id: 'traffic',
      label: `Пробки ${clampTraffic(factors.traffic)}/10`,
      coefficient: smoothing,
      detail: `сглаживание пиков ${Math.round(smoothing * 100)} %`,
    },
  ]
}

/**
 * Применяет внешние факторы к суточному ряду:
 * 1) масштабирует объём (выходной день × погода);
 * 2) «размазывает» пики к среднему (уровень пробок).
 * Сглаживание — выпуклая комбинация значения и среднего, поэтому суточный объём сохраняется.
 */
export function applyExternalFactors(hours: number[], factors: ExternalFactors): number[] {
  const multiplier = externalVolumeMultiplier(factors)
  const smoothing = trafficSmoothingRatio(factors)
  const scaled = hours.map((value) => value * multiplier)
  if (smoothing <= 0 || scaled.length === 0) return scaled
  const mean = scaled.reduce((sum, value) => sum + value, 0) / scaled.length
  return scaled.map((value) => value + (mean - value) * smoothing)
}

export function getHorizon(horizonId: HorizonId): Horizon {
  return HORIZONS.find((horizon) => horizon.id === horizonId) ?? HORIZONS[0]
}

export function getTimeInterval(intervalId: IntervalId): TimeInterval {
  return TIME_INTERVALS.find((interval) => interval.id === intervalId) ?? TIME_INTERVALS[0]
}

/** Устойчивый хеш строки — основа детерминированности мок-данных. */
function hashString(value: string): number {
  let hash = 2166136261
  for (let index = 0; index < value.length; index += 1) {
    hash ^= value.charCodeAt(index)
    hash = Math.imul(hash, 16777619)
  }
  return hash >>> 0
}

/** Быстрый seeded PRNG (mulberry32): одинаковые данные при каждом рендере. */
function mulberry32(seed: number): () => number {
  let state = seed
  return () => {
    state = (state + 0x6d2b79f5) | 0
    let t = Math.imul(state ^ (state >>> 15), 1 | state)
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

/** Доля пассажиропотока остановки, приходящаяся на конкретный маршрут. */
function routeShare(stop: TramStop, routeId: string): number {
  if (routeId === ALL_ROUTES_VALUE) return 1
  const routesCount = stop.routeIds.length
  if (routesCount <= 1) return 1
  const index = stop.routeIds.indexOf(routeId)
  if (index < 0) return 0
  const random = mulberry32(hashString(`${stop.id}|share|${index}`))
  return (1 / routesCount) * (0.8 + random() * 0.4)
}

interface StopSeries {
  stop: TramStop
  /** Базовый суточный ряд без учёта внешних факторов (основа для факта прошлого периода). */
  baseHours: number[]
  /** Пассажиропоток по часам суток с учётом внешних факторов (прогноз). */
  forecastHours: number[]
  /** Провозная способность остановки, пасс./час (мок). */
  capacityPerHour: number
}

/**
 * Прогнозы бэкенда по маршрутам: ключ — id маршрута, значение — 24 часовых прогноза.
 * Нужны, чтобы при фильтре «Все маршруты» остановка получала ряд своего маршрута,
 * а не среднее по всей сети.
 */
export type RoutePredictions = Record<string, number[]>

/**
 * Базовый суточный ряд остановки из реального прогноза бэкенда.
 * Прогноз отдаётся по маршруту целиком, поэтому на остановку приходится доля 1/N его остановок.
 * Возвращает null, если реальных данных нет — тогда работает расчётный профиль.
 */
function stopBaseHoursFromBackend(
  stop: TramStop,
  routeId: string,
  realHourlyPredictions?: number[],
  realRoutePredictions?: RoutePredictions,
): number[] | null {
  const ownRouteId = realRoutePredictions
    ? stop.routeIds.find((id) => realRoutePredictions[id]?.length === 24)
    : undefined
  const series = ownRouteId ? realRoutePredictions?.[ownRouteId] : realHourlyPredictions
  if (!series || series.length !== 24) return null
  const stops = getStopsForRoute(ownRouteId ?? routeId)
  const stopRatio = stops.length > 0 ? 1 / stops.length : 1
  return series.map((value) => value * stopRatio)
}

/**
 * Преобразует массив ForecastRecord из FastAPI эндпоинта /forecast в массив из 24 чисел по часам 0..23.
 */
export function recordsToHourlyArray(records: ForecastRecord[]): number[] {
  const hours = new Array<number>(24).fill(0)
  for (const record of records) {
    if (!record.datetime) continue
    const dt = new Date(record.datetime)
    const h = isNaN(dt.getHours()) ? parseInt(record.datetime.slice(11, 13), 10) : dt.getHours()
    if (h >= 0 && h < 24) {
      hours[h] = record.passengers
    }
  }
  return hours
}

/**
 * Провозная способность остановки, пасс./час: подвижной состав рассчитан на пиковый час
 * базового (без внешних факторов) спроса с запасом CAPACITY_HEADROOM.
 * Так заполняемость сопоставима и для реального прогноза бэкенда, и для расчётного профиля.
 */
function capacityFromBase(baseHours: number[]): number {
  const peak = baseHours.length > 0 ? Math.max(...baseHours) : 0
  return Math.max(1, Math.round(peak * CAPACITY_HEADROOM))
}

/** Строит суточный ряд для одной остановки: реальный прогноз бэкенда или расчётный профиль. */
function buildStopSeries(
  stop: TramStop,
  routeId: string,
  horizon: Horizon,
  factors: ExternalFactors,
  realHourlyPredictions?: number[],
  realRoutePredictions?: RoutePredictions,
): StopSeries {
  const backendHours = stopBaseHoursFromBackend(
    stop,
    routeId,
    realHourlyPredictions,
    realRoutePredictions,
  )
  if (backendHours) {
    return {
      stop,
      baseHours: backendHours,
      forecastHours: applyExternalFactors(backendHours, factors),
      capacityPerHour: capacityFromBase(backendHours),
    }
  }

  const share = routeShare(stop, routeId)
  const random = mulberry32(hashString(`${stop.id}|${routeId}|${horizon.id}`))
  const character = 0.9 + random() * 0.25
  const dailyVolume = stop.dailyBase * share * (1 + horizon.growth)
  const baseHours = HOUR_WEIGHTS.map((weight, hour) => {
    const base = (dailyVolume * weight) / WEIGHT_SUM
    const hourlyNoise = 0.88 + random() * 0.24
    const rushBoost = hour === 8 || hour === 18 ? 1.04 : 1
    return Math.max(0, base * character * hourlyNoise * rushBoost)
  })
  return {
    stop,
    baseHours,
    forecastHours: applyExternalFactors(baseHours, factors),
    capacityPerHour: capacityFromBase(baseHours),
  }
}

/** Приводит ряд к HourlyPoint (факт прошлого периода + доверительный интервал + заполняемость). */
function toPoint(params: {
  hour: number
  forecast: number
  actual: number
  bandWidth: number
  capacityPerHour: number
}): HourlyPoint {
  const { hour, forecast, actual, bandWidth, capacityPerHour } = params
  const load = capacityPerHour > 0 ? Math.min(120, (forecast / capacityPerHour) * 100) : 0
  return {
    hour,
    label: `${String(hour).padStart(2, '0')}:00`,
    actual: Math.round(actual),
    forecast: Math.round(forecast),
    low: Math.max(0, Math.round(forecast * (1 - bandWidth))),
    high: Math.round(forecast * (1 + bandWidth)),
    load,
  }
}



interface ForecastScope {
  stops: TramStop[]
  forecastHours: number[]
  actualHours: number[]
  capacityPerHour: number
}

/** Округляет суточный ряд до целых пассажиров: суммы KPI совпадают с точками графика. */
function roundSeries(hours: number[]): number[] {
  return hours.map((value) => Math.round(value))
}

/** Готовит суммарный ряд по выбранным остановкам (учитывает фильтры, внешние факторы и реальные предсказания). */
function buildScope(
  routeId: string,
  stopId: string,
  horizon: Horizon,
  factors: ExternalFactors,
  realHourlyPredictions?: number[],
  realRoutePredictions?: RoutePredictions,
): ForecastScope {
  const stopsInRoute = getStopsForRoute(routeId)
  const selectedStop = stopId === ALL_STOPS_VALUE ? undefined : getStop(stopId)
  const stops = selectedStop ? [selectedStop] : stopsInRoute

  // Весь маршрут / вся сеть + реальный прогноз бэкенда: агрегат уже посчитан по маршруту,
  // поэтому делить его по остановкам не нужно.
  if (!selectedStop && realHourlyPredictions && realHourlyPredictions.length === 24) {
    const previousFactor = 1 / (1 + horizon.growth)
    return {
      stops,
      forecastHours: roundSeries(applyExternalFactors(realHourlyPredictions, factors)),
      actualHours: roundSeries(realHourlyPredictions.map((val) => val * previousFactor)),
      capacityPerHour: capacityFromBase(realHourlyPredictions),
    }
  }

  const series = stops.map((stop) =>
    buildStopSeries(stop, routeId, horizon, factors, realHourlyPredictions, realRoutePredictions),
  )
  const actualRandom = mulberry32(hashString(`${routeId}|${stopId}|actual|${horizon.id}`))
  const baseHours = new Array<number>(24).fill(0)
  const forecastHours = new Array<number>(24).fill(0)
  const actualHours = new Array<number>(24).fill(0)
  let capacityPerHour = 0

  series.forEach((item) => {
    capacityPerHour += item.capacityPerHour
    item.baseHours.forEach((value, hour) => {
      baseHours[hour] += value
    })
    item.forecastHours.forEach((value, hour) => {
      forecastHours[hour] += value
    })
  })

  // Факт прошлого сопоставимого периода — статическая базовая константа БЕЗ влияния внешних факторов
  const previousFactor = 1 / (1 + horizon.growth)
  baseHours.forEach((value, hour) => {
    const noise = 0.94 + actualRandom() * 0.12
    actualHours[hour] = value * previousFactor * noise
  })

  return {
    stops,
    forecastHours: roundSeries(forecastHours),
    actualHours: roundSeries(actualHours),
    capacityPerHour,
  }
}


export interface BuildForecastParams {
  routeId: string
  stopId: string
  intervalId: IntervalId
  horizonId: HorizonId
  /** Корректирующие коэффициенты панели «Внешние факторы». */
  factors: ExternalFactors
  /** ISO-время формирования набора данных. */
  generatedAt: string
  /** Реальный прогноз по часам 0..23 из FastAPI `/forecast` для выбранного маршрута (или агрегат по сети). */
  realHourlyPredictions?: number[]
  /** Прогнозы бэкенда по маршрутам: нужны, чтобы раскладывать агрегат по остановкам. */
  realRoutePredictions?: RoutePredictions
}

/** Собирает набор данных для графика, KPI и экспорта: точка = час суток. */
export function buildForecast(params: BuildForecastParams): ForecastDataset {
  const {
    routeId,
    stopId,
    intervalId,
    horizonId,
    factors,
    generatedAt,
    realHourlyPredictions,
    realRoutePredictions,
  } = params
  const horizon = getHorizon(horizonId)
  const interval = getTimeInterval(intervalId)
  const scope = buildScope(routeId, stopId, horizon, factors, realHourlyPredictions, realRoutePredictions)

  const points = interval.hours.map((hour) =>
    toPoint({
      hour,
      forecast: scope.forecastHours[hour],
      actual: scope.actualHours[hour],
      bandWidth: horizon.bandWidth,
      capacityPerHour: scope.capacityPerHour,
    }),
  )

  const intervalTotal = points.reduce((sum, point) => sum + point.forecast, 0)
  const intervalActual = points.reduce((sum, point) => sum + point.actual, 0)
  const deltaPercent = intervalActual > 0 ? ((intervalTotal - intervalActual) / intervalActual) * 100 : 0
  const dayTotal = scope.forecastHours.reduce((sum, value) => sum + value, 0)
  const avgLoad = points.length > 0 ? points.reduce((sum, point) => sum + point.load, 0) / points.length : 0

  const dayPeakIndex = scope.forecastHours.reduce(
    (best, value, hour) => (value > best.value ? { hour, value } : best),
    { hour: 0, value: -1 },
  )
  const intervalPeakIndex = scope.forecastHours.reduce(
    (best, value, hour) => (interval.hours.includes(hour) && value > best.value ? { hour, value } : best),
    { hour: interval.hours[0] ?? 0, value: -1 },
  )

  const selectedStop = stopId === ALL_STOPS_VALUE ? undefined : getStop(stopId)
  const scopeLabel = selectedStop
    ? selectedStop.name
    : `Все остановки (${scope.stops.length}) · ${routeId === ALL_ROUTES_VALUE ? 'вся сеть' : 'в пределах маршрута'}`

  return {
    meta: {
      scopeId: selectedStop ? selectedStop.id : routeId === ALL_ROUTES_VALUE ? 'all-routes' : routeId,
      scopeLabel,
      routeLabel: formatRouteLabel(routeId),
      intervalLabel: interval.label,
      horizonLabel: horizon.label,
      horizonId,
      generatedAt,
      appliedFactors: describeExternalFactors(factors),
      volumeMultiplier: externalVolumeMultiplier(factors),
      trafficSmoothing: trafficSmoothingRatio(factors),
    },
    points,
    summary: {
      intervalTotal,
      intervalActual,
      deltaPercent,
      dayTotal,
      peak:
        intervalPeakIndex.value >= 0
          ? { label: `${String(intervalPeakIndex.hour).padStart(2, '0')}:00`, value: Math.round(intervalPeakIndex.value) }
          : null,
      dayPeak:
        dayPeakIndex.value >= 0
          ? { label: `${String(dayPeakIndex.hour).padStart(2, '0')}:00`, value: Math.round(dayPeakIndex.value) }
          : null,
      avgLoad,
      accuracy: horizon.accuracy,
    },
  }
}

/** Рейтинг остановок маршрута по суточному прогнозу — для блока «топ остановок» и размеров маркеров на карте. */
export function buildStopRanking(
  routeId: string,
  horizonId: HorizonId,
  factors: ExternalFactors,
  realHourlyPredictions?: number[],
  realRoutePredictions?: RoutePredictions,
): StopRankingRow[] {
  const horizon = getHorizon(horizonId)
  const series = getStopsForRoute(routeId).map((stop) =>
    buildStopSeries(stop, routeId, horizon, factors, realHourlyPredictions, realRoutePredictions),
  )
  const total = series.reduce(
    (sum, item) => sum + item.forecastHours.reduce((innerSum, value) => innerSum + value, 0),
    0,
  )

  return series
    .map((item) => {
      const dailyForecast = Math.round(item.forecastHours.reduce((sum, value) => sum + value, 0))
      const peakValue = Math.round(Math.max(...item.forecastHours))
      const peakHour = item.forecastHours.indexOf(Math.max(...item.forecastHours))
      const loads = item.forecastHours.map((value) =>
        item.capacityPerHour > 0 ? Math.min(120, (value / item.capacityPerHour) * 100) : 0,
      )
      const avgLoad = loads.reduce((sum, value) => sum + value, 0) / (loads.length || 1)
      return {
        stop: item.stop,
        dailyForecast,
        peakLabel: `${String(peakHour).padStart(2, '0')}:00`,
        peakValue,
        avgLoad,
        share: total > 0 ? (dailyForecast / total) * 100 : 0,
      }
    })
    .sort((left, right) => right.dailyForecast - left.dailyForecast)
}

