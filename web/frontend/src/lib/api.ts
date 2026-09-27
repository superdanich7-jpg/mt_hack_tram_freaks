/**
 * HTTP-клиент FastAPI-бэкенда прогноза пассажиропотока трамваев Москвы.
 * Контракт (см. api_spec.md): GET /routes и GET /forecast?route=&date=, Basic-авторизация.
 */

/**
 * Базовый адрес FastAPI-бэкенда (контракт: api_spec.md).
 * Переопределяется переменной окружения Vite, например:
 * `VITE_API_BASE_URL=http://127.0.0.1:8002 npm run dev`.
 */
export const API_BASE_URL: string =
  import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000'

/** Basic-авторизация бэкенда: admin / hackathon2025. */
export const AUTH_HEADERS: Record<string, string> = {
  Authorization: 'Basic YWRtaW46aGFja2F0aG9uMjAyNQ==',
}

/** Первый день доступного бэкенду периода прогноза. */
export const MIN_FORECAST_DATE = '2025-11-01'

/** Последний день доступного бэкенду периода прогноза. */
export const MAX_FORECAST_DATE = '2025-12-31'

/** Дата прогноза по умолчанию — начало периода. */
export const DEFAULT_FORECAST_DATE = MIN_FORECAST_DATE

export interface ForecastRecord {
  datetime: string
  passengers: number
}

/** Ответ /forecast: реализация отдаёт forecast[], спецификация описывает hours[]. */
export interface ForecastResponse {
  route?: string | number
  date?: string
  forecast?: ForecastRecord[]
  hours?: { hour: number; prediction: number }[]
}

/** Ответ /routes: список номеров либо объект с агрегатами по маршрутам. */
export interface RoutesResponse {
  routes?: (number | string | { route: number | string })[]
}

/** JSON-запрос с Basic-авторизацией; ошибочный статус превращаем в Error. */
async function requestJson<T>(path: string, signal?: AbortSignal): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`, { headers: AUTH_HEADERS, signal })
  if (!response.ok) {
    throw new Error(`${path} → ${response.status} ${response.statusText}`)
  }
  return (await response.json()) as T
}

/** Номер маршрута из ответа /routes (поддержаны числа, строки и объекты с полем route). */
function normalizeRoute(value: number | string | { route: number | string }): number | null {
  const raw = typeof value === 'object' && value !== null ? value.route : value
  const parsed = Number(String(raw).replace(/^route-/, ''))
  return Number.isFinite(parsed) ? parsed : null
}

/**
 * Список номеров маршрутов от бэкенда (GET /routes).
 * Пустой массив означает, что бэкенд не отдал ни одного маршрута.
 */
export async function fetchRoutes(signal?: AbortSignal): Promise<number[]> {
  const data = await requestJson<RoutesResponse | (number | string)[]>('/routes', signal)
  const list = Array.isArray(data) ? data : data?.routes ?? []
  return list.map(normalizeRoute).filter((route): route is number => route !== null)
}

/** Приводит ответ /forecast к записям {datetime, passengers}. */
function toForecastRecords(payload: ForecastResponse, date: string): ForecastRecord[] {
  if (Array.isArray(payload?.forecast) && payload.forecast.length > 0) {
    return payload.forecast
  }
  if (Array.isArray(payload?.hours)) {
    return payload.hours.map((item) => ({
      datetime: `${date}T${String(item.hour).padStart(2, '0')}:00:00`,
      passengers: Number(item.prediction) || 0,
    }))
  }
  return []
}

/**
 * Прогноз по одному маршруту на сутки (GET /forecast?route=&date=).
 * Возвращает 24 записи (по одной на час) либо пустой массив, если данных нет.
 */
export async function fetchForecast(
  route: string | number,
  date: string = DEFAULT_FORECAST_DATE,
  signal?: AbortSignal,
): Promise<ForecastRecord[]> {
  const cleanRoute = String(route).replace(/^route-/, '')
  const query = `?route=${encodeURIComponent(cleanRoute)}&date=${encodeURIComponent(date)}`
  const payload = await requestJson<ForecastResponse>(`/forecast${query}`, signal)
  return toForecastRecords(payload, date)
}
