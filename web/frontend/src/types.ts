/**
 * Общие типы дашборда прогноза пассажиропотока трамваев Москвы.
 * Структуры описывают как реальный прогноз FastAPI-бэкенда, так и расчётный профиль-фолбэк.
 */

export interface LatLng {
  lat: number
  lng: number
}

/** Маршрут трамвая. */
export interface TramRoute {
  id: string
  /** Номер маршрута для отображения: «А», «3», «6»... */
  number: string
  name: string
  /** Цвет маршрута на карте и в легенде. */
  color: string
  /** Идентификаторы остановок в порядке движения. */
  stopIds: string[]
}

/** Остановка трамвая. */
export interface TramStop {
  id: string
  name: string
  district: string
  coords: LatLng
  /** Маршруты, которые проходят через остановку. */
  routeIds: string[]
  /** Среднесуточный пассажиропоток остановки, пасс./сутки (мок). */
  dailyBase: number
}

/** Горизонт прогноза. */
export type HorizonId = 'day' | 'month' | 'year'

export interface Horizon {
  id: HorizonId
  /** Полная подпись для dropdown: «День (D+1)». */
  label: string
  /** Короткая подпись для бейджей. */
  shortLabel: string
  /** Рост пассажиропотока к горизонту прогноза (доля). */
  growth: number
  /** Точность модели прогноза, % (мок-значение, 100 − MAPE). */
  accuracy: number
  /** Ширина доверительного интервала, доля от прогноза. */
  bandWidth: number
}

/** Интервал времени внутри суток. */
export type IntervalId = 'all-day' | 'morning' | 'daytime' | 'evening' | 'night'

export interface TimeInterval {
  id: IntervalId
  label: string
  /** Часы суток, попадающие в интервал (в порядке отображения на графике). */
  hours: number[]
}

/** Одна точка ряда: час суток с фактом и прогнозом. */
export interface HourlyPoint {
  hour: number
  /** Подпись часа: «08:00». */
  label: string
  /** Факт прошлого сопоставимого периода, пасс./час. */
  actual: number
  /** Прогноз пассажиропотока, пасс./час. */
  forecast: number
  /** Нижняя граница доверительного интервала. */
  low: number
  /** Верхняя граница доверительного интервала. */
  high: number
  /** Заполняемость подвижного состава, %. */
  load: number
}

/** Метаданные набора мок-данных (используются в CSV и в шапке графиков). */
/** Погодный сценарий «Внешних факторов». */
export type WeatherId = 'clear' | 'rain' | 'snow'

export interface WeatherOption {
  id: WeatherId
  label: string
  /** Коэффициент влияния на объём пассажиропотока (1 = без изменений). */
  coefficient: number
  /** Краткое пояснение для интерфейса. */
  note: string
}

/** События на маршруте («Внешние факторы»). */
export type EventId = 'none' | 'roadwork' | 'major-event'

export interface EventOption {
  id: EventId
  label: string
  /** Множитель объёма пассажиропотока. */
  coefficient: number
  note: string
}

/**
 * Корректирующие коэффициенты, которые пользователь задаёт вручную.
 * Хранятся в состоянии Dashboard и прокидываются в генератор прогноза.
 */
export interface ExternalFactors {
  /** Выходной / праздничный день: снижает базовый объём пассажиропотока. */
  holiday: boolean
  /** Погодный сценарий: плохая погода увеличивает пассажиропоток трамваев. */
  weather: WeatherId
  /** События на маршруте (ремонты дорог, массовые мероприятия). */
  event: EventId
  /** Уровень пробок 0..10: размазывает (сглаживает) пиковые часы. */
  traffic: number
}

/** Применённый к прогнозу коэффициент — для чипов в UI и колонок в CSV. */
export interface AppliedFactor {
  id: string
  label: string
  /** Итоговое значение коэффициента (для пробок — доля сглаживания). */
  coefficient: number
  /** Читаемое описание влияния. */
  detail: string
}

export interface ForecastMeta {
  /** Идентификатор области данных: id остановки или 'all'. */
  scopeId: string
  /** Подпись области данных: остановка или «Все остановки маршрута». */
  scopeLabel: string
  routeLabel: string
  intervalLabel: string
  horizonLabel: string
  horizonId: HorizonId
  /** Время генерации мок-данных. */
  generatedAt: string
  /** Список применённых внешних факторов с коэффициентами. */
  appliedFactors: AppliedFactor[]
  /** Итоговый множитель объёма (выходной × погода). */
  volumeMultiplier: number
  /** Доля сглаживания пиков от уровня пробок (0..0.45). */
  trafficSmoothing: number
}

export interface ForecastSummary {
  /** Прогноз пассажиропотока внутри выбранного интервала, пасс. */
  intervalTotal: number
  /** Факт прошлого сопоставимого периода внутри интервала, пасс. */
  intervalActual: number
  /** Изменение к прошлому периоду, %. */
  deltaPercent: number
  /** Прогноз пассажиропотока за полные сутки, пасс. */
  dayTotal: number
  /** Пиковый час внутри выбранного интервала. */
  peak: { label: string; value: number } | null
  /** Пиковый час за полные сутки. */
  dayPeak: { label: string; value: number } | null
  /** Средняя заполняемость внутри интервала, %. */
  avgLoad: number
  /** Точность модели для выбранного горизонта, %. */
  accuracy: number
}

export interface ForecastDataset {
  meta: ForecastMeta
  /** Точки, попадающие в выбранный интервал времени (то, что видно на графике). */
  points: HourlyPoint[]
  summary: ForecastSummary
}

/** Строка рейтинга остановок (блок «топ по загрузке»). */
export interface StopRankingRow {
  stop: TramStop
  dailyForecast: number
  peakLabel: string
  peakValue: number
  avgLoad: number
  share: number
}

/** Режим компоновки экрана. */
export type LayoutMode = 'split' | 'map' | 'charts'

/** Состояние фильтров дашборда. */
export interface FiltersState {
  routeId: RouteFilterValue
  stopId: StopFilterValue
  intervalId: IntervalId
  horizonId: HorizonId
  /** Дата прогноза `YYYY-MM-DD` — уходит в параметр `date` запроса /forecast. */
  date: string
}

/** Источник данных дашборда: реальный бэкенд или детерминированный мок-генератор. */
export type ForecastSource = 'backend' | 'mock'

/** Значение фильтра «остановка» = конкретная остановка или все видимые. */
export type StopFilterValue = 'all' | string

/** Значение фильтра «маршрут» = конкретный маршрут или все. */
export type RouteFilterValue = 'all' | string

/** Запрос на центрирование карты (nonce меняется при каждом выборе остановки в фильтре). */
export interface MapFocusRequest {
  stopId: string
  nonce: number
}
