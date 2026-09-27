import type { LatLng, TramRoute, TramStop } from '../types'

/** Мок-остановки: координаты соответствуют реальным адресам Москвы, пассажиропоток — выдуманный. */
interface RawStop {
  id: string
  name: string
  district: string
  lat: number
  lng: number
  /** Среднесуточный пассажиропоток, пасс./сутки. */
  dailyBase: number
}

interface RawRoute {
  id: string
  number: string
  name: string
  color: string
  /** Остановки в порядке движения. */
  stopIds: string[]
}

const RAW_ROUTES: RawRoute[] = [
  {
    id: 'route-a',
    number: 'А',
    name: 'Чистые пруды — Дербеневская набережная',
    color: '#d47a5b', // спокойный терракотовый
    stopIds: [
      'a-chistye-prudy',
      'a-pokrovka',
      'a-yauzskie-vorota',
      'a-ustinsky-most',
      'a-novokuznetskaya',
      'a-paveletskaya',
      'a-kozhevnicheskaya',
      'a-derbenevskaya',
    ],
  },
  {
    id: 'route-3',
    number: '3',
    name: 'Чертаново — Павелецкая',
    color: '#5a9e78', // сдержанный зелёный
    stopIds: [
      't3-chertanovskaya',
      't3-balaklavsky',
      't3-varshavskaya',
      't3-nagatinskaya',
      'a-paveletskaya',
    ],
  },
  {
    id: 'route-6',
    number: '6',
    name: 'Богородское — Комсомольская площадь',
    color: '#9b82db', // приглушённый фиолетовый
    stopIds: [
      't6-preobrazhenskaya',
      't6-krasnobogatyrskaya',
      't6-sokolniki',
      't6-rusakovskaya',
      't6-krasnoselskaya',
      't6-komsomolskaya',
    ],
  },
  {
    id: 'route-27',
    number: '27',
    name: 'Дмитровская — Михалково',
    color: '#d1a852', // горчично-жёлтый
    stopIds: [
      't27-dmitrovskaya',
      't27-timiryazevskaya',
      't27-vuchetica',
      't27-koptevo',
      't27-mihalkovo',
    ],
  },
  {
    id: 'route-50',
    number: '50',
    name: 'Семёновская — Чистые пруды',
    color: '#5d87d8', // стальной синий
    stopIds: [
      't50-semenovskaya',
      't50-elektrozavodskaya',
      't50-baumanskaya',
      't50-kursky',
      'a-chistye-prudy',
    ],
  },
]

const RAW_STOPS: RawStop[] = [
  // Маршрут «А»
  { id: 'a-chistye-prudy', name: 'Чистые пруды', district: 'Басманный', lat: 55.7648, lng: 37.6445, dailyBase: 14200 },
  { id: 'a-pokrovka', name: 'Покровка, 27', district: 'Басманный', lat: 55.7603, lng: 37.648, dailyBase: 9800 },
  { id: 'a-yauzskie-vorota', name: 'Яузские Ворота', district: 'Басманный', lat: 55.7509, lng: 37.6431, dailyBase: 8600 },
  { id: 'a-ustinsky-most', name: 'Устьинский мост', district: 'Якиманка', lat: 55.7454, lng: 37.6383, dailyBase: 7400 },
  { id: 'a-novokuznetskaya', name: 'Новокузнецкая', district: 'Замоскворечье', lat: 55.7418, lng: 37.6293, dailyBase: 11300 },
  { id: 'a-paveletskaya', name: 'Павелецкая', district: 'Замоскворечье', lat: 55.7302, lng: 37.6397, dailyBase: 12600 },
  { id: 'a-kozhevnicheskaya', name: 'Кожевническая', district: 'Даниловский', lat: 55.7268, lng: 37.6445, dailyBase: 6900 },
  { id: 'a-derbenevskaya', name: 'Дербеневская набережная', district: 'Даниловский', lat: 55.7225, lng: 37.6502, dailyBase: 5200 },
  // Маршрут 3
  { id: 't3-chertanovskaya', name: 'Чертановская', district: 'Чертаново Северное', lat: 55.611, lng: 37.607, dailyBase: 12100 },
  { id: 't3-balaklavsky', name: 'Балаклавский проспект', district: 'Чертаново Северное', lat: 55.6305, lng: 37.596, dailyBase: 7300 },
  { id: 't3-varshavskaya', name: 'Варшавская', district: 'Нагорный', lat: 55.6533, lng: 37.621, dailyBase: 10400 },
  { id: 't3-nagatinskaya', name: 'Нагатинская', district: 'Нагатино-Садовники', lat: 55.6826, lng: 37.654, dailyBase: 9900 },
  // Маршрут 6
  { id: 't6-preobrazhenskaya', name: 'Преображенская площадь', district: 'Преображенское', lat: 55.796, lng: 37.715, dailyBase: 10800 },
  { id: 't6-krasnobogatyrskaya', name: 'Краснобогатырская', district: 'Богородское', lat: 55.812, lng: 37.7, dailyBase: 8100 },
  { id: 't6-sokolniki', name: 'Сокольники', district: 'Сокольники', lat: 55.789, lng: 37.679, dailyBase: 11900 },
  { id: 't6-rusakovskaya', name: 'Русаковская', district: 'Сокольники', lat: 55.783, lng: 37.679, dailyBase: 8800 },
  { id: 't6-krasnoselskaya', name: 'Красносельская', district: 'Красносельский', lat: 55.78, lng: 37.666, dailyBase: 9700 },
  { id: 't6-komsomolskaya', name: 'Комсомольская площадь', district: 'Красносельский', lat: 55.7757, lng: 37.655, dailyBase: 13400 },
  // Маршрут 27
  { id: 't27-dmitrovskaya', name: 'Дмитровская', district: 'Бутырский', lat: 55.8074, lng: 37.582, dailyBase: 10200 },
  { id: 't27-timiryazevskaya', name: 'Тимирязевская', district: 'Тимирязевский', lat: 55.8174, lng: 37.574, dailyBase: 12600 },
  { id: 't27-vuchetica', name: 'Улица Вучетича', district: 'Тимирязевский', lat: 55.823, lng: 37.563, dailyBase: 6400 },
  { id: 't27-koptevo', name: 'Коптево', district: 'Коптево', lat: 55.829, lng: 37.552, dailyBase: 7800 },
  { id: 't27-mihalkovo', name: 'Михалково', district: 'Головинский', lat: 55.8335, lng: 37.539, dailyBase: 5900 },
  // Маршрут 50
  { id: 't50-semenovskaya', name: 'Семёновская', district: 'Соколиная Гора', lat: 55.7832, lng: 37.719, dailyBase: 10400 },
  { id: 't50-elektrozavodskaya', name: 'Электрозаводская', district: 'Басманный', lat: 55.7815, lng: 37.704, dailyBase: 9100 },
  { id: 't50-baumanskaya', name: 'Бауманская', district: 'Басманный', lat: 55.7725, lng: 37.679, dailyBase: 11700 },
  { id: 't50-kursky', name: 'Курский вокзал', district: 'Басманный', lat: 55.7577, lng: 37.659, dailyBase: 13900 },
]

/** Маршруты трамвая (мок). */
export const ROUTES: TramRoute[] = RAW_ROUTES.map((route) => ({ ...route }))

/** Остановки с автоматически вычисленным списком маршрутов. */
export const STOPS: TramStop[] = RAW_STOPS.map((stop) => ({
  id: stop.id,
  name: stop.name,
  district: stop.district,
  coords: { lat: stop.lat, lng: stop.lng },
  dailyBase: stop.dailyBase,
  routeIds: ROUTES.filter((route) => route.stopIds.includes(stop.id)).map((route) => route.id),
}))

/** Полилинии трасс маршрутов для отрисовки на карте. */
export const ROUTE_PATHS: Record<string, LatLng[]> = ROUTES.reduce<Record<string, LatLng[]>>(
  (acc, route) => {
    acc[route.id] = route.stopIds
      .map((stopId) => STOPS.find((stop) => stop.id === stopId))
      .filter((stop): stop is TramStop => Boolean(stop))
      .map((stop) => stop.coords)
    return acc
  },
  {},
)

/** Центр карты — Москва (внутри Садового кольца). */
export const MOSCOW_CENTER: LatLng = { lat: 55.7522, lng: 37.6156 }

/** Значение фильтра «все маршруты». */
export const ALL_ROUTES_VALUE = 'all'

/** Значение фильтра «все остановки». */
export const ALL_STOPS_VALUE = 'all'

export function getStop(stopId: string): TramStop | undefined {
  return STOPS.find((stop) => stop.id === stopId)
}

export function getRoute(routeId: string): TramRoute | undefined {
  return ROUTES.find((route) => route.id === routeId)
}

/** Остановки выбранного маршрута (или все остановки при 'all'). */
export function getStopsForRoute(routeId: string): TramStop[] {
  if (routeId === ALL_ROUTES_VALUE) return STOPS
  const route = getRoute(routeId)
  if (!route) return []
  return route.stopIds
    .map((stopId) => getStop(stopId))
    .filter((stop): stop is TramStop => Boolean(stop))
}

/** Подпись маршрута: «Маршрут А · Чистые пруды — Дербеневская набережная». */
export function formatRouteLabel(routeId: string): string {
  if (routeId === ALL_ROUTES_VALUE) return 'Все маршруты'
  const route = getRoute(routeId)
  return route ? `Маршрут ${route.number} · ${route.name}` : 'Неизвестный маршрут'
}
