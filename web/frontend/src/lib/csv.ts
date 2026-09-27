import type { ForecastDataset } from '../types'
import { slugify } from './format'

/** Описание колонки CSV. */
export interface CsvColumn<T> {
  header: string
  value: (row: T) => string | number
}

const SEPARATOR = ';'
const BOM = '\uFEFF'

function escapeCell(value: string | number): string {
  const text = String(value)
  if (text.includes(SEPARATOR) || text.includes('"') || text.includes('\n')) {
    return `"${text.replace(/"/g, '""')}"`
  }
  return text
}

/**
 * Собирает CSV-строку. Разделитель «;» и BOM выбраны сознательно:
 * так файл корректно открывается в Excel с русской локалью.
 */
export function buildCsv<T>(rows: T[], columns: CsvColumn<T>[]): string {
  const header = columns.map((column) => escapeCell(column.header)).join(SEPARATOR)
  const body = rows.map((row) =>
    columns.map((column) => escapeCell(column.value(row))).join(SEPARATOR),
  )
  return BOM + [header, ...body].join('\r\n') + '\r\n'
}

/** Инициирует скачивание готового CSV в браузере. */
export function downloadCsv(filename: string, content: string): void {
  const blob = new Blob([content], { type: 'text/csv;charset=utf-8;' })
  const url = URL.createObjectURL(blob)
  const link = document.createElement('a')
  link.href = url
  link.download = filename
  document.body.appendChild(link)
  link.click()
  document.body.removeChild(link)
  URL.revokeObjectURL(url)
}

/** Колонки экспорта: ровно те данные, которые отображаются на графике и в таблице. */
export const FORECAST_CSV_COLUMNS: CsvColumn<ForecastDataset['points'][number]>[] = [
  { header: 'Час', value: (row) => row.label },
  { header: 'Факт (прошлый период), пасс.', value: (row) => Math.round(row.actual) },
  { header: 'Прогноз, пасс.', value: (row) => Math.round(row.forecast) },
  { header: 'Нижняя граница, пасс.', value: (row) => Math.round(row.low) },
  { header: 'Верхняя граница, пасс.', value: (row) => Math.round(row.high) },
  { header: 'Заполняемость, %', value: (row) => Math.round(row.load) },
]

/** Коэффициент в CSV — с десятичной запятой, как принято в русской локали Excel. */
function decimal(value: number, digits = 3): string {
  return value.toFixed(digits).replace('.', ',')
}

/** CSV текущего отображаемого набора мок-данных вместе с контекстом фильтров и внешних факторов. */
export function buildForecastCsv(dataset: ForecastDataset): string {
  const meta = dataset.meta
  const date = meta.generatedAt.slice(0, 10)
  const rows = dataset.points.map((point, index) => ({
    point,
    index,
  }))
  const columns: CsvColumn<(typeof rows)[number]>[] = [
    { header: 'Маршрут', value: () => meta.routeLabel },
    { header: 'Остановка', value: () => meta.scopeLabel },
    { header: 'Интервал времени', value: () => meta.intervalLabel },
    { header: 'Горизонт прогноза', value: () => meta.horizonLabel },
    { header: 'Дата формирования', value: () => date },
    { header: 'Множитель объёма (внешние факторы)', value: () => decimal(meta.volumeMultiplier) },
    { header: 'Сглаживание пиков пробками, %', value: () => Math.round(meta.trafficSmoothing * 100) },
    ...meta.appliedFactors.map((factor) => ({
      header: `Фактор: ${factor.label}`,
      value: () => decimal(factor.coefficient),
    })),
    ...FORECAST_CSV_COLUMNS.map((column) => ({ header: column.header, value: (row: (typeof rows)[number]) => column.value(row.point) })),
  ]
  return buildCsv(rows, columns)
}

/** Имя файла экспорта: tram_forecast_a-paveletskaya_day_2026-09-26.csv */
export function buildForecastFileName(dataset: ForecastDataset): string {
  const meta = dataset.meta
  const scope = slugify(meta.scopeId) || 'all'
  const date = meta.generatedAt.slice(0, 10)
  return `tram_forecast_${scope}_${meta.horizonId}_${date}.csv`
}
