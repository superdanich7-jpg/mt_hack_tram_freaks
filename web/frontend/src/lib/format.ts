const ruNumber = new Intl.NumberFormat('ru-RU')
const ruNumber1 = new Intl.NumberFormat('ru-RU', {
  minimumFractionDigits: 1,
  maximumFractionDigits: 1,
})
const ruDateTime = new Intl.DateTimeFormat('ru-RU', {
  day: '2-digit',
  month: '2-digit',
  year: 'numeric',
  hour: '2-digit',
  minute: '2-digit',
})

/** 12 345 → «12 345». */
export function formatNumber(value: number): string {
  return ruNumber.format(Math.round(value))
}

/** 12 345,6 → «12 345,6». */
export function formatNumber1(value: number): string {
  return ruNumber1.format(value)
}

/** 48,2 → «48,2 %». */
export function formatPercent(value: number, digits: 0 | 1 = 1): string {
  const text = digits === 0 ? formatNumber(value) : formatNumber1(value)
  return `${text} %`
}

/** 4 → «04:00». */
export function formatHourLabel(hour: number): string {
  return `${String(hour).padStart(2, '0')}:00`
}

/** +8,4 % / −3,1 %. */
export function formatSignedPercent(value: number): string {
  const sign = value > 0 ? '+' : value < 0 ? '−' : ''
  return `${sign}${formatNumber1(Math.abs(value))} %`
}

/** Компактный формат для осей графика: 12 400 → «12,4 тыс.». */
export function formatCompact(value: number): string {
  if (Math.abs(value) >= 1_000_000) return `${formatNumber1(value / 1_000_000)} млн`
  if (Math.abs(value) >= 10_000) return `${formatNumber1(value / 1000)} тыс.`
  return formatNumber(value)
}

/** ISO-строка → «26.09.2026, 14:05». */
export function formatDateTime(iso: string): string {
  const date = new Date(iso)
  if (Number.isNaN(date.getTime())) return '—'
  return ruDateTime.format(date)
}

/** Аккуратный слаг для имени файла: «Чистые пруды, 27» → «chistye-prudy-27». */
export function slugify(input: string): string {
  const map: Record<string, string> = {
    а: 'a', б: 'b', в: 'v', г: 'g', д: 'd', е: 'e', ё: 'e', ж: 'zh', з: 'z',
    и: 'i', й: 'y', к: 'k', л: 'l', м: 'm', н: 'n', о: 'o', п: 'p', р: 'r',
    с: 's', т: 't', у: 'u', ф: 'f', х: 'h', ц: 'c', ч: 'ch', ш: 'sh', щ: 'sch',
    ъ: '', ы: 'y', ь: '', э: 'e', ю: 'yu', я: 'ya',
  }
  return input
    .toLowerCase()
    .split('')
    .map((char) => map[char] ?? char)
    .join('')
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '')
}
