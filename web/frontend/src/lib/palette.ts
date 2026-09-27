/** Единая палитра тёмной темы (фон #1a1230, акценты #406fd2 / #f5f8fd). */
export const CHART_COLORS = {
  /** Основной цвет прогноза — акцентный синий. */
  forecast: '#406fd2',
  forecastFillTop: 'rgba(64, 111, 210, 0.45)',
  forecastFillBottom: 'rgba(64, 111, 210, 0.02)',
  /** Цвет линии факта прошлого периода — светлый акцент. */
  actual: '#f5f8fd',
  /** Цвет линии заполняемости — приглушённый фиолетовый. */
  load: '#a08bf0',
  grid: 'rgba(151, 140, 205, 0.18)',
  axis: '#8f96bd',
  tooltipBg: '#221839',
  tooltipBorder: '#362a5c',
  /** Цвет маркера выбранной остановки. */
  selected: '#f5f8fd',
} as const

/** Уровни загрузки остановки: от приглушённого тона к яркому светлому. */
export const LOAD_COLORS = {
  low: '#5b63a8',
  medium: '#406fd2',
  high: '#f5f8fd',
} as const

/** Цвет для бейджей загрузки (низкая / средняя / высокая). */
export function loadColor(load: number): string {
  if (load >= 70) return LOAD_COLORS.high
  if (load >= 45) return LOAD_COLORS.medium
  return LOAD_COLORS.low
}
