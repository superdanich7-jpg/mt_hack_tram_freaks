import { useCallback, useMemo, useState } from 'react'
import { Group, Panel, Separator } from 'react-resizable-panels'
import { ALL_STOPS_VALUE, ROUTES, getStop, getStopsForRoute } from '../data/routes'
import { INITIAL_FACTORS, INITIAL_FILTERS, useDashboardData } from '../hooks/useDashboardData'
import { API_BASE_URL } from '../lib/api'
import { buildForecastCsv, buildForecastFileName, downloadCsv } from '../lib/csv'
import { formatNumber1 } from '../lib/format'
import type { ExternalFactors, FiltersState, LayoutMode, MapFocusRequest } from '../types'
import ChartPanel from './ChartPanel'
import FactorsPanel from './FactorsPanel'
import FiltersBar from './FiltersBar'
import Header from './Header'
import MapPanel from './MapPanel'
import StatsPanel from './StatsPanel'
import './Dashboard.css'

export default function Dashboard() {
  const [filters, setFilters] = useState<FiltersState>(INITIAL_FILTERS)
  const [factors, setFactors] = useState<ExternalFactors>(INITIAL_FACTORS)
  const [layoutMode, setLayoutMode] = useState<LayoutMode>('split')
  const [focusNonce, setFocusNonce] = useState(0)

  const {
    dataset,
    ranking,
    generatedAt,
    routeId,
    stopId,
    date,
    routeIds,
    routesError,
    source,
    isLoading,
    error: forecastError,
    reload,
  } = useDashboardData(filters, factors)

  /** Маршруты фильтра: только те, по которым бэкенд отдал прогноз (иначе — локальный справочник). */
  const availableRoutes = useMemo(
    () => ROUTES.filter((route) => routeIds.includes(route.id)),
    [routeIds],
  )

  const availableStops = useMemo(() => getStopsForRoute(routeId), [routeId])

  /** Подпись источника данных для шапки и панели фильтров. */
  const sourceLabel =
    source === 'backend'
      ? `бэкенд ${API_BASE_URL} · прогноз на ${date}`
      : `расчётный профиль · запрос на ${date}`

  const selectedStop = stopId === ALL_STOPS_VALUE ? undefined : getStop(stopId)

  /** Запрос центрирования карты: обновляется только при выборе остановки в фильтре или рейтинге. */
  const focus = useMemo<MapFocusRequest | null>(() => {
    if (focusNonce === 0 || !selectedStop) return null
    return { stopId: selectedStop.id, nonce: focusNonce }
  }, [focusNonce, selectedStop])

  const handleFiltersChange = useCallback((patch: Partial<FiltersState>) => {
    setFilters((prev) => {
      const next: FiltersState = { ...prev, ...patch }
      const routeChanged = patch.routeId !== undefined
      if (routeChanged && next.stopId !== ALL_STOPS_VALUE) {
        const stopInRoute = getStopsForRoute(next.routeId).some((stop) => stop.id === next.stopId)
        if (!stopInRoute) next.stopId = ALL_STOPS_VALUE
      }
      return next
    })
    if (patch.routeId !== undefined) {
      setFocusNonce(0)
    } else if (patch.stopId !== undefined && patch.stopId !== ALL_STOPS_VALUE) {
      setFocusNonce((nonce) => nonce + 1)
    }
  }, [])

  /** Выбор остановки на карте или в рейтинге: пересчёт графика без перемещения карты. */
  const handleSelectStop = useCallback((stopId: string) => {
    setFilters((prev) => ({ ...prev, stopId }))
  }, [])

  const handleReset = useCallback(() => {
    setFilters(INITIAL_FILTERS)
    setFocusNonce(0)
  }, [])

  /** Изменение корректирующих коэффициентов панели «Внешние факторы». */
  const handleFactorsChange = useCallback((patch: Partial<ExternalFactors>) => {
    setFactors((prev) => ({ ...prev, ...patch }))
  }, [])

  const handleFactorsReset = useCallback(() => {
    setFactors(INITIAL_FACTORS)
  }, [])

  const handleExport = useCallback(() => {
    if (dataset.points.length === 0) return
    downloadCsv(buildForecastFileName(dataset), buildForecastCsv(dataset))
  }, [dataset])

  const mapPanel = (
    <MapPanel
      visibleStops={availableStops}
      ranking={ranking}
      activeRouteId={routeId}
      selectedStopId={stopId}
      focus={focus}
      onSelectStop={handleSelectStop}
    />
  )

  const chartPanel = <ChartPanel dataset={dataset} />

  const statsPanel = (
    <StatsPanel
      dataset={dataset}
      ranking={ranking}
      selectedStopId={stopId}
      onSelectStop={handleSelectStop}
    />
  )

  const chartAndStats = (
    <Group orientation="vertical" className="split-group">
      <Panel id="chart" defaultSize="62" minSize="28" className="split-panel">
        {chartPanel}
      </Panel>
      <Separator className="split-separator split-separator--horizontal">
        <span className="split-separator__grip" aria-hidden="true" />
      </Separator>
      <Panel id="stats" defaultSize="38" minSize="16" className="split-panel">
        {statsPanel}
      </Panel>
    </Group>
  )

  return (
    <div className="app-shell">
      <Header
        layoutMode={layoutMode}
        onLayoutModeChange={setLayoutMode}
        onExport={handleExport}
        exportDisabled={dataset.points.length === 0}
        exportRowCount={dataset.points.length}
        generatedAt={generatedAt}
        dataSourceLabel={sourceLabel}
      />

      <FiltersBar
        filters={{ ...filters, routeId, stopId }}
        routes={availableRoutes}
        availableStops={availableStops}
        selectedStop={selectedStop}
        pointsCount={dataset.points.length}
        accuracy={dataset.summary.accuracy}
        dataSource={source}
        isLoading={isLoading}
        error={forecastError || routesError}
        onReload={reload}
        onChange={handleFiltersChange}
        onReset={handleReset}
      />

      <FactorsPanel
        factors={factors}
        applied={dataset.meta.appliedFactors}
        volumeMultiplier={dataset.meta.volumeMultiplier}
        trafficSmoothing={dataset.meta.trafficSmoothing}
        onChange={handleFactorsChange}
        onReset={handleFactorsReset}
      />

      <main className={`app-body app-body--${layoutMode}`}>
        {layoutMode === 'split' && (
          <Group orientation="horizontal" className="split-group">
            <Panel id="map" defaultSize="52" minSize="22" className="split-panel">
              {mapPanel}
            </Panel>
            <Separator className="split-separator split-separator--vertical">
              <span className="split-separator__grip" aria-hidden="true" />
            </Separator>
            <Panel id="charts" defaultSize="48" minSize="26" className="split-panel">
              {chartAndStats}
            </Panel>
          </Group>
        )}

        {layoutMode === 'map' && mapPanel}

        {layoutMode === 'charts' && chartAndStats}
      </main>

      <footer className="app-footer">
        <span>
          {source === 'backend' ? (
            <>
              Данные бэкенда{' '}
              <code>
                {API_BASE_URL}/forecast?route={routeId}&amp;date={date}
              </code>
              {isLoading ? ' · обновление…' : ''}
            </>
          ) : (
            <>
              Бэкенд недоступен — расчётный профиль <code>src/data/forecast.ts</code>
              {isLoading ? ' · запрос к бэкенду…' : ''}
            </>
          )}
          {forecastError ? ` · ${forecastError}` : ''}
          {routesError ? ` · /routes: ${routesError}` : ''}
        </span>
        <span className="app-footer__hint">
          {dataset.meta.scopeLabel} · {dataset.meta.intervalLabel} · {dataset.meta.horizonLabel} ·
          факторы: {dataset.meta.appliedFactors.map((factor) => factor.label).join(' / ')} · объём
          ×{formatNumber1(dataset.meta.volumeMultiplier)} · сглаживание{' '}
          {Math.round(dataset.meta.trafficSmoothing * 100)} %
        </span>
      </footer>
    </div>
  )
}
