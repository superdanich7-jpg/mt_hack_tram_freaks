import { useEffect, useMemo } from 'react'
import {
  CircleMarker,
  MapContainer,
  Polyline,
  Popup,
  TileLayer,
  Tooltip as LeafletTooltip,
  useMap,
} from 'react-leaflet'
import { ALL_ROUTES_VALUE, MOSCOW_CENTER, ROUTE_PATHS, ROUTES, getRoute } from '../data/routes'
import { formatNumber, formatPercent } from '../lib/format'
import { CHART_COLORS, loadColor } from '../lib/palette'
import type { LatLng, MapFocusRequest, StopRankingRow, TramStop } from '../types'
import './MapPanel.css'

interface MapPanelProps {
  visibleStops: TramStop[]
  ranking: StopRankingRow[]
  activeRouteId: string
  selectedStopId: string
  focus: MapFocusRequest | null
  onSelectStop: (stopId: string) => void
}

function toPositions(path: LatLng[]): [number, number][] {
  return path.map((point) => [point.lat, point.lng])
}

/** Держит размер карты в актуальном состоянии при перетаскивании разделителя сплит-режима. */
function MapAutoResize() {
  const map = useMap()

  useEffect(() => {
    const container = map.getContainer()
    const observer = new ResizeObserver(() => map.invalidateSize())
    observer.observe(container)
    return () => observer.disconnect()
  }, [map])

  return null
}

/** Центрирует карту по выбранной остановке и подгоняет границы под видимые остановки маршрута. */
function MapController({
  visibleStops,
  focus,
}: {
  visibleStops: TramStop[]
  focus: MapFocusRequest | null
}) {
  const map = useMap()
  const boundsKey = useMemo(() => visibleStops.map((stop) => stop.id).join('|'), [visibleStops])

  useEffect(() => {
    if (!focus) return
    const stop = visibleStops.find((item) => item.id === focus.stopId)
    if (!stop) return
    map.setView([stop.coords.lat, stop.coords.lng], Math.max(map.getZoom(), 14), { animate: true })
  }, [focus, map, visibleStops])

  useEffect(() => {
    if (visibleStops.length < 2) return
    const points = visibleStops.map((stop) => [stop.coords.lat, stop.coords.lng] as [number, number])
    map.fitBounds(points, { padding: [48, 48], animate: true })
  }, [boundsKey, visibleStops, map])

  return null
}

export default function MapPanel({
  visibleStops,
  ranking,
  activeRouteId,
  selectedStopId,
  focus,
  onSelectStop,
}: MapPanelProps) {
  const metricsByStop = useMemo(() => {
    const map = new Map<string, StopRankingRow>()
    ranking.forEach((row) => map.set(row.stop.id, row))
    return map
  }, [ranking])

  return (
    <section className="map-panel">
      <div className="map-panel__badges">
        <span className="map-panel__badge">
          Остановок: <b>{visibleStops.length}</b>
        </span>
        <span className="map-panel__badge map-panel__badge--hint">
          Клик по остановке → пересчёт графика
        </span>
      </div>

      <MapContainer
        center={[MOSCOW_CENTER.lat, MOSCOW_CENTER.lng]}
        zoom={12}
        minZoom={9}
        maxZoom={18}
        className="map-panel__canvas"
        scrollWheelZoom
      >
        <TileLayer
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
          attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
          subdomains="abc"
          maxZoom={19}
        />
        <MapAutoResize />
        <MapController visibleStops={visibleStops} focus={focus} />

        {ROUTES.map((route) => {
          const isActive = activeRouteId === ALL_ROUTES_VALUE || activeRouteId === route.id
          return (
            <Polyline
              key={route.id}
              positions={toPositions(ROUTE_PATHS[route.id] ?? [])}
              pathOptions={{
                color: route.color,
                weight: isActive ? 4 : 2.5,
                opacity: isActive ? 0.9 : 0.22,
                dashArray: isActive ? undefined : '6 6',
                lineCap: 'round',
              }}
            >
              <LeafletTooltip sticky>
                Маршрут {route.number} · {route.name}
              </LeafletTooltip>
            </Polyline>
          )
        })}

        {visibleStops.map((stop) => {
          const metrics = metricsByStop.get(stop.id)
          const load = metrics?.avgLoad ?? 0
          const isSelected = stop.id === selectedStopId
          const route = getRoute(stop.routeIds[0] ?? '')
          const badgeColor =
            stop.routeIds.length > 1 ? CHART_COLORS.selected : route?.color ?? CHART_COLORS.forecast

          return (
            <CircleMarker
              key={stop.id}
              center={[stop.coords.lat, stop.coords.lng]}
              radius={isSelected ? 11 : 5.5 + (load / 100) * 7}
              pathOptions={{
                color: isSelected ? CHART_COLORS.selected : '#1a1230',
                weight: isSelected ? 3 : 1.5,
                fillColor: loadColor(load),
                fillOpacity: isSelected ? 1 : 0.85,
              }}
              eventHandlers={{ click: () => onSelectStop(stop.id) }}
            >
              <Popup>
                <div className="map-popup">
                  <p className="map-popup__title">{stop.name}</p>
                  <p className="map-popup__meta">
                    {stop.district} · маршруты:{' '}
                    <b style={{ color: badgeColor }}>
                      {stop.routeIds.map((id) => getRoute(id)?.number).join(', ')}
                    </b>
                  </p>
                  <ul className="map-popup__list">
                    <li>
                      Суточный прогноз: <b>{formatNumber(metrics?.dailyForecast ?? 0)} пасс.</b>
                    </li>
                    <li>
                      Пик: <b>{metrics?.peakLabel ?? '—'}</b> · {formatNumber(metrics?.peakValue ?? 0)} пасс./ч
                    </li>
                    <li>
                      Средняя заполняемость: <b>{formatPercent(load)}</b>
                    </li>
                  </ul>
                  <button type="button" onClick={() => onSelectStop(stop.id)}>
                    Показать на графике
                  </button>
                </div>
              </Popup>
              <LeafletTooltip direction="top" offset={[0, -6]} opacity={1}>
                {stop.name} · {formatPercent(load)}
              </LeafletTooltip>
            </CircleMarker>
          )
        })}
      </MapContainer>

      <div className="map-panel__legend">
        {ROUTES.map((route) => (
          <span key={route.id} className="map-panel__legend-item">
            <span className="map-panel__legend-dot" style={{ background: route.color }} />
            {route.number}
          </span>
        ))}
        <span className="map-panel__legend-scale">
          <span className="map-panel__legend-dot" style={{ background: loadColor(20) }} />
          <span className="map-panel__legend-dot" style={{ background: loadColor(55) }} />
          <span className="map-panel__legend-dot" style={{ background: loadColor(85) }} />
          загрузка
        </span>
      </div>

      {visibleStops.length === 0 && (
        <div className="map-panel__empty">Для выбранного маршрута нет остановок в мок-данных</div>
      )}
    </section>
  )
}
