const NS = 'http://www.w3.org/2000/svg';
function element(tag, attrs = {}, value = '') {
  const node = document.createElementNS(NS, tag);
  for (const [key, val] of Object.entries(attrs)) node.setAttribute(key, val);
  if (value) node.textContent = value;
  return node;
}
export function createMapGeometry(roads, data) {
  const points = [...roads.nodes, ...data.buses];
  const minLon = Math.min(...points.map(n => n.lon)), maxLon = Math.max(...points.map(n => n.lon));
  const minLat = Math.min(...points.map(n => n.lat)), maxLat = Math.max(...points.map(n => n.lat));
  const cosine = Math.cos((minLat + maxLat) / 2 * Math.PI / 180);
  const scale = Math.min(800 / ((maxLon - minLon) * cosine), 460 / (maxLat - minLat));
  const dx = (900 - (maxLon - minLon) * cosine * scale) / 2;
  const dy = (560 - (maxLat - minLat) * scale) / 2;
  const project = point => [dx + (point.lon - minLon) * cosine * scale, dy + (maxLat - point.lat) * scale];
  const path = roads.roads.map(line => line.map(([lon, lat], i) => `${i ? 'L' : 'M'}${project({lon, lat}).map(v => v.toFixed(2)).join(',')}`).join('')).join('');
  return {project, path, pixelsPerKM: scale / 111.2};
}
export function renderMap(svg, geometry, snapshot, layers, selected, onSelect) {
  const key = svg.dataset.mapKey;
  const focused = document.activeElement?.getAttribute('data-map-focus');
  svg.replaceChildren();
  svg.append(element('title', {}, `Bản đồ Đống Đa · ${snapshot.method} · bước ${snapshot.step}`));
  svg.append(element('desc', {}, 'Xám: trạm hiện hữu. Đỏ: trạm mới; điểm lớn hơn có công suất lớn hơn. Chọn trạm hoặc thanh cái để xem chi tiết.'));
  const defs = element('defs'), gradient = element('radialGradient', {id: `heat-${key}`});
  gradient.append(element('stop', {offset: '0%', 'stop-color': '#e58b2b', 'stop-opacity': '.75'}), element('stop', {offset: '100%', 'stop-color': '#f8ce69', 'stop-opacity': '0'}));
  defs.append(gradient); svg.append(defs);
  svg.append(element('path', {d: geometry.path, class: 'road-base'}), element('path', {d: geometry.path, class: 'road-line'}));
  const heat = element('g', {'data-layer': 'demand', 'aria-hidden': 'true'});
  if (layers.demand) snapshot.remaining.forEach((demand, i) => {
    if (demand < .01) return;
    const [cx, cy] = geometry.project(layers.nodes[i]);
    heat.append(element('circle', {cx, cy, r: 10 + demand * 14, fill: `url(#heat-${key})`, opacity: .1 + demand * .5}));
  });
  svg.append(heat);
  const service = element('g', {'data-layer': 'service', 'aria-hidden': 'true'});
  if (layers.service) snapshot.stations.forEach(s => {
    const [cx, cy] = geometry.project(s.node);
    service.append(element('circle', {cx, cy, r: s.radiusKM * geometry.pixelsPerKM, class: 'service-circle', 'data-service-station': s.id}));
  });
  svg.append(service);
  const station = selected?.kind === 'station' && selected.mapKey === key ? snapshot.stations.find(s => s.id === selected.id) : null;
  if (layers.grid && station) {
    const bus = snapshot.buses.find(b => b.id === station.busId), [x1, y1] = geometry.project(station.node), [x2, y2] = geometry.project(bus);
    svg.append(element('line', {x1, y1, x2, y2, class: 'connection-line'}));
  }
  function button(group, type, id, label) {
    group.setAttribute('role', 'button'); group.setAttribute('tabindex', '0'); group.setAttribute('aria-label', label);
    group.setAttribute('data-map-focus', `${key}-${type}-${id}`);
    group.setAttribute('aria-pressed', String(selected?.mapKey === key && selected?.kind === type && selected?.id === id));
    group.append(element('title', {}, label));
    group.addEventListener('click', () => onSelect({kind: type, id, mapKey: key}));
    group.addEventListener('keydown', e => {if (e.key === 'Enter' || e.key === ' ') {e.preventDefault(); onSelect({kind: type, id, mapKey: key});}});
  }
  const grid = element('g', {'data-layer': 'grid'});
  if (layers.grid) snapshot.buses.forEach(bus => {
    const [x, y] = geometry.project(bus);
    const group = element('g', {class: `bus-marker${bus.overloaded ? ' overloaded' : ''}`});
    const fill = bus.overloaded ? '#c73838' : bus.utilization > .8 ? '#d1812c' : '#57958a';
    group.append(element('circle', {cx: x, cy: y, r: 12, fill: 'transparent'}), element('path', {d: `M${x},${y - 6}l6,6 -6,6 -6,-6Z`, fill, stroke: 'white', 'stroke-width': 1.5}));
    button(group, 'bus', bus.id, `Thanh cái ${bus.id}, tải sạc ${(bus.utilization * 100).toFixed(1)}% dư địa${bus.overloaded ? ', quá tải' : ''}`);
    grid.append(group);
  });
  svg.append(grid);
  const markers = element('g', {'data-layer': 'stations'});
  snapshot.stations.forEach(s => {
    const [cx, cy] = geometry.project(s.node), radius = 3.5 + Math.sqrt(s.powerKW) * .12;
    const isSelected = selected?.mapKey === key && selected?.kind === 'station' && selected?.id === s.id;
    const group = element('g', {class: `station-marker ${s.existing ? 'existing' : 'new'}${isSelected ? ' selected' : ''}`, 'data-station-id': s.id});
    group.append(element('circle', {cx, cy, r: Math.max(13, radius + 4), fill: 'transparent'}));
    if (isSelected || s.id === snapshot.targetId || s.id === snapshot.donorId) group.append(element('circle', {cx, cy, r: radius + 5, class: 'selection-ring'}));
    group.append(element('circle', {cx, cy, r: radius, class: 'station-dot'}));
    if (isSelected) group.append(element('text', {x: cx + radius + 8, y: cy - 7, class: 'map-station-label'}, s.id));
    button(group, 'station', s.id, `${s.id}, ${s.existing ? 'trạm hiện hữu' : 'trạm mới'}, ${s.chargers} bộ sạc, ${s.powerKW} kW`);
    markers.append(group);
  });
  svg.append(markers);
  svg.append(element('text', {x: 30, y: 30, class: 'map-location'}, 'ĐỐNG ĐA · HÀ NỘI'), element('text', {x: 860, y: 30, class: 'map-location'}, 'N ↑'));
  if (focused?.startsWith(key + '-')) svg.querySelector(`[data-map-focus="${focused}"]`)?.focus({preventScroll: true});
}
