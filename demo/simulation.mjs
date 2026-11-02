// Deterministic illustrative rules. No policy inference, training or paper replay.
export const ACTIONS = [
  ['Tạo trạm · lợi ích', 'Tìm nút trống có tiềm năng mở rộng phạm vi phục vụ; xét thêm tải vùng nếu bật chiến lược cân bằng.'],
  ['Tạo trạm · nhu cầu', 'Tìm nút trống có tín hiệu nhu cầu chưa phục vụ cao trong kịch bản giả lập.'],
  ['Bổ sung · lợi ích', 'Thêm một bộ sạc mẫu tại trạm có khả năng mở rộng phạm vi phục vụ.'],
  ['Bổ sung · nhu cầu', 'Thêm một bộ sạc mẫu tại trạm gần vùng còn nhiều nhu cầu.'],
  ['Di dời bộ sạc', 'Chuyển một bộ sạc từ trạm có nhu cầu thấp sang trạm có nhu cầu cao; giữ nguyên số trạm.'],
];
export const MAX_STEPS = 48;
export function hash(value) {
  let h = 2166136261;
  for (const c of String(value)) h = Math.imul(h ^ c.charCodeAt(0), 16777619);
  return (h >>> 0) / 4294967296;
}
export function projectGraph(data) {
  const lon = data.nodes.map(n => n.lon), lat = data.nodes.map(n => n.lat);
  const left = Math.min(...lon), top = Math.max(...lat);
  const width = Math.max(...lon) - left, height = top - Math.min(...lat);
  const correction = Math.cos(top * Math.PI / 180);
  const scale = Math.min(610 / (width * correction), 390 / height);
  const offsetX = (760 - width * correction * scale) / 2;
  const offsetY = (510 - height * scale) / 2;
  const project = (x, y) => [offsetX + (x - left) * correction * scale, offsetY + (top - y) * scale];
  return { nodes: data.nodes.map(n => {
    const [x, y] = project(n.lon, n.lat);
    const col = Math.min(2, Math.floor((n.lon - left) / width * 3));
    const row = Math.min(5, Math.floor((top - n.lat) / height * 6));
    return { ...n, x, y, zone: row * 3 + col, demand: .3 + hash(n.id) * .7 };
  }), paths: data.roads.map(line => line.map((p, i) => `${i ? 'L' : 'M'}${project(...p).map(v => v.toFixed(2)).join(',')}`).join('')).join('') };
}
const distance = (a, b) => Math.hypot(a.x - b.x, a.y - b.y);
const capacity = zone => 7 + hash(`capacity-${zone}`) * 5;
function loads(stations) {
  const values = Array(18).fill(0);
  for (const station of stations) values[station.node.zone] += station.chargers;
  return values;
}
function unmet(node, stations, demand) {
  let service = 0;
  for (const station of stations) service += station.chargers * Math.exp(-distance(node, station.node) / 37);
  return Math.exp(-service / (2.7 * demand));
}
function initialStations(nodes) {
  const sorted = [...nodes].sort((a, b) => hash(`initial-${a.id}`) - hash(`initial-${b.id}`));
  const picked = [sorted[0]];
  while (picked.length < 5) {
    const next = sorted.filter(n => !picked.includes(n)).sort((a, b) => Math.min(...picked.map(p => distance(b, p))) - Math.min(...picked.map(p => distance(a, p))))[0];
    picked.push(next);
  }
  return picked.map((node, i) => ({ id: i + 1, node, chargers: 2, initial: true }));
}
export function generateRun(nodes, {budget = 100, demand = 1, strategy = 'balanced'} = {}) {
  if (!nodes.length) throw new Error('Không có nút đường bộ để tạo kịch bản.');
  const stations = initialStations(nodes);
  const baseline = nodes.reduce((sum, n) => sum + n.demand * unmet(n, stations, demand), 0);
  const snapshots = [];
  let spent = 0;
  const candidates = [...nodes].sort((a, b) => hash(a.id) - hash(b.id)).slice(0, 180);
  function snapshot(action, target, message) {
    const grid = loads(stations);
    snapshots.push({ stations: stations.map(s => ({...s})), spent: Number(spent.toFixed(2)), action, target, message,
      remaining: 100 * nodes.reduce((sum, n) => sum + n.demand * unmet(n, stations, demand), 0) / baseline,
      overloaded: grid.filter((load, i) => load > capacity(i)).length });
  }
  snapshot(null, null, 'Khởi tạo 5 trạm mẫu. Vị trí không phải kế hoạch hiện hữu trong bài báo.');
  const sequence = [0, 1, 0, 2, 1, 3, 0, 4];
  for (let t = 0; t < MAX_STEPS; t++) {
    const action = sequence[t % sequence.length];
    const cost = action < 2 ? 2.8 : action < 4 ? .9 : .4;
    if (spent + cost > budget + 1e-8) break;
    const grid = loads(stations);
    const occupied = new Set(stations.map(s => s.node.id));
    const score = (node, isDemand) => {
      const need = node.demand * unmet(node, stations, demand);
      const isolation = Math.min(...stations.map(s => distance(node, s.node))) / 100;
      const pressure = (grid[node.zone] + (action < 2 ? 3 : 1)) / capacity(node.zone);
      return (isDemand ? need * 2 : need + isolation * .5) - (strategy === 'balanced' ? Math.max(0, pressure - .65) * 1.5 : 0);
    };
    let target;
    if (action < 2) {
      const free = candidates.filter(n => !occupied.has(n.id));
      if (!free.length) break;
      const node = free.sort((a, b) => score(b, action === 1) - score(a, action === 1))[0];
      target = {id: stations.length + 1, node, chargers: 3, initial: false};
      stations.push(target);
    } else if (action < 4) {
      target = [...stations].sort((a, b) => score(b.node, action === 3) - score(a.node, action === 3))[0];
      target.chargers += 1;
    } else {
      const donor = stations.filter(s => s.chargers > 1).sort((a, b) => score(a.node, true) - score(b.node, true))[0];
      target = [...stations].filter(s => s !== donor).sort((a, b) => score(b.node, true) - score(a.node, true))[0];
      if (!donor || !target) break;
      donor.chargers -= 1;
      target.chargers += 1;
    }
    spent += cost;
    snapshot(action, target.id, `${ACTIONS[action][0]} → trạm M${String(target.id).padStart(2, '0')}, nút ${target.node.id}. ${ACTIONS[action][1]}`);
  }
  return {snapshots, endReason: snapshots.length - 1 === MAX_STEPS ? 'Hoàn tất 48 bước minh họa.' : 'Dừng: ngân sách giả lập không đủ cho hành động kế tiếp.'};
}
