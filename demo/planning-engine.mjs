// Offline illustrative trajectories. These rules are not trained DQN/GA policies.
export const METHODS = {
  dqn: {name: 'DQN đề xuất', short: 'DQN', sequence: [0, 1, 0, 2, 3, 4]},
  ga: {name: 'Giải thuật di truyền (GA)', short: 'GA', sequence: [0, 1, 0, 2, 1, 3, 0, 4]},
  benefit: {name: 'Greedy-Benefit', short: 'Greedy-Benefit', sequence: [0]},
  demand: {name: 'Greedy-Demand', short: 'Greedy-Demand', sequence: [1]},
};
export const ACTION_NAMES = ['Xây mới theo lợi ích', 'Xây mới theo nhu cầu', 'Bổ sung theo lợi ích', 'Bổ sung theo nhu cầu', 'Di dời bộ sạc'];
export const MAX_STEPS = 72;
const sum = values => values.reduce((a, b) => a + b, 0);
const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
const ratio = (value, base) => value / Math.max(base, 1e-9);
function hash(value) {let h = 2166136261; for (const c of String(value)) h = Math.imul(h ^ c.charCodeAt(0), 16777619); return (h >>> 0) / 4294967296;}
export function haversine(a, b) {
  const r = Math.PI / 180, dy = (b.lat - a.lat) * r, dx = (b.lon - a.lon) * r;
  return 6371 * 2 * Math.asin(Math.min(1, Math.sqrt(Math.sin(dy / 2) ** 2 + Math.cos(a.lat * r) * Math.cos(b.lat * r) * Math.sin(dx / 2) ** 2)));
}
export function createContext(data) {
  if (data.initialStations.length !== 5 || data.settings.CHARGING_POWER.length !== 12) throw new Error('Dữ liệu trạm đầu vào không hợp lệ.');
  const nodes = data.nodes.map((node, i) => ({...node, i}));
  const nodeMap = new Map(nodes.map(n => [n.id, n]));
  const busMap = new Map(data.buses.map(b => [b.id, b]));
  const distances = nodes.map((a, i) => Float64Array.from(nodes.map((b, j) => i === j ? 0 : haversine(a, b))));
  const ctx = {data, nodes, nodeMap, busMap, distances, settings: data.settings};
  ctx.initial = data.initialStations.map((s, i) => ({id: `S${String(i + 1).padStart(2, '0')}`, nodeId: s.nodeId, counts: [...s.counts], existing: true}));
  ctx.base = evaluateRaw(ctx, ctx.initial);
  ctx.baseScore = rawScore(ctx, ctx.base);
  return ctx;
}
export function influenceRadiusKM(powerKW, settings) {
  const scaleKW = 1000 * settings.capacity_unit;
  if (!Number.isFinite(powerKW) || powerKW < 0 || !(scaleKW > 0)) throw new Error('Công suất hoặc thang bán kính không hợp lệ.');
  return settings.RADIUS_MAX / (1 + Math.exp(-powerKW / scaleKW));
}
export function stationCostBreakdown(ctx, nodeId, counts) {
  const landAreaM2 = sum(counts) * ctx.settings.evs_parking_area;
  const landCost = landAreaM2 * ctx.nodeMap.get(nodeId).landPrice;
  const equipmentCost = sum(counts.map((n, i) => n * ctx.settings.INSTALL_FEE[i]));
  return {landAreaM2, landCost, equipmentCost, total: landCost + equipmentCost};
}
export function stationCost(ctx, nodeId, counts) {
  return stationCostBreakdown(ctx, nodeId, counts).total;
}
function queueMinutes(arrival, service) {
  if (arrival < 1e-9) return 0;
  const rho = arrival / Math.max(service, 1e-8), N = clamp(Math.ceil(arrival), 1, 100);
  let full, length;
  if (Math.abs(rho - 1) < 1e-7) {full = 1 / (N + 1); length = N / 2;}
  else if (rho > 1) {const u = 1 / rho; full = (1 - u) / (1 - u ** (N + 1)); length = (N - (N + 1) * u + u ** (N + 1)) / ((1 - u) * (1 - u ** (N + 1)));}
  else {full = rho ** N * (1 - rho) / (1 - rho ** (N + 1)); length = rho / (1 - rho) - (N + 1) * rho ** (N + 1) / (1 - rho ** (N + 1));}
  return Math.max(0, length / Math.max(arrival * (1 - full), 1e-8) - 1 / Math.max(service, 1e-8)) * 60;
}
function evaluateRaw(ctx, plan) {
  const stations = plan.map(s => {
    const node = ctx.nodeMap.get(s.nodeId), bus = ctx.busMap.get(node.busId);
    const powerKW = sum(s.counts.map((n, i) => n * ctx.settings.CHARGING_POWER[i]));
    const costs = stationCostBreakdown(ctx, s.nodeId, s.counts);
    return {...s, counts: [...s.counts], node, busId: bus.id, powerKW, chargers: sum(s.counts),
      cost: costs.total, landAreaM2: costs.landAreaM2, landCost: costs.landCost, equipmentCost: costs.equipmentCost,
      radiusKM: influenceRadiusKM(powerKW, ctx.settings),
      connectionKM: haversine(node, bus), arrival: 0, waitMinutes: 0, covered: 0};
  });
  const coverage = [], remaining = [];
  let travel = 0, benefitNodes = 0;
  for (const node of ctx.nodes) {
    let nearest = null, nearestDistance = Infinity, count = 0, absorption = 0;
    for (const station of stations) {
      const distance = ctx.distances[node.i][station.node.i];
      if (distance < nearestDistance) {nearestDistance = distance; nearest = station;}
      if (distance < station.radiusKM) {count++; station.covered++; absorption += station.powerKW / 1000 * Math.exp(-ctx.settings.DEMAND_BETA * distance);}
    }
    coverage.push(count);
    remaining.push(node.demand * Math.exp(-ctx.settings.DEMAND_ETA * absorption));
    for (let k = 1; k <= count; k++) benefitNodes += 1 / k;
    const arrival = node.demand * .08; // Explicit illustrative arrival-rate conversion.
    nearest.arrival += arrival;
    travel += arrival * nearestDistance / 40;
  }
  let waiting = 0, charging = 0;
  for (const station of stations) {
    const service = station.powerKW / ctx.settings.BATTERY;
    station.waitMinutes = queueMinutes(station.arrival, service);
    waiting += station.arrival * station.waitMinutes / 60;
    charging += station.arrival / Math.max(service, 1e-8);
  }
  const avg = sum(coverage) / coverage.length;
  const fairness = 1 / (1 + Math.sqrt(sum(coverage.map(n => (n - avg) ** 2)) / coverage.length));
  const benefit = (benefitNodes / ctx.nodes.length + sum(stations.map(s => s.covered / ctx.nodes.length)) / stations.length) / 2;
  const buses = ctx.data.buses.map(b => {
    const evMW = sum(stations.filter(s => s.busId === b.id).map(s => s.powerKW)) / 1000;
    return {...b, evMW, utilization: ratio(evMW, b.availableMW), overloaded: evMW > b.availableMW + 1e-9};
  });
  const gridPenalty = sum(stations.map(s => Math.min(1, s.connectionKM / 2))) / stations.length + sum(buses.map(b => Math.min(10, Math.max(0, b.evMW - b.availableMW) / Math.max(.001, b.availableMW)))) / buses.length;
  return {stations, buses, coverage, remaining, travel, waiting, charging, fairness, benefit, gridPenalty};
}
function rawScore(ctx, raw) {
  const b = ctx.base || raw, alpha = ctx.settings.alpha;
  const cost = (alpha * ratio(raw.travel, b.travel) + (1 - alpha) * (ratio(raw.waiting, b.waiting) + ratio(raw.charging, b.charging))) / 3;
  return (ratio(raw.benefit, b.benefit) - cost + raw.fairness) / 3 - raw.gridPenalty;
}
export function evaluatePlan(ctx, plan, spent = 0) {
  const raw = evaluateRaw(ctx, plan), alpha = ctx.settings.alpha, b = ctx.base;
  const normalizedCost = (alpha * ratio(raw.travel, b.travel) + (1 - alpha) * (ratio(raw.waiting, b.waiting) + ratio(raw.charging, b.charging))) / 3;
  const baselineCost = (alpha + (1 - alpha) * 2) / 3;
  return {...raw, metrics: {
    score: 100 + (rawScore(ctx, raw) - ctx.baseScore) / Math.max(Math.abs(ctx.baseScore), 1e-9) * 100,
    benefit: ratio(raw.benefit, b.benefit) * 100,
    cost: normalizedCost / baselineCost * 100,
    fairness: ratio(raw.fairness, b.fairness) * 100,
    stations: raw.stations.length, chargers: sum(raw.stations.map(s => s.chargers)),
    budget: spent / ctx.settings.BUDGET * 100, budgetTotal: ctx.settings.BUDGET, spent, overloaded: raw.buses.filter(b => b.overloaded).length,
    powerKW: sum(raw.stations.map(s => s.powerKW)),
  }};
}
export function generatePlanningRun(ctx, method) {
  if (!METHODS[method]) throw new Error('Phương pháp không được hỗ trợ.');
  let plan = ctx.initial.map(s => ({...s, counts: [...s.counts]})), relocationPaid = 0;
  const initialFee = sum(plan.map(s => stationCost(ctx, s.nodeId, s.counts)));
  const snapshots = [];
  const capture = (step, action, targetId, message, donorId = null) => {
    const spent = Math.max(0, sum(plan.map(s => stationCost(ctx, s.nodeId, s.counts))) - initialFee + relocationPaid);
    const frame = evaluatePlan(ctx, plan, spent);
    snapshots.push({...frame, step, action, targetId, donorId, message, method});
  };
  capture(0, null, null, 'Khởi tạo từ 5 trạm hiện hữu trong kế hoạch đầu vào.');
  let stopReason = `Hoàn tất ${MAX_STEPS} bước giả lập.`;
  for (let step = 1; step <= MAX_STEPS; step++) {
    const previous = snapshots.at(-1), action = METHODS[method].sequence[(step - 1) % METHODS[method].sequence.length];
    const buildLevel = [8, 9, 10][Math.floor(hash(`${method}-${step}`) * 3)];
    const buildCount = method === 'demand' ? 8 : method === 'benefit' ? 6 : 7;
    const occupied = new Set(plan.map(s => s.nodeId));
    const candidates = action < 2 ? ctx.nodes.filter(n => !occupied.has(n.id)) : plan.filter(s => sum(s.counts) < ctx.settings.K).map(s => ctx.nodeMap.get(s.nodeId));
    const busLoads = new Map(previous.buses.map(b => [b.id, b]));
    const ranked = candidates.map(node => {
      const currentPower = previous.stations.find(s => s.nodeId === node.id)?.powerKW || 0;
      const candidatePower = action < 2 ? ctx.settings.CHARGING_POWER[buildLevel] * buildCount : currentPower + (action < 4 ? ctx.settings.CHARGING_POWER[step % 12] : 0);
      const candidateRadius = influenceRadiusKM(candidatePower, ctx.settings);
      const neighbors = ctx.nodes.filter(n => ctx.distances[node.i][n.i] < candidateRadius);
      const gain = sum(neighbors.map(n => n.demand / (1 + previous.coverage[n.i]))) / Math.max(1, neighbors.length);
      const need = previous.remaining[node.i];
      let value = action === 1 || action === 3 ? need : gain;
      if (method === 'dqn') value -= Math.max(0, busLoads.get(node.busId).utilization - .5) * .8;
      if (method === 'ga') value = .6 * gain + .4 * need + hash(`${step}-${node.id}`) * .12;
      return {node, value};
    }).sort((a, b) => b.value - a.value || a.node.id.localeCompare(b.node.id));
    if (!ranked.length) {stopReason = 'Không còn vị trí hoặc trạm đủ điều kiện.'; break;}
    const next = plan.map(s => ({...s, counts: [...s.counts]}));
    let target, donor = null, newRelocationPaid = relocationPaid, detail;
    if (action < 2) {
      const node = ranked[0].node;
      const counts = Array(12).fill(0);
      counts[buildLevel] = buildCount;
      target = {id: `S${String(next.length + 1).padStart(2, '0')}`, nodeId: node.id, counts, existing: false};
      next.push(target); detail = `Xây mới ${target.id} tại nút ${node.id}`;
    } else if (action < 4) {
      target = next.find(s => s.nodeId === ranked[0].node.id);
      target.counts[step % 12]++;
      detail = `Bổ sung 1 bộ sạc ${ctx.settings.CHARGING_POWER[step % 12]} kW tại ${target.id}`;
    } else {
      const donorCandidates = [...previous.stations].filter(s => s.chargers > 1).sort((a, b) => a.arrival / a.powerKW - b.arrival / b.powerKW);
      const source = donorCandidates[0];
      const destination = [...previous.stations].filter(s => s.id !== source?.id && s.chargers < ctx.settings.K).sort((a, b) => b.waitMinutes - a.waitMinutes)[0];
      if (!source || !destination) {stopReason = 'Không có cặp trạm phù hợp để di dời.'; break;}
      donor = next.find(s => s.id === source.id); target = next.find(s => s.id === destination.id);
      const level = donor.counts.findIndex(n => n > 0);
      donor.counts[level]--; target.counts[level]++;
      newRelocationPaid += ctx.settings.INSTALL_FEE[level] * ctx.settings.RELOCATION_FACTOR;
      detail = `Di dời 1 bộ sạc ${ctx.settings.CHARGING_POWER[level]} kW từ ${donor.id} sang ${target.id}`;
    }
    const spent = sum(next.map(s => stationCost(ctx, s.nodeId, s.counts))) - initialFee + newRelocationPaid;
    if (spent > ctx.settings.BUDGET + 1e-8) {stopReason = 'Dừng vì chi phí hành động tiếp theo vượt ngân sách.'; break;}
    plan = next; relocationPaid = newRelocationPaid;
    capture(step, action, target.id, `${ACTION_NAMES[action]}: ${detail}.`, donor?.id);
  }
  const bestIndex = snapshots.reduce((best, frame, i) => frame.metrics.score > snapshots[best].metrics.score ? i : best, 0);
  return {method, snapshots, bestIndex, stopReason};
}
export function stationCSV(snapshot, powers) {
  const escape = v => {
    let s = String(v ?? '');
    if (/^[=+\-@]/.test(s)) s = `'${s}`;
    return /[",\r\n]/.test(s) ? `"${s.replaceAll('"', '""')}"` : s;
  };
  const header = ['method', 'step', 'data_mode', 'station_id', 'node_id', 'existing', 'latitude', 'longitude', ...powers.map(p => `chargers_${p}_kw`), 'chargers_total', 'power_kw', 'radius_km', 'cost_unit', 'station_cost_million_vnd', 'land_cost_million_vnd', 'equipment_cost_million_vnd', 'land_area_m2', 'budget_total_million_vnd', 'budget_used_million_vnd', 'wait_minutes_estimated', 'bus_id', 'bus_ev_load_mw', 'bus_available_mw', 'bus_overloaded'];
  const rows = snapshot.stations.map(s => {
    const bus = snapshot.buses.find(b => b.id === s.busId);
    return [snapshot.method, snapshot.step, 'illustrative_no_model', s.id, s.nodeId, s.existing, s.node.lat, s.node.lon,
      ...s.counts, s.chargers, s.powerKW, s.radiusKM, 'million_vnd', s.cost, s.landCost, s.equipmentCost, s.landAreaM2, snapshot.metrics.budgetTotal, snapshot.metrics.spent, s.waitMinutes, s.busId, bus.evMW, bus.availableMW, bus.overloaded];
  });
  return '\uFEFF' + [header, ...rows].map(row => row.map(escape).join(',')).join('\r\n');
}
