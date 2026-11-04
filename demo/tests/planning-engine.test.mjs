import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {createContext, generatePlanningRun, evaluatePlan, stationCSV, influenceRadiusKM, stationCostBreakdown, METHODS} from '../planning-engine.mjs';
const data=JSON.parse(readFileSync(new URL('../assets/planning-data.json',import.meta.url),'utf8'));
const ctx=createContext(data);
const runs=Object.fromEntries(Object.keys(METHODS).map(method=>[method,generatePlanningRun(ctx,method)]));
test('radius uses kW consistently, grows with current power, and has the declared sigmoid limit',()=>{
  assert.ok(Math.abs(influenceRadiusKM(60,ctx.settings)*1000-514.9955016194)<1e-6);
  assert.ok(Math.abs(influenceRadiusKM(800,ctx.settings)*1000-689.9744811276)<1e-6);
  assert.ok(Math.abs(influenceRadiusKM(1000,ctx.settings)*1000-731.05857863)<1e-6);
  assert.equal(influenceRadiusKM(0,ctx.settings),ctx.settings.RADIUS_MAX/2);
  assert.ok(influenceRadiusKM(2000,ctx.settings)>influenceRadiusKM(1000,ctx.settings));
  assert.ok(influenceRadiusKM(100000,ctx.settings)<=ctx.settings.RADIUS_MAX);
  const first=runs.dqn.snapshots[0].stations[0];
  const changed=structuredClone(ctx.initial);changed[0].counts[0]++;
  const next=evaluatePlan(ctx,changed).stations[0];
  assert.ok(next.radiusKM>first.radiusKM);
  assert.equal(next.powerKW,first.powerKW+3);
  assert.ok(Math.abs(next.radiusKM-influenceRadiusKM(next.powerKW,ctx.settings))<1e-12);
});
test('station costs are itemized in million VND and the budget is 900 billion VND',()=>{
  assert.equal(ctx.settings.COST_UNIT,'million_vnd');assert.equal(ctx.settings.BUDGET,900000);
  const first=ctx.initial[0],costs=stationCostBreakdown(ctx,first.nodeId,first.counts);
  assert.equal(costs.landAreaM2,30);
  assert.ok(Math.abs(costs.landCost-2643.93999)<1e-6);
  assert.equal(costs.equipmentCost,286);
  assert.ok(Math.abs(costs.total-2929.93999)<1e-6);
  assert.equal(costs.total,costs.landCost+costs.equipmentCost);
});
test('shared actual five-station baseline and normalized metrics',()=>{
  assert.equal(data.nodes.length,639);assert.equal(data.initialStations.length,5);
  assert.deepEqual(ctx.settings.CHARGING_POWER,[3,7,11,20,22,30,60,80,120,150,180,250]);
  for(const run of Object.values(runs)){
    const first=run.snapshots[0];assert.equal(first.stations.length,5);
    assert.deepEqual(first.stations.map(s=>s.nodeId),data.initialStations.map(s=>s.nodeId));
    for(const key of ['score','benefit','cost','fairness'])assert.ok(Math.abs(first.metrics[key]-100)<1e-6,key);
    assert.equal(first.metrics.spent,0);
  }
});
test('every snapshot is internally consistent and budget-feasible; actions conserve the correct quantities',()=>{
  for(const run of Object.values(runs)){
    assert.ok(run.snapshots.length>5);
    for(const [i,frame] of run.snapshots.entries()){
      for(const [key,value] of Object.entries(frame.metrics))assert.ok(Number.isFinite(value),key);
      assert.ok(frame.metrics.budget<=100+1e-8);
      assert.equal(frame.metrics.stations,frame.stations.length);
      assert.equal(frame.metrics.chargers,frame.stations.reduce((s,n)=>s+n.counts.reduce((x,y)=>x+y,0),0));
      assert.equal(frame.metrics.overloaded,frame.buses.filter(b=>b.evMW>b.availableMW+1e-9).length);
      assert.equal(new Set(frame.stations.map(s=>s.nodeId)).size,frame.stations.length);
      for(const station of frame.stations){
        assert.equal(station.counts.length,12);assert.ok(station.counts.every(n=>Number.isInteger(n)&&n>=0));
        assert.ok(station.chargers>0&&station.chargers<=100);assert.ok(Number.isFinite(station.waitMinutes));
        assert.equal(station.powerKW,station.counts.reduce((a,n,j)=>a+n*ctx.settings.CHARGING_POWER[j],0));
        assert.ok(frame.buses.some(b=>b.id===station.busId));
        assert.equal(station.radiusKM,influenceRadiusKM(station.powerKW,ctx.settings));
        assert.equal(station.cost,station.landCost+station.equipmentCost);
      }
      if(i===0)continue;
      const prev=run.snapshots[i-1];
      if(frame.action<2)assert.equal(frame.metrics.stations,prev.metrics.stations+1);
      else assert.equal(frame.metrics.stations,prev.metrics.stations);
      if(frame.action===2||frame.action===3)assert.equal(frame.metrics.chargers,prev.metrics.chargers+1);
      if(frame.action===4){assert.equal(frame.metrics.chargers,prev.metrics.chargers);assert.equal(frame.metrics.powerKW,prev.metrics.powerKW);assert.ok(frame.donorId);}
    }
    assert.equal(run.snapshots[run.bestIndex].metrics.score,Math.max(...run.snapshots.map(s=>s.metrics.score)));
    assert.deepEqual(run.snapshots[0].stations.map(s=>s.counts),data.initialStations.map(s=>s.counts));
  }
});
test('method scenarios differ and repeated generation is deterministic',()=>{
  assert.notDeepEqual(runs.dqn.snapshots.at(-1).stations.map(s=>s.nodeId),runs.ga.snapshots.at(-1).stations.map(s=>s.nodeId));
  assert.deepEqual(generatePlanningRun(ctx,'dqn'),runs.dqn);
});
test('CSV exports exactly the selected snapshot and all 12 power columns, with illustrative provenance',()=>{
  const frame=runs.dqn.snapshots[12],csv=stationCSV(frame,ctx.settings.CHARGING_POWER);
  assert.ok(csv.startsWith('\uFEFFmethod,step,data_mode'));
  const lines=csv.slice(1).split('\r\n');assert.equal(lines.length,frame.stations.length+1);
  const headings=lines[0].split(',');
  assert.ok(!headings.includes('station_cost_units'));
  assert.ok(headings.includes('station_cost_million_vnd'));
  for(const power of ctx.settings.CHARGING_POWER)assert.ok(headings.includes('chargers_'+power+'_kw'));
  const first=lines[1].split(',');assert.equal(first[1],'12');assert.equal(first[2],'illustrative_no_model');
  assert.equal(Number(first[headings.indexOf('power_kw')]),frame.stations[0].powerKW);
  assert.equal(first[headings.indexOf('cost_unit')],'million_vnd');
  assert.equal(Number(first[headings.indexOf('station_cost_million_vnd')]),frame.stations[0].cost);
  assert.equal(Number(first[headings.indexOf('land_cost_million_vnd')])+Number(first[headings.indexOf('equipment_cost_million_vnd')]),frame.stations[0].cost);
  assert.equal(Number(first[headings.indexOf('budget_total_million_vnd')]),900000);
});
test('bus overload follows actual power and available capacity, with no simulated hidden clipping',()=>{
  const altered=structuredClone(data);altered.buses=altered.buses.map(b=>({...b,availableMW:.00001}));
  const low=createContext(altered),frame=evaluatePlan(low,low.initial);
  assert.ok(frame.metrics.overloaded>0);
  for(const bus of frame.buses.filter(b=>b.overloaded))assert.ok(bus.utilization>1);
});
