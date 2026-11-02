import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {generateRun,projectGraph} from '../simulation.mjs';

const graph = projectGraph(JSON.parse(readFileSync(new URL('../assets/roads.json',import.meta.url),'utf8')));
test('same scenario produces the same snapshots without changing input geometry',()=>{
  const before = JSON.stringify(graph.nodes);
  const a=generateRun(graph.nodes),b=generateRun(graph.nodes);
  assert.deepEqual(a,b);
  assert.equal(JSON.stringify(graph.nodes),before);
  assert.equal(a.snapshots[0].stations.length,5);
  assert.equal(a.snapshots[0].remaining,100);
  assert.equal(a.snapshots[0].spent,0);
});
test('all scenario budgets stay within limits and preserve charger conservation on relocation',()=>{
  for(const budget of [40,70,100]) for(const strategy of ['balanced','demand']) {
    const {snapshots}=generateRun(graph.nodes,{budget,strategy,demand:1.5});
    assert.ok(snapshots.length>1);
    assert.deepEqual(new Set(snapshots.slice(1).map(s=>s.action)),new Set([0,1,2,3,4]));
    snapshots.forEach((s,i)=>{
      assert.ok(s.spent<=budget);
      assert.ok(Number.isFinite(s.remaining) && s.remaining>=0);
      assert.ok(s.overloaded>=0 && s.overloaded<=18);
      assert.equal(new Set(s.stations.map(s=>s.node.id)).size,s.stations.length);
      assert.ok(s.stations.every(s=>s.chargers>=1));
      if(s.action===4) {
        const previous=snapshots[i-1];
        assert.equal(s.stations.length,previous.stations.length);
        assert.equal(s.stations.reduce((n,s)=>n+s.chargers,0),previous.stations.reduce((n,s)=>n+s.chargers,0));
      }
    });
  }
});
test('budget, demand and grid strategy affect generated outcomes',()=>{
  const full=generateRun(graph.nodes);
  assert.equal(full.snapshots.length,49);
  const small=generateRun(graph.nodes,{budget:40});
  assert.ok(small.snapshots.length<full.snapshots.length);
  assert.notDeepEqual(generateRun(graph.nodes,{strategy:'demand'}),full);
  assert.notDeepEqual(generateRun(graph.nodes,{demand:1.5}),full);
});
