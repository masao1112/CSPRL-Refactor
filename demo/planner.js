import {METHODS, createContext, generatePlanningRun, stationCSV} from './planning-engine.mjs';
import {createMapGeometry, renderMap} from './map-renderer.mjs';
const $ = id => document.getElementById('planner-' + id);
const fmt = (v, digits = 1) => Number(v).toLocaleString('vi-VN', {maximumFractionDigits: digits, minimumFractionDigits: digits});
const money = value => fmt(value, 2) + ' triệu đồng';
const billion = value => fmt(value / 1000, 3) + ' tỷ đồng';
const esc = value => String(value).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const state = {ready:false, ctx:null, geometry:null, runs:{}, method:'dqn', index:0, playing:false, timer:null, selection:{kind:'station',id:'S01',mapKey:'main'}};
const controls = ['method','play','step','reset','best','export','timeline','station-select','compare-a','compare-b','export-a','export-b'];
function mainRun() {return state.runs[state.method];}
function snapshotFor(key) {
  if (key === 'main') return mainRun().snapshots[state.index];
  const run = state.runs[$(key === 'a' ? 'compare-a' : 'compare-b').value];
  return run.snapshots[run.bestIndex];
}
function layers() {return {demand:$('layer-demand').checked, service:$('layer-service').checked, grid:$('layer-grid').checked, nodes:state.ctx.nodes};}
function draw(key) {
  if (!state.ready) return;
  const id = key === 'main' ? 'main-map' : key === 'a' ? 'compare-map-a' : 'compare-map-b';
  renderMap($(id), state.geometry, snapshotFor(key), layers(), state.selection, selectEntity);
}
function selectEntity(selection) {state.selection = selection; ['main','a','b'].forEach(draw); renderInspector();}
function pause() {
  clearTimeout(state.timer);state.timer=null;state.playing=false;
  $('play').textContent = state.ready && state.index === mainRun().snapshots.length-1 ? '↺ Chạy lại' : '▶ Chạy';
}
function schedule() {
  if (!state.playing) return;
  state.timer = setTimeout(() => {
    state.index++;
    renderMain();
    if (state.index >= mainRun().snapshots.length-1) pause(); else schedule();
  }, Number($('speed').value));
}
function renderMain() {
  const run=mainRun(),frame=snapshotFor('main'),m=frame.metrics;
  for (const key of ['score','benefit','cost','fairness']) $('metric-'+key).textContent=fmt(m[key]);
  $('metric-stations').textContent=m.stations;$('metric-chargers').textContent=m.chargers;
  $('metric-budget').textContent=fmt(m.budget)+'%';$('metric-overloaded').textContent=m.overloaded;
  $('metric-overloaded').classList.toggle('status-alert',m.overloaded>0);
  $('station-count-note').textContent='5 hiện hữu · '+(m.stations-5)+' bổ sung';
  $('total-power').textContent=fmt(m.powerKW/1000,2)+' MW lắp đặt';
  $('budget-amount').textContent=fmt(m.spent,2)+' / '+fmt(state.ctx.settings.BUDGET,0)+' triệu đồng';
  $('budget-summary').textContent='Ngân sách mở rộng B = '+fmt(state.ctx.settings.BUDGET,0)+' triệu đồng = '+fmt(state.ctx.settings.BUDGET/1000,0)+' tỷ đồng. Không cộng chi phí hạ tầng hiện hữu vào phần ngân sách đã dùng.';
  $('bus-count-note').textContent='Trên '+frame.buses.length+' thanh cái trong repo';
  $('timeline').max=run.snapshots.length-1;$('timeline').value=state.index;
  $('timeline').setAttribute('aria-valuetext','Bước '+state.index+' trên '+(run.snapshots.length-1));
  $('step-label').textContent=String(state.index).padStart(2,'0')+' / '+String(run.snapshots.length-1).padStart(2,'0');
  $('current-action').textContent=frame.message;
  $('spent-label').textContent='Đã dùng '+money(m.spent)+' ('+fmt(m.budget)+'%)';
  $('budget-progress').value=Math.max(0,Math.min(100,m.budget));
  $('timeline-state').textContent=state.index===0?'Trạng thái ban đầu':state.index===run.snapshots.length-1?run.stopReason:state.index===run.bestIndex?'Đang xem kế hoạch tốt nhất của chuỗi':'Đang xem bước '+state.index;
  $('best-step-note').textContent='Tốt nhất: bước '+run.bestIndex+' · R = '+fmt(run.snapshots[run.bestIndex].metrics.score);
  $('step').disabled=state.index===run.snapshots.length-1;
  const events=run.snapshots.slice(Math.max(0,state.index-4),state.index+1);
  $('event-log').replaceChildren(...events.map(event=>{
    const li=document.createElement('li'),time=document.createElement('time'),description=document.createElement('span');
    time.textContent='Bước '+String(event.step).padStart(2,'0');description.textContent=event.message;
    li.append(time,description);return li;
  }));
  if(state.selection.mapKey==='main' && state.selection.kind==='station' && !frame.stations.some(s=>s.id===state.selection.id))state.selection.id='S01';
  draw('main');
  if (state.selection.mapKey==='main') renderInspector();
  if (!state.playing) pause();
}
const metricRows=[
  ['score','R(p) ↑',1],['benefit','Lợi ích ↑',1],['cost','Chi phí bất tiện ↓',1],['fairness','Công bằng ↑',1],
  ['stations','Số trạm',0],['chargers','Số bộ sạc',0],['budget','Ngân sách sử dụng (%)',1],['overloaded','Thanh cái quá tải ↓',0],
];
function renderComparison() {
  ['a','b'].forEach(key=>{
    const frame=snapshotFor(key);
    $('compare-'+key+'-caption').textContent='Kế hoạch tốt nhất · bước '+frame.step+' · '+frame.metrics.stations+' trạm · '+fmt(frame.metrics.budget)+'% ngân sách';
    draw(key);
  });
  const a=snapshotFor('a'),b=snapshotFor('b'),headers=$('comparison-table').querySelectorAll('thead th');
  headers[1].textContent='A · '+METHODS[a.method].short;headers[2].textContent='B · '+METHODS[b.method].short;
  $('comparison-body').replaceChildren(...metricRows.map(([key,label,digits])=>{
    const row=document.createElement('tr'),th=document.createElement('th');th.scope='row';th.textContent=label;row.append(th);
    for(const value of [a.metrics[key],b.metrics[key],b.metrics[key]-a.metrics[key]]) {
      const cell=document.createElement('td');cell.textContent=fmt(value,digits);row.append(cell);
    }
    return row;
  }));
  if(state.selection.mapKey!=='main') renderInspector();
}
function renderInspector() {
  const selection=state.selection,frame=snapshotFor(selection.mapKey);
  const context=selection.mapKey==='main'?'Bản đồ chính':selection.mapKey==='a'?'Phương án A':'Phương án B';
  $('inspector-source').textContent=context+' · bước '+frame.step;
  $('station-select').replaceChildren(...frame.stations.map(s=>{
    const option=document.createElement('option');option.value=s.id;option.textContent=s.id+' · '+(s.existing?'hiện hữu':'trạm mới')+' · '+fmt(s.powerKW,0)+' kW';return option;
  }));
  const detailRow=(label,value)=>'<div><dt>'+esc(label)+'</dt><dd>'+esc(value)+'</dd></div>';
  let html,brief;
  if(selection.kind==='bus') {
    const bus=frame.buses.find(b=>b.id===selection.id);
    if(!bus){state.selection={kind:'station',id:'S01',mapKey:'main'};renderInspector();return;}
    $('inspector-title').textContent='Chi tiết thanh cái';
    html='<div class="station-heading"><h4>Bus '+esc(bus.id)+'</h4><span>22 kV</span></div><p class="detail-footnote">'+esc(bus.name)+'</p><dl class="detail-list">'+
      detailRow('Tọa độ',fmt(bus.lat,6)+', '+fmt(bus.lon,6))+
      detailRow('Tải sạc',fmt(bus.evMW,3)+' MW')+
      detailRow('Dư địa sau phụ tải nền',fmt(bus.availableMW,3)+' MW')+
      detailRow('Tải sạc / dư địa',fmt(bus.utilization*100)+'%')+
      detailRow('Trạng thái',bus.overloaded?'QUÁ TẢI':'Trong dư địa')+
      detailRow('Phụ tải nền CSV',fmt(bus.baseLoadMW,3)+' MW')+
      detailRow('Định mức giới hạn',fmt(bus.ratedMVA,2)+' MVA')+
      detailRow('Trạm đấu nối',frame.stations.filter(s=>s.busId===bus.id).length)+'</dl><p class="detail-footnote">Màu thanh cái dùng tải sạc / dư địa sau phụ tải nền. Dữ liệu lưới mô phỏng trong repo; không chạy power flow.'+(bus.fallback?' Định mức dùng fallback 10 MVA.':'')+'</p>';
    brief=context+' · Bus '+bus.id+' · '+bus.name+' · tải sạc '+fmt(bus.evMW,3)+' / '+fmt(bus.availableMW,3)+' MW dư địa · '+(bus.overloaded?'QUÁ TẢI':'trong dư địa')+'.';
  } else {
    let station=frame.stations.find(s=>s.id===selection.id);
    if(!station){station=frame.stations[0];state.selection={...selection,id:station.id};}
    $('station-select').value=station.id;$('inspector-title').textContent='Chi tiết trạm';
    const bus=frame.buses.find(b=>b.id===station.busId);
    html='<div class="station-heading"><h4>'+esc(station.id)+'</h4><span>'+(station.existing?'Trạm hiện hữu':'Trạm bổ sung')+'</span></div><dl class="detail-list">'+
      detailRow('Vĩ độ / kinh độ',fmt(station.node.lat,6)+' / '+fmt(station.node.lon,6))+
      detailRow('Nút đường bộ',station.nodeId)+
      detailRow('Công suất lắp đặt',fmt(station.powerKW,0)+' kW')+
      detailRow('Tổng số bộ sạc',station.chargers)+
      detailRow('Bán kính theo công suất',fmt(station.radiusKM*1000,1)+' m')+
      detailRow('Tổng chi phí trạm',money(station.cost))+
      detailRow('Tương đương',billion(station.cost))+
      detailRow('Chi phí mặt bằng',money(station.landCost))+
      detailRow('Chi phí thiết bị',money(station.equipmentCost))+
      detailRow('Diện tích mặt bằng',fmt(station.landAreaM2,0)+' m²')+
      detailRow('Đơn giá đất',money(station.node.landPrice)+'/m²')+
      detailRow('Chờ ước tính',fmt(station.waitMinutes)+' phút')+
      detailRow('Nhu cầu nút còn lại',fmt(frame.remaining[station.node.i],3)+' / '+fmt(station.node.demand,3))+
      detailRow('Thanh cái đấu nối','Bus '+bus.id)+
      detailRow('Khoảng cách đấu nối',fmt(station.connectionKM*1000,0)+' m')+
      detailRow('Tải sạc / dư địa bus',fmt(bus.utilization*100)+'%'+(bus.overloaded?' · QUÁ TẢI':''))+
      '</dl><h4 class="charger-title">Cấu hình bộ sạc · 12 mức công suất</h4><div class="charger-grid">'+
      state.ctx.settings.CHARGING_POWER.map((power,i)=>'<div><span>'+power+' kW</span><strong>'+station.counts[i]+'</strong></div>').join('')+
      '</div><p class="detail-footnote">Bán kính r = Rmax / [1 + exp(−C/C₀)]; C = '+fmt(station.powerKW,0)+' kW, C₀ = '+fmt(1000*state.ctx.settings.capacity_unit,0)+' kW, Rmax = '+fmt(state.ctx.settings.RADIUS_MAX*1000,0)+' m. Tính lại từ cấu hình ở bước đang xem.</p><p class="detail-footnote">Chi phí trạm = '+fmt(station.landAreaM2,0)+' m² × '+money(station.node.landPrice)+'/m² + '+money(station.equipmentCost)+' thiết bị. '+esc(bus.name)+'. Thời gian chờ là ước tính của demo.</p>';
    brief=context+' · '+station.id+' · nút '+station.nodeId+' · '+station.chargers+' bộ sạc · '+fmt(station.powerKW,0)+' kW · bán kính '+fmt(station.radiusKM*1000)+' m · '+money(station.cost)+' · Bus '+bus.id+' · chờ ước tính '+fmt(station.waitMinutes)+' phút.';
  }
  $('inspector-content').innerHTML=html;
  if(selection.mapKey!=='main') $('compare-detail').textContent=brief+' Chi tiết đầy đủ đang hiển thị trong bảng “'+$('inspector-title').textContent+'” phía trên.';
}
function exportSnapshot(key) {
  const frame=snapshotFor(key),csv=stationCSV(frame,state.ctx.settings.CHARGING_POWER);
  const name='CSPRL_DongDa_'+METHODS[frame.method].short+'_buoc_'+String(frame.step).padStart(2,'0')+'_gia_lap.csv';
  const url=URL.createObjectURL(new Blob([csv],{type:'text/csv;charset=utf-8;'})),link=document.createElement('a');
  link.href=url;link.download=name;document.body.append(link);link.click();link.remove();
  setTimeout(()=>URL.revokeObjectURL(url),1500);$('export-status').textContent='Đã xuất '+frame.stations.length+' trạm · '+name;
}
$('play').addEventListener('click',()=>{
  if(state.playing){pause();return;}
  if(state.index===mainRun().snapshots.length-1)state.index=0;
  state.playing=true;$('play').textContent='Ⅱ Tạm dừng';renderMain();schedule();
});
$('step').addEventListener('click',()=>{pause();state.index=Math.min(state.index+1,mainRun().snapshots.length-1);renderMain();});
$('reset').addEventListener('click',()=>{pause();state.index=0;state.selection={kind:'station',id:'S01',mapKey:'main'};renderMain();});
$('best').addEventListener('click',()=>{pause();state.index=mainRun().bestIndex;state.selection={kind:'station',id:mainRun().snapshots[state.index].targetId||'S01',mapKey:'main'};renderMain();});
$('timeline').addEventListener('input',()=>{pause();state.index=Number($('timeline').value);state.selection={...state.selection,mapKey:'main'};renderMain();});
$('method').addEventListener('change',()=>{pause();state.method=$('method').value;state.index=0;state.selection={kind:'station',id:'S01',mapKey:'main'};renderMain();});
$('speed').addEventListener('change',()=>{if(state.playing){clearTimeout(state.timer);schedule();}});
for(const id of ['layer-demand','layer-service','layer-grid'])$(id).addEventListener('change',()=>{if(state.ctx)['main','a','b'].forEach(draw);});
for(const id of ['compare-a','compare-b'])$(id).addEventListener('change',()=>{state.selection={kind:'station',id:'S01',mapKey:id==='compare-a'?'a':'b'};renderComparison();});
$('station-select').addEventListener('change',()=>selectEntity({kind:'station',id:$('station-select').value,mapKey:state.selection.mapKey}));
$('export').addEventListener('click',()=>exportSnapshot('main'));
$('export-a').addEventListener('click',()=>exportSnapshot('a'));
$('export-b').addEventListener('click',()=>exportSnapshot('b'));
document.addEventListener('visibilitychange',()=>{if(document.hidden)pause();});
async function load() {
  state.ready=false;
  controls.forEach(id=>$(id).disabled=true);$('load-status').hidden=false;$('load-status').textContent='Đang đọc dữ liệu Đống Đa ngoại tuyến…';
  try {
    const [roads,data]=await Promise.all(['assets/roads.json','assets/planning-data.json'].map(async path=>{
      const response=await fetch(path);if(!response.ok)throw new Error('Không đọc được '+path);return response.json();
    }));
    state.ctx=createContext(data);state.geometry=createMapGeometry(roads,data);
    for(const method of Object.keys(METHODS)){
      $('load-status').textContent='Chuẩn bị kịch bản giả lập: '+METHODS[method].name+'…';
      await new Promise(resolve=>setTimeout(resolve,0));
      state.runs[method]=generatePlanningRun(state.ctx,method);
    }
    state.ready=true;controls.forEach(id=>$(id).disabled=false);$('load-status').hidden=true;
    $('dataset-status').textContent=data.nodes.length+' nút nhu cầu · 5 trạm hiện hữu · '+data.buses.length+' thanh cái · dữ liệu cục bộ';
    renderMain();renderComparison();
  }catch(error){
    state.ctx=null;controls.forEach(id=>$(id).disabled=true);
    $('load-status').textContent='Không tải được dữ liệu. Hãy mở trang bằng HTTP server cục bộ. ';
    const retry=document.createElement('button');retry.textContent='Thử lại';retry.addEventListener('click',load);$('load-status').append(retry);
    $('dataset-status').textContent='Dữ liệu chưa sẵn sàng';
    console.error('Planning data unavailable:',error);
  }
}
const disclosure = document.getElementById('planning-demo');
let initialized = false;
disclosure.addEventListener('toggle', () => {
  if (!disclosure.open) {pause(); return;}
  if (!initialized) {initialized = true; load();}
});
if (disclosure.open) {initialized = true; load();}
