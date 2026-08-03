const $=s=>document.querySelector(s);
const esc=s=>String(s==null?'':s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
// A user-typed name (e.g. an individual's) embedded as a JS-string ARGUMENT in an inline onclick:
// JSON.stringify makes a valid JS string literal (so an apostrophe like O'Brien can't break it),
// and esc neutralises it for the double-quoted HTML attribute. Use jarg(x), never '${esc(x)}', for
// any onclick string argument.
const jarg=s=>esc(JSON.stringify(s==null?'':String(s)));
// Word-initial capitals only: "townsend's chipmunk" -> "Townsend's Chipmunk" (never Townsend'S),
// "band-tailed pigeon" -> "Band-tailed Pigeon" (bird-guide style keeps the hyphenated tail lower).
const cap1=s=>s.replace(/(^|[\s(])(\S)/g,(m,p,c)=>p+c.toUpperCase());
// encodeURI leaves apostrophes alone, and these paths land inside url('...') in style attributes.
const media=p=>'/media/'+encodeURI(p).replace(/'/g,'%27');
// playful pseudo-taxonomic labels (flavour only)
const LATIN={'raccoon':'Procyon lotor','american crow':'Corvus brachyrhynchos','eastern gray squirrel':'Sciurus carolinensis',
  'dark-eyed junco':'Junco hyemalis','domestic cat':'Felis catus','virginia opossum':'Didelphis virginiana',
  'spotted towhee':'Pipilo maculatus','brown rat':'Rattus norvegicus','steller’s jay':'Cyanocitta stelleri',
  'black-capped chickadee':'Poecile atricapillus','house finch':'Haemorhous mexicanus','american robin':'Turdus migratorius',
  'european starling':'Sturnus vulgaris','northern flicker':'Colaptes auratus','varied thrush':'Ixoreus naevius',
  'band-tailed pigeon':'Patagioenas fasciata','eastern cottontail':'Sylvilagus floridanus','house sparrow':'Passer domesticus',
  'townsend’s chipmunk':'Neotamias townsendii'};
const latinOf=n=>LATIN[(n||'').toLowerCase().replace(/'/g,'’')]||'';
const fmtDur=s=>{ s=Math.round(s||0); return s>=60?`${Math.floor(s/60)}:${String(s%60).padStart(2,'0')}`:`${s}s`; };

/* ---------- live-connection indicator ----------
   The recurring live polls (stats, live/now, cameras, header, naming) shouldn't shout on every
   transient blip. Instead they call connFail() on a failed fetch and connOK() on a good one; a
   single subtle "reconnecting…" pill appears in the masthead after a failure and self-clears on the
   next success. Reference-counted across the several pollers so one recovering doesn't hide a pill
   another still needs. Created lazily (dashboard.html isn't ours to edit). */
let __connDown=0;
function __connPill(){
  let el=document.getElementById('connpill');
  if(!el){
    const top=document.querySelector('.mast .top');
    if(!top) return null;
    el=document.createElement('span');
    el.id='connpill'; el.className='conn-pill'; el.hidden=true;
    el.innerHTML='<span class="cp-dot"></span>reconnecting…';
    el.title='The live feed of stats briefly stopped responding. Retrying automatically — nothing to do.';
    top.appendChild(el);
  }
  return el;
}
function connFail(){ __connDown++; const el=__connPill(); if(el) el.hidden=false; }
function connOK(){ if(__connDown===0) return; __connDown=0; const el=document.getElementById('connpill'); if(el) el.hidden=true; }

/* Fire-and-forget POST feedback: dim+disable the just-clicked button while its request is in
   flight, restore it after. Pass a DOM node, or nothing to grab the current inline-handler's
   target. Returns a restore() to call in a finally. No-op if there's no button to mark. */
function busyBtn(btn){
  const el=btn||(typeof event!=='undefined'&&event?(event.currentTarget||event.target):null);
  if(!el||el.tagName!=='BUTTON') return ()=>{};
  const wasDisabled=el.disabled, prevOp=el.style.opacity;
  el.disabled=true; el.style.opacity='.45'; el.style.cursor='progress';
  let done=false;
  return ()=>{ if(done) return; done=true; el.disabled=wasDisabled; el.style.opacity=prevOp; el.style.cursor=''; };
}

/* ---------- clip player (lightbox) ----------
   Three shapes, one box: a single clip; a PLAYLIST of clips (auto-advances on 'ended', filmstrip
   + ‹ › + arrow keys to scrub); or a stitched HIGHLIGHT REEL — one mp4 whose filmstrip cells are
   CHAPTERS (seek points inside the same video) rather than separate files. */
let __clips=[], __ci=0, __reelMode=false, __chapters=null;
function playClips(clips,i,title){
  clips=(clips||[]).filter(c=>c&&c.clip_path);
  if(!clips.length) return;
  // Playlists play in wall-clock order regardless of how the caller ranked them.
  if(clips.length>1) clips=clips.slice().sort((a,b)=>String(a.start||'').localeCompare(String(b.start||'')));
  __clips=clips; __ci=Math.max(0,Math.min(i||0,clips.length-1)); __reelMode=clips.length>1; __chapters=null;
  const box=$('#clipbox'); if(!box) return;
  $('#clip-title').textContent=title||(__reelMode?`Reel · ${clips.length} clips`:'Clip');
  box.hidden=false;
  const strip=$('#clip-strip'); strip.style.display=__reelMode?'flex':'none';
  strip.innerHTML=__reelMode?__clips.map((c,k)=>`
    <div class="fs" data-i="${k}" onclick="reelGo(${k})">
      <div class="ft" style="background-image:url('${c.thumb?media(c.thumb):''}')"><span class="dur">${fmtDur(c.seconds)}</span></div>
      <div class="lab">${esc(fmtClock(c.start)||'')}${c.species?' · '+esc(nameOf(c.species)):''}</div>
    </div>`).join(''):'';
  const v=$('#clip-video');
  v.ontimeupdate=null;
  v.onended=()=>{ if(__reelMode && __ci<__clips.length-1) reelStep(1); };
  document.addEventListener('keydown',clipKeys);
  showClip();
}
/* The stitched highlight reel: one video, chapter filmstrip. `man` is /api/reel's manifest. */
function playHighlightReel(man,title){
  if(!man||!man.clip_path||!(man.segments||[]).length) return;
  __clips=[{clip_path:man.clip_path, seconds:man.seconds}]; __ci=0; __reelMode=false; __chapters=man.segments;
  const box=$('#clipbox'); if(!box) return;
  $('#clip-title').textContent=title||'Highlight Reel';
  box.hidden=false;
  const strip=$('#clip-strip'); strip.style.display='flex';
  strip.innerHTML=__chapters.map((s,k)=>`
    <div class="fs" data-i="${k}" onclick="reelSeek(${k})">
      <div class="ft" style="background-image:url('${s.thumb?media(s.thumb):''}')"><span class="dur">${fmtDur(s.seconds)}</span></div>
      <div class="lab">${esc(fmtClock(s.start)||'')}${s.species?' · '+esc(nameOf(s.species)):''}</div>
    </div>`).join('');
  const v=$('#clip-video');
  // Same story as the Dispatch hero: prefer a real reel frame; a chapter thumb is a small crop
  // and stretching it across the player is what made the video look broken before it played.
  const pos=man.poster_path||__chapters[0].thumb;
  if(pos) v.poster=media(pos); else v.removeAttribute('poster');
  v.src=media(man.clip_path);
  v.onended=null;
  v.ontimeupdate=()=>__reelSyncChapter();
  const pr=v.play(); if(pr&&pr.catch) pr.catch(()=>{});
  document.addEventListener('keydown',clipKeys);
  __reelSyncChapter(true);
}
function __chapterAt(t){ let k=0; (__chapters||[]).forEach((s,i)=>{ if(t>=s.at-0.3) k=i; }); return k; }
function __reelSyncChapter(force){
  if(!__chapters) return;
  const v=$('#clip-video'), k=__chapterAt(v.currentTime||0), s=__chapters[k];
  if(!force && __ci===k) return;
  __ci=k;
  $('#clip-cap').innerHTML=`${s.species?`<span class="nm">${esc(nameOf(s.species))}</span>`:''}
    ${(s.individuals||[]).map(n=>`<span class="mono" style="font-size:12px;color:var(--gilt)">${esc(cap1(n))}</span>`).join(' ')}
    ${s.start?`<span class="mono" style="font-size:12px">${esc(fmtClock(s.start))}</span>`:''}
    <span class="mono" style="font-size:12px;margin-left:auto;color:var(--faint)">moment ${k+1} / ${__chapters.length}</span>`;
  const prev=$('#clip-prev'), next=$('#clip-next');
  prev.style.display=next.style.display='flex';
  prev.disabled=k<=0; next.disabled=k>=__chapters.length-1;
  $('#clip-strip').querySelectorAll('.fs').forEach(el=>{
    const on=+el.dataset.i===k; el.classList.toggle('on',on); if(on) el.scrollIntoView({inline:'center',block:'nearest'}); });
}
function reelSeek(i){
  if(!__chapters||i<0||i>=__chapters.length) return;
  const v=$('#clip-video'); v.currentTime=__chapters[i].at+0.01;
  const pr=v.play(); if(pr&&pr.catch) pr.catch(()=>{});
  __ci=i; __reelSyncChapter(true);
}
function showClip(){
  const c=__clips[__ci]; if(!c) return;
  const v=$('#clip-video');
  // Poster = the clip's best frame, shown while the H.264 transcode streams in on first view
  // (a cold clip can take a few seconds; cached instantly after).
  if(c.thumb) v.poster=media(c.thumb); else v.removeAttribute('poster');
  v.src=media(c.clip_path);
  const pr=v.play(); if(pr&&pr.catch) pr.catch(()=>{});
  $('#clip-cap').innerHTML=`${c.species?`<span class="nm">${esc(nameOf(c.species))}</span>`:''}
    ${c.start?`<span class="mono" style="font-size:12px">${esc(fmtClock(c.start))}</span>`:''}
    ${c.seconds?`<span class="mono" style="font-size:12px">· ${fmtDur(c.seconds)}</span>`:''}
    ${c.dets?`<span class="mono" style="font-size:12px">· ${c.dets} detection${c.dets===1?'':'s'}</span>`:''}
    ${c.conf!=null?`<span class="c mono" style="font-size:12px">· ~${Math.round(c.conf*100)}%</span>`:''}
    ${__reelMode?`<span class="mono" style="font-size:12px;margin-left:auto;color:var(--faint)">${__ci+1} / ${__clips.length}</span>`:''}`;
  const prev=$('#clip-prev'), next=$('#clip-next');
  prev.style.display=next.style.display=__reelMode?'flex':'none';
  prev.disabled=__ci<=0; next.disabled=__ci>=__clips.length-1;
  $('#clip-strip').querySelectorAll('.fs').forEach(el=>{
    const on=+el.dataset.i===__ci; el.classList.toggle('on',on); if(on) el.scrollIntoView({inline:'center',block:'nearest'}); });
}
function reelStep(d){
  if(__chapters){ reelSeek(__ci+d); return; }
  const n=__ci+d; if(n<0||n>=__clips.length) return; __ci=n; showClip();
}
function reelGo(i){ if(i>=0&&i<__clips.length){ __ci=i; showClip(); } }
function closeClipbox(){ const m=$('#clipbox'); if(!m) return; m.hidden=true;
  const v=$('#clip-video'); v.pause(); v.onended=null; v.ontimeupdate=null; v.removeAttribute('src'); v.load();
  __chapters=null;
  document.removeEventListener('keydown',clipKeys); }
function clipKeys(e){ if(e.key==='Escape') closeClipbox(); else if(e.key==='ArrowRight') reelStep(1); else if(e.key==='ArrowLeft') reelStep(-1); }
/* small ▶ overlay markup for a thumbnail that has clip(s) behind it */
const playBadge=(n)=>`<span class="play-badge sm" data-play></span>${n>1?`<span class="clip-count">${n} clips</span>`:''}`;

const VIEWS=['live','dispatch','behavior','indiv','calendar','cat'];
function show(v, fromHash){
  closeSettings();
  VIEWS.concat('explore').forEach(k=>{ const s=$('#view-'+k); if(s) s.classList.toggle('on',v===k); });
  VIEWS.forEach(k=>{ const t=$('#tab-'+k); if(t) t.classList.toggle('on',v===k); });
  // Deep-linkable tabs + a working Back button: the view lives in the URL hash. Programmatic
  // hash writes echo a hashchange we must NOT re-show (it would double-load the view).
  if(!fromHash && VIEWS.includes(v) && location.hash!=='#'+v){ __hashQuiet=v; location.hash=v; }
  syncLiveStreams();
  if(v==='cat') loadCatalogue();
  if(v==='dispatch') loadDispatch();
  if(v==='behavior') loadBehavior();
  if(v==='indiv') loadIndividuals();
  if(v==='calendar') loadCalendar();
  if(v==='live') refreshWhoshere();
}
let __hashQuiet=null;
window.addEventListener('hashchange',()=>{
  const v=location.hash.replace('#','');
  if(__hashQuiet===v){ __hashQuiet=null; return; }
  if(VIEWS.includes(v)) show(v, true);
});
/* The MJPEG live streams are open-ended HTTP responses -- left attached while another tab is
   on screen they stream (and decode) forever for nobody, which on a phone is real battery and
   LAN traffic. Attach each pane's stream only while the Live tab is visible. */
function syncLiveStreams(){
  const on=!!document.querySelector('#view-live.on');
  document.querySelectorAll('.live-pane').forEach(p=>{
    const img=p.querySelector('.frame img'); if(!img) return;
    const want='/stream.mjpg?source='+encodeURIComponent(p.dataset.source||'');
    if(on){ if(!img.getAttribute('src')) img.src=want; }
    else if(img.getAttribute('src')){ img.removeAttribute('src'); }
  });
}

/* ---------- camera controls ---------- */
const CONTROLS=[
  {key:'exposure',label:'Exposure',min:-13,max:0,step:1,auto:'auto_exposure'},
  {key:'gain',label:'Gain',min:0,max:255,step:1},
  {key:'focus',label:'Focus',min:0,max:1023,step:5,auto:'autofocus'},
  {key:'brightness',label:'Brightness',min:0,max:255,step:1},
  {key:'contrast',label:'Contrast',min:0,max:255,step:1},
  {key:'saturation',label:'Saturation',min:0,max:255,step:1},
  {key:'sharpness',label:'Sharpness',min:0,max:255,step:1},
  {key:'wb',label:'White balance',min:2000,max:8000,step:100,auto:'auto_wb'},
];
function buildControls(){
  $('#controls').innerHTML=CONTROLS.map(c=>`
    <div class="ctrl" data-k="${c.key}">
      <div class="row">
        <span class="name">${c.label}</span>
        <span style="display:flex;gap:10px;align-items:center">
          ${c.auto?`<label class="auto"><input type="checkbox" data-auto="${c.auto}"> auto</label>`:''}
          <span class="val" data-val>—</span>
        </span>
      </div>
      <input type="range" min="${c.min}" max="${c.max}" step="${c.step}" data-slider>
    </div>`).join('');
  $('#controls').querySelectorAll('.ctrl').forEach(el=>{
    const key=el.dataset.k, sl=el.querySelector('[data-slider]'), val=el.querySelector('[data-val]');
    sl.addEventListener('input',()=>{ val.textContent=sl.value; touchedAt[key]=Date.now(); sendControl(key,parseFloat(sl.value)); });
    const auto=el.querySelector('[data-auto]');
    if(auto) auto.addEventListener('change',()=>{ sl.disabled=auto.checked; touchedAt[key]=Date.now(); sendAuto(auto.dataset.auto,auto.checked,sl); });
  });
}
let postTimer={}, touchedAt={};   // touchedAt[key] = Date.now() of the user's last interaction with that control
function postCamera(obj){ fetch('/api/camera?source='+encodeURIComponent(LIVE.sel||''),{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(obj)}); }
function sendControl(key,v){
  let p;
  if(key==='exposure')p={exposure:v};
  else if(key==='gain')p={gain:v};
  else if(key==='focus')p={AUTOFOCUS:0,FOCUS:v};
  else if(key==='wb')p={AUTO_WB:0,WB_TEMPERATURE:v};
  else p={[key.toUpperCase()]:v};
  clearTimeout(postTimer[key]); postTimer[key]=setTimeout(()=>postCamera(p),60);
}
function sendAuto(autoKey,on,slider){
  if(autoKey==='auto_exposure')postCamera({exposure:on?null:parseFloat(slider.value)});
  else if(autoKey==='autofocus')postCamera(on?{AUTOFOCUS:1}:{AUTOFOCUS:0,FOCUS:parseFloat(slider.value)});
  else if(autoKey==='auto_wb')postCamera(on?{AUTO_WB:1}:{AUTO_WB:0,WB_TEMPERATURE:parseFloat(slider.value)});
}
/* The live cameras. LIVE.sel is the camera the controls + "who's here" act on (click a pane to
   change it); LIVE.primary is the main feed the masthead period/coords come from. A single-camera
   rig has one pane and behaves exactly as before. */
let LIVE={ cams:[], sel:null, primary:null };
async function loadCameras(){
  let d; try{ d=await fetch('/api/cameras').then(r=>r.json()); connOK(); }catch(e){ connFail(); return; }
  const cams=d.cameras||[];
  if(!cams.length) return;
  LIVE.cams=cams; LIVE.primary=d.primary||cams[0].source;
  if(!LIVE.sel || !cams.find(c=>c.source===LIVE.sel)) LIVE.sel=LIVE.primary;
  const grid=$('#live-grid'); if(!grid) return;
  grid.classList.toggle('single', cams.length<=1);
  grid.innerHTML=cams.map(c=>`
    <figure class="live-pane${c.source===LIVE.sel?' sel':''}" data-source="${esc(c.source)}" onclick="selectCamera(${jarg(c.source)})">
      <div class="frame"><i></i>
        <img alt="live feed" onerror="paneFeed(${jarg(c.source)},false)">
        <div class="feed-msg"><b>Camera feed unavailable</b><span>This camera may be warming up, or isn&rsquo;t running.</span></div>
      </div>
      <figcaption class="cap">
        <span class="lbl">${esc(c.name||c.source)}${c.network?' · net':''}${c.primary&&cams.length>1?' · main':''}</span>
        <button class="gear" type="button" onclick="event.stopPropagation();openSettings(${jarg(c.source)})" title="Camera settings">&#9881;</button>
      </figcaption>
    </figure>`).join('');
  syncLiveStreams();          // streams attach only while the Live tab is on screen
  checkFeeds();
}
function paneFor(source){ return [...document.querySelectorAll('.live-pane')].find(p=>p.dataset.source===source); }
function paneFeed(source,up){ const p=paneFor(source); if(p){ const f=p.querySelector('.frame'); if(f) f.classList.toggle('offline',!up); } }
async function checkFeeds(){
  // An MJPEG <img> goes black (not "errored") when frames merely stall, so poll /snapshot.jpg per
  // camera -- it 503s whenever that camera's capture thread has no current frame. Only worth
  // asking while the Live tab is actually visible.
  if(!document.querySelector('#view-live.on')) return;
  for(const c of LIVE.cams){
    try{ const r=await fetch('/snapshot.jpg?source='+encodeURIComponent(c.source),{cache:'no-store'}); paneFeed(c.source,r.ok); }
    catch(e){ paneFeed(c.source,false); }
  }
}
function selectCamera(source){
  if(LIVE.sel===source) return;
  LIVE.sel=source; touchedAt={};   // a fresh camera's read-back shouldn't be suppressed by the last one's edits
  document.querySelectorAll('.live-pane').forEach(p=>p.classList.toggle('sel',p.dataset.source===source));
  refreshWhoshere();               // re-scope "who's here" to the newly selected camera
  if(!$('#settings').hidden) refreshControls();
}
function camName(source){ const c=(LIVE.cams||[]).find(x=>x.source===source); return c?(c.name||c.source):source; }

/* The masthead period/coords come from the PRIMARY camera (period is global -- one sun). */
async function refreshHeader(){
  const src=LIVE.primary; if(!src) return;
  let v; try{ v=await fetch('/api/camera?source='+encodeURIComponent(src)).then(r=>r.json()); connOK(); }catch(e){ connFail(); return; }
  if(v.period){ window.__period=v.period; $('#period').textContent=v.period; $('#cap-period').textContent=v.period; }
  if(v.lat!=null && v.lon!=null){ const f=(x,p,n)=>`${Math.abs(x).toFixed(3)}° ${x>=0?p:n}`; $('#coords').textContent=`${f(v.lat,'N','S')} · ${f(v.lon,'E','W')}`; }
  /* Rig warning strip: a wedged camera outranks a battery warning (it already IS the
     consequence). Text comes verbatim from the rig (powerguard.py) so the three surfaces --
     console, HUD, here -- always tell the same story. */
  const warn=$('#rigwarn');
  if(warn){
    const wedged=v.wedge&&v.wedge.message, batt=v.power&&v.power.warning;
    if(wedged){ warn.textContent='⚠ '+v.wedge.message; warn.className='rig-warn wedge'; warn.hidden=false; }
    else if(batt){ warn.textContent='⚠ '+v.power.warning; warn.className='rig-warn'; warn.hidden=false; }
    else warn.hidden=true;
  }
}

/* The Instrument Panel controls act on the SELECTED camera. A networked camera exposes no
   settable controls, so we show a note instead of sliders. Only refreshed while the modal is open. */
async function refreshControls(){
  const src=LIVE.sel; if(!src) return;
  $('#instr-cam').textContent = (LIVE.cams.length>1?'· '+camName(src):'');
  let v; try{ v=await fetch('/api/camera?source='+encodeURIComponent(src)).then(r=>r.json()); connOK(); }catch(e){ connFail(); return; }
  const net=!!v.network;
  $('#instr-net').hidden=!net;
  $('#controls').style.display=net?'none':'';
  if(net) return;
  CONTROLS.forEach(c=>{
    const el=$(`.ctrl[data-k="${c.key}"]`); if(!el)return;
    const sl=el.querySelector('[data-slider]'), val=el.querySelector('[data-val]'), au=el.querySelector('[data-auto]');
    // Hardware lock: the driver rejects writes to this control (e.g. this webcam can't set
    // manual FOCUS), so the slider stays locked no matter what -- see probe_writable_controls.
    const locked = !!(v.writable && v.writable[c.key]===false);
    el.classList.toggle('locked', locked);
    el.title = locked ? `${c.label} can't be set on this camera (the driver rejects it).` : '';
    if(au){
      // Mirror the camera's auto/manual state only UNTIL the user takes control of this control;
      // afterwards the checkbox is user-owned and the camera read-back never touches it again.
      // (This webcam keeps reporting AUTOFOCUS=1.0 even after we set it to 0 -- the old code
      // re-checked the box and disabled the Focus slider a few seconds after the user went manual.)
      const a=v[c.auto];
      if(!touchedAt[c.key] && a!=null && document.activeElement!==au){
        au.checked = c.auto==='auto_exposure' ? !(Math.abs(a-0.25)<0.05) : (a>=0.5);
      }
    }
    sl.disabled = locked || (au ? au.checked : false);   // hardware lock OR the local auto checkbox
    if(locked){ val.textContent='locked'; return; }      // value read-back is meaningless when locked
    // Don't fight the user: for ~10s after they last touched a control, leave its slider value
    // alone -- the read-back can briefly lag the change we just POSTed and would yank it back.
    if(Date.now()-(touchedAt[c.key]||0) < 10000) return;
    let raw=v[c.key];
    if(raw!=null && !document.activeElement.isEqualNode(sl)){ sl.value=raw; val.textContent=Math.round(raw); }
  });
}

/* ---------- live stats + species ---------- */
async function refreshLive(){
  let s; try{ s=await fetch('/api/stats').then(r=>r.json()); connOK(); }catch(e){ connFail(); return; }
  maybeFirstRun(s);
  if(s.period){ $('#period').textContent=s.period; }
  if(s.span){ $('#span').textContent=`${s.span.start.slice(0,10)} → ${s.span.end.slice(0,10)}`; }
  $('#cap-period').textContent=(window.__period||'live');
  const days=(s.by_day||[]).length;
  $('#tallies').innerHTML=[
    ['observations',(s.total_crops||0).toLocaleString(),'obs'],
    ['visits <small>est</small>',s.total_visits||0,'visits'],
    ['species',(s.by_class||[]).length,'species'],
    ['days afield',days,'days'],
  ].map(([k,n,go])=>`<div class="tally" data-go="${go}"><div class="n">${n}</div><div class="k lbl">${k}</div></div>`).join('');
  $('#tallies').querySelectorAll('.tally').forEach(el=>el.onclick=()=>goTally(el.dataset.go));

  // Feature the most recent IDENTIFIED visitor; fall back to the newest detection if none are classified yet.
  renderLatest((s.latest||[]).find(x=>x.species) || (s.latest||[])[0]);

  let o; try{ o=await fetch('/api/species').then(r=>r.json()); connOK(); }catch(e){ connFail(); return; }
  const sp=(o.species||[]);
  window.__species=sp;
  const top=sp.slice(0,3), rare=sp.slice().reverse().filter(x=>!top.includes(x)).slice(0,3);
  $('#most').innerHTML=top.map((x,i)=>card(x,i+1)).join('')||'<p class="empty">No specimens yet.</p>';
  $('#least').innerHTML=rare.map(x=>card(x,null)).join('')||'<p class="empty">—</p>';
  bindCards();
}
/* ---------- "Who's here now?": name the live visit as it happens ----------
   One name tags the current visit's crops with that individual (a live solo confirm that feeds
   the appearance templates); two or more record who came TOGETHER without stamping a single name
   on both animals (the pair gotcha). The span is resolved server-side, so the client only sends
   the names it recognises. State: WH_SEL = the names currently picked. */
let WH_CAST=[], WH_SEL=new Set(), WH_BUSY=false, WH_CHIPSIG='';
async function refreshWhoshere(){
  if(!document.querySelector('#view-live.on')) return;   // the panel lives on the Live tab only
  let d; try{ d=await fetch('/api/live/now?source='+encodeURIComponent(LIVE.sel||'')).then(r=>r.json()); connOK(); }catch(e){ connFail(); return; }
  const sec=$('#whoshere'); if(!sec) return;
  sec.hidden=false;
  const wc=$('#wh-cam'); if(wc) wc.textContent=((LIVE.cams||[]).length>1?' · '+camName(LIVE.sel):'');
  WH_CAST=d.cast||[];
  const v=d.visit||{};
  $('#wh-span').textContent = v.count
    ? (v.active ? `active now · ${v.count} frame${v.count===1?'':'s'} this visit`
                : `quiet — last seen ${timeAgo(v.latest)}`)
    : 'all quiet — log it anyway and it attaches to the next frames';
  whRenderChips();
  whRenderRecent(d.recent||[]);
}
function whRenderChips(){
  const names=[...new Set([...WH_CAST, ...WH_SEL])];
  // Only rebuild the chip row when its contents change, so a 6s refresh can't wipe a hover or
  // re-trigger the fade. The Log button's enabled state is cheap, so always sync it.
  const sig=names.map(n=>(WH_SEL.has(n)?'*':'')+n).join('|');
  if(sig!==WH_CHIPSIG){
    WH_CHIPSIG=sig;
    $('#wh-cast').innerHTML = names.length
      ? names.map(n=>`<button type="button" class="wh-chip${WH_SEL.has(n)?' on':''}" onclick="whToggle(${jarg(n)})">${esc(cap1(n))}</button>`).join('')
      : '<span class="lbl" style="opacity:.6">No named critters yet — add the first below.</span>';
  }
  const log=$('#wh-log'); if(log) log.disabled = WH_SEL.size===0 || WH_BUSY;
}
function whToggle(n){ if(WH_SEL.has(n)) WH_SEL.delete(n); else WH_SEL.add(n); whRenderChips(); }
function whAddName(){
  const inp=$('#wh-new'); if(!inp) return;
  const n=(inp.value||'').trim(); if(!n) return;
  // Reuse an existing name if it only differs by case, so "notch" doesn't fork from "Notch".
  const hit=[...WH_CAST, ...WH_SEL].find(x=>x.toLowerCase()===n.toLowerCase());
  WH_SEL.add(hit||n); inp.value=''; whRenderChips(); inp.focus();
}
async function whLog(){
  if(!WH_SEL.size || WH_BUSY) return;
  WH_BUSY=true; whRenderChips();
  const names=[...WH_SEL];
  try{
    const r=await fetch('/api/live/sighting',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({names, source:LIVE.sel})}).then(r=>r.json());
    if(r.error){ whMsg(r.error,true); }
    else{
      const who=names.map(cap1).join(' + ');
      whMsg(r.multi
        ? `Logged ${who} together — co-presence noted.`
        : `Logged ${who}${r.stamped?` · tagged ${r.stamped} frame${r.stamped===1?'':'s'}`:''}.`);
      WH_SEL.clear();
    }
  }catch(e){ whMsg('Could not save: '+e,true); }
  finally{ WH_BUSY=false; refreshWhoshere(); }
}
function whMsg(t,err){
  const m=$('#wh-msg'); if(!m) return;
  m.textContent=t; m.className='wh-msg '+(err?'err':'ok');
  if(!err) setTimeout(()=>{ if(m.textContent===t){ m.textContent=''; m.className='wh-msg'; } },6000);
}
function whRenderRecent(list){
  const box=$('#wh-recent'); if(!box) return;
  if(!list.length){ box.innerHTML=''; return; }
  box.innerHTML='<div class="lbl wh-rec-h">Recently logged</div>'
    +list.map(s=>{
      const who=(s.names||[]).map(cap1).join(' + ');
      const tag=s.stamped? `tagged ${s.stamped}` : ((s.names||[]).length>1?'together':'');
      return `<div class="wh-rec"><span class="who">${esc(who)}</span>`
        +`<span class="when lbl">${esc(fmtClock(s.observed_at)||reidWhen(s.observed_at))}</span>`
        +(tag?`<span class="lbl tag">${esc(tag)}</span>`:'')+`</div>`;
    }).join('');
}

function timeAgo(iso){
  if(!iso) return '';
  const t=new Date(iso).getTime(); if(isNaN(t)) return '';
  const s=Math.max(0,(Date.now()-t)/1000);
  if(s<60) return 'just now';
  if(s<3600){ const m=Math.floor(s/60); return `${m} min ago`; }
  if(s<86400){ const h=Math.floor(s/3600); return `${h} hr${h>1?'s':''} ago`; }
  const d=Math.floor(s/86400); return `${d} day${d>1?'s':''} ago`;
}
function renderLatest(d){
  const el=$('#latest'); if(!el) return;
  if(!d){ el.innerHTML='<p class="empty" style="padding:20px">No visitors logged yet.</p>'; el.onclick=null; return; }
  // No species yet => the live rig saved it but the classifier (classify.py --watch) hasn't
  // named it. Show "Identifying…" rather than the coarse detector label ("animal").
  const name=d.species||(d.detection_class==='animal'?'Identifying…':(d.detection_class||'unknown'));
  const latin=d.species?latinOf(d.species):'';
  const conf=(d.species_confidence!=null)?d.species_confidence:null;
  el.innerHTML=`
    <div class="latest-eyebrow lbl">Most Recent Visitor</div>
    <div class="latest-photo playable" style="background-image:url('${d.crop_path?media(d.crop_path):''}')">${d.clip?playBadge(1):''}</div>
    <div class="latest-info">
      <div class="latest-name">${esc(cap1(name))}</div>
      <div class="latest-latin">${esc(latin)}</div>
      <div class="latest-when"><span>${esc(timeAgo(d.timestamp))}</span>${conf!=null?`<span class="dot">·</span><span class="conf">~${Math.round(conf*100)}%</span>`:''}${d.clip?`<span class="dot">·</span><span>${fmtDur(d.clip.seconds)} clip</span>`:''}</div>
    </div>`;
  el.onclick = d.species ? ()=>{ show('cat'); openSheet(d.species); } : null;
  // The ▶ on the photo plays the clip that was rolling for this visitor; the rest of the card
  // still opens the species sheet. (No clip => no badge, card behaves as before.)
  const pb=el.querySelector('[data-play]');
  if(pb) pb.onclick=(e)=>{ e.stopPropagation(); playClips([d.clip],0,cap1(name)); };
  flagLatest(d, el);   // async: appends an "off-pattern hour" chip only when behaviour disagrees
}
/* Live two-axis awareness (species level): if the most recent visitor arrived OUTSIDE its
   species' typical window, say so right on the card. Quiet when everything fits — the
   disagreement is the information (PLAN.md). Windows come from /api/behavior, cached 5 min. */
let __bwinCache=null, __bwinAt=0;
async function behaviorWindows(){
  if(__bwinCache && Date.now()-__bwinAt<300000) return __bwinCache;
  try{
    const d=await fetch('/api/behavior').then(r=>r.json());
    const m={}; (d.species||[]).forEach(s=>{ if(s.typical_window && s.n_visits>=3) m[s.species]=s.typical_window; });
    __bwinCache=m; __bwinAt=Date.now(); return m;
  }catch(e){ return __bwinCache||{}; }
}
function hourInWin(h,w){ return w.start_hour<=w.end_hour ? (h>=w.start_hour&&h<=w.end_hour) : (h>=w.start_hour||h<=w.end_hour); }
async function flagLatest(d,el){
  try{
    if(!d||!d.species||!d.timestamp) return;
    const wins=await behaviorWindows(); const w=wins[d.species]; if(!w) return;
    const h=new Date(d.timestamp).getHours();
    if(isNaN(h)||hourInWin(h,w)) return;
    const info=el.querySelector('.latest-info'); if(!info||info.querySelector('.offpat')) return;
    const f=document.createElement('div'); f.className='offpat';
    f.style.cssText='margin-top:8px;font-size:12px;color:#d9a441;border:1px solid rgba(217,164,65,.4);border-radius:4px;padding:4px 8px;display:inline-block';
    f.textContent=`⚑ off-pattern hour — ${nameOf(d.species)} usually ${String(w.start_hour).padStart(2,'0')}–${String(w.end_hour).padStart(2,'0')}h`;
    info.appendChild(f);
  }catch(e){ /* a missing chip is never worth breaking the live card */ }
}
function card(x,rank){
  const latin=latinOf(x.species);
  return `<div class="card" data-sp="${esc(x.species)}" style="animation-delay:${(rank||1)*40}ms">
    <div class="thumb" style="background-image:url('${x.sample?media(x.sample):''}')">${rank?`<span class="rank">№${rank}</span>`:''}</div>
    <div class="body">
      <div class="common">${esc(cap1(x.species))}</div>
      <div class="latin">${esc(latin)}</div>
      <div class="meta"><span class="count">${x.count.toLocaleString()}<small> obs</small></span><span class="conf">~${Math.round((x.avg_conf||0)*100)}%</span></div>
    </div></div>`;
}
function bindCards(){ document.querySelectorAll('.card[data-sp]').forEach(c=>c.onclick=()=>{ show('cat'); openSheet(c.dataset.sp); }); }

/* ---------- catalogue + species sheet ---------- */
async function loadCatalogue(){
  let o; try{ o=await fetch('/api/species').then(r=>r.json()); }catch(e){ $('#catalogue').innerHTML='<p class="empty">Could not load the catalogue.</p>'; return; }
  const sp=o.species||[]; window.__species=sp;
  $('#cat-count').textContent=`${sp.length} species · ${(o.total||0).toLocaleString()} observations`;
  $('#catalogue').innerHTML=sp.map((x,i)=>card(x,i+1)).join('')||'<p class="empty">Catalogue is empty.</p>';
  bindCards();
}
let LABELS=[];
fetch('/api/labels').then(r=>r.json()).then(l=>LABELS=l).catch(()=>{});
async function openSheet(name){
  $('#cat-index').style.display='none'; $('#cat-sheet').style.display='block';
  $('#sheet-name').textContent=cap1(name); $('#sheet-latin').textContent=latinOf(name);
  $('#sheet-crops').innerHTML='<p class="empty">Loading plates…</p>';
  let rows; try{ rows=await fetch('/api/species/'+encodeURIComponent(name)).then(r=>r.json()); }catch(e){ rows=[]; }
  if(!rows.length){ $('#sheet-crops').innerHTML='<p class="empty">No plates.</p>'; return; }
  $('#sheet-crops').innerHTML=rows.map(r=>cropTile(r,name)).join('');
}
function cropTile(r,name,tag){
  const v=r.verified; const cls=v===1?'v-1':v===0?'v-0':'';
  const stamp=v===1?'<span class="stamp v1">✓ confirmed</span>':v===0?'<span class="stamp v0">✗ wrong</span>':'';
  const opts=['<option value="">— correct to —</option>'].concat(
    LABELS.map(l=>`<option value="${esc(l)}"${l===name?' selected':''}>${esc(cap1(l))}</option>`),
    ['<option value="__other__">+ other (type a label)…</option>']).join('');
  return `<div class="crop ${cls}" data-id="${r.id}">
    <img loading="lazy" src="${media(r.crop_path)}" alt="">
    ${stamp}
    ${tag?`<div class="rv-tag lbl">${esc(tag)}</div>`:''}
    <div class="ft"><span class="c">${Math.round((r.confidence||0)*100)}%</span>
      <span class="acts">
        <button class="b up" title="confirm" onclick="act(${r.id},'verify',this)">✓</button>
        <button class="b dn" title="wrong" onclick="act(${r.id},'reject',this)">✗</button>
        <button class="b" title="correct" onclick="toggleEdit(this)">✎</button>
      </span></div>
    <select onchange="correct(${r.id},this.value)">${opts}</select>
  </div>`;
}
/* Needs Review: the prioritized "most likely mislabeled" crops (suspect species, junk labels,
   day-species-at-night, both-models-unsure), pulled across all species into the same sheet and the
   same ✓/✗/✎ controls. Opens from the Catalogue. */
async function openReview(){
  show('cat');
  $('#cat-index').style.display='none'; $('#cat-sheet').style.display='block';
  $('#sheet-name').textContent='Needs Review'; $('#sheet-latin').textContent='most likely mislabeled';
  $('#sheet-crops').innerHTML='<p class="empty">Gathering the suspect plates…</p>';
  let d; try{ d=await fetch('/api/review').then(r=>r.json()); }catch(e){ d={crops:[]}; }
  const rows=d.crops||[];
  if(!rows.length){ $('#sheet-crops').innerHTML='<p class="empty">Nothing flagged for review — every label has been checked.</p>'; return; }
  const head=`<p class="hint" style="color:var(--faint);font-style:italic">Showing ${rows.length} of ${d.total} flagged · sorted most-suspect first · ✓ confirm · ✗ wrong · ✎ correct the identification.</p>`;
  $('#sheet-crops').innerHTML=head+rows.map(r=>cropTile(r, r.species, cap1(r.species)+' · '+r.reason)).join('');
}
function toggleEdit(btn){ btn.closest('.crop').classList.toggle('editing'); }
async function act(id,action,btn){
  const restore=busyBtn(btn);
  try{ await fetch('/api/detection/'+id,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({action})}); connOK(); }
  catch(e){ connFail(); restore(); return; }   // leave the crop untouched; the subtle pill flags the blip
  restore();
  const crop=btn.closest('.crop');
  crop.classList.remove('v-1','v-0'); crop.querySelectorAll('.stamp').forEach(s=>s.remove());
  if(action==='verify'){ crop.classList.add('v-1'); crop.insertAdjacentHTML('afterbegin','<span class="stamp v1">✓ confirmed</span>'); }
  if(action==='reject'){ crop.classList.add('v-0'); crop.insertAdjacentHTML('afterbegin','<span class="stamp v0">✗ wrong</span>'); crop.classList.add('editing'); }
}
async function correct(id,species){
  if(species==='__other__'){ species=(prompt('New label for this crop (e.g. cat food):')||'').trim(); }
  if(!species)return;
  const sel=(typeof event!=='undefined'&&event)?(event.currentTarget||event.target):null;   // the <select> that changed
  if(sel){ sel.disabled=true; sel.style.opacity='.45'; }
  try{ await fetch('/api/detection/'+id,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({action:'correct',species})}); connOK(); }
  catch(e){ connFail(); if(sel){ sel.disabled=false; sel.style.opacity=''; } return; }
  const crop=document.querySelector(`.crop[data-id="${id}"]`);
  if(crop){ crop.style.transition='opacity .4s'; crop.style.opacity='.25'; }
}
function closeSheet(){ $('#cat-sheet').style.display='none'; $('#cat-index').style.display='block'; loadCatalogue(); }

/* ---------- explorer: drill into the field tallies ---------- */
let exploreStack=[], visitsData=[], obsState=null;
function goTally(t){
  if(t==='species'){ show('cat'); return; }   // species => the existing Specimen Catalogue
  exploreStack=[];
  explore(t, {}, t==='obs'?'Observations':t==='visits'?'Visits':'Days Afield');
}
function explore(screen,params,title,sub){
  exploreStack.push({screen,params:params||{},title:title||'',sub:sub||''});
  renderExplore();
}
function exploreBack(){ exploreStack.pop(); exploreStack.length?renderExplore():show('live'); }
function renderExplore(){
  const cur=exploreStack[exploreStack.length-1]; if(!cur)return;
  show('explore');
  $('#explore-title').textContent=cur.title;
  $('#explore-sub').textContent=cur.sub||'';
  const body=$('#explore-body'); body.innerHTML='<p class="empty">Loading…</p>';
  ({obs:renderObs,visits:renderVisits,days:renderDays,day:renderDay}[cur.screen]||(()=>{}))(cur.params,body);
}
const fmtDateTime=iso=>{ const d=new Date(iso); return isNaN(d)?iso:d.toLocaleString(undefined,{month:'short',day:'numeric',hour:'numeric',minute:'2-digit'}); };
const fmtDay=ymd=>{ const d=new Date(ymd+'T12:00:00'); return isNaN(d)?ymd:d.toLocaleDateString(undefined,{weekday:'short',month:'short',day:'numeric',year:'numeric'}); };

async function renderObs(params,body){
  obsState={day:params.day||null,species:params.species||null,start:params.start||null,end:params.end||null,offset:0};
  body.innerHTML='<div class="crops" id="obs-grid"></div><div class="more" id="obs-more"></div>';
  await loadMoreObs();
}
async function loadMoreObs(){
  const s=obsState; if(!s)return;
  const qs=new URLSearchParams();
  ['day','species','start','end'].forEach(k=>{ if(s[k]) qs.set(k,s[k]); });
  qs.set('offset',s.offset); qs.set('limit',60);
  let o; try{ o=await fetch('/api/crops?'+qs).then(r=>r.json()); }catch(e){ const m=$('#obs-more'); if(m) m.innerHTML='<p class="empty">Could not load observations.</p>'; return; }
  s.offset+=o.crops.length;
  const grid=$('#obs-grid'); if(!grid)return;
  grid.insertAdjacentHTML('beforeend', o.crops.map(obsTile).join(''));
  if(!grid.children.length) grid.innerHTML='<p class="empty">No observations here.</p>';
  $('#explore-sub').textContent=`${(o.total||0).toLocaleString()} total`;
  if(!s.species) grid.querySelectorAll('.crop[data-sp]').forEach(el=>el.onclick=()=>explore('obs',{species:el.dataset.sp},cap1(el.dataset.sp)));
  const more=$('#obs-more'); if(more) more.innerHTML = s.offset<o.total ? `<button class="back" onclick="loadMoreObs()">Load more &middot; ${(o.total-s.offset).toLocaleString()} left</button>` : '';
}
function obsTile(r){
  const sp=r.species?cap1(r.species):'unidentified';
  const conf=r.species_confidence!=null?r.species_confidence:r.confidence;
  const v=r.verified;
  return `<div class="crop ${v===1?'v-1':v===0?'v-0':''}"${r.species?` data-sp="${esc(r.species)}"`:''} title="${esc(fmtDateTime(r.timestamp))}">
    <img loading="lazy" src="${media(r.crop_path)}" alt="">
    <div class="ft"><span class="c">${Math.round((conf||0)*100)}%</span><span class="obs-sp">${esc(sp)}</span></div>
  </div>`;
}

async function renderVisits(params,body){
  const qs=params.day?('?day='+encodeURIComponent(params.day)):'';
  let o; try{ o=await fetch('/api/visits'+qs).then(r=>r.json()); }catch(e){ body.innerHTML='<p class="empty">Could not load visits.</p>'; return; }
  visitsData=o.visits||[];
  $('#explore-sub').textContent = o.window
    ? `the latest ${(o.total||0).toLocaleString()} visits`
    : `${(o.total||0).toLocaleString()} visit${o.total===1?'':'s'}`;
  if(!visitsData.length){ body.innerHTML='<p class="empty">No visits.</p>'; return; }
  body.innerHTML='<div class="cards">'+visitsData.map(visitCard).join('')+'</div>';
  // A visit with clips plays its video on click (what you asked for); one without falls back to
  // its crop grid (the old behaviour, for visits before clip recording was on).
  body.querySelectorAll('[data-vi]').forEach(el=>{ const v=visitsData[+el.dataset.vi];
    el.onclick=(v.clips&&v.clips.length)
      ? ()=>playClips(v.clips,0,`${cap1(v.title||'animal')} · ${fmtDateTime(v.start)}`)
      : ()=>explore('obs',{start:v.start,end:v.end},`Visit · ${cap1(v.title||'animal')}`,fmtDateTime(v.start)); });
}
function visitCard(v,i){
  const nclips=(v.clips||[]).length;
  const sp=(v.title&&v.title!=='animal')?v.title:'';
  const inds=(v.individuals||[]).map(n=>`<span class="vl-ind">${esc(cap1(n))}</span>`).join('');
  // The curation tools (confirm/correct species, name the individual) hide behind the ✎ so the
  // card itself stays a reading surface: who, when, how long, play. stopPropagation keeps the
  // whole label layer from triggering the card's play/drill.
  const footer=`<div class="vlabel" onclick="event.stopPropagation()">
      <button class="gear vlabel-toggle" onclick="this.closest('.vlabel').classList.toggle('open')" title="confirm or correct this visit's labels">✎ label</button>
      <div class="vlabel-tools">
        <div style="display:flex;gap:5px;align-items:center;flex-wrap:wrap;margin-top:6px">
          <button class="gear" onclick="postVisitLabel(_visitTarget(${i}),{verify:true},()=>visitSaved(${i},'✓ species confirmed'))" title="confirm this species for the whole visit">✓ sp</button>
          ${speciesSelect('vsp-'+i,sp)}
          <button class="gear" onclick="explorerSpecies(${i})" title="correct the species for the whole visit">correct</button></div>
        <div style="display:flex;gap:5px;align-items:center;margin-top:6px">${reidInput('vn-'+i,'name the individual…')}<button class="gear" onclick="explorerName(${i})">Name</button></div>
        <span id="vst-${i}" class="lbl" style="opacity:.8;min-height:14px"></span>
      </div>
    </div>`;
  return `<div class="card vcard" data-vi="${i}">
    <div class="thumb playable" style="background-image:url('${v.rep_crop?media(v.rep_crop):''}')">${nclips?playBadge(nclips):''}</div>
    <div class="body">
      <div class="common">${esc(cap1(v.title||'animal'))} ${inds}</div>
      <div class="latin" style="font-style:normal">${esc(fmtDateTime(v.start))}</div>
      <div class="meta"><span class="count">${v.count}<small> obs</small></span><span class="conf">${v.minutes>=1?Math.round(v.minutes)+' min':'brief'}</span></div>
      ${footer}
    </div></div>`;
}
function _visitTarget(i){ const v=visitsData[i]||{}; return {source:v.source,start:v.start,end:v.end}; }
function visitSaved(i,msg){ const s=document.getElementById('vst-'+i); if(s){ s.textContent=msg; s.style.color='var(--ok)'; } }
function explorerName(i){
  const inp=document.getElementById('vn-'+i); const name=(inp&&inp.value||'').trim();
  if(!name){ if(inp) inp.focus(); return; }
  postVisitLabel(_visitTarget(i),{name},()=>visitSaved(i,'✓ named '+name));
}
function explorerSpecies(i){
  visitSpeciesCorrect(_visitTarget(i),'vsp-'+i,r=>visitSaved(i,'✓ '+nameOf(r.species_set||r.dominant_species||'')));
}

async function renderDays(params,body){
  let s; try{ s=await fetch('/api/stats').then(r=>r.json()); }catch(e){ body.innerHTML='<p class="empty">Could not load.</p>'; return; }
  const days=(s.by_day||[]).slice().reverse();
  $('#explore-sub').textContent=`${days.length} day${days.length===1?'':'s'}`;
  if(!days.length){ body.innerHTML='<p class="empty">No days yet.</p>'; return; }
  body.innerHTML='<div class="daygrid">'+days.map(dayCard).join('')+'</div>';
  body.querySelectorAll('[data-day]').forEach(el=>el.onclick=()=>explore('day',{date:el.dataset.day},fmtDay(el.dataset.day)));
}
function dayCard(d){
  const nsp=Object.keys(d.classes||{}).length;
  return `<div class="daycard" data-day="${esc(d.day)}">
    <div class="daycard-date">${esc(fmtDay(d.day))}</div>
    <div class="daycard-stats"><span><b>${(d.crops||0).toLocaleString()}</b> obs</span><span><b>${d.visits||0}</b> visits</span><span><b>${nsp}</b> species</span></div>
  </div>`;
}

async function renderDay(params,body){
  const date=params.date;
  let s; try{ s=await fetch('/api/stats').then(r=>r.json()); }catch(e){ s={}; }
  const d=(s.by_day||[]).find(x=>x.day===date)||{crops:0,visits:0,classes:{}};
  $('#explore-sub').textContent=`${(d.crops||0).toLocaleString()} obs · ${d.visits||0} visits`;
  // The night FOLLOWING day d runs dusk(d) -> dawn(d+1), so it's anchored on d+1 in the digest.
  const nextDay=(dt=>{const x=new Date(dt+'T12:00:00'); x.setDate(x.getDate()+1); return ymdLocal(x);})(date);
  const chips=Object.entries(d.classes||{}).map(([sp,n])=>`<button class="chip" data-sp="${esc(sp)}">${esc(cap1(sp))} <i>${n}</i></button>`).join('');
  body.innerHTML=`
    <div class="day-summary">
      <button class="back" id="day-visits">${d.visits||0} visits this day &rarr;</button>
      <button class="back" onclick="openDispatchAt(${jarg(date)},'day')" title="the day's dispatch — reel, timeline, roll">☀ day dispatch</button>
      <button class="back" onclick="openDispatchAt(${jarg(nextDay)},'night')" title="the night that FOLLOWED this day (dusk to dawn)">☾ that night&rsquo;s dispatch</button>
      ${chips?`<div class="chips">${chips}</div>`:''}
    </div>
    <div class="crops" id="obs-grid"></div><div class="more" id="obs-more"></div>`;
  const dv=$('#day-visits'); if(dv) dv.onclick=()=>explore('visits',{day:date},`Visits · ${fmtDay(date)}`);
  body.querySelectorAll('.chip[data-sp]').forEach(el=>el.onclick=()=>{ show('cat'); openSheet(el.dataset.sp); });
  obsState={day:date,species:null,start:null,end:null,offset:0};
  await loadMoreObs();
}

/* ---------- shared: species glyphs (calendar + dispatch) ---------- */
// Non-visitor labels -- mirror of stats._NON_CRITTER, which is also where an operator adds the
// individual names their own household gets labelled with. Hidden from glances.
const NONCRITTER=new Set(['bricks','brick','blur','blurry','cat food','catfood','food',
  'door','porch','broom','chair','fence','wall','table','plant','pot','hose','shadow','reflection','leaf','leaves',
  'rock','stick','sticks','ground','tree','bush','person','people','human','vehicle','car','unknown','unidentified',
  'nothing','empty','none','background','n/a','na','','not an animal']);
// The server (stats._NON_CRITTER) is the source of truth: merge its list in on load so this set
// never drifts out of sync with the backend. The literal above is only a fallback if the fetch fails.
fetch('/api/denylist').then(r=>r.json()).then(a=>{(a||[]).forEach(x=>NONCRITTER.add(String(x).toLowerCase()))}).catch(()=>{});
const GLYPH_MAP={'raccoon':'🦝','domestic cat':'🐱','cat':'🐱','brown rat':'🐀','rat':'🐀',
  'eastern cottontail':'🐰','american crow':'🐦‍⬛','animal':'🐾'};
// {e:emoji} where a clear one exists, else {m:monogram} -- so any species the classifier emits
// still renders. Opossum has no emoji, so it gets a small-caps monogram.
function glyphInfo(sp){
  const k=(sp||'').toLowerCase().replace(/’/g,"'");
  if(GLYPH_MAP[k]) return {e:GLYPH_MAP[k]};
  if(/squirrel|chipmunk/.test(k)) return {e:'🐿️'};
  if(k.includes('opossum')) return {m:'Op'};
  if(/sparrow|finch|wren|warbler|junco|towhee|thrush|jay|crow|raven|robin|starling|flicker|pigeon|dove|chickadee|hummingbird|waxwing|siskin|bushtit|nuthatch|woodpecker|blackbird|grosbeak|kinglet|tanager|swallow|\bbird\b/.test(k)) return {e:'🐦'};
  if(/\brat\b|mouse|vole|mole|shrew/.test(k)) return {e:'🐀'};
  if(/rabbit|cottontail|hare/.test(k)) return {e:'🐰'};
  if(/coyote|fox|\bdog\b/.test(k)) return {e:'🐾'};
  if(/deer|elk/.test(k)) return {e:'🦌'};
  if(/skunk/.test(k)) return {e:'🦨'};
  const m=k.split(/[\s-]+/).slice(0,2).map(w=>w[0]||'').join('').toUpperCase()||'?';
  return {m};
}
const glyphHTML=g=>g.e?g.e:`<b class="mono-gl">${esc(g.m)}</b>`;
const nameOf=sp=>sp==='animal'?'Unidentified':cap1(sp);
const fmtHourJS=h=>`${(h%12)||12}${h<12?'am':'pm'}`;
const fmtClock=iso=>{ const d=new Date(iso); return isNaN(d)?'':d.toLocaleTimeString(undefined,{hour:'numeric',minute:'2-digit'}); };

/* ---------- motion strip (phase-4 clip-track signal) ----------
   The robust signals LEAD: approach/retreat/steady + straightness-in-words + moving%. Speed is
   bbox-SIZE-confounded and unvalidated, so it's a faint secondary figure only, never the headline.
   Fetched lazily per shown visit (never for a whole list at once) — see fillVisitMotion. */
const APPROACH_WORD={approach:'approaching',retreat:'moving off',steady:'holding steady'};
// straightness is 0..1 (path directness); describe it in words rather than a bare number.
function straightWord(s){ if(s==null) return ''; return s>=0.8?'a direct line':s>=0.55?'a fairly direct path':s>=0.3?'a meandering path':'a wandering path'; }
// Build the one-line strip from a visit_motion payload; '' when there's nothing to show.
function motionStrip(m){
  if(!m||!(m.tracks>0)) return '';
  const bits=[];
  const app=APPROACH_WORD[m.approach]; if(app) bits.push(`<b>${app}</b>`);
  const sw=straightWord(m.straightness); if(sw) bits.push(sw);
  if(m.moving_frac!=null) bits.push(`moving ${Math.round(m.moving_frac*100)}%`);
  // Speed: a hushed, size-confounded aside — omitted unless we have nothing more robust to say.
  const spd=(m.avg_speed!=null && !bits.length)?`<span class="ms-spd" title="uncalibrated; bbox-size-confounded — not an absolute speed">~${(+m.avg_speed).toFixed(2)} px/f</span>`:'';
  if(!bits.length && !spd) return '';
  const track=`${m.tracks} track${m.tracks===1?'':'s'}`;
  return `<div class="motion-strip" title="from ${track} of clip motion in this visit — direction &amp; path are the trustworthy signals; speed is uncalibrated">`
    +`<span class="ms-paw">🐾</span><span class="ms-txt">${bits.join(' · ')}${spd?' · '+spd:''}</span></div>`;
}
// Lazily fetch a single visit's motion into a placeholder <div data-vm="ID">; drops the box when
// there's nothing (tracks==0) so we never render an empty strip. Guarded so a fetch blip is silent.
async function fillVisitMotion(vid, box){
  if(!box || vid==null) return;
  let m; try{ m=await fetch('/api/visit/motion?visit_id='+encodeURIComponent(vid)).then(r=>r.json()); }
  catch(e){ box.remove(); return; }
  const html=motionStrip(m);
  if(html) box.outerHTML=html; else box.remove();
}
// Wire every [data-vm] placeholder under `root`, capped so a huge list can't fire N calls at once.
const MOTION_FETCH_CAP=40;
function wireVisitMotion(root){
  if(!root) return;
  const boxes=[...root.querySelectorAll('[data-vm]')];
  if(boxes.length>MOTION_FETCH_CAP){
    console.log(`[motion] capping lazy visit-motion fetches at ${MOTION_FETCH_CAP} of ${boxes.length} shown`);
    boxes.slice(MOTION_FETCH_CAP).forEach(b=>b.remove());   // leave the rest strip-less rather than storm the server
  }
  boxes.slice(0,MOTION_FETCH_CAP).forEach(b=>fillVisitMotion(b.dataset.vm, b));
}
// Per-individual motion FINGERPRINT (aggregate across a named individual's clip-tracks). Same
// priority order as the visit strip: direction tendency + straightness-in-words lead; the clip
// count is context; speed is omitted (aggregate size-confound is worse). '' when tracks==0.
function individualFP(m){
  if(!m||!(m.tracks>0)) return '';
  const bits=[];
  const dirs=[['approach',m.approach,'approaches'],['retreat',m.retreat,'retreats'],['steady',m.steady,'holds steady']]
    .filter(d=>d[1]>0).sort((a,b)=>b[1]-a[1]);
  if(dirs.length){
    const top=dirs[0];
    bits.push(`usually ${top[2]} <span class="fp-n">(${top[1]})</span>`
      +dirs.slice(1).map(d=>` · ${d[2]} <span class="fp-n">(${d[1]})</span>`).join(''));
  }
  const sw=straightWord(m.straightness); if(sw) bits.push(sw.replace(/^a /,''));   // "direct line" reads better bare here
  if(m.moving_frac!=null) bits.push(`moving ${Math.round(m.moving_frac*100)}%`);
  if(!bits.length) return '';
  bits.push(`${m.tracks} clip${m.tracks===1?'':'s'} of motion`);
  return `<span class="fp-paw">🐾</span>${bits.join(' · ')}`;
}
// Lazily fill an individual's fingerprint into a placeholder <span data-im="NAME">; drop if empty.
async function fillIndivMotion(name, box){
  if(!box || !name) return;
  let m; try{ m=await fetch('/api/individual/motion?individual='+encodeURIComponent(name)).then(r=>r.json()); }
  catch(e){ box.remove(); return; }
  const html=individualFP(m);
  if(html) box.innerHTML=html; else box.remove();
}
function wireIndivMotion(root){
  if(!root) return;
  // The named cast is small (placeholders are excluded at render), so no cap is needed here.
  root.querySelectorAll('[data-im]').forEach(b=>fillIndivMotion(b.dataset.im, b));
}

/* ---------- The Dispatch (period digest) ----------
   The landing page, rebuilt around the two questions people actually open it with:
   "what happened?" (the condensed highlight reel) and "who came, when?" (the visit timeline).
   Species aggregates (the Roll), the plate, and the cast roll call follow underneath.
   DSP.date=null shows the latest completed period; ‹ › walk earlier days via the digest's
   prev/next anchors. The stitched reel is fetched alongside and re-polled while building. */
let DSP={edition:'auto', date:null};
let __dspSeq=0, __reelTimer=null;
function setDispatch(ed){ DSP.edition=ed; loadDispatch(); }
function dspNav(date, edition){ DSP={edition:edition||DSP.edition, date:date||null}; loadDispatch(); }
function dspLatest(){ DSP={edition:'auto', date:null}; loadDispatch(); }
function openDispatchAt(date, edition){ dspNav(date, edition); show('dispatch'); }

async function loadDispatch(){
  const body=$('#dispatch-body'); body.innerHTML='<p class="empty">Loading the dispatch…</p>';
  clearTimeout(__reelTimer);
  const seq=++__dspSeq;
  const qs='edition='+encodeURIComponent(DSP.edition)+(DSP.date?'&date='+encodeURIComponent(DSP.date):'');
  let d, rc, rl;
  try{
    [d, rc, rl] = await Promise.all([
      fetch('/api/digest?'+qs).then(r=>r.json()),
      fetch('/api/rollcall').then(r=>r.json()).catch(()=>({cast:[]})),
      fetch('/api/reel?'+qs).then(r=>r.ok?r.json():null).catch(()=>null),   // older server -> no reel
    ]);
  }catch(e){ body.innerHTML='<p class="empty">Could not load the dispatch.</p>'; return; }
  if(seq!==__dspSeq) return;                    // a newer navigation superseded this load
  window.__digest=d;
  renderDispatch(d, rc, rl);
  if(rl && rl.status==='building') pollReel(qs, seq, 40);
}
/* While the server stitches the condensed cut, quietly re-ask and swap the reel section in
   place when it's ready -- never rebuilding the whole page under the reader's scroll. */
function pollReel(qs, seq, tries){
  clearTimeout(__reelTimer);
  if(tries<=0 || seq!==__dspSeq) return;
  __reelTimer=setTimeout(async()=>{
    if(seq!==__dspSeq) return;
    let r2=null;
    try{ r2=await fetch('/api/reel?'+qs).then(r=>r.ok?r.json():null); }catch(e){ /* retry below */ }
    if(seq!==__dspSeq) return;
    if(r2 && r2.status==='ready'){
      window.__reelMan=r2;
      const el=document.getElementById('reel-sec');
      if(el) el.innerHTML=reelSection(window.__digest||{}, r2);
    } else {
      pollReel(qs, seq, tries-1);
    }
  }, 5000);
}
/* The named cast. Faces seen in the last days lead (with their overdue flag when a regular
   has gone quiet); long-gone individuals compress to one soft line instead of a wall of
   rust-red OVERDUE cards. Clicking a face jumps to the Individuals tab. */
function rollcallSection(rc){
  const cast=(rc&&rc.cast)||[];
  if(!cast.length) return '';
  const fresh=cast.filter(c=>c.days_since!=null && c.days_since<=10)
                  .sort((a,b)=>a.days_since-b.days_since);
  const gone=cast.filter(c=>!(c.days_since!=null && c.days_since<=10));
  const sinceText=c=> c.days_since==null ? '' :
    c.days_since===0 ? 'seen today' : c.days_since===1 ? 'seen yesterday' : `${c.days_since} days ago`;
  const card=c=>{
    const av=c.crop?`style="background-image:url('${media(c.crop)}')"`:'';
    return `<div class="rc-card${c.overdue?' overdue':''}" onclick="show('indiv')"
        title="${esc(c.id)} — last seen ${esc((c.last_seen||'').slice(0,16).replace('T',' '))}">
      <div class="rc-av" ${av}></div>
      <div class="rc-info">
        <div class="rc-name">${esc(c.id)}${c.overdue?'<span class="rc-flag">overdue</span>':''}</div>
        <div class="rc-sub lbl">${esc(nameOf(c.species||'·'))}</div>
        <div class="rc-since">${esc(sinceText(c))}</div>
      </div></div>`;
  };
  const goneLine=gone.length?`<p class="rc-gone">Not seen in a while: ${gone.map(c=>
    `<span onclick="show('indiv')" title="last seen ${esc((c.last_seen||'').slice(0,10))}">${esc(c.id)}`
    +`${c.days_since!=null?` <i>(${c.days_since}d)</i>`:''}</span>`).join(' · ')}</p>`:'';
  const note=fresh.length?`${fresh.length} about lately`:'nobody about lately';
  return `<h2 class="sec">Cast Roll Call <span class="n">${note}</span></h2>
    ${fresh.length?`<div class="rollcall">${fresh.map(card).join('')}</div>`:''}${goneLine}`;
}
function dispatchHeader(d){
  const ed=d.edition||DSP.edition;
  const masthead = ed==='night' ? 'The Morning Dispatch' : ed==='day' ? 'The Evening Dispatch' : 'The Dispatch';
  const range = d.start ? `${fmtDateTime(d.start)} – ${fmtDateTime(d.end)}` : '';
  const back = d.latest===false ? ` · <span class="dsp-latest" onclick="dspLatest()">back to the latest ↻</span>` : '';
  const nav=(dt,dir,lab)=> dt
    ? `<button class="dsp-arrow" onclick="dspNav(${jarg(dt)},${jarg(d.edition||'')})" title="${lab}">${dir}</button>`
    : `<button class="dsp-arrow" disabled>${dir}</button>`;
  return `<div class="dsp-head">
    <div>
      <div class="dsp-title">${esc(masthead)}<span style="color:var(--gilt)"> · </span>${esc(d.title||'')}</div>
      <div class="dsp-sub">${esc(range)}${d.backed_off?' · last period with activity':''}${back}</div>
    </div>
    <div class="dsp-nav">
      ${nav(d.prev_date,'‹','the previous '+esc(d.edition||'period'))}
      <div class="dsp-toggle">
        <button class="${ed==='night'?'on':''}" onclick="setDispatch('night')">☾ Night</button>
        <button class="${ed==='day'?'on':''}" onclick="setDispatch('day')">☀ Day</button>
      </div>
      ${nav(d.next_date,'›','the next '+esc(d.edition||'period'))}
    </div>
  </div>`;
}
/* The reel section: the stitched condensed cut when it's ready (one video, chapter strip),
   the old every-clip playlist while it builds or on a server without /api/reel. */
function reelSection(d, rl){
  const playlist=window.__reel||[];
  const ready=rl&&rl.status==='ready'&&(rl.segments||[]).length;
  if(!ready && !playlist.length) return '';
  const ed=d.edition==='night'?'night':'day';
  if(ready){
    // Hero backdrop: a real full-size frame lifted out of the stitched reel. The old fallbacks
    // are CROPS — tight animal cutouts, measured as small as 96×103 — and at ~1500px hero width
    // that's a 4–16× upscale, i.e. the wall of blurry ovals. Kept only for reels built before
    // posters existed, which have no poster_path.
    const poster=rl.poster_path||(d.plate&&d.plate.crop_path)||(rl.segments.find(s=>s.thumb)||{}).thumb;
    const chaps=rl.segments.map((s,i)=>`
      <div class="fs" data-i="${i}" onclick="reelChap(${i})">
        <div class="ft" style="background-image:url('${s.thumb?media(s.thumb):''}')"><span class="dur">${fmtDur(s.seconds)}</span></div>
        <div class="lab">${esc(fmtClock(s.start))} · ${esc(nameOf(s.species||'·'))}${(s.individuals||[]).length?' · '+esc(cap1(s.individuals[0])):''}</div>
      </div>`).join('');
    return `<h2 class="sec">Highlight Reel <span class="n">the ${ed} in ${fmtDur(rl.seconds)}</span></h2>
      <div class="panel reelhero" id="reelhero" onclick="reelHeroPlay()" style="background-image:url('${poster?media(poster):''}')">
        <div class="big-play">&#9654;</div>
        <div class="scrim">
          <div class="rh-eyebrow lbl">Watch the ${ed}</div>
          <div class="rh-title">The best moments, in ${fmtDur(rl.seconds)}</div>
          <div class="rh-meta">${rl.segments.length} moments · condensed from ${rl.n_source_clips||playlist.length} clips · press play</div>
        </div></div>
      <div class="filmstrip" id="dsp-strip">${chaps}</div>
      <div class="reel-links lbl">
        ${playlist.length?`<span class="reel-link" onclick="reelPlayAll()">▷ watch every clip (${playlist.length})</span> · `:''}
        <a class="reel-link" href="${media(rl.clip_path)}" download>⤓ save the reel</a>
      </div>`;
  }
  const total=playlist.reduce((a,c)=>a+(c.seconds||0),0);
  const poster=(d.plate&&d.plate.crop_path)||playlist[0].thumb;
  const note = rl&&rl.status==='building'
    ? `<div class="reel-links lbl"><span class="reel-note">✂ a condensed cut is being stitched — it will appear here in a minute or two</span></div>` : '';
  return `<h2 class="sec">Highlight Reel <span class="n">${playlist.length} clip${playlist.length>1?'s':''} · ${fmtDur(total)}</span></h2>
    <div class="panel reelhero" id="reelhero" onclick="reelPlayAll()" style="background-image:url('${poster?media(poster):''}')">
      <div class="big-play">&#9654;</div>
      <div class="scrim">
        <div class="rh-eyebrow lbl">Watch the ${ed}</div>
        <div class="rh-title">The visitors, in sequence</div>
        <div class="rh-meta">${playlist.length} clips · ${fmtDur(total)} of footage · press play</div>
      </div></div>
    <div class="filmstrip" id="dsp-strip">${playlist.map((c,i)=>`
      <div class="fs" data-i="${i}" onclick="reelPlayAll(${i})"><div class="ft" style="background-image:url('${c.thumb?media(c.thumb):''}')"><span class="dur">${fmtDur(c.seconds)}</span></div>
        <div class="lab">${esc(fmtClock(c.start))} · ${esc(nameOf(c.species||'·'))}</div></div>`).join('')}</div>
    ${note}`;
}
function reelHeroPlay(){
  if(window.__reelMan) playHighlightReel(window.__reelMan,'Highlight Reel — '+((window.__digest||{}).title||''));
  else reelPlayAll();
}
function reelChap(i){
  if(!window.__reelMan) return;
  playHighlightReel(window.__reelMan,'Highlight Reel — '+((window.__digest||{}).title||''));
  if(i>0) reelSeek(i);
}
function reelPlayAll(i){ playClips(window.__reel||[], i||0, 'Every clip — '+((window.__digest||{}).title||'')); }
/* The visit timeline: the period's comings and goings in order — when, who (species + any
   named individuals), how long, what the motion looked like, and the clips to prove it. */
function visitLogSection(d){
  const log=d.visit_log||[];
  if(!log.length) return '';
  window.__vlog=log;
  const rows=log.map((v,i)=>{
    const sp=(v.species||[]).map(nameOf).join(' + ')||'Unidentified';
    const g=glyphInfo((v.species||[])[0]||'');
    const inds=(v.individuals||[]).map(n=>`<span class="vl-ind">${esc(cap1(n))}</span>`).join('');
    const nclips=(v.clips||[]).length;
    const dur=v.minutes>=1.5?`${Math.round(v.minutes)} min`:'brief';
    return `<div class="vl-row" data-vl="${i}" title="${nclips?'watch this visit':'see this visit&rsquo;s photos'}">
      <div class="vl-time"><b>${esc(fmtClock(v.start))}</b><span>${dur}</span></div>
      <div class="vl-thumb playable" style="background-image:url('${v.rep_crop?media(v.rep_crop):''}')">${nclips?'<span class="play-badge sm"></span>':''}</div>
      <div class="vl-main">
        <div class="vl-name"><span class="gx">${glyphHTML(g)}</span> ${esc(sp)} ${inds}</div>
        ${motionStrip(v.motion)||''}
      </div>
      <div class="vl-right"><span class="vl-obs">${v.count}<small> obs</small></span>${nclips?`<span class="vl-clips">▶ ${nclips} clip${nclips>1?'s':''}</span>`:''}</div>
    </div>`;
  }).join('');
  const first=fmtClock(log[0].start), last=fmtClock(log[log.length-1].start);
  return `<h2 class="sec">The Visits <span class="n">${log.length} · first ${esc(first)} · last ${esc(last)}</span></h2>
    <div class="vlog">${rows}</div>
    <p class="vlog-more"><span onclick="goTally('visits')">Browse the full visit archive →</span></p>`;
}
function vlOpen(i){
  const v=(window.__vlog||[])[i]; if(!v) return;
  const title=`${(v.species||[]).map(nameOf).join(' + ')||'Visit'} · ${fmtClock(v.start)}`;
  if((v.clips||[]).length) playClips(v.clips.slice(0,16), 0, title);   // busiest 16, played in order
  else explore('obs',{start:v.start,end:v.end},title,fmtDateTime(v.start));
}
function renderDispatch(d, rc, rl){
  const body=$('#dispatch-body');
  const roll=rollcallSection(rc||{});
  if(!d || (d.empty && !d.start)){ body.innerHTML=dispatchHeader(d||{})+`<p class="empty">${esc((d&&d.reason)||'Nothing to report yet.')}</p>`+roll; return; }
  let html=dispatchHeader(d);
  const flags=[];
  (d.novel||[]).forEach(sp=>{ const s=(d.species||[]).find(x=>x.species===sp); const n=s&&s.novelty;
    const lead = n&&n.first_ever ? 'First ever recorded' : (n&&n.days_since ? `First in ${n.days_since} days` : 'Notable');
    flags.push(`<span class="flag new">❋ ${lead}: ${esc(nameOf(sp))}</span>`); });
  (d.quiet||[]).forEach(q=>{ flags.push(`<span class="flag quiet">— No ${esc(nameOf(q.species))} this ${esc(d.edition)} (usually ${Math.round(q.frac*100)}% of ${esc(d.edition)}s)</span>`); });
  if(d.moon){ flags.push(`<span class="flag moon">${d.moon.glyph} ${esc(d.moon.name)} · ${d.moon.illum_pct}% lit</span>`); }
  if(flags.length) html+=`<div class="lede">${flags.join('')}</div>`;
  if(d.empty){ body.innerHTML=html+`<p class="empty">A quiet ${esc(d.edition)} — no visitors recorded.</p>`+roll; return; }
  window.__reel=d.reel||[];
  window.__reelMan=(rl&&rl.status==='ready')?rl:null;
  html+=`<div id="reel-sec">${reelSection(d, rl)}</div>`;
  html+=visitLogSection(d);
  html+=roll;
  const t=[['visits',(d.visits||0).toLocaleString()],['species',(d.species||[]).length],
    ['busiest hour',d.busiest_hour?fmtHourJS(d.busiest_hour.hour):'—']];
  html+=`<div class="tallies">${t.map(([k,v])=>`<div class="tally" style="cursor:default"><div class="n">${v}</div><div class="k lbl">${k}</div></div>`).join('')}</div>`;
  if(d.plate){ const p=d.plate;
    html+=`<div class="panel hero">
      <div class="hero-photo playable" style="background-image:url('${p.crop_path?media(p.crop_path):''}')">${p.clip?playBadge(1):''}</div>
      <div class="hero-info">
        <div class="hero-eyebrow lbl">Plate of the ${d.edition==='night'?'Night':'Day'}</div>
        <div class="hero-name">${esc(nameOf(p.species))}</div>
        <div class="hero-latin">${esc(latinOf(p.species))}</div>
        <div class="hero-meta">${esc(fmtClock(p.time))}<br><span class="c">~${Math.round((p.conf||0)*100)}%</span> · sharpest frame</div>
      </div></div>`;
  }
  window.__rollClips=(d.species||[]).map(s=> s.clip?{...s.clip, species:s.species}:null);
  html+=`<h2 class="sec">The Roll <span class="n">${(d.species||[]).length} species</span></h2>`;
  html+=`<div class="roll">${(d.species||[]).map(entryRow).join('')||'<p class="empty" style="padding:18px">—</p>'}</div>`;
  body.innerHTML=html;
  body.querySelectorAll('.entry[data-sp]').forEach(el=>el.onclick=()=>{ show('cat'); openSheet(el.dataset.sp); });
  body.querySelectorAll('[data-vl]').forEach(el=>el.onclick=()=>vlOpen(+el.dataset.vl));
  // ▶ on the plate (when a clip caught the sharpest frame) and on each roll row.
  const pp=body.querySelector('.hero-photo .play-badge'); if(pp&&d.plate&&d.plate.clip) pp.onclick=(e)=>{ e.stopPropagation(); playClips([d.plate.clip],0,nameOf(d.plate.species)); };
  body.querySelectorAll('.entry .play-badge[data-ci]').forEach(el=>el.onclick=(e)=>{ e.stopPropagation(); const c=(window.__rollClips||[])[+el.dataset.ci]; if(c) playClips([c],0,nameOf(c.species||'')); });
}
function entryRow(s,i){
  const n=s.novelty||{}, badges=[];
  if(n.first_ever) badges.push('<span class="badge new">New</span>');
  else if((n.days_since||0)>=3) badges.push(`<span class="badge gap">first in ${n.days_since}d</span>`);
  if((s.streak||0)>=3) badges.push(`<span class="badge streak">${s.streak} in a row</span>`);
  const active=new Set(s.active_hours||[]);
  const hrs=s.hours||[], max=Math.max(1,...hrs);
  const clock=hrs.map((c,h)=>{const ht=c?Math.max(2,Math.round(Math.sqrt(c/max)*18)):1;return `<span class="hbar${active.has(h)?' on':''}" style="height:${ht}px" title="${fmtHourJS(h)}: ${c}"></span>`;}).join('');
  const g=glyphInfo(s.species), clickable=s.species!=='animal';
  return `<div class="entry"${clickable?` data-sp="${esc(s.species)}"`:' style="cursor:default"'}>
    <div class="entry-thumb${s.clip?' playable':''}" style="background-image:url('${s.rep_crop?media(s.rep_crop):''}')">${s.clip?`<span class="play-badge sm" data-ci="${i}"></span>`:''}</div>
    <div class="entry-main">
      <div class="entry-name"><span class="gx">${glyphHTML(g)}</span> ${esc(nameOf(s.species))} <span class="latin">${esc(latinOf(s.species))}</span></div>
      <div class="entry-when">${fmtClock(s.first)}–${fmtClock(s.last)}${s.typical?` · <span class="typ">usually ${esc(s.typical)}</span>`:''}</div>
      ${badges.length?`<div class="badges">${badges.join('')}</div>`:''}
    </div>
    <div class="entry-right">
      <div class="entry-count">${s.visits}<small> visit${s.visits===1?'':'s'}</small></div>
      <div class="clock" title="all-time activity by hour; gold = this period">${clock}</div>
    </div>
  </div>`;
}

/* ---------- Calendar ---------- */
let calYear=null, calMonth=null, calByDay={};
const ymdLocal=dt=>{const p=n=>String(n).padStart(2,'0');return dt.getFullYear()+'-'+p(dt.getMonth()+1)+'-'+p(dt.getDate());};
/* ---------- behaviour (phase 4: visits, profiles, the two-axis read-out) ---------- */
async function loadBehavior(){
  const body=$('#behavior-body'); body.innerHTML='<p class="empty">Reading the field notes…</p>';
  let d; try{ d=await fetch('/api/behavior').then(r=>r.json()); }
  catch(e){ body.innerHTML='<p class="empty">Could not load behaviour.</p>'; return; }
  renderBehavior(d);
}
function behaviorClock(hours, win){
  const max=Math.max(1,...hours);
  const inWin=h=> win ? (win.start_hour<=win.end_hour ? (h>=win.start_hour&&h<=win.end_hour)
                                                       : (h>=win.start_hour||h<=win.end_hour)) : false;
  return hours.map((c,h)=>{ const ht=c?Math.max(2,Math.round(Math.sqrt(c/max)*18)):1;
    return `<span title="${fmtHourJS(h)}: ${c} visit(s)" style="display:inline-block;width:4px;height:${ht}px;`
      +`background:${inWin(h)?'var(--gilt,#c8a45a)':'rgba(255,255,255,.22)'}"></span>`; }).join('');
}
function renderBehavior(d){
  const body=$('#behavior-body');
  if(!d || d.need_rebuild){ body.innerHTML='<p class="empty">No visits yet — they appear here as animals come and go (the visit ledger refreshes automatically when the app stops).</p>'; return; }
  if(!d.visits){ body.innerHTML='<p class="empty">No visits recorded yet.</p>'; return; }
  const flags=(d.flags||[]).filter(f=>f.verdict==='DISAGREES');
  let html=`<div class="tallies">
    <div class="tally" style="cursor:default"><div class="n">${d.visits.toLocaleString()}</div><div class="k lbl">visits</div></div>
    <div class="tally" style="cursor:default"><div class="n">${d.species.length}</div><div class="k lbl">species</div></div>
    <div class="tally" style="cursor:default"><div class="n">${flags.length}</div><div class="k lbl">off-pattern</div></div></div>`;
  if(flags.length){
    html+=`<h2 class="sec">Off-Pattern <span class="n">appearance vs behaviour</span></h2>`;
    html+=`<div class="lede">`+flags.map(f=>{
      // Lead with the note that actually DISAGREES (a dwell-flagged visit's first note can read
      // "arrived 14h ... OK", which looked like a false alarm), and date the sighting.
      const note=(f.notes||[]).find(n=>/UNUSUAL/.test(n))||(f.notes&&f.notes[0])||'unusual';
      return `<span class="flag quiet">⚑ ${esc(nameOf(f.species))} · ${esc((f.started_at||'').slice(5,16).replace('T',' '))} — ${esc(note)}</span>`;
    }).join('')+`</div>`;
  }
  html+=`<h2 class="sec">Field Notes <span class="n">${d.species.length} species</span></h2>`;
  html+=`<div style="display:flex;flex-direction:column;gap:8px">`+d.species.map(s=>{
    const hours=Array.from({length:24},(_,h)=>s.arrival_hours[h]||s.arrival_hours[String(h)]||0);
    const win=s.typical_window;
    const wtxt=win?`${String(win.start_hour).padStart(2,'0')}–${String(win.end_hour).padStart(2,'0')}h`:'—';
    const dwell=s.dwell_median_s>=60?Math.round(s.dwell_median_s/60)+'m':Math.round(s.dwell_median_s)+'s';
    return `<div class="panel" style="display:flex;align-items:center;justify-content:space-between;gap:14px;padding:10px 14px">
      <div><div style="font-weight:600">${esc(nameOf(s.species))}</div>
        <div class="lbl" style="opacity:.72">${s.n_visits} visits · ${s.visits_per_day}/day · dwell ~${dwell} · usually ${wtxt}</div></div>
      <div style="display:flex;align-items:flex-end;gap:1px;height:20px" title="arrivals by hour (0–23h); highlighted = typical window">${behaviorClock(hours,win)}</div>
    </div>`;
  }).join('')+`</div>`;
  if((d.co_occurrence||[]).length){
    html+=`<h2 class="sec">Seen Together <span class="n">who shares a visit</span></h2>`;
    html+=`<div class="lede">`+d.co_occurrence.map(c=>
      `<span class="flag">${esc(nameOf(c.a))} + ${esc(nameOf(c.b))} · ${c.n}</span>`).join('')+`</div>`;
  }
  body.innerHTML=html;
}

/* ---------- individuals (phase 3: suggest-confirm loop + hand-label the clusters) ---------- */
async function loadIndividuals(){
  const body=$('#indiv-body'); body.innerHTML='<p class="empty">Gathering the suspects…</p>';
  let d=null,q=null;
  try{
    [q,d]=await Promise.all([
      fetch('/api/reid/queue').then(r=>r.json()).catch(()=>null),
      fetch('/api/individuals').then(r=>r.json())]);
  }catch(e){ body.innerHTML='<p class="empty">Could not load individuals.</p>'; return; }
  renderIndividuals(d,q);
}

/* The review queue: every recent visit gets a "who is this?" suggestion; you confirm or
   correct. Each confirmation becomes a new appearance template, so suggestions sharpen as the
   cast grows. Before anything is confirmed, name the bootstrap visit-GROUPS instead. */
let REID_BOOT=[];
function reidWhen(ts){ return ts? ts.slice(5,16).replace('T',' ') : '?'; }
function reidInput(id,ph){ return `<input id="${id}" placeholder="${ph}" style="width:110px;padding:5px 8px;background:rgba(0,0,0,.25);border:1px solid rgba(255,255,255,.2);border-radius:4px;color:inherit" onkeydown="if(event.key==='Enter')this.nextElementSibling.click()">`; }
/* Clips that rolled during each queued visit, stashed at render time so a card's "▶ N clips"
   button can replay them in the lightbox without re-fetching. Reset each time the queue renders. */
let REID_VISIT_CLIPS={};
function reidPlayClips(vid){ const c=REID_VISIT_CLIPS[vid]||[]; if(c.length) playClips(c,0,`visit #${vid}`); }
function reidCard(v){
  const mins=v.dwell_s>=90? Math.round(v.dwell_s/60)+' min' : (v.dwell_s||0)+'s';
  const thumb=v.rep_crop? `<img src="/media/${encodeURI(v.rep_crop)}" loading="lazy" style="width:84px;height:84px;object-fit:cover;border-radius:6px">` : '';
  const multiEv=[v.co_present_frames? `${v.co_present_frames} still frame(s)`:'', v.co_present_clips? `${v.co_present_clips} clip(s)`:''].filter(Boolean).join(' + ');
  const multi=v.multi? `<span class="flag" style="background:rgba(255,170,60,.16);border-color:rgba(255,170,60,.4)" title="${multiEv||'two animals'} show two animals at once — co-arrival is behaviour signal; the appearance suggestion is a blend">2+ animals</span>` : '';
  let sugg='', act='';
  if(v.confirmed_as){
    sugg=`<span class="flag" style="background:rgba(90,200,120,.15);border-color:rgba(90,200,120,.45)">= ${esc(v.confirmed_as)} ✓</span>`;
    act=`<button class="gear" onclick="reidConfirm(${v.visit_id},null,true)" title="unconfirm this visit">Clear</button>`;
  }else if(v.auto_as){
    // Named by the nightly auto-assign pass (high similarity + clear margin). Review by
    // exception: ✓ promotes it to a real confirmation (feeds templates); ✗ clears it AND pins
    // the visit so the next nightly run leaves it alone (the reject tombstone).
    const top=(v.candidates||[])[0];
    const pct=(top&&top.name===v.auto_as)? ` ${Math.round(top.similarity*100)}%` : '';
    sugg=`<span class="flag" style="background:rgba(120,200,255,.13);border-color:rgba(120,200,255,.42)" title="named automatically by the nightly pass — it cleared both the similarity and the runner-up-margin bar. Auto names never feed the matching templates until you ✓ them.">auto: <b>${esc(v.auto_as)}</b>${pct}</span>`;
    act=`<button class="gear" onclick="reidConfirm(${v.visit_id},${jarg(v.auto_as)})" title="yes, it's ${esc(v.auto_as)} — promote to a confirmed template">✓ keep</button>
         <button class="gear" onclick="reidConfirm(${v.visit_id},null,true,true)" title="not ${esc(v.auto_as)} — clear the auto name; the nightly pass won't re-name this visit">✗ not them</button>
         ${reidInput('rq-'+v.visit_id,'or who…')}<button class="gear" onclick="reidConfirm(${v.visit_id})">Name</button>`;
  }else if((v.candidates||[]).length){
    const top=v.candidates[0];
    const rest=v.candidates.slice(1).map(c=>`${esc(c.name)} ${Math.round(c.similarity*100)}%`).join(' · ');
    sugg=`<span class="flag" title="nearest confirmed visit: #${top.via_visit} (${reidWhen(top.via_started)})">looks like <b>${esc(top.name)}</b> ${Math.round(top.similarity*100)}%</span>`
        +(v.novel? `<span class="flag" style="background:rgba(255,120,90,.14);border-color:rgba(255,120,90,.4)" title="best match is below the novelty threshold">possibly someone new</span>`:'')
        +(rest? `<span class="lbl" style="opacity:.6">${rest}</span>`:'');
    act=`<button class="gear" onclick="reidConfirm(${v.visit_id},${jarg(top.name)})" title="yes, it's ${esc(top.name)}">✓ ${esc(top.name)}</button>
         ${reidInput('rq-'+v.visit_id,'or who…')}<button class="gear" onclick="reidConfirm(${v.visit_id})">Name</button>`;
  }else{
    sugg=`<span class="lbl" style="opacity:.65">${esc(v.note||'no suggestion yet')}</span>`;
    act=`${reidInput('rq-'+v.visit_id,'who is this…')}<button class="gear" onclick="reidConfirm(${v.visit_id})">Name</button>`;
  }
  // Clip-space match: a SEPARATE signal from the un-blended tracklets — the only way a never-solo
  // pair member (Elliot) gets named in a new visit. Shown distinctly; offers a confirm when it's
  // the only suggestion on offer.
  const clipTop=(v.clip_candidates||[])[0];
  const clipSugg=clipTop?`<span class="flag" style="background:rgba(120,160,220,.16);border-color:rgba(120,160,220,.45)" title="clip-space appearance match (from un-blended individuals) — a separate signal from the still match">clip-match <b>${esc(clipTop.name)}</b> ${Math.round(clipTop.similarity*100)}%</span>`:'';
  if(clipTop && !v.confirmed_as && !(v.candidates||[]).length){
    act=`<button class="gear" onclick="reidConfirm(${v.visit_id},${jarg(clipTop.name)})" title="confirm from the clip-space match">✓ ${esc(clipTop.name)}</button> `+act;
  }
  const sp=v.species||'raccoon', spId='rqsp-'+v.visit_id;
  const spRow=`<div style="display:flex;gap:6px;align-items:center;margin-top:8px;border-top:1px solid var(--rule);padding-top:8px;flex-wrap:wrap">
    <span class="lbl" style="opacity:.72">species: <b>${esc(nameOf(sp))}</b></span>
    <button class="gear" onclick="postVisitLabel({visit_id:${v.visit_id}},{verify:true},loadIndividuals)" title="confirm this species for the whole visit">✓</button>
    ${speciesSelect(spId,sp)}
    <button class="gear" onclick="visitSpeciesCorrect({visit_id:${v.visit_id}},'${spId}',loadIndividuals)">correct</button></div>`;
  const ub=v.multi?`<div style="margin-top:8px"><button class="gear" onclick="toggleUnblend(${v.visit_id})" title="separate the animals (from the clips, or the stills when there are none), then name each">⚖ un-blend the animals</button>
    <div id="ub-${v.visit_id}" style="display:none;margin-top:8px"></div></div>`:'';
  // Naming evidence: a one-click "play this visit's clips" button + a strip of its sharpest crops
  // (each enlarges via the global /media/ lightbox), so the eye has more than the one hero angle.
  REID_VISIT_CLIPS[v.visit_id]=v.clips||[];
  const nclip=(v.clips||[]).length;
  const clipBtn=nclip?`<button class="gear" onclick="reidPlayClips(${v.visit_id})" title="watch the clip${nclip>1?'s':''} that rolled during this visit">▶ ${nclip} clip${nclip>1?'s':''}</button>`:'';
  const cropTiles=(v.crops||[]).map(c=>`<img src="/media/${encodeURI(c)}" loading="lazy" title="click to enlarge" style="width:58px;height:58px;object-fit:cover;border-radius:4px;cursor:zoom-in">`).join('');
  const strip=(nclip||cropTiles)?`<div style="display:flex;gap:5px;align-items:center;flex-wrap:wrap;margin-top:8px">${clipBtn}${cropTiles}</div>`:'';
  return `<div class="panel" style="padding:10px 14px">
    <div style="display:flex;align-items:center;gap:12px;flex-wrap:wrap">
      ${thumb}
      <div style="min-width:150px"><div style="font-weight:600">visit <span style="opacity:.7">#${v.visit_id}</span> · ${reidWhen(v.started_at)}</div>
        <div class="lbl" style="opacity:.72">${mins} · ${v.n_crops} crops · ${v.n_embedded} embedded</div></div>
      <div style="display:flex;gap:6px;align-items:center;flex-wrap:wrap;flex:1">${sugg} ${clipSugg} ${multi}</div>
      <div style="display:flex;gap:6px;align-items:center">${act}</div>
    </div>
    ${strip}
    <div data-vm="${esc(v.visit_id)}"></div>
    ${spRow}
    ${ub}
  </div>`;
}
/* Un-blend: cluster a multi-animal visit's tracklets into its animals, name each. Two bases,
   same card: the CLIPS basis clusters clip tracklets (labels land on the tracklets, clip-space);
   the STILLS basis (d.basis==='stills', when the clips can't separate) chains the visit's saved
   crops into per-animal tracklets and labels stamp detection ids directly. */
let REID_UNBLEND={};
async function toggleUnblend(vid){
  const box=document.getElementById('ub-'+vid); if(!box) return;
  if(box.style.display!=='none'){ box.style.display='none'; return; }
  box.style.display=''; await renderUnblend(vid);
}
async function renderUnblend(vid){
  const box=document.getElementById('ub-'+vid); if(!box) return;
  box.innerHTML='<p class="lbl" style="opacity:.7">Separating the animals…</p>';
  let d; try{ d=await fetch('/api/reid/unblend?visit_id='+vid).then(r=>r.json()); }
  catch(e){ box.innerHTML='<p class="lbl">Could not un-blend.</p>'; return; }
  REID_UNBLEND[vid]=d.groups||[];
  if(!REID_UNBLEND[vid].length){ box.innerHTML=`<p class="lbl" style="opacity:.7">${esc(d.note||'nothing to separate yet — needs clip tracklets (clipmotion + clipembed) or embedded still crops (embed.py --co-present)')}</p>`; return; }
  const co=(d.co_present&&d.co_present.names)||[];
  const anySugg=REID_UNBLEND[vid].some(g=>(g.suggestion||[]).length);
  const hint=co.length>=2
    ? (co.length===2
      ? `The two biggest groups are the pair you logged. ✓ a suggested name and the other is filled in by elimination — or just tap a name onto each.`
      : `The biggest groups are the ${co.length} animals you logged. ✓ what the matcher resolves; tap the rest on by eye.`)
    : anySugg
    ? `The two biggest groups are usually the pair. ✓ a match to confirm, or correct it — each label sharpens the next visit.`
    : `The biggest groups are usually the animals — name each clean single animal. Once each is named once, future multi visits will <b>auto-suggest</b> them.`;
  // If a human logged who was here, surface that pair up top — it's what drives the one-click
  // assign + elimination on the groups below.
  const coLog=co.length>=2
    ? `<div class="ub-colog">📓 You logged <b>${co.map(n=>esc(cap1(n))).join(' + ')}</b> here together${d.co_present.observed_at?` · ${esc(fmtClock(d.co_present.observed_at)||'')}`:''}.</div>`
    : '';
  box.innerHTML=`<div class="lbl" style="opacity:.75;margin-bottom:6px">${d.n_tracklets} ${d.basis==='stills'?'still':'clip'} tracklet(s) → ${REID_UNBLEND[vid].length} group(s). ${hint}</div>`
    +coLog
    +REID_UNBLEND[vid].map((g,i)=>reidUnblendGroup(vid,g,i)).join('');
}
function reidUnblendGroup(vid,g,i){
  const thumbs=(g.rep_crops||[]).slice(0,8).map(c=>
    `<img src="/media/${encodeURI(c)}" loading="lazy" style="width:58px;height:58px;object-fit:cover;border-radius:4px">`).join('')
    || '<span class="lbl" style="opacity:.5">no thumbnails yet — they appear once this visit&rsquo;s clips or crops have been processed.</span>';
  const lab=g.label?`<span class="flag" style="background:rgba(90,200,120,.15);border-color:rgba(90,200,120,.45)">= ${esc(g.label)} ✓</span>`:'';
  const iid='ubn-'+vid+'-'+i;
  const top=(g.suggestion||[])[0];
  const sugg=(!g.label&&top)?`<button class="gear" onclick="reidUnblendConfirm(${vid},${i},${jarg(top.name)})" title="clip-space match — confirm this group is ${esc(top.name)}">✓ ${esc(top.name)} ${Math.round(top.similarity*100)}%</button>`:'';
  // Elimination from the co-presence log: the OTHER cluster matched, so by your logged pair this
  // one must be ${g.co_elim}. The recommended one-click — distinct from a raw appearance match.
  const elim=(!g.label&&g.co_elim)?`<button class="gear ub-elim" onclick="reidUnblendConfirm(${vid},${i},${jarg(g.co_elim)})" title="from your co-presence log — the other group is the pair member, so this one is ${esc(g.co_elim)}">★ ${esc(cap1(g.co_elim))} · from your log</button>`:'';
  // Quick-pick the logged pair onto this cluster (no typing), minus any name already offered above.
  const quick=(!g.label?(g.co_names||[]):[]).filter(n=>n!==g.co_elim && !(top&&n===top.name))
    .map(n=>`<button class="gear" onclick="reidUnblendConfirm(${vid},${i},${jarg(n)})" title="you logged this pair as here together">＋ ${esc(cap1(n))}</button>`).join('');
  return `<div class="panel" style="display:flex;align-items:center;gap:10px;padding:8px 10px;margin-bottom:6px;flex-wrap:wrap">
    <div style="min-width:96px"><div style="font-weight:600">group ${i+1}</div><div class="lbl" style="opacity:.7">${g.n} tracklet(s)${g.n_crops?` · ${g.n_crops} crop(s)`:''} · coh ${g.cohesion}</div></div>
    <div style="display:flex;gap:3px;flex-wrap:wrap;flex:1">${thumbs}</div>
    <div style="display:flex;gap:6px;align-items:center;flex-wrap:wrap">${lab}${elim}${sugg}${quick}${reidInput(iid,'name…')}<button class="gear" onclick="reidUnblendLabel(${vid},${i})">Name</button></div>
  </div>`;
}
/* Which ids this group's label lands on: clip tracklets (clips basis) or detection ids (stills). */
function unblendBody(g,name){
  return JSON.stringify(g.track_ids?{track_ids:g.track_ids,name}:{detection_ids:g.detection_ids,name});
}
function reidUnblendConfirm(vid,i,name){
  const g=(REID_UNBLEND[vid]||[])[i]; if(!g) return;
  const restore=busyBtn();   // the clicked ✓/★/＋ suggestion button
  fetch('/api/reid/unblend/label',{method:'POST',headers:{'Content-Type':'application/json'},
    body:unblendBody(g,name)}).then(r=>r.json()).then(r=>{
      if(r.error){ restore(); alert(r.error); return; } renderUnblend(vid);
    }).catch(e=>{ restore(); connFail(); });
}
async function reidUnblendLabel(vid,i){
  const restore=busyBtn();   // the clicked "Name" button (grab before await)
  const inp=document.getElementById('ubn-'+vid+'-'+i); const name=(inp&&inp.value||'').trim();
  if(!name){ restore(); if(inp) inp.focus(); return; }
  const g=(REID_UNBLEND[vid]||[])[i]; if(!g){ restore(); return; }
  try{
    const r=await fetch('/api/reid/unblend/label',{method:'POST',headers:{'Content-Type':'application/json'},
      body:unblendBody(g,name)}).then(r=>r.json());
    if(r.error){ restore(); alert(r.error); return; }
    renderUnblend(vid);
  }catch(e){ restore(); connFail(); }
}
function reidBootCard(g,i){
  const span=`${reidWhen(g.started[0])} → ${reidWhen(g.started[g.started.length-1])}`;
  const nmulti=(g.multi||[]).filter(Boolean).length;
  const thumbs=(g.crops||[]).slice(0,6).map(c=>
    `<img src="/media/${encodeURI(c)}" loading="lazy" style="width:64px;height:64px;object-fit:cover;border-radius:4px">`).join('');
  return `<div class="panel" style="display:flex;align-items:center;gap:12px;padding:10px 14px;flex-wrap:wrap">
    <div style="min-width:170px"><div style="font-weight:600">look-alike group ${i+1}</div>
      <div class="lbl" style="opacity:.72">${g.visits.length} visit(s) · ${span} · cohesion ${g.cohesion}${nmulti? ` · ${nmulti}× 2+ animals`:''}</div></div>
    <div style="display:flex;gap:4px;flex-wrap:wrap;flex:1">${thumbs}</div>
    <div style="display:flex;gap:6px;align-items:center">${reidInput('rg-'+i,'name this individual…')}<button class="gear" onclick="reidNameGroup(${i})">Name all</button></div>
  </div>`;
}
/* Re-fit: once a cast exists, sort the unconfirmed remainder into "looks like <name>" buckets
   (bulk-confirm) and candidate-new-individual groups, and flag anyone named only on a pair visit. */
let REID_REFIT=null;
function reidFitCard(name,bucket){
  const vs=bucket.visits||[];
  const thumbs=vs.slice(0,8).map(x=>x.rep_crop?
    `<img src="/media/${encodeURI(x.rep_crop)}" loading="lazy" title="visit #${x.visit_id} · ${Math.round(x.similarity*100)}%" style="width:58px;height:58px;object-fit:cover;border-radius:4px">`:'').join('');
  const lo=Math.round(vs[vs.length-1].similarity*100), hi=Math.round(vs[0].similarity*100);
  return `<div class="panel" style="display:flex;align-items:center;gap:12px;padding:10px 14px;flex-wrap:wrap">
    <div style="min-width:150px"><div style="font-weight:600">looks like ${esc(name)}</div>
      <div class="lbl" style="opacity:.72">${vs.length} unconfirmed visit(s) · ${lo}–${hi}% match</div></div>
    <div style="display:flex;gap:4px;flex-wrap:wrap;flex:1">${thumbs}</div>
    <div style="display:flex;gap:6px;align-items:center">
      <button class="gear" onclick="reidConfirmFit(${jarg(name)})" title="confirm all ${vs.length} visits as ${esc(name)}">✓ All ${vs.length} as ${esc(name)}</button></div>
  </div>`;
}
function reidNovelCard(g,i){
  const span=`${reidWhen(g.started[0])} → ${reidWhen(g.started[g.started.length-1])}`;
  const nmulti=(g.multi||[]).filter(Boolean).length;
  const thumbs=(g.crops||[]).slice(0,8).map(c=>
    `<img src="/media/${encodeURI(c)}" loading="lazy" style="width:58px;height:58px;object-fit:cover;border-radius:4px">`).join('');
  return `<div class="panel" style="display:flex;align-items:center;gap:12px;padding:10px 14px;flex-wrap:wrap">
    <div style="min-width:170px"><div style="font-weight:600">possible new individual ${i+1}</div>
      <div class="lbl" style="opacity:.72">${g.visits.length} visit(s) · ${span} · cohesion ${g.cohesion}${nmulti?` · ${nmulti}× 2+ animals`:''}</div></div>
    <div style="display:flex;gap:4px;flex-wrap:wrap;flex:1">${thumbs}</div>
    <div style="display:flex;gap:6px;align-items:center">${reidInput('rn-'+i,'name…')}<button class="gear" onclick="reidNameNovel(${i})">Name all</button></div>
  </div>`;
}
function reidRefitHTML(refit){
  if(!refit) return '';
  const fitNames=Object.keys(refit.fits||{});
  if(!fitNames.length && !(refit.novel_groups||[]).length && !(refit.untemplated||[]).length) return '';
  let html=`<h2 class="sec">Fit to the Cast <span class="n">sort the rest by who they resemble — confirm in bulk</span></h2>`;
  if((refit.untemplated||[]).length){
    html+=`<p class="lbl" style="margin:2px 0 10px;color:var(--rust)">⚠ ${refit.untemplated.map(esc).join(', ')} ${refit.untemplated.length>1?'were':'was'} confirmed only on a <b>multi-animal visit</b>, so the appearance template is a blend of two raccoons and can't be matched. Confirm a <b>solo</b> visit for ${refit.untemplated.length>1?'each':'them'} (find one in the queue below without a “2+ raccoons” badge) to make ${refit.untemplated.length>1?'them':'it'} findable.</p>`;
  }
  if(fitNames.length){
    html+=`<div style="display:flex;flex-direction:column;gap:8px;margin-bottom:12px">${
      fitNames.sort((a,b)=>refit.fits[b].visits.length-refit.fits[a].visits.length)
        .map(n=>reidFitCard(n,refit.fits[n])).join('')}</div>`;
  }
  if((refit.novel_groups||[]).length){
    html+=`<p class="lbl" style="opacity:.75;margin:8px 0 6px">Looks like nobody on file yet — ${refit.n_novel} visit(s) clustered into candidate new individuals:</p>`;
    html+=`<div style="display:flex;flex-direction:column;gap:8px;margin-bottom:14px">${refit.novel_groups.map(reidNovelCard).join('')}</div>`;
  }
  return html;
}
function reidQueueHTML(q){
  if(!q||(!q.queue.length&&!q.bootstrap.length&&!q.refit)) return '';
  REID_VISIT_CLIPS={};
  let html=`<h2 class="sec">Who Is This? <span class="n">confirm or correct — each answer sharpens the next guess</span></h2>`;
  if(q.unembedded>0) html+=`<p class="lbl" style="opacity:.7;margin:2px 0 10px">⚠ ${q.unembedded} recent crops aren't analysed for appearance yet — naming suggestions sharpen once the re-ID step has run (see the README's “Individual re-identification”).</p>`;
  if(q.bootstrap.length){
    html+=`<p class="lbl" style="opacity:.75;margin:4px 0 10px">Nothing confirmed yet, so here are the corpus' look-alike <b>visit groups</b> (each is probably one animal — your eye decides; skip the 2+-animal ones first pass). Naming a group confirms every visit in it.</p>`;
    html+=`<div style="display:flex;flex-direction:column;gap:8px;margin-bottom:14px">${q.bootstrap.map(reidBootCard).join('')}</div>`;
  }
  html+=reidRefitHTML(q.refit);
  if(q.queue.length){
    const cast=(q.cast||[]).map(c=>`<span class="flag"${c.n_auto?` title="${c.n_auto} recent visit${c.n_auto>1?'s':''} auto-named ${esc(c.name)} by the nightly pass — ✓/✗ them on the cards below"`:''}>${esc(c.name)} · ${c.n_visits} visit${c.n_visits>1?'s':''}${c.n_auto?` <span style="opacity:.65">+${c.n_auto} auto</span>`:''}</span>`).join(' ');
    if(cast) html+=`<h2 class="sec">Visit-by-Visit <span class="n">${q.queue.length} recent</span></h2><div class="lede" style="margin-bottom:8px">The cast so far: ${cast}</div>`;
    html+=`<div style="display:flex;flex-direction:column;gap:8px">${q.queue.map(reidCard).join('')}</div>`;
  }
  return html;
}
/* Shared per-visit labelling (species confirm/correct + individual name), used by BOTH the
   "Who is this?" queue (targets a visit_id) and the Explorer Visits list (targets a
   source+start+end span). One backend: POST /api/visit/label. */
function speciesSelect(id, current){
  const opts=['<option value="">— correct species… —</option>']
    .concat(LABELS.map(l=>`<option value="${esc(l)}"${l===current?' selected':''}>${esc(cap1(l))}</option>`),
            ['<option value="__other__">+ other…</option>']).join('');
  return `<select id="${id}" style="padding:4px 6px;background:rgba(0,0,0,.25);border:1px solid rgba(255,255,255,.2);border-radius:4px;color:inherit;font:inherit;max-width:150px">${opts}</select>`;
}
async function postVisitLabel(target, body, after){
  const restore=busyBtn();   // dim the clicked confirm/correct/Name button while it saves
  try{
    const r=await fetch('/api/visit/label',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify(Object.assign({}, target, body))}).then(r=>r.json());
    if(r.error){ restore(); alert(r.error); return; }
    restore();
    if(after) after(r);
  }catch(e){ restore(); connFail(); }
}
function visitSpeciesCorrect(target, selectId, after){
  const sel=document.getElementById(selectId); if(!sel) return;
  let sp=sel.value;
  if(sp==='__other__'){ sp=(prompt('New species label (e.g. striped skunk):')||'').trim(); }
  if(!sp) return;
  postVisitLabel(target, {species:sp}, after);
}
async function reidConfirmMany(visitIds,name){
  try{
    for(const vid of visitIds)
      await fetch('/api/reid/confirm',{method:'POST',headers:{'Content-Type':'application/json'},
        body:JSON.stringify({visit_id:vid,name})});
    loadIndividuals();
  }catch(e){ alert('Could not save: '+e); }
}
async function reidConfirmFit(name){
  const b=REID_REFIT&&REID_REFIT.fits&&REID_REFIT.fits[name]; if(!b) return;
  const ids=b.visits.map(x=>x.visit_id);
  if(!confirm(`Confirm all ${ids.length} visits as ${name}?`)) return;
  reidConfirmMany(ids,name);
}
async function reidNameNovel(i){
  const inp=document.getElementById('rn-'+i);
  const name=(inp&&inp.value||'').trim();
  if(!name){ if(inp) inp.focus(); return; }
  const g=REID_REFIT&&REID_REFIT.novel_groups&&REID_REFIT.novel_groups[i]; if(!g) return;
  reidConfirmMany(g.visits,name);
}
async function reidConfirm(vid,name,clear,reject){
  const restore=busyBtn();   // the clicked ✓/Name/Clear button — grab it before any await/confirm
  if(clear&&!confirm(reject?'Clear this auto-name? The nightly pass won\'t re-name this visit.':'Unconfirm this visit?')){ restore(); return; }
  if(!clear&&!name){
    const inp=document.getElementById('rq-'+vid);
    name=(inp&&inp.value||'').trim();
    if(!name){ restore(); if(inp) inp.focus(); return; }
  }
  try{
    const r=await fetch('/api/reid/confirm',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({visit_id:vid,name:clear?null:name,reject:!!reject})}).then(r=>r.json());
    if(r.error){ restore(); alert(r.error); return; }
    loadIndividuals();   // re-renders the whole list (button goes away with it); no restore needed
  }catch(e){ restore(); connFail(); }
}
async function reidNameGroup(i){
  const inp=document.getElementById('rg-'+i);
  const name=(inp&&inp.value||'').trim();
  if(!name){ if(inp) inp.focus(); return; }
  const g=REID_BOOT[i]; if(!g) return;
  try{
    for(const vid of g.visits)
      await fetch('/api/reid/confirm',{method:'POST',headers:{'Content-Type':'application/json'},
        body:JSON.stringify({visit_id:vid,name})});
    loadIndividuals();
  }catch(e){ alert('Could not save: '+e); }
}
function indivRow(g){
  const span=(g.first_seen&&g.last_seen)?`${g.first_seen.slice(5,10)} → ${g.last_seen.slice(5,10)}`:'';
  const thumbs=(g.crops||[]).map(c=>
    `<img src="/media/${encodeURI(c)}" loading="lazy" style="width:64px;height:64px;object-fit:cover;border-radius:4px">`).join('');
  // onclick string args go through jarg (JSON.stringify + esc), NOT '${esc(x)}': esc does not
  // neutralize a single quote, so a name with a ' would break out of the JS string literal and
  // inject code. See the esc/jarg note at the top of this file.
  const act = g.placeholder
    ? `<input id="nm-${esc(g.id)}" placeholder="name… (e.g. Notch)" style="width:130px;padding:5px 8px;background:rgba(0,0,0,.25);border:1px solid rgba(255,255,255,.2);border-radius:4px;color:inherit">
       <button class="gear" onclick="nameIndiv(${jarg(g.id)})">Name</button>`
    : `<button class="gear" onclick="togglePoses(${jarg(g.id)})" title="cluster this individual's crops into characteristic poses">Poses</button>
       <button class="gear" onclick="toggleClips(${jarg(g.id)})" title="watch this individual's behaviour clips">Clips</button>
       <button class="gear" onclick="nameIndiv(${jarg(g.id)})" title="rename">Rename</button>
       <button class="gear" onclick="clearIndiv(${jarg(g.id)})" title="unassign these crops">Clear</button>
       <input id="nm-${esc(g.id)}" placeholder="new name…" style="width:90px;padding:5px 8px;background:rgba(0,0,0,.25);border:1px solid rgba(255,255,255,.2);border-radius:4px;color:inherit">`;
  return `<div class="panel" style="padding:10px 14px">
    <div style="display:flex;align-items:center;gap:12px;flex-wrap:wrap">
      <div style="min-width:150px"><div style="font-weight:600">${g.placeholder?'<span style="opacity:.65">'+esc(g.id)+'</span>':esc(g.id)}</div>
        <div class="lbl" style="opacity:.72">${esc(nameOf(g.species||''))} · ${g.n_crops} crops · ${esc(span)}</div>
        ${g.placeholder?'':`<div class="motion-fp" data-im="${esc(g.id)}"></div>`}</div>
      <div style="display:flex;gap:4px;flex-wrap:wrap;flex:1">${thumbs}</div>
      <div style="display:flex;gap:6px;align-items:center">${act}</div>
    </div>
    ${g.placeholder?'':`<div id="poses-${esc(g.id)}" style="display:none;margin-top:10px;border-top:1px solid var(--rule);padding-top:10px"></div>
    <div id="clips-${esc(g.id)}" style="display:none;margin-top:10px;border-top:1px solid var(--rule);padding-top:10px"></div>`}
  </div>`;
}
async function toggleClips(name){
  const box=document.getElementById('clips-'+name); if(!box) return;
  if(box.style.display!=='none'){ box.style.display='none'; return; }
  box.style.display=''; box.innerHTML='<p class="lbl" style="opacity:.7">Gathering '+esc(name)+"'s clips…</p>";
  let d; try{ d=await fetch('/api/reid/clips?individual='+encodeURIComponent(name)).then(r=>r.json()); }
  catch(e){ box.innerHTML='<p class="lbl">Could not load clips.</p>'; return; }
  const clips=d.clips||[];
  if(!clips.length){ box.innerHTML='<p class="lbl" style="opacity:.7">No clips overlap '+esc(name)+"'s visits yet.</p>"; return; }
  box.innerHTML=`<div class="lbl" style="opacity:.75;margin-bottom:8px">${esc(name)}'s footage — ${d.n_clips} clip(s), newest first. ⚠ marks clips during a 2+-animal visit (shows ${esc(name)} with another).</div>`
    +`<div style="display:flex;flex-wrap:wrap;gap:12px">`+clips.map(c=>
      `<div style="flex:0 0 auto;width:240px"><video src="/media/${encodeURI(c.clip_path)}" controls preload="metadata" style="width:240px;border-radius:5px;background:#000"></video>`
      +`<div class="lbl" style="opacity:.6;font-family:var(--mono);font-size:11px;margin-top:2px">${reidWhen(c.started_at)}${c.duration_s?` · ${Math.round(c.duration_s)}s`:''}${c.multi?' · ⚠ 2+':''}</div></div>`).join('')
    +`</div>`;
}
async function togglePoses(name){
  const box=document.getElementById('poses-'+name); if(!box) return;
  if(box.style.display!=='none'){ box.style.display='none'; return; }
  box.style.display=''; box.innerHTML='<p class="lbl" style="opacity:.7">Clustering '+esc(name)+"'s poses…</p>";
  let d; try{ d=await fetch('/api/reid/poses?individual='+encodeURIComponent(name)).then(r=>r.json()); }
  catch(e){ box.innerHTML='<p class="lbl">Could not load poses.</p>'; return; }
  const poses=d.poses||[];
  if(!poses.length){ box.innerHTML='<p class="lbl" style="opacity:.7">Not enough embedded crops to cluster poses yet.</p>'; return; }
  box.innerHTML=`<div class="lbl" style="opacity:.75;margin-bottom:8px">${esc(name)}'s characteristic poses — ${d.n_groups} group(s) of crops that share a body posture/viewpoint (the biggest ${poses.length} shown):</div>`
    +`<div style="display:flex;flex-wrap:wrap;gap:14px">`+poses.map((p,i)=>
      `<div style="flex:0 0 auto"><div class="lbl" style="opacity:.6;font-family:var(--mono);font-size:11px;margin-bottom:3px">pose ${i+1} · ${p.n} crops</div>`
      +`<div style="display:flex;gap:3px;flex-wrap:wrap;max-width:300px">`+(p.rep_crops||[]).map(c=>
        `<img src="/media/${encodeURI(c)}" loading="lazy" style="width:60px;height:60px;object-fit:cover;border-radius:4px">`).join('')+`</div></div>`).join('')
    +`</div>`;
}
function renderIndividuals(d,q){
  const body=$('#indiv-body');
  REID_BOOT=(q&&q.bootstrap)||[];
  REID_REFIT=(q&&q.refit)||null;
  const queueHTML=reidQueueHTML(q);
  const groups=(d&&d.groups)||[];
  if(!groups.length&&!queueHTML){ body.innerHTML='<p class="empty">No individuals to name yet. As more animals visit, look-alike groups will appear here for you to name — this is the slowest part, since it needs a good number of clear photos first.</p>'; return; }
  if(!groups.length){ body.innerHTML=queueHTML; wireVisitMotion(body); return; }
  const named=groups.filter(g=>!g.placeholder), prop=groups.filter(g=>g.placeholder);
  let html=queueHTML;
  if(named.length){
    html+=`<h2 class="sec">The Cast <span class="n">${named.length} named</span></h2>`;
    html+=`<div style="display:flex;flex-direction:column;gap:8px">${named.map(indivRow).join('')}</div>`;
  }
  html+=`<h2 class="sec">Crop-Level Look-Alike Clusters <span class="n">${prop.length} proposed — the visit queue above is the sharper tool</span></h2>`;
  html+=`<p class="lbl" style="opacity:.75;margin:4px 0 14px">Each group is a set of crops that LOOK alike (appearance clustering — proposals, not certainties).
    Naming a group adds it to the cast; giving two groups the same name merges them. Through-glass appearance is weak — when unsure, leave it unnamed.</p>`;
  html+=`<div style="display:flex;flex-direction:column;gap:8px">${prop.slice(0,30).map(indivRow).join('')}</div>`;
  if(prop.length>30) html+=`<p class="empty">…and ${prop.length-30} smaller groups (name the big ones first).</p>`;
  body.innerHTML=html;
  wireVisitMotion(body);   // lazily fill the per-visit motion strips on the "Who Is This?" cards
  wireIndivMotion(body);   // and the per-individual motion fingerprint on the Cast rows
}
async function nameIndiv(from){
  const inp=document.getElementById('nm-'+from);
  const to=(inp&&inp.value||'').trim();
  if(!to){ if(inp) inp.focus(); return; }
  await postIndiv(from,to);
}
async function clearIndiv(from){
  if(!confirm(`Unassign all crops labelled "${from}"?`)) return;
  await postIndiv(from,null);
}
async function postIndiv(from,to){
  const restore=busyBtn();   // dim the clicked Rename/Clear button while it saves
  try{
    const r=await fetch('/api/individual',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({from,to})}).then(r=>r.json());
    if(r.error){ restore(); alert(r.error); return; }
    loadIndividuals();   // re-renders; button removed
  }catch(e){ restore(); connFail(); }
}

async function loadCalendar(){
  let s; try{ s=await fetch('/api/stats').then(r=>r.json()); }
  catch(e){ $('#calendar-body').innerHTML='<p class="empty">Could not load the calendar.</p>'; return; }
  calByDay={}; (s.by_day||[]).forEach(d=>{ calByDay[d.day]=d; });
  if(calYear==null){
    const days=Object.keys(calByDay).sort();
    const base=days.length?new Date(days[days.length-1]+'T12:00:00'):new Date();
    calYear=base.getFullYear(); calMonth=base.getMonth();
  }
  renderCalendar();
}
function calNav(delta){ calMonth+=delta; if(calMonth<0){calMonth=11;calYear--;} else if(calMonth>11){calMonth=0;calYear++;} renderCalendar(); }
function glyphsForDay(entry){
  if(!entry||!entry.classes) return '';
  return Object.entries(entry.classes).filter(([sp])=>!NONCRITTER.has(sp.toLowerCase()))
    .sort((a,b)=>b[1]-a[1]).slice(0,4)
    .map(([sp,nn])=>{const g=glyphInfo(sp);return `<span class="gl" title="${esc(nameOf(sp))} · ${nn}">${glyphHTML(g)}${nn>1?`<i>${nn>99?'99+':nn}</i>`:''}</span>`;}).join('');
}
function renderCalendar(){
  const body=$('#calendar-body');
  const first=new Date(calYear,calMonth,1);
  const start=new Date(calYear,calMonth,1); start.setDate(1-first.getDay());   // back to Sunday
  const monthName=first.toLocaleDateString(undefined,{month:'long',year:'numeric'});
  const todayStr=ymdLocal(new Date());
  const wd=['Sun','Mon','Tue','Wed','Thu','Fri','Sat'].map(w=>`<div class="cal-wd">${w}</div>`).join('');
  const monthSpecies={};
  let cells='';
  for(let i=0;i<42;i++){
    const dt=new Date(start); dt.setDate(start.getDate()+i);
    const ymd=ymdLocal(dt), inMonth=dt.getMonth()===calMonth, entry=calByDay[ymd];
    if(inMonth&&entry&&entry.classes) Object.entries(entry.classes).forEach(([sp,nn])=>{ if(!NONCRITTER.has(sp.toLowerCase())) monthSpecies[sp]=(monthSpecies[sp]||0)+nn; });
    cells+=`<div class="cal-cell${inMonth?'':' out'}${ymd===todayStr?' today':''}${entry&&inMonth?' has':''}"${entry&&inMonth?` data-day="${ymd}"`:''}>
      <div class="cal-d">${dt.getDate()}</div>
      ${entry&&inMonth?`<div class="cal-v" title="${entry.visits||0} visits">${entry.visits||0}</div>`:''}
      <div class="cal-gl">${inMonth?glyphsForDay(entry):''}</div>
    </div>`;
  }
  const legend=Object.entries(monthSpecies).sort((a,b)=>b[1]-a[1])
    .map(([sp])=>{const g=glyphInfo(sp);return `<span class="li"><b>${glyphHTML(g)}</b> ${esc(nameOf(sp))}</span>`;}).join('');
  body.innerHTML=`
    <div class="cal-head">
      <button class="nav" onclick="calNav(-1)" title="Previous month">‹</button>
      <div class="mo">${esc(monthName)}</div>
      <button class="nav" onclick="calNav(1)" title="Next month">›</button>
    </div>
    <div class="cal-grid">${wd}${cells}</div>
    ${legend?`<div class="cal-legend">${legend}</div>`:'<p class="empty" style="padding:16px 2px">No visitors recorded this month.</p>'}`;
  body.querySelectorAll('[data-day]').forEach(el=>el.onclick=()=>explore('day',{date:el.dataset.day},fmtDay(el.dataset.day)));
}

/* ---------- settings popout (scoped to the selected camera) ---------- */
function openSettings(source){ if(source) selectCamera(source); const m=$('#settings'); if(m) m.hidden=false; refreshControls(); }
function closeSettings(){ const m=$('#settings'); if(m) m.hidden=true; }
document.addEventListener('keydown',e=>{ if(e.key==='Escape') closeSettings(); });

/* ---------- boot ---------- */
buildControls();
async function refreshNaming(){
  let n; try{ n=await fetch('/api/naming').then(r=>r.json()); connOK(); }catch(e){ connFail(); return; }
  const el=document.getElementById('naming'); if(!el) return;
  const map={ loading:['#d9a23b','Identifier: warming up…'],
              ready:['#5b8c5a','Identifier: on'],
              stopped:['#b4503f','Identifier: stopped'] };
  const m=map[n.state];
  if(!m){ el.style.display='none'; return; }
  el.style.display='';
  el.innerHTML='<span style="color:'+m[0]+'">●</span> '+m[1];
  el.title = n.state==='loading'
      ? 'The species identifier is loading its model (about a minute). New visitors get named once it is ready.'
      : n.state==='stopped' ? 'The species identifier is not running right now.'
      : 'New visitors are being named automatically.';
}
/* ---------- first-run orientation: shown only while the database is still empty ---------- */
let __firstRunLanded=false;   // so the empty-DB redirect to Live fires at most once per page load
function maybeFirstRun(s){
  const el=document.getElementById('firstrun'); if(!el) return;
  const empty=!(s&&s.total_crops);
  if(empty && localStorage.getItem('cc-introDismissed')!=='1'){
    // A brand-new, empty rig should see the welcome card (it lives in the Live view), not the
    // empty Dispatch we land on by default. Redirect once; afterwards the user can navigate freely.
    if(!__firstRunLanded){ __firstRunLanded=true; show('live'); }
    if(!el.dataset.filled){
      el.innerHTML='<div class="fr-card"><div class="fr-title">Welcome to your Backyard Observatory</div>'
        +'<p>The camera is watching the yard right now &mdash; there&rsquo;s nothing else to start. As real animals visit, this log fills in by itself: their photographs, the species name, and over days, who comes and when.</p>'
        +'<p class="fr-soft">It can take a while for the first visitor to appear. Leave it running and check back later.</p>'
        +'<button class="fr-x" type="button" onclick="dismissIntro()">Got it</button></div>';
      el.dataset.filled='1';
    }
    el.hidden=false;
  } else { el.hidden=true; }
}
function dismissIntro(){ localStorage.setItem('cc-introDismissed','1'); const el=document.getElementById('firstrun'); if(el) el.hidden=true; }

loadCameras(); refreshLive(); refreshHeader(); refreshNaming(); refreshWhoshere();
// Land on the view in the URL hash when there is one (deep links / refresh keep their tab);
// else the daily Dispatch ("who visited"). maybeFirstRun still redirects empty rigs to Live.
{ const h=location.hash.replace('#','');
  if(VIEWS.includes(h) && h!=='dispatch') show(h, true); else loadDispatch(); }
setInterval(refreshLive,6000);
setInterval(refreshHeader,4000);
setInterval(refreshNaming,4000);
setInterval(checkFeeds,8000);
setInterval(refreshWhoshere,6000);
setInterval(()=>{ const m=$('#settings'); if(m && !m.hidden) refreshControls(); },2000);   // live controls while the panel's open
/* ---------- lightbox: click any crop/frame to enlarge ---------- */
/* Delegated so it covers every served image across all tabs (review queue, cast, species
   browser, explorer) with no per-image wiring -- and any future <img src="/media/..."> too.
   The thumbnails are full-res crops sized down by CSS, so the same src shown at natural size
   IS the enlargement; no separate big-image endpoint needed. */
(function(){
  const lb=document.createElement('div');
  lb.id='lightbox';
  lb.innerHTML='<div class="lb-hint">← → or click image to browse · Esc to close</div>'
              +'<img alt="enlarged crop"><div class="lb-cap"></div>';
  document.body.appendChild(lb);   // this script runs at end of <body>, so body exists
  const img=lb.querySelector('img'), cap=lb.querySelector('.lb-cap');
  let gallery=[], idx=-1;
  function captionFor(el){
    // Nearest visit/individual label if there is one, else the filename; + position in the set.
    const card=el.closest('.panel'); let label='';
    if(card){ const h=card.querySelector('div[style*="font-weight:600"]'); if(h) label=h.textContent.trim(); }
    const file=decodeURI((el.getAttribute('src')||'').split('/').pop());
    const base=label? label+' — '+file : file;
    return gallery.length>1? `${base}   ·   ${idx+1} / ${gallery.length}` : base;
  }
  function showIdx(i){
    if(!gallery.length) return;
    idx=(i+gallery.length)%gallery.length;        // wrap both directions
    img.src=gallery[idx].src; cap.textContent=captionFor(gallery[idx]);
  }
  function openFrom(el){
    // The gallery is every crop/frame currently on the page, in document order, so ← → walk the
    // strip you clicked from (a visit's crops, the cast, the explorer) without per-view wiring.
    gallery=[...document.querySelectorAll('img[src*="/media/"]')].filter(x=>!lb.contains(x));
    const at=gallery.indexOf(el);
    gallery=gallery.length?gallery:[el];
    showIdx(at<0?0:at); lb.classList.add('open');
  }
  function close(){ lb.classList.remove('open'); img.removeAttribute('src'); gallery=[]; idx=-1; }
  lb.addEventListener('click',e=>{ e.target===img? showIdx(idx+1) : close(); });  // image=next, backdrop=close
  document.addEventListener('keydown',e=>{
    if(!lb.classList.contains('open')) return;
    if(e.key==='Escape') close();
    else if(e.key==='ArrowRight'){ e.preventDefault(); showIdx(idx+1); }
    else if(e.key==='ArrowLeft'){ e.preventDefault(); showIdx(idx-1); }
  });
  document.addEventListener('click',e=>{
    const t=e.target;
    if(t&&t.tagName==='IMG'&&!lb.contains(t)&&/\/media\//.test(t.getAttribute('src')||'')){
      e.preventDefault();
      openFrom(t);
    }
  });
})();
