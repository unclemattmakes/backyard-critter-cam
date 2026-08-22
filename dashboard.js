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

/* ---------- operator/viewer split (server-enforced; see web.py _is_operator) ----------
   When the rig has an operator_token configured, this browser is a VIEWER until the token is
   entered once (footer link). The token rides on every POST via this one wrapper — the ~16
   fetch POST call sites stay untouched. No token configured on the rig = everyone operates,
   exactly as before. */
const __origFetch=window.fetch.bind(window);
window.fetch=function(url,opts){
  const tok=localStorage.getItem('cc-operator-token');
  if(tok){   // every request, not just POSTs: GET /api/role must see it to report truthfully
    opts=opts||{};
    opts.headers=Object.assign({},opts.headers,{'X-Operator-Token':tok});
  }
  // WHO is typing rides on every human verdict. The server reads `logged_by` on the label /
  // confirm / sighting / life-event paths; injecting it here means no call site can forget it,
  // and attribution is the one thing that CANNOT be added later — a verdict recorded anonymously
  // is anonymous forever (db.py refuses to invent provenance after the fact).
  const who=localStorage.getItem('cc-labeler');
  if(who && opts && String(opts.method||'').toUpperCase()==='POST' && typeof opts.body==='string'){
    try{ const b=JSON.parse(opts.body);
      if(b && typeof b==='object' && !Array.isArray(b) && b.logged_by===undefined){
        b.logged_by=who; opts=Object.assign({},opts,{body:JSON.stringify(b)});
      }
    }catch(e){ /* not JSON — leave it exactly as the caller built it */ }
  }
  return __origFetch(url,opts);
};
/* The name tag itself: free text, this browser only, no account. Shown in the footer beside the
   operator control so a household can tell whose call a label was. */
function labelerName(){ return localStorage.getItem('cc-labeler')||''; }
function refreshLabeler(){
  const el=document.getElementById('labeler-link'); if(!el) return;
  const who=labelerName();
  el.textContent = who ? `labelling as ${who}` : 'sign your labels';
  el.title = who ? 'Your verdicts are recorded under this name. Click to change it.'
                 : 'Optional: put a name on the labels you confirm, so a household can tell whose call was whose.';
}
function labelerEdit(){
  const who=(prompt('Label as (a first name is plenty; blank to clear):', labelerName())||'').trim();
  if(who) localStorage.setItem('cc-labeler', who.slice(0,40));
  else localStorage.removeItem('cc-labeler');
  refreshLabeler();
}
async function refreshRole(){
  let r; try{ r=await fetch('/api/role').then(x=>x.json()); }catch(e){ return; }
  window.__role=r;
  document.body.classList.toggle('viewer', r.split && !r.operator);
  const el=document.getElementById('role-link');
  if(el){
    if(!r.split){ el.hidden=true; }
    else{
      el.hidden=false;
      el.textContent=r.operator?'operator ✓':'viewing — unlock';
      el.title=r.operator?'This browser holds the operator token. Click to forget it.'
                         :'Reading and playing everything; edits are off. Enter the operator token to curate from this device.';
    }
  }
}
function roleToggle(){
  const r=window.__role||{};
  if(r.operator&&localStorage.getItem('cc-operator-token')){
    localStorage.removeItem('cc-operator-token'); refreshRole(); return;
  }
  const box=document.getElementById('role-entry'); if(!box) return;
  box.hidden=!box.hidden;
  if(!box.hidden) box.querySelector('input').focus();
}
async function roleSave(){
  const inp=document.querySelector('#role-entry input'); if(!inp) return;
  const tok=(inp.value||'').trim(); if(!tok){ inp.focus(); return; }
  localStorage.setItem('cc-operator-token',tok);
  inp.value='';
  document.getElementById('role-entry').hidden=true;
  await refreshRole();
  const r=window.__role||{};
  if(r.split&&!r.operator){ localStorage.removeItem('cc-operator-token'); alert('That token was not accepted — still viewing.'); refreshRole(); }
}

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
/* One JSON fetch that THROWS on HTTP failure too, so "the server is down" can never render as a
   designed empty state — "every label has been checked" over a network blip is a lie, and it's
   exactly the wrong lie for a family member checking remotely. Pair with errEmpty for a retry. */
async function fetchJSON(url){
  const r=await fetch(url);
  if(!r.ok) throw new Error('HTTP '+r.status);
  return r.json();
}
function errEmpty(retryExpr){
  return `<p class="empty">Could not reach the observatory — <span style="text-decoration:underline;cursor:pointer" onclick="${retryExpr}">retry</span>.</p>`;
}

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
/* A clip the disk budget pruned plays from the backup archive via its stable id URL (the
   server restores it out of that day's zip on first view). Everything else streams as before. */
const clipUrl=c=>c.archived?('/archive/clip/'+encodeURIComponent(c.id)):media(c.clip_path);
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
      <div class="lab">${esc(fmtClock(c.start)||'')}${c.species?' · '+esc(nameOf(c.species)):''}${c.archived?' <span class="arch">archive</span>':''}</div>
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
  v.src=clipUrl(c);
  const pr=v.play(); if(pr&&pr.catch) pr.catch(()=>{});
  $('#clip-cap').innerHTML=`${c.species?`<span class="nm">${esc(nameOf(c.species))}</span>`:''}
    ${c.start?`<span class="mono" style="font-size:12px">${esc(fmtClock(c.start))}</span>`:''}
    ${c.seconds?`<span class="mono" style="font-size:12px">· ${fmtDur(c.seconds)}</span>`:''}
    ${c.dets?`<span class="mono" style="font-size:12px">· ${c.dets} detection${c.dets===1?'':'s'}</span>`:''}
    ${c.conf!=null?`<span class="c mono" style="font-size:12px">· ~${Math.round(c.conf*100)}%</span>`:''}
    ${clipVideoBadge(c)}
    ${c.archived?`<span class="mono" style="font-size:12px;color:var(--gilt)" title="the local copy was pruned for disk space; this plays the backup archive's copy">· from the archive</span>`:''}
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

const VIEWS=['live','visits','favorites','dispatch','behavior','indiv','calendar','seasons','cat'];
let __curView='visits';   // the tab currently on screen (explore screens remember it as "back home")
function show(v, fromHash){
  closeSettings();
  if(VIEWS.includes(v)) __curView=v;
  VIEWS.concat('explore').forEach(k=>{ const s=$('#view-'+k); if(s) s.classList.toggle('on',v===k); });
  VIEWS.forEach(k=>{ const t=$('#tab-'+k); if(t) t.classList.toggle('on',v===k); });
  // Deep-linkable tabs + a working Back button: the view lives in the URL hash. Programmatic
  // hash writes echo a hashchange we must NOT re-show (it would double-load the view). A hash
  // already DEEPER inside this view (#dispatch/2026-08-07/night when showing 'dispatch') is
  // kept — the caller wrote it on purpose just before showing the tab.
  if(!fromHash && VIEWS.includes(v) && !(location.hash||'').startsWith('#'+v+'/')) setHash('#'+v);
  syncLiveStreams();
  if(v==='cat') loadCatalogue();
  if(v==='visits') loadVisitsTab();
  if(v==='favorites') loadFavorites();
  if(v==='dispatch') loadDispatch();
  if(v==='behavior') loadBehavior();
  if(v==='indiv') loadIndividuals();
  if(v==='calendar') loadCalendar();
  if(v==='seasons') loadSeasons();
  if(v==='live') refreshWhoshere();
}
/* ---------- the hash router ----------
   Tabs were always deep-linkable; since 2026-08-08 the SHAREABLE drill-downs are too:
     #profile/<name>        one animal's whole record        (#profile/Stan)
     #day/<YYYY-MM-DD>      one day's summary                (#day/2026-08-07)
     #species/<name>        one species' catalogue sheet     (#species/raccoon)
     #dispatch/<date>/<ed>  a dated dispatch                 (#dispatch/2026-08-07/night)
   That makes "look at Stan" a URL you can text (LAN reach applies — see the README's security
   section), a refresh stop eating your drill-down, and the browser Back button walk your real
   trail instead of teleporting to the previous tab. The remaining explorer screens (obs
   windows, day-filtered visit lists, Days Afield) stay memory-only: Back from those lands on
   the nearest URL-addressed ancestor, which is where you drilled in from.
   Programmatic writes mark __hashQuiet (the FULL hash string, not just a view name) so their
   own hashchange echo doesn't double-render. */
let __hashQuiet=null;
function setHash(h){
  if(location.hash===h) return;
  __hashQuiet=h.replace(/^#/,'');
  location.hash=h;
}
function parseHash(h){
  const parts=(h||'').replace(/^#/,'').split('/');
  const kind=parts[0];
  if(!kind) return null;
  if(VIEWS.includes(kind)&&parts.length===1) return {kind:'view', view:kind};
  if(kind==='profile'&&parts[1]) return {kind:'profile', name:decodeURIComponent(parts[1])};
  if(kind==='day'&&/^\d{4}-\d{2}-\d{2}$/.test(parts[1]||'')) return {kind:'day', date:parts[1]};
  if(kind==='species'&&parts[1]) return {kind:'species', name:decodeURIComponent(parts[1])};
  if(kind==='dispatch'&&/^\d{4}-\d{2}-\d{2}$/.test(parts[1]||''))
    return {kind:'dispatch', date:parts[1],
            edition:['day','night','auto'].includes(parts[2])?parts[2]:'auto'};
  return null;
}
function applyHash(h){
  const p=parseHash(h);
  if(!p) return false;
  if(p.kind==='view'){ show(p.view, true); return true; }
  if(p.kind==='profile'){ exploreStack=[]; explore('profile',{name:p.name},cap1(p.name),'',true); return true; }
  if(p.kind==='day'){ exploreStack=[]; explore('day',{date:p.date},fmtDay(p.date),'',true); return true; }
  if(p.kind==='species'){ show('cat', true); openSheet(p.name, true); return true; }
  if(p.kind==='dispatch'){ DSP={edition:p.edition, date:p.date}; show('dispatch', true); return true; }
  return false;
}
window.addEventListener('hashchange',()=>{
  const h=location.hash.replace('#','');
  if(__hashQuiet===h){ __hashQuiet=null; return; }
  applyHash(location.hash);
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
  // Auto checkboxes start CHECKED (slider disabled): the rig asserts auto focus/WB/exposure at
  // every camera open, so before any read-back arrives, "auto" IS the camera's state. A camera
  // genuinely in manual (a deliberate config lock) unchecks them on the first refresh.
  $('#controls').innerHTML=CONTROLS.map(c=>`
    <div class="ctrl" data-k="${c.key}">
      <div class="row">
        <span class="name">${c.label}</span>
        <span style="display:flex;gap:10px;align-items:center">
          ${c.auto?`<label class="auto"><input type="checkbox" data-auto="${c.auto}" checked> auto</label>`:''}
          <span class="val" data-val>—</span>
        </span>
      </div>
      <input type="range" min="${c.min}" max="${c.max}" step="${c.step}" data-slider ${c.auto?'disabled':''}>
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
/* Camera MANAGEMENT state (the editable list); see the block near openCameras(). */
let CAMS={rows:[],pending:false,manageable:false,canSecret:false,editing:null};
async function loadCameras(){
  let d; try{ d=await fetch('/api/cameras').then(r=>r.json()); connOK(); }catch(e){ connFail(); return; }
  // The same payload carries whether this client may manage cameras at all (operator) and
  // whether it may set a password (loopback). Viewers never see the button.
  CAMS.manageable=!!d.manageable; CAMS.canSecret=!!d.can_set_credentials;
  const camBtn=$('#cams-open'); if(camBtn) camBtn.hidden=!CAMS.manageable;
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
  if(!$('#settings').hidden){ refreshControls(); loadZones(); }
}
function camName(source){ const c=(LIVE.cams||[]).find(x=>x.source===source); return c?(c.name||c.source):source; }

/* The masthead period/coords come from the PRIMARY camera (period is global -- one sun). */
async function refreshHeader(){
  const src=LIVE.primary; if(!src) return;
  let v; try{ v=await fetch('/api/camera?source='+encodeURIComponent(src)).then(r=>r.json()); connOK(); }catch(e){ connFail(); return; }
  if(v.period){ window.__period=v.period; $('#period').textContent=v.period; $('#cap-period').textContent=v.period; }
  if(v.lat!=null && v.lon!=null){ const f=(x,p,n)=>`${Math.abs(x).toFixed(3)}° ${x>=0?p:n}`; $('#coords').textContent=`${f(v.lat,'N','S')} · ${f(v.lon,'E','W')}`; }
  // First-run checklist state: frames flowing (frame_w publishes with the first real frame) + geo.
  window.__frCamera = v.frame_w ? true : (v.frame_w===undefined ? false : null);
  window.__frGeo = (v.lat!=null && v.lon!=null);
  /* Rig warning strip: a wedged camera outranks a battery warning (it already IS the
     consequence). Text comes verbatim from the rig (powerguard.py) so the three surfaces --
     console, HUD, here -- always tell the same story. */
  const warn=$('#rigwarn');
  if(warn){
    const wedged=v.wedge&&v.wedge.message, batt=v.power&&v.power.warning;
    const ev=window.__evalstatus;                 // filled by refreshEvalStatus(), boot + slow poll
    if(wedged){ warn.textContent='⚠ '+v.wedge.message; warn.className='rig-warn wedge'; warn.hidden=false; }
    else if(batt){ warn.textContent='⚠ '+v.power.warning; warn.className='rig-warn'; warn.hidden=false; }
    else if(ev && ev.ok===false){
      warn.textContent='⚠ Last night’s eval found a regression ('+(ev.regressions||[]).join(', ')
        +') — auto-naming paused itself; see reports/'+(ev.artifact||'');
      warn.className='rig-warn'; warn.hidden=false;
    }
    else warn.hidden=true;
  }
  /* Masthead chip: "observation in progress" only while a critter is actually on-cam
     (animal_active = a critter-class detection within cfg.on_cam_window_s, published by the
     rig); otherwise the rig is watching an empty stage, and the chip says so. */
  const chip=$('#livechip');
  if(chip){
    const on=!!v.animal_active;
    chip.classList.toggle('on',on);
    const t=$('#livechip-txt'); if(t) t.textContent = on ? 'observation in progress' : 'monitoring in progress';
  }
}

/* The nightly eval gate's verdict (/api/evalstatus). Fetched at boot and on a slow poll -- the
   artifact only changes when the ~2pm batch writes one, so anything faster is noise. The verdict
   shows two ways: a masthead warning when a metric regressed (refreshHeader's chain, below
   wedge/battery), and a one-liner in the Instrument Panel either way. */
async function refreshEvalStatus(){
  let ev; try{ ev=await fetch('/api/evalstatus').then(r=>r.json()); }catch(e){ return; }
  window.__evalstatus=ev;
  const line=$('#instr-eval');
  if(!line) return;
  if(!ev.available){ line.hidden=true; return; }
  const when=(ev.run_at||'').slice(0,10);
  line.hidden=false;
  line.textContent = ev.ok===false
    ? `Nightly eval (${when}): REGRESSION — ${(ev.regressions||[]).join(', ')}. Auto-naming paused itself; the full diff is in reports/${ev.artifact}.`
    : ev.ok===true
    ? `Nightly eval (${when}): no regression past tolerance.`
    : `Nightly eval (${when}): ran without a baseline to diff against (this run IS the baseline).`;
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
      // Mirror the rig's auto/manual state only UNTIL the user takes control of this control;
      // afterwards the checkbox is user-owned and the read-back never touches it again (it can
      // briefly lag the change we just POSTed). For the auto keys the rig publishes what it
      // last COMMANDED, not the driver's raw get() -- every cam we've had lies there (one
      // reported AUTOFOCUS=1.0 forever after we set 0; the current one answers -1/0 for the
      // auto props no matter the real mode) -- so the mirrored value is trustworthy.
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
      whMsg(r.as_viewer
        ? `Logged ${who} — noted for the operator to review (viewer logs never stamp).`
        : r.group
        ? `Logged ${who} — family stamp on ${r.stamped||0} frame${r.stamped===1?'':'s'}, counted as several animals.`
        : r.multi
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
async function openSheet(name,fromHash){
  if(!fromHash) setHash('#species/'+encodeURIComponent(name));
  $('#cat-index').style.display='none'; $('#cat-sheet').style.display='block';
  $('#sheet-name').textContent=cap1(name); $('#sheet-latin').textContent=latinOf(name);
  $('#sheet-crops').innerHTML='<p class="empty">Loading plates…</p>';
  let rows;
  try{ rows=await fetchJSON('/api/species/'+encodeURIComponent(name)); connOK(); }
  catch(e){ connFail(); $('#sheet-crops').innerHTML=errEmpty(`openSheet(${jarg(name)})`); return; }
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
    <img loading="lazy" src="${media(r.crop_path)}" alt="${esc(cap1(name||'crop'))}">
    ${stamp}
    ${tag?`<div class="rv-tag lbl">${esc(tag)}</div>`:''}
    ${favHeart(r.favorite, `favCrop(this,${r.id})`)}
    <div class="ft"><span class="c">${Math.round((r.confidence||0)*100)}%</span>
      <span class="acts">
        <button class="b up" title="confirm" onclick="act(${r.id},'verify',this)">✓</button>
        <button class="b dn" title="wrong" onclick="act(${r.id},'reject',this)">✗</button>
        <button class="b" title="correct" onclick="toggleEdit(this)">✎</button>
      </span></div>
    <select onchange="correct(${r.id},this)">${opts}</select>
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
  let d;
  try{ d=await fetchJSON('/api/review'); connOK(); }
  catch(e){ connFail(); $('#sheet-crops').innerHTML=errEmpty('openReview()'); return; }
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
/* '+ other…' used to open window.prompt — unstylable, miserable on a phone, and silently
   swallowed by some in-app browsers. The select swaps for an inline input + save instead;
   Escape restores the select. */
function otherSpeciesInline(sel, save){
  const wrap=document.createElement('span');
  wrap.style.cssText='display:inline-flex;gap:4px;align-items:center';
  wrap.innerHTML=`<input placeholder="new species label…" style="width:130px;padding:4px 6px;background:rgba(0,0,0,.25);border:1px solid rgba(255,255,255,.25);border-radius:4px;color:inherit;font:inherit">
    <button class="gear">save</button>`;
  const inp=wrap.querySelector('input'), go=wrap.querySelector('button');
  go.onclick=()=>{ const v=(inp.value||'').trim(); if(!v){ inp.focus(); return; } save(v); };
  inp.onkeydown=e=>{ if(e.key==='Enter') go.click(); if(e.key==='Escape'){ sel.value=''; wrap.replaceWith(sel); } };
  sel.replaceWith(wrap);
  inp.focus();
}
async function correct(id,sel){
  const species=sel&&sel.value;
  if(species==='__other__'){ otherSpeciesInline(sel, v=>_correctPost(id,v,null)); return; }
  if(!species)return;
  _correctPost(id,species,sel);
}
async function _correctPost(id,species,sel){
  if(sel){ sel.disabled=true; sel.style.opacity='.45'; }
  try{ await fetch('/api/detection/'+id,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({action:'correct',species})}); connOK(); }
  catch(e){ connFail(); if(sel){ sel.disabled=false; sel.style.opacity=''; } return; }
  const crop=document.querySelector(`.crop[data-id="${id}"]`);
  if(crop){ crop.style.transition='opacity .4s'; crop.style.opacity='.25'; }
}
function closeSheet(){ setHash('#cat'); $('#cat-sheet').style.display='none'; $('#cat-index').style.display='block'; loadCatalogue(); }

/* ---------- explorer: drill into the field tallies ---------- */
let exploreStack=[], visitsData=[], obsState=null, __exploreFrom='live';
function goTally(t){
  if(t==='species'){ show('cat'); return; }   // species => the existing Specimen Catalogue
  if(t==='visits'){ show('visits'); return; } // visits => their own first-class tab now
  exploreStack=[];
  explore(t, {}, t==='obs'?'Observations':'Days Afield');
}
function explore(screen,params,title,sub,fromHash){
  if(!exploreStack.length) __exploreFrom=__curView;   // Back lands on the tab you drilled in from
  exploreStack.push({screen,params:params||{},title:title||'',sub:sub||''});
  // The two shareable drill-downs write themselves into the URL (deep link + refresh survival +
  // a truthful browser Back); the rest of the explorer stays memory-only — see the router note.
  if(!fromHash){
    if(screen==='profile'&&params&&params.name) setHash('#profile/'+encodeURIComponent(String(params.name)));
    else if(screen==='day'&&params&&params.date) setHash('#day/'+params.date);
  }
  renderExplore();
}
function exploreBack(){
  const cur=exploreStack[exploreStack.length-1];
  const encoded=cur&&((cur.screen==='profile'&&cur.params.name)||(cur.screen==='day'&&cur.params.date));
  exploreStack.pop();
  if(encoded&&parseHash(location.hash)){ history.back(); return; }  // hashchange renders the ancestor
  exploreStack.length?renderExplore():show(__exploreFrom||'live');
}
function renderExplore(){
  const cur=exploreStack[exploreStack.length-1]; if(!cur)return;
  show('explore');
  $('#explore-title').textContent=cur.title;
  $('#explore-sub').textContent=cur.sub||'';
  const body=$('#explore-body'); body.innerHTML='<p class="empty">Loading…</p>';
  ({obs:renderObs,visits:renderVisits,days:renderDays,day:renderDay,profile:renderProfile}[cur.screen]||(()=>{}))(cur.params,body);
}
/* Open one individual's profile — every visit they're stamped into, their photos, their clips
   (archived ones included). Reachable from any surface that shows a name. */
function openProfile(name){ if(!name) return; exploreStack=[]; explore('profile',{name},cap1(String(name))); }
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
  ['day','species','start','end','individual'].forEach(k=>{ if(s[k]) qs.set(k,s[k]); });
  qs.set('offset',s.offset); qs.set('limit',60);
  let o; try{ o=await fetch('/api/crops?'+qs).then(r=>r.json()); }catch(e){ const m=$('#obs-more'); if(m) m.innerHTML='<p class="empty">Could not load observations.</p>'; return; }
  s.offset+=o.crops.length;
  const grid=$('#obs-grid'); if(!grid)return;
  grid.insertAdjacentHTML('beforeend', o.crops.map(obsTile).join(''));
  if(!grid.children.length) grid.innerHTML='<p class="empty">No observations here.</p>';
  // On the profile page the grid is one section of many, so the count goes on its own heading
  // rather than the explorer's subtitle.
  const pn=$('#prof-photo-n');
  if(pn) pn.textContent=`${(o.total||0).toLocaleString()} total`;
  else $('#explore-sub').textContent=`${(o.total||0).toLocaleString()} total`;
  // Drill into a species from a mixed grid. This used to sit on the WHOLE tile, which is mostly
  // photograph -- so "let me see that one bigger" and "show me every raccoon" were the same click,
  // and the navigation won. It lives on the species name in the footer now: still one tap, but a
  // tap at something that reads like a link, and the photo is free to just be a photo.
  if(!s.species) grid.querySelectorAll('.crop[data-sp] .obs-sp').forEach(el=>{
    const sp=el.closest('.crop').dataset.sp;
    el.classList.add('clickable');
    el.title=`see every ${cap1(sp)} photograph`;
    el.onclick=e=>{ e.stopPropagation(); explore('obs',{species:sp},cap1(sp)); };
  });
  const more=$('#obs-more'); if(more) more.innerHTML = s.offset<o.total ? `<button class="back" onclick="loadMoreObs()">Load more &middot; ${(o.total-s.offset).toLocaleString()} left</button>` : '';
}
function obsTile(r){
  const sp=r.species?cap1(r.species):'unidentified';
  const conf=r.species_confidence!=null?r.species_confidence:r.confidence;
  const v=r.verified;
  return `<div class="crop ${v===1?'v-1':v===0?'v-0':''}"${r.species?` data-sp="${esc(r.species)}"`:''} title="${esc(fmtDateTime(r.timestamp))}">
    <img loading="lazy" src="${media(r.crop_path)}" alt="${esc(sp)} · ${esc(fmtDateTime(r.timestamp))}">
    ${favHeart(r.favorite, `favCrop(this,${r.id})`)}
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
  wireVisitCards(body);
}
/* A visit with clips plays its video on click (what you asked for); one without falls back to
   its crop grid (the old behaviour, for visits before clip recording was on). Shared by the
   Visits tab, the day-filtered explorer, and the individual profile. */
function wireVisitCards(root){
  root.querySelectorAll('[data-vi]').forEach(el=>{ const v=visitsData[+el.dataset.vi]; if(!v) return;
    el.onclick=(v.clips&&v.clips.length)
      ? ()=>playClips(v.clips,0,`${cap1(v.title||'animal')} · ${fmtDateTime(v.start)}`)
      : ()=>explore('obs',{start:v.start,end:v.end},`Visit · ${cap1(v.title||'animal')}`,fmtDateTime(v.start)); });
}
function visitCard(v,i){
  const nclips=(v.clips||[]).length;
  const narch=(v.clips||[]).filter(c=>c.archived).length;
  const sp=(v.title&&v.title!=='animal')?v.title:'';
  const inds=(v.individuals||[]).map(n=>`<span class="vl-ind clickable" title="open ${esc(cap1(n))}'s profile" onclick="event.stopPropagation();openProfile(${jarg(n)})">${esc(cap1(n))}</span>`).join('');
  // The curation tools (confirm/correct species, name the individual) hide behind the ✎ so the
  // card itself stays a reading surface: who, when, how long, play. stopPropagation keeps the
  // whole label layer from triggering the card's play/drill. The tools are BUILT on first open
  // (vlabelOpen): rendered eagerly, the ~40-option species corrector times a 269-card Visit Log
  // was most of the landing page's DOM.
  const footer=`<div class="vlabel" onclick="event.stopPropagation()">
      <button class="gear vlabel-toggle" onclick="vlabelOpen(this,${i})" title="confirm or correct this visit's labels">✎ label</button>
      <div class="vlabel-tools"></div>
    </div>`;
  return `<div class="card vcard" data-vi="${i}">
    <div class="thumb playable" style="background-image:url('${v.rep_crop?media(v.rep_crop):''}')">${nclips?playBadge(nclips):''}${favHeart(v.favorite, `favVisit(this,${jarg(v.source)},${jarg(v.start)},${jarg(v.end)})`)}${narch?`<span class="arch-badge" title="${narch===nclips?'all':narch} of these clips play from the backup archive">archive</span>`:''}</div>
    <div class="body">
      <div class="common">${esc(cap1(v.title||'animal'))} ${inds}</div>
      <div class="latin" style="font-style:normal">${esc(fmtDateTime(v.start))}</div>
      <div class="meta"><span class="count">${v.count}<small> obs</small></span><span class="conf">${v.minutes>=1?Math.round(v.minutes)+' min':'brief'}</span></div>
      ${footer}
    </div></div>`;
}
/* Build a visit card's label tools the first time its ✎ opens. The card index is the key into
   visitsData, same as every other per-card action (_visitTarget). */
function vlabelOpen(btn,i){
  const wrap=btn.closest('.vlabel'); if(!wrap) return;
  const tools=wrap.querySelector('.vlabel-tools');
  if(tools && !tools.dataset.built){
    tools.dataset.built='1';
    const v=(visitsData||[])[i]||{};
    const sp=(v.title&&v.title!=='animal')?v.title:'';
    tools.innerHTML=`
        <div style="display:flex;gap:5px;align-items:center;flex-wrap:wrap;margin-top:6px">
          <button class="gear" onclick="postVisitLabel(_visitTarget(${i}),{verify:true},()=>visitSaved(${i},'✓ species confirmed'))" title="confirm this species for the whole visit">✓ sp</button>
          ${speciesSelect('vsp-'+i,sp)}
          <button class="gear" onclick="explorerSpecies(${i})" title="correct the species for the whole visit">correct</button></div>
        <div style="display:flex;gap:5px;align-items:center;margin-top:6px">${reidInput('vn-'+i,'name the individual…')}<button class="gear" onclick="explorerName(${i})">Name</button></div>
        <span id="vst-${i}" class="lbl" style="opacity:.8;min-height:14px"></span>`;
  }
  wrap.classList.toggle('open');
}
/* ---------- individual profile: one animal's whole record ---------- */
async function renderProfile(params,body){
  const name=params.name;
  let p; try{ p=await fetch('/api/individual/profile?name='+encodeURIComponent(name)).then(r=>r.json()); }
  catch(e){ body.innerHTML='<p class="empty">Could not load the profile.</p>'; return; }
  if(!p.found){ body.innerHTML=`<p class="empty">No records for ${esc(cap1(name))} yet.</p>`; return; }
  $('#explore-sub').textContent=p.species?cap1(p.species):'';
  // Archived clips whose backup zip is unreachable right now (drive unmounted, or pruned before
  // the backups began) would be dead play buttons — drop them, but say how many are gone.
  let lost=0;
  (p.visits||[]).forEach(v=>{ const n=(v.clips||[]).length;
    v.clips=(v.clips||[]).filter(c=>!c.archived||c.archive_ok!==false); lost+=n-v.clips.length; });
  const nclips=p.visits.reduce((a,v)=>a+(v.clips||[]).length,0);
  const narch=p.visits.reduce((a,v)=>a+(v.clips||[]).filter(c=>c.archived).length,0);
  const stamps=Object.entries(p.stamp_mix||{}).map(([k,n])=>`${n.toLocaleString()} ${k==='human'?'confirmed':k}`).join(' · ');
  const depart=(p.status&&p.status.status==='departed')
    ? `<span class="flag" style="opacity:.85">moved on${p.status.effective_date?' · last resident day '+esc(p.status.effective_date):''}</span>` : '';
  const comp=(p.companions||[]).map(c=>
    `<button class="cast-chip" onclick="openProfile(${jarg(c.name)})">${esc(cap1(c.name))}<small>×${c.n_visits}</small></button>`).join('');
  /* WHAT THEY MOSTLY DO. The per-visit tag has been computed since the coarse behaviour pass and
     discarded every time; this is the total. Two honesty rules baked in: it prints how many
     visits it could NOT tag (a visit with no overlapping clip has no motion features, and on this
     corpus that is about half of them), and below three tagged visits it shows counts instead of
     percentages -- "100% fed here" off one visit is a sentence about arithmetic. */
  const bh=p.behaviour;
  const bhRow=(bh&&bh.n_tagged)?(()=>{
    const parts=Object.entries(bh.share||{}).map(([k,v])=>
      bh.thin?`${bh.tags[k]}× ${esc(k)}`:`${Math.round(v*100)}% ${esc(k)}`);
    const why=`From ${bh.n_tagged} of ${bh.n_visits} visits — the rest had no clip overlapping them, so no motion to read. `+
      (bh.thin?`Too few to quote a percentage, so these are counts.`:``)+
      ` The tag is a coarse reading of three motion features, not a verdict.`;
    return `<div class="lbl" style="opacity:.78" title="${esc(why)}">Mostly: ${parts.join(' · ')}`+
      `<span style="opacity:.6"> · ${bh.n_tagged}/${bh.n_visits} visits readable</span>${infoDot(why)}</div>`;
  })():'';
  const refs=(p.references||[]).map(r=>
    `<img loading="lazy" src="${media(r.crop_path)}" alt="reference photo" title="${esc(r.kind==='video_frame'?'phone video frame':'phone photo')}${r.captured_at?' · '+esc(fmtDateTime(r.captured_at)):''}${r.note?' · '+esc(r.note):''}">`).join('');
  visitsData=p.visits;
  body.innerHTML=`
    <div class="profile-head panel">
      <div class="tallies" style="margin:0">
        <div class="tally" style="cursor:default"><div class="n">${p.n_visits}</div><div class="k lbl">visits</div></div>
        <div class="tally" style="cursor:default"><div class="n">${p.n_crops.toLocaleString()}</div><div class="k lbl">photographs</div></div>
        <div class="tally" style="cursor:default"><div class="n">${nclips}${narch?` <small>${narch} archived</small>`:''}</div><div class="k lbl">clips</div></div>
      </div>
      <div class="lbl" style="opacity:.78">first seen ${esc(fmtDateTime(p.first_seen))} · last seen ${esc(fmtDateTime(p.last_seen))}${stamps?` · labels: ${stamps}`:''}${p.unfiled?` · ${p.unfiled} newest photos not yet filed into a visit`:''}</div>
      ${depart}
      ${bhRow}
      ${lost?`<div class="lbl" style="opacity:.6">${lost} clip${lost===1?'':'s'} pruned before the backups began (or the backup drive is unreachable) — not shown.</div>`:''}
      ${comp?`<div class="castrow"><span class="lbl">Seen together with</span>${comp}</div>`:''}
      ${refs?`<div class="profile-refs"><span class="lbl">Reference shots — phone (identity certified by hand)</span><div class="refstrip">${refs}</div></div>`:''}
    </div>
    ${profileEventsHTML(name, p.events||[])}
    <h2 class="sec">Visits <span class="n">${p.visits.length<p.n_visits?`the newest ${p.visits.length} of ${p.n_visits}`:'each card plays its clips'}</span></h2>
    <div class="cards">${p.visits.map(visitCard).join('')}</div>
    <h2 class="sec">Photographs <span class="n" id="prof-photo-n"></span></h2>
    <div class="crops" id="obs-grid"></div><div class="more" id="obs-more"></div>`;
  wireVisitCards(body);
  obsState={day:null,species:null,start:null,end:null,individual:name,offset:0};
  await loadMoreObs();
}
/* The field notebook: this animal's STORY as dated notes (life_events). The knowledge that
   used to evaporate — kits first emerged, an injury noticed, a debut night — gets a dated line
   a future reader (or next spring's comparison) can find. Append-only; correct a wrong note
   with another note, the way a paper notebook works. */
function profileEventsHTML(name, events){
  const rows=events.map(e=>`<div class="fr-row" style="margin:4px 0"><span class="lbl" style="opacity:.6;margin-right:6px">${esc(e.event_date||String(e.created_at||'').slice(0,10))}</span>${esc(e.note)}${e.labeled_by?` <span class="lbl" style="opacity:.5">— ${esc(e.labeled_by)}</span>`:''}</div>`).join('');
  return `<h2 class="sec">Field Notebook <span class="n">${events.length?events.length+' entr'+(events.length===1?'y':'ies'):'the story, in dated notes'}</span></h2>
    <div class="panel" style="padding:10px 14px">
      ${rows||'<p class="empty" style="padding:0">No entries yet — the first litter, a limp, a debut night: write it down while you remember it.</p>'}
      <div class="ev-add" style="display:flex;gap:5px;align-items:center;flex-wrap:wrap;margin-top:8px">
        <input type="date" id="ev-date" style="padding:4px 6px;background:rgba(0,0,0,.25);border:1px solid rgba(255,255,255,.2);border-radius:4px;color:inherit;font:inherit">
        <input id="ev-note" placeholder="what happened…" maxlength="500" style="flex:1;min-width:180px;padding:5px 8px;background:rgba(0,0,0,.25);border:1px solid rgba(255,255,255,.2);border-radius:4px;color:inherit" onkeydown="if(event.key==='Enter')this.nextElementSibling.click()">
        <button class="gear" onclick="profileAddEvent(${jarg(name)})">add to the notebook</button>
      </div>
    </div>`;
}
async function profileAddEvent(name){
  const note=($('#ev-note')&&$('#ev-note').value||'').trim();
  if(!note){ if($('#ev-note')) $('#ev-note').focus(); return; }
  const date=($('#ev-date')&&$('#ev-date').value)||null;
  const restore=busyBtn();
  try{
    const r=await fetch('/api/individual/event',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({name,note,date})}).then(x=>x.json());
    if(r.error){ restore(); alert(r.error); return; }
    openProfile(name);   // re-render with the new entry
  }catch(e){ restore(); connFail(); }
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
  let s;
  try{ s=await fetchJSON('/api/stats'); connOK(); }
  catch(e){ connFail(); body.innerHTML=errEmpty(`explore('day',{date:${jarg(date)}})`); return; }
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

/* ---------- the Visits tab: the default landing view ----------
   The same visit cards as the old explorer screen, promoted to a first-class tab (it's the
   "scroll around and see what happened" surface), with the named cast across the top so any
   individual is one tap from their full profile. */
async function loadVisitsTab(){
  const body=$('#visits-body'); if(!body) return;
  let o; try{ o=await fetch('/api/visits').then(r=>r.json()); connOK(); }
  catch(e){ connFail(); body.innerHTML='<p class="empty">Could not load visits.</p>'; return; }
  visitsData=o.visits||[];
  const note=o.window
    ? `the latest ${(o.total||0).toLocaleString()} — older ones via the Calendar or Days Afield`
    : `${(o.total||0).toLocaleString()} visit${o.total===1?'':'s'}`;
  if(!visitsData.length){
    body.innerHTML='<p class="empty">No visits yet — cards appear here as animals come and go.</p>';
    return;
  }
  body.innerHTML=`
    <h2 class="sec">The Visit Log <span class="n">${note}</span></h2>
    <div class="castrow" id="visits-cast"></div>
    <div class="cards">${visitsData.map(visitCard).join('')}</div>`;
  wireVisitCards(body);
  loadCastStrip();
}
/* The named cast (rollcall order: overdue first, then most recently seen). */
async function loadCastStrip(){
  const el=$('#visits-cast'); if(!el) return;
  let d; try{ d=await fetch('/api/rollcall').then(r=>r.json()); }catch(e){ return; }
  const cast=d.cast||[];
  if(!cast.length){ el.remove(); return; }
  el.innerHTML='<span class="lbl">Look up an individual</span>'+cast.map(c=>{
    const ago=c.days_since==null?'':(c.days_since===0?'today':c.days_since===1?'yesterday':c.days_since+'d ago');
    return `<button class="cast-chip${c.overdue?' overdue':''}" onclick="openProfile(${jarg(c.id)})"
      title="${esc(cap1(c.id))} — ${c.n_crops.toLocaleString()} photos over ${c.nights} day${c.nights===1?'':'s'}${c.overdue?' · overdue (past their usual gap)':''}">
      ${c.crop?`<span class="cc-face" style="background-image:url('${media(c.crop)}')"></span>`:''}${esc(cap1(c.id))}${ago?`<small>${ago}</small>`:''}</button>`;
  }).join('');
}

/* ---------- FAVOURITES: "keep this one" ----------
   The one control on this dashboard that makes no claim about the animal. Every other verdict
   here -- ✓ confirmed, a corrected species, a name -- is evidence that feeds the models, which is
   why they all carry attribution and why the server gates them. A ♡ is taste: nothing downstream
   reads it, it is undone by tapping it again, and it can never be mistaken for a confirmed label.

   A crop is kept by its detection id. A VISIT is kept by source + the moment it started, never by
   a visit id: the Visit Log re-clusters visits out of raw detections on every request and visits.py
   renumbers the ledger, so an id is not a handle that survives the week (the same reason the ✎
   label tools post source+start+end). See the `favorites` table comment in db.py. */
function favHeart(on, call, cls){
  const yes = !!on;
  return `<button class="fav${yes?' on':''}${cls?' '+cls:''}" type="button" aria-pressed="${yes?'true':'false'}"
    aria-label="${yes?'Remove from favourites':'Add to favourites'}"
    title="${yes?'kept — tap to remove':'keep this one'}"
    onclick="event.stopPropagation();${call}">${yes?'♥':'♡'}</button>`;
}
async function favPost(body){
  const r=await fetch('/api/favorite',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify(body)}).then(x=>x.json());
  if(r&&r.error) throw new Error(r.error);
  return r;
}
/* Flip one heart. The button is the source of truth for its own state (.on), so this works for a
   tile rendered by any of the four surfaces without any of them tracking a favourites list. */
async function favToggle(btn, key){
  if(!btn) return;
  const want=!btn.classList.contains('on');
  const restore=busyBtn(btn);
  try{ await favPost(Object.assign({on:want}, key)); connOK(); }
  catch(e){ connFail(); restore(); return; }
  restore();
  btn.classList.toggle('on',want);
  btn.textContent = want?'♥':'♡';
  btn.setAttribute('aria-pressed', want?'true':'false');
  btn.setAttribute('aria-label', want?'Remove from favourites':'Add to favourites');
  btn.title = want?'kept — tap to remove':'keep this one';
  // On the gallery itself an un-kept item DIMS rather than vanishing: a mis-tap on a phone is
  // undone by tapping the same heart again, instead of hunting the log for the photo you lost.
  const item=btn.closest('.fav-wrap, .crop, .card');
  if(item && document.querySelector('#view-favorites.on')){
    item.style.transition='opacity .35s'; item.style.opacity = want?'' : '.3';
  }
}
function favCrop(btn,id){ favToggle(btn,{kind:'detection',detection_id:id}); }
function favVisit(btn,source,start,end){ favToggle(btn,{kind:'visit',source,start,end}); }

/* A favourite's identity, carried on the note element so the note can be saved without threading
   the whole record through an inline handler. */
function favKeyAttrs(f){
  return f.kind==='visit'
    ? `data-kind="visit" data-source="${esc(f.source)}" data-start="${esc(f.started_at)}"`
    : `data-kind="detection" data-det="${esc(f.detection_id)}"`;
}
function favKeyOf(el){
  return el.dataset.kind==='visit'
    ? {kind:'visit', source:el.dataset.source, start:el.dataset.start}
    : {kind:'detection', detection_id:+el.dataset.det};
}
/* The caption: why this one was worth keeping -- the part a photo cannot say for itself, and the
   part that is gone in a year if nobody writes it down (the same argument as the field notebook on
   a profile page). Click to write, Enter to save, Escape to abandon. */
function favNoteHTML(f){
  const who=f.labeled_by?`<span class="fav-by">kept by ${esc(f.labeled_by)}</span>`:'';
  return `<div class="fav-note${f.note?'':' blank'}" ${favKeyAttrs(f)} data-note="${esc(f.note||'')}"
    ${f.labeled_by?`data-by="${esc(f.labeled_by)}"`:''}
    role="button" tabindex="0" title="click to write why this one is worth keeping"
    onclick="event.stopPropagation();favNoteEdit(this)"
    onkeydown="if(event.key==='Enter'||event.key===' '){event.preventDefault();event.stopPropagation();favNoteEdit(this);}"
    ><span class="fav-note-txt">${f.note?esc(f.note):'add a note…'}</span>${who}</div>`;
}
function favNoteRender(el){
  const note=el.dataset.note||'';
  const who=el.dataset.by?`<span class="fav-by">kept by ${esc(el.dataset.by)}</span>`:'';
  el.classList.toggle('blank',!note);
  el.innerHTML=`<span class="fav-note-txt">${note?esc(note):'add a note…'}</span>${who}`;
}
function favNoteEdit(el){
  if(el.querySelector('input')) return;             // already editing
  el.innerHTML='<input class="fav-note-in" maxlength="300" placeholder="why this one…">';
  const inp=el.querySelector('input');
  inp.value=el.dataset.note||'';
  inp.focus(); inp.select();
  let done=false;
  const finish=async(save)=>{
    if(done) return; done=true;                     // Enter fires blur too -- save once
    const val=(inp.value||'').trim();
    if(save && val!==(el.dataset.note||'')){
      try{ await favPost(Object.assign({on:true, note:val}, favKeyOf(el))); connOK(); el.dataset.note=val; }
      catch(e){ connFail(); }
    }
    favNoteRender(el);
  };
  inp.onkeydown=e=>{ e.stopPropagation();
                     if(e.key==='Enter'){ e.preventDefault(); finish(true); }
                     else if(e.key==='Escape'){ e.preventDefault(); finish(false); } };
  inp.onblur=()=>finish(true);
  inp.onclick=e=>e.stopPropagation();
}

async function loadFavorites(){
  const body=$('#favorites-body'); if(!body) return;
  let d;
  try{ d=await fetchJSON('/api/favorites'); connOK(); }
  catch(e){ connFail(); body.innerHTML=errEmpty('loadFavorites()'); return; }
  const favs=d.favorites||[];
  const head=`<h2 class="sec">Favourites <span class="n">${
    favs.length?`${d.visits} visit${d.visits===1?'':'s'} · ${d.crops} photograph${d.crops===1?'':'s'}`
              :'the things worth keeping'}</span></h2>`;
  if(!favs.length){
    body.innerHTML=head+`<p class="empty">Nothing kept yet — tap the ♡ on any visit card or
      photograph and it lands here. <span style="text-decoration:underline;cursor:pointer"
      onclick="show('visits')">Open the visit log</span> and start an album.</p>`;
    return;
  }
  const vis=favs.filter(f=>f.kind==='visit'&&f.visit);
  const crops=favs.filter(f=>f.kind==='detection'&&f.crop);
  const gone=favs.filter(f=>f.gone);
  // The kept visits reuse the Visit Log's own card AND its click-to-play wiring, so a card looks
  // and behaves identically in both places. visitsData is repointed at them while this tab is on
  // screen -- the same contract renderProfile uses for an individual's visits.
  visitsData=vis.map(f=>f.visit);
  body.innerHTML=head+
    `<p class="fav-hint">A ♡ is only ever taste — it changes no label, and nothing downstream reads it.</p>`+
    (vis.length?`<h2 class="sec">Kept Visits <span class="n">each card plays its clips</span></h2>
      <div class="cards">${vis.map((f,i)=>`<div class="fav-wrap">${visitCard(f.visit,i)}${favTrunc(f.visit)}${favNoteHTML(f)}</div>`).join('')}</div>`:'')+
    (crops.length?`<h2 class="sec">Kept Photographs <span class="n">${crops.length}</span></h2>
      <div class="crops">${crops.map(favCropTile).join('')}</div>`:'')+
    (gone.length?`<h2 class="sec">No Longer In The Log <span class="n">${gone.length}</span></h2>
      <p class="fav-hint">Kept, and then the observations behind them were removed — a purge, or a
        camera that no longer exists. Listed rather than silently dropped: a favourite that
        vanishes without a word is worse than one that says so.</p>
      <div class="crops">${gone.map(favGoneTile).join('')}</div>`:'');
  wireVisitCards(body);
}
/* A visit so long the album's scan bound cut it short (stats._FAV_VISIT_SCAN_ROWS). Rare, and
   said out loud rather than drawn as a shorter visit than the Visit Log shows for the same span. */
function favTrunc(v){
  return (v&&v.truncated)
    ? `<div class="fav-trunc lbl">longer than this card can count — showing the first ${v.count.toLocaleString()} photographs</div>`
    : '';
}
function favCropTile(f){
  const c=f.crop||{};
  const sp=c.species?cap1(c.species):'unidentified';
  const when=c.timestamp?fmtDateTime(c.timestamp):'';
  const who=c.individual?`<span class="obs-sp clickable" title="open ${esc(cap1(c.individual))}&#39;s profile"
      onclick="event.stopPropagation();openProfile(${jarg(c.individual)})">${esc(cap1(c.individual))}</span>`:'';
  return `<div class="crop fav-item" data-id="${c.id}">
    <img loading="lazy" src="${media(c.crop_path)}" alt="${esc(sp)} · ${esc(when)}">
    ${favHeart(true, `favCrop(this,${c.id})`)}
    <div class="ft"><span class="c">${esc(sp)}</span>${who}</div>
    <div class="fav-when lbl">${esc(when)}</div>
    ${favNoteHTML(f)}
  </div>`;
}
function favGoneTile(f){
  const what=f.kind==='visit'
    ? `a ${esc(f.source||'camera')} visit · ${esc(f.started_at?fmtDateTime(f.started_at):'')}`
    : `photograph #${esc(f.detection_id)}`;
  const call=f.kind==='visit'
    ? `favVisit(this,${jarg(f.source)},${jarg(f.started_at)},${jarg(f.ended_at||f.started_at)})`
    : `favCrop(this,${f.detection_id})`;
  return `<div class="crop fav-item fav-gone">
    <div class="fav-gone-face">—</div>
    ${favHeart(true, call)}
    <div class="ft"><span class="c">${what}</span></div>
    <div class="fav-when lbl">kept ${esc(fmtDateTime(f.created_at))}${f.labeled_by?' · '+esc(f.labeled_by):''}</div>
    ${f.note?`<div class="fav-note" style="cursor:default">${esc(f.note)}</div>`:''}
  </div>`;
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
  if(m.tag) bits.push(`<b>${esc(m.tag)}</b>`);   // the what-they-DID word leads: fed here / passed through / lingered
  const app=APPROACH_WORD[m.approach]; if(app&&!m.tag) bits.push(`<b>${app}</b>`);
  else if(app) bits.push(app);
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
/* A dated dispatch lives in the URL (#dispatch/DATE/EDITION — shareable, refresh-proof);
   "latest" is just the plain #dispatch tab. */
function _dspHash(){ setHash(DSP.date?`#dispatch/${DSP.date}/${DSP.edition}`:'#dispatch'); }
function setDispatch(ed){ DSP.edition=ed; _dspHash(); loadDispatch(); }
function dspNav(date, edition){ DSP={edition:edition||DSP.edition, date:date||null}; _dspHash(); loadDispatch(); }
function dspLatest(){ DSP={edition:'auto', date:null}; _dspHash(); loadDispatch(); }
function openDispatchAt(date, edition){ dspNav(date, edition); show('dispatch'); }

async function loadDispatch(){
  const body=$('#dispatch-body'); body.innerHTML='<p class="empty">Loading the dispatch…</p>';
  clearTimeout(__reelTimer);
  const seq=++__dspSeq;
  const qs='edition='+encodeURIComponent(DSP.edition)+(DSP.date?'&date='+encodeURIComponent(DSP.date):'');
  /* The digest text renders the moment it (and the roll call) arrive; the REEL streams into its
     own slot afterwards. It used to sit in the same Promise.all, which held the whole page on a
     bare "Loading…" for as long as /api/reel took — measured 2026-08-08 at 6s with the reel
     already BUILT, and a first-view ffmpeg stitch takes minutes. The readable half is ready in
     ~0.3s; serve it. */
  let d, rc;
  try{
    [d, rc] = await Promise.all([
      fetch('/api/digest?'+qs).then(r=>r.json()),
      fetch('/api/rollcall').then(r=>r.json()).catch(()=>({cast:[]})),
    ]);
  }catch(e){ body.innerHTML='<p class="empty">Could not load the dispatch.</p>'; return; }
  if(seq!==__dspSeq) return;                    // a newer navigation superseded this load
  window.__digest=d;
  renderDispatch(d, rc, {status:'loading'});
  let rl=null;
  try{ rl=await fetch('/api/reel?'+qs).then(r=>r.ok?r.json():null); }catch(e){ rl=null; }  // older server -> no reel
  if(seq!==__dspSeq) return;
  if(rl && rl.status==='ready') window.__reelMan=rl;
  const el=document.getElementById('reel-sec');  // swap the reel slot only; never yank the page
  if(el) el.innerHTML=reelSection(d, rl);        // under the reader's scroll
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
  const sinceText=c=>{ const base=c.days_since==null ? '' :
    c.days_since===0 ? 'seen today' : c.days_since===1 ? 'seen yesterday' : `${c.days_since} days ago`;
    return base && c.via_group ? base+' (with the kits)' : base; };
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
    ? `<div class="reel-links lbl"><span class="reel-note">✂ a condensed cut is being stitched — it will appear here in a minute or two</span></div>`
    : rl&&rl.status==='loading'
    ? `<div class="reel-links lbl"><span class="reel-note">… checking for a condensed cut</span></div>` : '';
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
  if(d.coverage&&d.coverage.dark_minutes>=30){
    const cv=d.coverage, hrs=cv.dark_minutes>=90?Math.round(cv.dark_minutes/6)/10+' h':cv.dark_minutes+' min';
    flags.push(`<span class="flag" style="background:rgba(255,120,90,.12);border-color:rgba(255,120,90,.4)" title="the coverage ledger says the camera was not watching for part of this period — a dark camera is not an empty yard, so read tonight's absence claims (quiet regulars, first/last) accordingly">⚠ camera dark ${esc(hrs)} of this ${esc(d.edition)}${infoDot('The rig records when each camera is actually watching (open / lost / reconnected / stopped). This period has a known dark stretch, so “no X tonight” may mean “camera was down”, not “X didn’t come”. Periods before the ledger existed say nothing either way.')}</span>`);
  }
  if(d.moon){ flags.push(`<span class="flag moon">${d.moon.glyph} ${esc(d.moon.name)} · ${d.moon.illum_pct}% lit</span>`); }
  if(flags.length) html+=`<div class="lede">${flags.join('')}</div>`;
  if(d.empty){ body.innerHTML=html+`<p class="empty">A quiet ${esc(d.edition)} — no visitors recorded.</p>`+roll; return; }
  window.__reel=d.reel||[];
  window.__reelMan=(rl&&rl.status==='ready')?rl:null;
  html+=`<div id="reel-sec">${reelSection(d, rl)}</div>`;
  html+=visitLogSection(d);
  html+=roll;
  const t=[['visits',(d.visits||0).toLocaleString()],
    [(d.n_surprising?'species (+'+d.n_surprising+' to verify)':'species'),(d.species||[]).length-(d.n_surprising||0)],
    ['busiest hour',d.busiest_hour?fmtHourJS(d.busiest_hour.hour):'—']];
  /* AT LEAST N AT ONCE. A floor three times over -- the detector's recall on a huddle is ~0.39,
     the counting is greedy, and the stills only see instants something was saved. So it says
     "at least", it never names anybody, and it never claims a litter is intact: counting kits
     per mother was tested against the one eye-verified four-animal window and killed, because
     the detector produced ONE box where a human counted four. */
  if(d.crowd&&d.crowd.n>=2){
    const c=d.crowd, mix=Object.entries(c.by_species||{}).filter(([,v])=>v>=1)
      .map(([k,v])=>`${v}× ${nameOf(k)}`).join(', ');
    t.push(['at least '+c.n+' at once',
      `<span title="${esc(`The busiest instant of this ${d.edition}: ${c.n} separate bodies in one frame on ${c.source||'a camera'} at ${(c.at||'').slice(11,19)}${mix?' — '+mix:''}. A LOWER bound — the detector misses animals in a huddle (recall ~0.39), the count is deliberately greedy, and the saved stills only catch instants something was written. It says how many bodies, never whose.`)}">${(c.at||'').slice(11,16)}</span>`]);
  }
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
  const nSurp=d.n_surprising||0, nSp=(d.species||[]).length;
  html+=`<h2 class="sec">The Roll <span class="n">${nSp-nSurp} species${nSurp?` + ${nSurp} surprising`:''}</span>${nSurp?infoDot('A "surprising" species is one whose own record says it is almost never active at this hour — a goldfinch at 2 AM is nearly always a mislabeled crop of something else (kit-melee crops get forced onto the nearest species). They are listed at the bottom as questions, not counted as fauna; their crops are exactly the ones worth correcting in the Catalogue.'):''}</h2>`;
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
  if(s.surprising) badges.push(`<span class="badge" style="background:rgba(255,170,60,.16);border:1px solid rgba(255,170,60,.4)" title="${esc(s.surprise_note||'')}">⚠ surprising — verify</span>`);
  if(n.first_ever&&!s.surprising) badges.push('<span class="badge new">New</span>');
  else if((n.days_since||0)>=3&&!s.surprising) badges.push(`<span class="badge gap">first in ${n.days_since}d</span>`);
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

/* ---------- Seasons: the longitudinal view ----------
   Weekly per-species sparklines + first/last dates + the accumulation curve. The Calendar
   answers "what happened on the 4th?"; this answers "what is my yard DOING this year?" — the
   crows collapsing after mid-July, raccoon traffic doubling when the kits emerged, whether a
   new species is still turning up. Next year, "same week last year" lands here. */
async function loadSeasons(){
  const body=$('#seasons-body'); body.innerHTML='<p class="empty">Leafing back through the weeks…</p>';
  let d;
  try{ d=await fetchJSON('/api/seasons'); connOK(); }
  catch(e){ connFail(); body.innerHTML=errEmpty('loadSeasons()'); return; }
  if(!(d.species||[]).length){ body.innerHTML='<p class="empty">No visits yet — the weekly picture draws itself as the record grows.</p>'; return; }
  const weeks=d.weeks||[];
  const wkLabel=w=>w.slice(5);           // '2026-W27' -> 'W27'
  const spark=(vals)=>{
    const max=Math.max(1,...vals);
    return vals.map((v,i)=>`<span title="${esc(weeks[i])}: ${v} visit${v===1?'':'s'}" style="display:inline-block;width:10px;height:${v?Math.max(2,Math.round(Math.sqrt(v/max)*26)):1}px;background:${v?'var(--gilt)':'rgba(255,255,255,.12)'};opacity:${v?Math.max(.45,v/max):1}"></span>`).join('');
  };
  let html=`<h2 class="sec">The Weeks <span class="n">${weeks.length} week${weeks.length===1?'':'s'} · ${d.species.length} species</span></h2>
    <div class="lbl" style="opacity:.6;margin:-4px 0 10px">${weeks.length?esc(wkLabel(weeks[0])+' → '+wkLabel(weeks[weeks.length-1])):''} · bar height = visits that week</div>
    <div style="display:flex;flex-direction:column;gap:8px">`;
  html+=d.species.map(s=>`<div class="panel" style="display:flex;align-items:center;justify-content:space-between;gap:14px;padding:10px 14px;flex-wrap:wrap">
      <div style="min-width:190px"><div style="font-weight:600">${esc(nameOf(s.species))}</div>
        <div class="lbl" style="opacity:.72">${s.n_visits} visits · first ${esc(s.first||'?')} · last ${esc(s.last||'?')}</div></div>
      <div style="display:flex;align-items:flex-end;gap:2px;height:28px;overflow-x:auto" title="visits per ISO week">${spark(s.weekly)}</div>
    </div>`).join('');
  html+=`</div>`;
  const acc=d.accumulation||[];
  if(acc.length>1){
    const w=680,h=120,px=30,py=12;
    const t0=new Date(acc[0].date).getTime(), t1=new Date(acc[acc.length-1].date).getTime()||t0+1;
    const maxN=acc[acc.length-1].n_species;
    const pts=acc.map(a=>{
      const x=px+((new Date(a.date).getTime()-t0)/Math.max(1,t1-t0))*(w-px-10);
      const y=h-py-(a.n_species/maxN)*(h-2*py);
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    }).join(' ');
    const marks=acc.map(a=>{
      const x=px+((new Date(a.date).getTime()-t0)/Math.max(1,t1-t0))*(w-px-10);
      const y=h-py-(a.n_species/maxN)*(h-2*py);
      return `<circle cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" r="3" fill="var(--gilt)"><title>${esc(a.date)} · #${a.n_species}: ${esc(nameOf(a.species))}</title></circle>`;
    }).join('');
    html+=`<h2 class="sec">Species Accumulation <span class="n">have you seen everything this yard gets?</span>${infoDot('Each dot is a species’ FIRST-EVER appearance in the record. A curve still climbing means the yard is still introducing itself; a flat tail means the regular cast is known and anything new now is real news.')}</h2>
      <div class="panel" style="padding:10px 14px;overflow-x:auto">
        <svg viewBox="0 0 ${w} ${h}" style="width:100%;max-width:${w}px;display:block" role="img" aria-label="species accumulation curve">
          <line x1="${px}" y1="${h-py}" x2="${w-8}" y2="${h-py}" stroke="var(--rule2)"/>
          <polyline points="${pts}" fill="none" stroke="var(--gilt)" stroke-opacity=".6" stroke-width="1.5"/>
          ${marks}
          <text x="2" y="${py+6}" fill="var(--faint)" font-size="9" font-family="var(--mono)">${maxN}</text>
        </svg>
        <div class="lbl" style="opacity:.6;margin-top:4px">${esc(acc[0].date)} → ${esc(acc[acc.length-1].date)} · hover a dot for the debut</div>
      </div>`;
  }
  body.innerHTML=html;
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
    const sa=s.sun_anchor;
    const saTxt=sa?` · ${sa.median_offset_min>=0?'~'+fmtOffset(sa.median_offset_min)+' after':'~'+fmtOffset(-sa.median_offset_min)+' before'} ${sa.anchor}`:'';
    return `<div class="panel" style="display:flex;align-items:center;justify-content:space-between;gap:14px;padding:10px 14px">
      <div><div style="font-weight:600">${esc(nameOf(s.species))}</div>
        <div class="lbl" style="opacity:.72">${s.n_visits} visits · ${s.visits_per_day}/day · dwell ~${dwell} · usually ${wtxt}${saTxt?esc(saTxt):''}${sa?infoDot('Arrivals anchored to the sun instead of the clock: the season-proof reading. This species’ median arrival is '+fmtOffset(Math.abs(sa.median_offset_min))+' '+(sa.median_offset_min>=0?'after':'before')+' civil '+sa.anchor+' (n='+sa.n+'). Clock hours smear as sunset walks across the season; “40 minutes after dusk” stays true all year.'):''}</div></div>
      <div style="display:flex;align-items:flex-end;gap:1px;height:20px" title="arrivals by hour (0–23h); highlighted = typical window">${behaviorClock(hours,win)}</div>
    </div>`;
  }).join('')+`</div>`;
  if((d.co_occurrence||[]).length){
    html+=`<h2 class="sec">Seen Together <span class="n">who shares a visit</span></h2>`;
    html+=`<div class="lede">`+d.co_occurrence.map(c=>
      `<span class="flag">${esc(nameOf(c.a))} + ${esc(nameOf(c.b))} · ${c.n}</span>`).join('')+`</div>`;
  }
  html+=politicsHTML(d.politics);
  html+=moonChartHTML(d.moon);
  body.innerHTML=html;
}
/* Yard politics: the DIRECTIONAL half of "seen together" — who avoids whom, who yields the
   yard. Observational (the payload's own caveat rides in the ⓘ). */
function politicsHTML(p){
  if(!p) return '';
  const sup=(p.suppression||[]), yld=(p.yields||[]);
  if(!sup.length&&!yld.length) return '';
  let html=`<h2 class="sec">Yard Politics <span class="n">who avoids whom, who gives way</span>${infoDot((p.note||'')+' — Suppression: after A has been, B’s next arrival takes X× longer than B’s usual gap. Yielding: B was mid-visit when A arrived and left within three minutes.')}</h2><div class="lede">`;
  html+=sup.map(s=>`<span class="flag quiet" title="${s.n} A-then-B sequences; B's usual gap ${s.baseline_min} min, after ${esc(nameOf(s.a))} it stretches to ${s.after_min} min">${esc(nameOf(s.b))} stays away ${s.factor}× longer after ${esc(nameOf(s.a))} · n=${s.n}</span>`).join('');
  html+=yld.map(y=>`<span class="flag" style="background:rgba(255,170,60,.12);border-color:rgba(255,170,60,.35)" title="${y.n_yield} of ${y.n_encounters} mid-visit encounters ended within 3 minutes of the arrival">${esc(nameOf(y.a))} arrives → ${esc(nameOf(y.b))} leaves (${Math.round(y.rate*100)}% of ${y.n_encounters})</span>`).join('');
  return html+`</div>`;
}
const fmtOffset=m=>{ m=Math.round(Math.abs(m)); return m>=90?`${Math.floor(m/60)}h${String(m%60).padStart(2,'0')}`:`${m} min`; };
/* Nocturnal visits per night vs moon illumination — the glyph upgraded to a question with an
   answer either way ("no lunar effect in this lit yard" is itself a finding worth having). */
function moonChartHTML(moon){
  const rows=(moon&&moon.nights)||[];
  if(rows.length<10) return '';
  const w=680,h=140,px=34,py=14;
  const maxV=Math.max(1,...rows.map(r=>r.n_visits));
  const pts=rows.filter(r=>r.illum_pct!=null).map(r=>{
    const x=px+(r.illum_pct/100)*(w-px-10);
    const y=h-py-(r.n_visits/maxV)*(h-2*py);
    return `<circle cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" r="3.5" fill="var(--gilt)" fill-opacity=".55"><title>${esc(r.night)} · ${r.n_visits} visit${r.n_visits===1?'':'s'} · moon ${r.illum_pct}%</title></circle>`;
  }).join('');
  return `<h2 class="sec">Moonlight <span class="n">does a brighter night change the traffic?</span>${infoDot((moon&&moon.note)||'')}</h2>
    <div class="panel" style="padding:10px 14px;overflow-x:auto">
      <svg viewBox="0 0 ${w} ${h}" style="width:100%;max-width:${w}px;display:block" role="img" aria-label="nocturnal visits per night against moon illumination">
        <line x1="${px}" y1="${h-py}" x2="${w-8}" y2="${h-py}" stroke="var(--rule2)"/>
        <line x1="${px}" y1="${py}" x2="${px}" y2="${h-py}" stroke="var(--rule2)"/>
        <text x="${px}" y="${h-2}" fill="var(--faint)" font-size="9" font-family="var(--mono)">new</text>
        <text x="${w-34}" y="${h-2}" fill="var(--faint)" font-size="9" font-family="var(--mono)">full</text>
        <text x="2" y="${py+6}" fill="var(--faint)" font-size="9" font-family="var(--mono)">${maxV}</text>
        ${pts}
      </svg>
      <div class="lbl" style="opacity:.6;margin-top:4px">each dot is one night · ${rows.length} nights</div>
    </div>`;
}

/* ---------- individuals (phase 3: suggest-confirm loop + hand-label the clusters) ---------- */
/* Which slice of the queue is on screen. 'recent' is the default and is what this panel has
   always shown; the other modes are opt-in filters over the WHOLE pool, and we PAGINATE them
   rather than asking for a huge limit (the server rebuilds the matcher per call). */
let REID_MODE='recent', REID_OFFSET=0;
const REID_LIMIT=30;
let __indivSeq=0;
async function loadIndividuals(){
  const body=$('#indiv-body'); body.innerHTML='<p class="empty">Gathering the suspects…</p>';
  const qs=`?mode=${encodeURIComponent(REID_MODE)}&offset=${REID_OFFSET}&limit=${REID_LIMIT}`
          +(REID_SINCE_H===null?'':`&since_h=${REID_SINCE_H}`);
  const seq=++__indivSeq;
  /* The queue can take SECONDS when the server's cache is cold — 24s measured 2026-08-08 after
     a 2,900-crop family night (the matcher rebuilds whenever the DB changed; the nightly batch
     now pre-warms it, but a mid-day label invalidates again). The cast overview is milliseconds.
     Render what's ready; stream the queue into the page when it lands. */
  const qP=fetch('/api/reid/queue'+qs).then(r=>r.json()).catch(()=>null);
  let d=null;
  try{ d=await fetch('/api/individuals').then(r=>r.json()); }catch(e){ d=null; }
  if(seq!==__indivSeq) return;                  // superseded by a mode/page change
  if(d) renderIndividuals(d, {__pending:true});
  else body.innerHTML='<p class="empty">Matching the recent visits against the cast — this can take a moment after a busy night…</p>';
  const q=await qP;
  if(seq!==__indivSeq) return;
  if(!d && !q){ body.innerHTML='<p class="empty">Could not load individuals.</p>'; return; }
  renderIndividuals(d,q);
}
function reidSetMode(m){ if(m===REID_MODE) return; REID_MODE=m; REID_OFFSET=0; loadIndividuals(); }
function reidPage(delta){
  const n=REID_OFFSET+delta*REID_LIMIT;
  REID_OFFSET=n<0?0:n;
  loadIndividuals();
  const el=document.getElementById('reid-modes'); if(el) el.scrollIntoView({block:'nearest'});
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
/* Temporal context — FOR YOUR EYES ONLY. Adjacent same-camera visits inside an hour turn out to
   be the same animal 67–77% of the time against a 28% base rate, which is worth knowing while
   you look at the photo. It is NOT worth scoring: as a ranking input, under session blocking, it
   fired three times and was wrong three times — a same-night neighbour and a same-night template
   are the same confound. So this renders next to the suggestion and never enters it. */
function reidContextChip(v){
  const c=v.context; if(!c) return '';
  const m=Math.round((c.gap_s||0)/60);
  const when=m<1?'moments':(m===1?'1 min':m+' min');
  const rel=c.direction==='after'?'after':'before';
  const why='Context only — this never changes the appearance ranking. Adjacent visits on the same camera inside an hour are the same animal about 70% of the time, against a 28% base rate, but as an automatic rule it was wrong every time it fired. It renders here for your eye and nowhere else.';
  return `<span class="flag" style="background:rgba(200,180,255,.13);border-color:rgba(200,180,255,.38)" title="${esc(why)}">started ${when} ${rel} the visit you named <b>${esc(c.name)}</b>${infoDot(why)}</span>`;
}
/* WHAT IS ACTUALLY IN THIS VIDEO — three states, and the difference between the last two is the
   entire point of the badge.

   A clip's `dets`/`conf` do not always describe the video. On the trail cam they describe the
   STILL PHOTO that triggered the recording (measured: 1,089 of 1,101 trail-cam clips carry a
   max_confidence byte-identical to some still's), and that camera starts rolling ~2-3 s after
   the photo — long enough for a close animal to leave the frame. The 2026-08-09 01:48 opossum
   visit is the case that prompted this: its stills score 0.93 and its linked 4 s video is empty
   in every frame.

   So "no tracks / no detections" must NOT be rendered as "nothing was there" until something has
   actually looked. `video_checked` is that distinction, and it is why absence of evidence gets
   its own wording instead of borrowing the confident one. */
function clipVideoBadge(c){
  if(!c) return '';
  const F='font-size:12px';
  if(!c.video_checked){
    return `<span class="mono" style="${F};color:var(--faint)" title="Nothing has run a detector over this video's frames yet, so the counts above describe the photo that TRIGGERED the recording — not what the video shows. On the trail cam the camera starts ~2-3s after that photo, so the animal is sometimes already gone.">· video not checked</span>`;
  }
  if(!c.video_dets){
    // DELIBERATELY HEDGED. We cannot say "no animal in this video" and be honest: measured over
    // the 1,244 clips overlapping a human-confirmed animal visit, max-confidence runs p10 0.50 /
    // median 0.80, while the audited empty-video phantom peaks at 0.502. Every bar that rejects
    // the phantom also rejects ~20% of REAL animals, so absence is unprovable from confidence
    // alone. What IS true and worth saying is the thing that started this: the numbers beside
    // this clip describe the still that triggered it, not the video.
    const weak=(c.video_conf!=null)?` (best ${Math.round(c.video_conf*100)}%)`:'';
    return `<span class="mono" style="${F};color:var(--gilt)" title="A detector ran over this video's own frames and found nothing it could call an animal with confidence — but that is NOT proof the yard was empty. On this corpus a fifth of clips containing a confirmed animal also score this low, and the detector fires on dark IR background at similar scores. Treat the stills as the evidence and this video as unconfirmed.">· video unconfirmed${weak}</span>`;
  }
  return `<span class="mono" style="${F};color:var(--ok,#7ec87e)" title="found by running a detector over this video's own frames, independently of the still that triggered it">· ${c.video_dets} in video${c.video_conf!=null?` ~${Math.round(c.video_conf*100)}%`:''}</span>`;
}

function reidCard(v){
  const mins=v.dwell_s>=90? Math.round(v.dwell_s/60)+' min' : (v.dwell_s||0)+'s';
  const thumb=v.rep_crop? `<img src="/media/${encodeURI(v.rep_crop)}" loading="lazy" style="width:84px;height:84px;object-fit:cover;border-radius:6px">` : '';
  const multiEv=[v.co_present_frames? `${v.co_present_frames} still frame(s)`:'', v.co_present_clips? `${v.co_present_clips} clip(s)`:''].filter(Boolean).join(' + ');
  const multiWhy=`${multiEv||'The evidence'} shows two animals at once. Who-arrives-with-whom is real behaviour signal, but the appearance suggestion for this visit is a BLEND of the animals present — name the visit by its main animal, or skip it.`;
  const multi=v.multi? `<span class="flag" style="background:rgba(255,170,60,.16);border-color:rgba(255,170,60,.4)" title="${esc(multiWhy)}">2+ animals${infoDot(multiWhy)}</span>` : '';
  let sugg='', act='';
  if(v.confirmed_as){
    sugg=`<span class="flag" style="background:rgba(90,200,120,.15);border-color:rgba(90,200,120,.45)">= ${esc(v.confirmed_as)} ✓</span>`;
    act=`<button class="gear" onclick="reidConfirm(${v.visit_id},null,true)" title="unconfirm this visit">Clear</button>`;
  }else if(v.rejected){
    // The reject TOMBSTONE (individual_source 'human', id NULL). Without this chip a rejected
    // visit re-renders identical to one nobody has looked at, so the click leaves no trace and
    // the same card asks the same question forever.
    sugg=`<span class="flag" style="background:rgba(150,150,160,.16);border-color:rgba(180,180,190,.4)" title="you left this one unnamed — the nightly auto-assign pass skips it">you left this unnamed</span>`;
    act=`<button class="gear" onclick="reidUnreject(${v.visit_id})" title="undo — put this visit back in play for the nightly pass">↺ undo</button>
         ${reidInput('rq-'+v.visit_id,'or who…')}<button class="gear" onclick="reidConfirm(${v.visit_id})">Name</button>`;
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
  }else if(v.cross_source){
    // A visit from a camera no confirmed template comes from. Measured on this corpus: every
    // trail-cam raccoon prototype scores a median 0.249 (max 0.363) against every glass-door
    // template, and trail-cam-to-trail-cam similarity is FLAT (0.510 for visits minutes apart vs
    // 0.514 for visits days apart — no identity structure to threshold). A top-1 here is noise,
    // and "possibly someone new" would state something about the ANIMAL that is really a fact
    // about the CAMERA. So say the camera thing, and offer your eye instead of a suggestion.
    sugg=`<span class="flag" style="background:rgba(150,150,160,.16);border-color:rgba(180,180,190,.4)" title="no confirmed template comes from this camera, and appearance does not carry between cameras here — cross-camera match scores are indistinguishable from noise. Naming this visit is your judgement of the photo, not a machine suggestion.">no cross-camera match is possible <span style="opacity:.7">· ${esc(v.source||'this camera')}</span></span>`;
    act=`${reidInput('rq-'+v.visit_id,'who is this…')}<button class="gear" onclick="reidConfirm(${v.visit_id})">Name</button>`;
  }else if((v.candidates||[]).length){
    const top=v.candidates[0];
    const rest=v.candidates.slice(1).map(c=>`${esc(c.name)} ${Math.round(c.similarity*100)}%`).join(' · ');
    // The offer carries the LAPSE state of the name it proposes. "Looks like Stan 79%" reads the
    // same whether Stan was confirmed last night or five weeks ago, and those are not the same
    // offer: past the crossing the match is at or below guessing the commonest name. Labelled,
    // never filtered — gating suggestions on template age was measured and rejected.
    sugg=`<span class="flag" title="nearest confirmed visit: #${top.via_visit} (${reidWhen(top.via_started)})">looks like <b>${esc(top.name)}</b> ${Math.round(top.similarity*100)}%</span>`
        +lapseBadge(top.lapse)
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
  // Suppressed on a cross-camera visit for the same reason the still match is: the clip templates
  // come from the other camera too, so a number there is noise wearing a decimal point.
  const clipTop=v.cross_source? null : (v.clip_candidates||[])[0];
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
        <div class="lbl" style="opacity:.72">${mins} · ${v.n_crops} crops · ${v.n_embedded} analysed</div></div>
      <div style="display:flex;gap:6px;align-items:center;flex-wrap:wrap;flex:1">${sugg} ${clipSugg} ${multi} ${reidContextChip(v)}</div>
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
  if(!REID_UNBLEND[vid].length){ box.innerHTML=`<p class="lbl" style="opacity:.7">${esc(d.note||'nothing to separate yet — needs analysed clip tracks (clipmotion + clipembed) or analysed still crops (embed.py --co-present)')}</p>`; return; }
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
  box.innerHTML=`<div class="lbl" style="opacity:.75;margin-bottom:6px">${d.n_tracklets} separated ${d.basis==='stills'?'still':'clip'} track(s) → ${REID_UNBLEND[vid].length} group(s). ${hint}</div>`
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
    <div style="min-width:96px"><div style="font-weight:600">group ${i+1}</div><div class="lbl" style="opacity:.7">${g.n} track(s)${g.n_crops?` · ${g.n_crops} crop(s)`:''} · similarity ${g.cohesion}</div></div>
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
      <div class="lbl" style="opacity:.72">${g.visits.length} visit(s) · ${span} · look-alike ${g.cohesion}${nmulti? ` · ${nmulti}× 2+ animals`:''}</div></div>
    <div style="display:flex;gap:4px;flex-wrap:wrap;flex:1">${thumbs}</div>
    <div style="display:flex;gap:6px;align-items:center">${reidInput('rg-'+i,'name this individual…')}<button class="gear" onclick="reidNameGroup(${i})">Name all</button></div>
  </div>`;
}
/* Re-fit: once a cast exists, sort the unconfirmed remainder into "looks like <name>" buckets
   (bulk-confirm) and candidate-new-individual groups, and flag anyone named only on a pair visit. */
let REID_REFIT=null;
/* Bulk confirms stop at this similarity. A UI guard, not a measured operating point: the card
   below happily offered "all 39 as The Dude" over a 36–68% match range, and wholesale stamping
   in the lookalike zone is exactly how a label set rots (the labels are the only thing the eval
   says works — see Template Freshness). Below the floor you confirm one at a time, eyes on. */
const BULK_FLOOR=0.70;
function reidFitCard(name,bucket){
  const vs=bucket.visits||[];
  const thumbs=vs.slice(0,8).map(x=>x.rep_crop?
    `<img src="/media/${encodeURI(x.rep_crop)}" loading="lazy" alt="visit ${x.visit_id} thumbnail" title="visit #${x.visit_id} · ${Math.round(x.similarity*100)}%" style="width:58px;height:58px;object-fit:cover;border-radius:4px">`:'').join('');
  const lo=Math.round(vs[vs.length-1].similarity*100), hi=Math.round(vs[0].similarity*100);
  const strong=vs.filter(x=>(x.similarity||0)>=BULK_FLOOR).length;
  const weak=vs.length-strong;
  const pct=Math.round(BULK_FLOOR*100);
  const act=strong
    ? `<button class="gear" onclick="reidConfirmFit(${jarg(name)},this)" title="confirm the ${strong} visit(s) matching ${esc(name)} at ${pct}%+ in one go${weak?`. The ${weak} weaker one(s) stay in the queue for one-by-one review — bulk-stamping the lookalike zone is how label sets rot`:''}">✓ the ${strong} strong (≥${pct}%) as ${esc(name)}</button>`
    : `<span class="lbl" style="opacity:.7" title="every match here is below ${pct}% — that's the lookalike zone, so confirm these one at a time in the queue below">all under ${pct}% — review one by one</span>`;
  return `<div class="panel" style="display:flex;align-items:center;gap:12px;padding:10px 14px;flex-wrap:wrap">
    <div style="min-width:150px"><div style="font-weight:600">looks like ${esc(name)}</div>
      <div class="lbl" style="opacity:.72">${vs.length} unconfirmed visit(s) · ${lo}–${hi}% match${weak&&strong?` · ${weak} below the ${pct}% bulk floor`:''}</div></div>
    <div style="display:flex;gap:4px;flex-wrap:wrap;flex:1">${thumbs}</div>
    <div style="display:flex;gap:6px;align-items:center">${act}</div>
  </div>`;
}
function reidNovelCard(g,i){
  const span=`${reidWhen(g.started[0])} → ${reidWhen(g.started[g.started.length-1])}`;
  const nmulti=(g.multi||[]).filter(Boolean).length;
  const thumbs=(g.crops||[]).slice(0,8).map(c=>
    `<img src="/media/${encodeURI(c)}" loading="lazy" style="width:58px;height:58px;object-fit:cover;border-radius:4px">`).join('');
  return `<div class="panel" style="display:flex;align-items:center;gap:12px;padding:10px 14px;flex-wrap:wrap">
    <div style="min-width:170px"><div style="font-weight:600">possible new individual ${i+1}</div>
      <div class="lbl" style="opacity:.72">${g.visits.length} visit(s) · ${span} · look-alike ${g.cohesion}${nmulti?` · ${nmulti}× 2+ animals`:''}</div></div>
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
/* THE FUNNEL, computed live on every load. Where this species' visits actually go — total, how
   many carry an appearance prototype, how many a human has confirmed, and how many of those are
   USABLE templates (solo visits only; a confirmation on a two-animal visit blends two raccoons
   and can never be matched against). The gap between "addressable" and "auto-named" is the
   automation's shortfall, stated as a number instead of a feeling. */
function reidFunnelHTML(f){
  if(!f||!f.visits) return '';
  const step=(n,label,title)=>`<span title="${esc(title)}"><b style="font-family:var(--mono)">${n}</b> <span style="opacity:.72">${esc(label)}</span></span>`;
  const chain=[
    step(f.visits,'visits','every visit of this species in the database'),
    step(f.with_prototype,'with a prototype','have enough embedded crops to be compared at all'),
    step(f.confirmed,'confirmed by you','a human named the visit — the only labels that become templates'),
    step(f.templates,'usable templates','confirmed AND solo: a two-animal visit blends two prototypes, so it is excluded'),
  ].join(' <span style="opacity:.4">→</span> ');
  const side=[`<b style="font-family:var(--mono)">${f.addressable}</b> addressable`,
              `<b style="font-family:var(--mono)">${f.multi_animal}</b> multi-animal`,
              `<b style="font-family:var(--mono)">${f.auto_named}</b> auto-named`,
              f.rejected?`<b style="font-family:var(--mono)">${f.rejected}</b> you left unnamed`:''].filter(Boolean).join(' · ');
  const srcs=(f.by_source||[]).length>1? `<div class="lbl" style="opacity:.72;margin-top:4px">`+
    f.by_source.map(s=>`<span title="${s.templated?'confirmed templates exist for this camera':'NO confirmed template comes from this camera — visits here cannot be matched from appearance, whatever the score says'}">${esc(s.source)}: ${s.visits} visits · ${s.confirmed} confirmed${s.templated?'':' · <b>no templates</b>'}</span>`).join(' &nbsp;·&nbsp; ')+`</div>` : '';
  const funnelWhy='Where this species’ visits actually go. Visits: every visit in the database. With a prototype: enough analysed crops to be compared at all. Confirmed by you: a human named the visit — the only labels that become templates. Usable templates: confirmed AND solo (a two-animal visit blends two animals and can never be matched). Addressable: has a prototype, not yet confirmed, not multi-animal, not left-unnamed — the visits the automatic tier is allowed to look at. The gap between addressable and auto-named is the automation’s shortfall, as a number instead of a feeling.';
  return `<div class="panel" style="padding:8px 14px;margin-bottom:10px">
    <div class="lbl" style="opacity:.85">${chain}${infoDot(funnelWhy)}</div>
    <div class="lbl" style="opacity:.72;margin-top:4px" title="addressable = has a prototype, not yet confirmed, not multi-animal, not left-unnamed — the visits the automatic tier is allowed to look at">${side}</div>
    ${srcs}</div>`;
}
/* TEMPLATE FRESHNESS, and the LAPSE state that now rides with every name.
   Appearance identity decays: session-blocked leave-one-visit-out top-1 goes 0.741 → 0.482 →
   0.259 as the newest usable template ages 0 → 7 → 14 days, against a 0.345 majority-class
   baseline — so somewhere between 10 days (0.403) and 14 (0.259) the matcher stops beating
   "just say the commonest name". That crossing is what `lapse.state` means, and the expected
   top-1 now comes from the SERVER (individuals.identity_lapse) instead of a table hard-coded
   here: the numbers that used to sit in this file were the session-LEAKED ones (0.82 for a
   same-night template), which is the flattering half of the very finding it was quoting. */
function reidFreshTone(days,staleDays,lapse){
  const st=lapse&&lapse.state;
  if(st==='none'||st==='lapsed') return ['rgba(255,120,90,.16)','rgba(255,120,90,.45)'];
  if(st==='fading')              return ['rgba(255,190,80,.15)','rgba(255,190,80,.42)'];
  if(st==='fresh')               return ['rgba(90,200,120,.14)','rgba(90,200,120,.4)'];
  const cut=staleDays||14;       // no server lapse block (an older payload): fall back to the age
  if(days==null) return ['rgba(255,120,90,.16)','rgba(255,120,90,.45)'];
  if(days>=cut)  return ['rgba(255,120,90,.16)','rgba(255,120,90,.45)'];
  if(days>=cut/2)return ['rgba(255,190,80,.15)','rgba(255,190,80,.42)'];
  return ['rgba(90,200,120,.14)','rgba(90,200,120,.4)'];
}
/* One badge, everywhere a name is offered. Says the thing the interface used to leave unsaid:
   the matcher can no longer vouch for this animal, and a human re-anchor is what fixes it. */
function lapseBadge(lapse){
  if(!lapse||lapse.state==='fresh') return '';
  const t={none:['no template','rgba(255,120,90,.16)','rgba(255,120,90,.45)'],
           lapsed:['lapsed','rgba(255,120,90,.16)','rgba(255,120,90,.45)'],
           fading:['fading','rgba(255,190,80,.15)','rgba(255,190,80,.42)']}[lapse.state];
  if(!t) return '';
  return `<span class="flag" style="background:${t[1]};border-color:${t[2]};margin-left:4px"
    title="${esc(lapse.why||'')}">${t[0]}</span>`;
}
/* THE ROSTER. Animals move on, and a stale template is not the same fact as a departed animal —
   you can't confirm a fresh visit for a raccoon that stopped coming. Marking someone gone (with
   the last day you saw them) stops the nightly pass writing that name onto LATER visits, and
   nothing else: they stay in the cast, stay rankable, stay suggestible, and every visit from
   before that date is still theirs to be named. */
function reidDepartedHTML(c){
  const on=c.departed_on?` ${esc(c.departed_on)}`:'';
  return `<span class="flag" style="background:rgba(255,255,255,.05);border-color:var(--rule2);opacity:.85"
    title="${esc(c.name)} is marked as no longer visiting${c.departed_on?`, last here ${esc(c.departed_on)}`:' (no date given, so the nightly pass will never write this name)'}.${c.status_note?' Note: '+esc(c.status_note):''} Visits that started on or before that day can still be named ${esc(c.name)} — by you or by the nightly pass. Later ones cannot: the templates outlive the animal, and a match against them is some other raccoon.">${esc(c.name)} · <b>moved on</b>${on}
    <button class="gear" style="padding:0 5px;margin-left:4px" title="Put ${esc(c.name)} back on the roster" onclick="reidSetResident(${jarg(c.name)})">↺</button></span>`;
}
function reidFreshnessHTML(cast,staleDays){
  // Departed individuals sort to the END: this list is a priority queue for whose template needs
  // refreshing, and nobody can refresh a template for an animal that no longer comes.
  const list=(cast||[]).slice().sort((a,b)=>{
    const ga=a.status==='departed'?1:0, gb=b.status==='departed'?1:0;
    if(ga!==gb) return ga-gb;
    const x=a.days_since_template, y=b.days_since_template;
    if(x==null&&y==null) return b.n_visits-a.n_visits;
    if(x==null) return -1; if(y==null) return 1;
    return y-x;
  });
  if(!list.length) return '';
  const chips=list.map(c=>{
    if(c.status==='departed') return reidDepartedHTML(c);
    const d=c.days_since_template, [bg,bd]=reidFreshTone(d,staleDays,c.lapse);
    const age=d==null?'no usable template':(d<1?'today':Math.round(d)+'d ago');
    const warn=(d==null||d>=(staleDays||14))?' ⚠':'';
    const gone=`<button class="gear" style="padding:0 5px;margin-left:4px" title="${esc(c.name)} isn't coming back? Record the last day you saw them. The nightly pass then stops writing this name onto later visits — everything else is unchanged." onclick="reidSetDeparted(this,${jarg(c.name)},${jarg((c.last_seen||'').slice(0,10))})">moved on?</button>`;
    const et=c.lapse&&c.lapse.expected_top1;
    const freshWhy=`${c.name}: ${c.n_visits} confirmed visit(s), ${c.n_templates||0} of them usable as templates (solo). Newest template ${d==null?'does not exist':reidWhen(c.newest_template)}. ${c.lapse?c.lapse.why:`Identification against a template this old is roughly ${et==null?'nil':et} top-1 — confirm a fresh solo visit for ${c.name} to reset it.`}`;
    return `<span class="flag" style="background:${bg};border-color:${bd}" title="${esc(freshWhy)}">${esc(c.name)} · <b>${age}</b>${warn}<span style="opacity:.6"> · ${c.n_templates||0}/${c.n_visits}</span>${infoDot(freshWhy)}${gone}</span>`;
  }).join(' ');
  return `<h2 class="sec">Template Freshness <span class="n">who the matcher can still recognise — stalest first</span></h2>
    <p class="lbl" style="opacity:.75;margin:2px 0 6px">Appearance goes stale fast on this animal: identification is ~0.74 correct against a template from the same night, ~0.48 at a week, and ~0.26 at a fortnight — which is <em>below</em> simply guessing the commonest name (~0.34). That crossing is what “lapsed” means. The number is days since that individual's newest <b>confirmed solo</b> visit, and the second pair is usable templates / confirmations. If one of them has simply stopped coming, say so — a template outlives the animal, and the nightly pass has no other way to find out.</p>
    <div class="lede" style="margin-bottom:10px">${chips}</div>`;
}
function reidSetDeparted(btn,name,lastSeen){
  /* Inline <input type=date> in place of the old window.prompt: the native date picker makes
     this a two-tap phone flow with validation for free, and it can't be suppressed by an
     in-app browser the way system dialogs can. Prefilled with the animal's last-seen day —
     the answer it almost always is. */
  const wrap=document.createElement('span');
  wrap.style.cssText='display:inline-flex;gap:4px;align-items:center;margin-left:4px';
  wrap.innerHTML=`<input type="date" value="${esc(lastSeen||'')}" style="padding:2px 4px;background:rgba(0,0,0,.25);border:1px solid rgba(255,255,255,.25);border-radius:4px;color:inherit;font:inherit"
      title="the last day ${esc(name)} was here — visits up to and including it can still be named ${esc(name)}; later ones never get auto-named ${esc(name)} again">
    <button class="gear" style="padding:0 5px">✓ moved on</button>
    <button class="gear" style="padding:0 5px" title="never mind">✕</button>`;
  const [ok,cancel]=wrap.querySelectorAll('button');
  const inp=wrap.querySelector('input');
  ok.onclick=async()=>{ const d=(inp.value||'').trim(); if(!d){ inp.focus(); return; }
    await postIndivStatus({name, status:'departed', effective_date:d}); };
  cancel.onclick=()=>{ wrap.replaceWith(btn); };
  btn.replaceWith(wrap);
  inp.focus();
}
async function reidSetResident(name){
  if(!confirm(`Put ${name} back on the roster? The nightly pass may name recent visits ${name} again.`)) return;
  await postIndivStatus({name, status:'resident'});
}
async function postIndivStatus(body){
  const restore=busyBtn();   // dim the clicked roster button while it saves
  try{
    const r=await fetch('/api/individual/status',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify(body)}).then(r=>r.json());
    if(r.error){ restore(); alert(r.error); return; }
    loadIndividuals();   // re-renders the panel; the button goes with it
  }catch(e){ restore(); connFail(); }
}
const REID_MODE_LABELS={
  recent:['recent','the newest visits first — what this panel has always shown'],
  unreviewed_auto:['auto, unreviewed','visits the nightly pass named that you have neither kept nor rejected. Each is one click from a real template or a tombstone.'],
  ambiguous:['ambiguous','the appearance match is strong enough to be somebody on file, but too close between two of them for the machine to call. Exactly what the automatic tier refuses — and the most informative click available, because your eye can settle it and it cannot.'],
  stale:['stale templates','visits whose best candidate is an individual nobody has confirmed lately. Confirming one refreshes the template the next week of matching stands on.'],
};
function reidModesHTML(q){
  const modes=q.modes||['recent'];
  const tabs=modes.map(m=>{
    const [label,tip]=REID_MODE_LABELS[m]||[m,''];
    const on=m===q.mode;
    return `<button class="gear" onclick="reidSetMode(${jarg(m)})" title="${esc(tip)}" style="${on?'background:rgba(120,200,255,.18);border-color:rgba(120,200,255,.5);font-weight:600':''}">${esc(label)}${on?` · ${q.n_matched}`:''}</button>`;
  }).join(' ');
  const from=(q.n_matched?q.offset+1:0), to=Math.min(q.offset+q.limit,q.n_matched);
  const prev=q.offset>0?`<button class="gear" onclick="reidPage(-1)">← newer</button>`:'';
  const next=(q.offset+q.limit)<q.n_matched?`<button class="gear" onclick="reidPage(1)">older →</button>`:'';
  const pager=q.n_matched?`<span class="lbl" style="opacity:.7">${from}–${to} of ${q.n_matched}</span> ${prev} ${next}`:'';
  const modesWhy=(q.modes||['recent']).map(m=>{ const [l,t]=REID_MODE_LABELS[m]||[m,'']; return `${l}: ${t}`; }).join('  •  ');
  return `<div id="reid-modes" style="display:flex;gap:6px;align-items:center;flex-wrap:wrap;margin:2px 0 10px">${tabs}${infoDot('Four slices of the same review pool — '+modesWhy)}
    <span style="flex:1"></span>${pager}</div>`;
}
/* ---------- ONE AT A TIME: the focused per-visit review ----------------------------------
   The queue above renders 30 cards and asks for 30 verdicts at once, which is a wall, not a
   question. This asks ONE question about ONE visit with everything the rig knows in front of
   you -- every crop, every clip, and the same minutes as seen by the OTHER camera.

   The cross-camera panel is the part that isn't cosmetic. A trail-cam visit cannot be
   appearance-matched at all (its prototypes score a median 0.249 against every glass-door
   template, and trail-cam-to-trail-cam similarity is flat), so the only thing that can ever
   name one is a human noticing the glass door saw the same animal in the same minutes. That
   pairing exists for 109 of 521 glass-door raccoon visits and was shown nowhere until now.

   Visit ids churn: visits.py rebuilds and renumbers from scratch (labels live on DETECTIONS and
   survive, ids do not). So a dossier that comes back empty is EXPECTED, not an error -- the flow
   skips it and says the list went stale rather than showing a broken card. */
let REID_FOCUS={ids:[],i:0,on:false,cache:{},stale:false};

function focusStart(ids){
  if(!ids||!ids.length) return;
  REID_FOCUS={ids:ids.slice(),i:0,on:true,cache:{},stale:false};
  focusRender();
  const el=document.getElementById('reid-focus');
  if(el) el.scrollIntoView({behavior:'smooth',block:'start'});
}
function focusExit(){ REID_FOCUS.on=false; loadIndividuals(); }
function focusGo(d){
  const f=REID_FOCUS;
  const next=f.i+d;
  if(next<0||next>=f.ids.length){ focusRender(); return; }
  f.i=next; focusRender();
}
async function focusFetch(vid){
  if(REID_FOCUS.cache[vid]) return REID_FOCUS.cache[vid];
  const d=await fetch('/api/reid/dossier?visit_id='+encodeURIComponent(vid))
    .then(r=>r.json()).catch(()=>null);
  if(d) REID_FOCUS.cache[vid]=d;
  return d;
}
async function focusRender(){
  const host=document.getElementById('reid-focus');
  if(!host||!REID_FOCUS.on) return;
  const f=REID_FOCUS, vid=f.ids[f.i];
  host.innerHTML=`<div class="panel" style="padding:14px"><p class="empty">Loading visit ${f.i+1} of ${f.ids.length}…</p></div>`;
  const d=await focusFetch(vid);
  // A renumbered-away visit: skip forward to the next one that still exists.
  if(!d||!d.visit_id){
    f.stale=true;
    if(f.i+1<f.ids.length){ f.i++; focusRender(); return; }
    host.innerHTML=`<div class="panel" style="padding:14px">
      <p class="empty">This review list is out of date — the visits were renumbered by a rebuild since it was built. Nothing was lost (names live on the crops); the list just needs rebuilding.</p>
      <button class="gear" onclick="focusExit()">↺ rebuild the list</button></div>`;
    return;
  }
  host.innerHTML=focusHTML(d);
}
function focusEvidence(d,idPrefix){
  REID_VISIT_CLIPS[d.visit_id]=d.clips||[];
  const n=(d.clips||[]).length;
  // Say up front how many of this visit's clips actually SHOW the animal, so the reader isn't
  // sent to play an empty video that the still evidence promised something in.
  const checked=(d.clips||[]).filter(x=>x.video_checked);
  const withAnimal=checked.filter(x=>x.video_dets);
  let clipNote='';
  if(n && checked.length===n && !withAnimal.length)
    clipNote=`<span class="lbl" style="opacity:.75;color:var(--gilt)">no animal confirmed in any of these videos — go by the stills</span>`;
  else if(n && checked.length)
    clipNote=`<span class="lbl" style="opacity:.65">${withAnimal.length} of ${checked.length} videos confirm the animal</span>`;
  else if(n)
    clipNote=`<span class="lbl" style="opacity:.55">video not checked yet</span>`;
  const clipBtn=n?`<button class="gear" onclick="reidPlayClips(${d.visit_id})" title="watch the clip${n>1?'s':''} from this visit">▶ ${n} clip${n>1?'s':''}</button>${clipNote}`:'';
  const tiles=(d.crops||[]).map(c=>`<img src="/media/${encodeURI(c.path||c)}" loading="lazy" title="${esc((c.at||'').slice(11,19))}${c.conf?' · '+Math.round(c.conf*100)+'%':''} — click to enlarge" style="width:74px;height:74px;object-fit:cover;border-radius:5px;cursor:zoom-in">`).join('');
  const more=(d.n_crops||0)>(d.crops||[]).length
    ? `<span class="lbl" style="opacity:.6">showing the ${(d.crops||[]).length} sharpest of ${d.n_crops}</span>`:'';
  return `<div style="display:flex;gap:6px;align-items:center;flex-wrap:wrap;margin-top:8px">${clipBtn}${tiles}</div>
          ${more?`<div style="margin-top:5px">${more}</div>`:''}`;
}
function focusHTML(d){
  const f=REID_FOCUS;
  const mins=d.n_crops||0;
  const when=fmtClock(d.started_at);
  const day=(d.started_at||'').slice(0,10);
  const top=(d.candidates||[])[0];
  const clipTop=(d.clip_candidates||[])[0];

  // What the machine thinks -- stated with its own confidence, never as the default answer.
  let hint='';
  if(d.confirmed_as) hint=`<span class="flag" style="background:rgba(90,200,120,.15);border-color:rgba(90,200,120,.45)">already confirmed: <b>${esc(d.confirmed_as)}</b></span>`;
  else if(d.cross_source) hint=`<span class="flag" style="background:rgba(150,150,160,.16);border-color:rgba(180,180,190,.4)" title="no confirmed template comes from this camera; appearance does not carry between cameras here, so any number would be noise">no machine guess is possible for this camera — your eye decides</span>`;
  else if(top) hint=`<span class="flag">best guess <b>${esc(top.name)}</b> ${Math.round(top.similarity*100)}%</span>`
    +(d.novel?`<span class="flag" style="background:rgba(255,120,90,.14);border-color:rgba(255,120,90,.4)">might be someone new</span>`:'')
    +((d.candidates||[]).slice(1,3).map(c=>`<span class="lbl" style="opacity:.6">${esc(c.name)} ${Math.round(c.similarity*100)}%</span>`).join(' '));
  else hint=`<span class="lbl" style="opacity:.65">${esc(d.note||'no guess yet')}</span>`;
  if(clipTop) hint+=`<span class="flag" style="background:rgba(120,160,220,.16);border-color:rgba(120,160,220,.45)" title="a separate signal: appearance in CLIP space, from the un-blended tracklets">clip-match <b>${esc(clipTop.name)}</b> ${Math.round(clipTop.similarity*100)}%</span>`;
  if(d.auto_as) hint+=`<span class="flag" style="background:rgba(120,200,255,.13);border-color:rgba(120,200,255,.42)">the nightly pass guessed <b>${esc(d.auto_as)}</b></span>`;
  if(d.multi){
    const ev=[d.co_present_frames?`${d.co_present_frames} still frame(s)`:'',d.co_present_clips?`${d.co_present_clips} clip(s)`:''].filter(Boolean).join(' + ');
    hint+=`<span class="flag" style="background:rgba(255,170,60,.16);border-color:rgba(255,170,60,.4)" title="${esc((ev||'The evidence')+' shows two animals at once, so a single name describes only the main one. Family labels like “Stan + Kits” are the honest answer here.')}">2+ animals in this visit</span>`;
  }

  // The answer buttons. Solo names first (these are what can become a template), then the
  // family labels, which are right for a mother-and-kits visit and are recorded but never
  // become templates -- a blended prototype would teach the matcher four animals as one.
  const solo=(d.cast||[]).filter(c=>c.name.indexOf(' + ')<0);
  const group=(d.cast||[]).filter(c=>c.name.indexOf(' + ')>=0);
  const btn=c=>{
    const stale=(c.days_since_template!=null&&c.days_since_template>14);
    const tip=c.n_templates?`${c.n_templates} template(s), newest ${c.days_since_template!=null?Math.round(c.days_since_template)+'d old':'—'}${stale?' — naming this visit would refresh a template that has aged past useful':''}`:'recorded, but this label never becomes a matching template';
    return `<button class="gear" onclick="focusAnswer(${jarg(c.name)})" title="${esc(tip)}"
      style="${top&&top.name===c.name?'border-color:rgba(90,200,120,.6);':''}padding:7px 12px">${esc(cap1(c.name))}${stale&&c.n_templates?' <span style="opacity:.6">·stale</span>':''}</button>`;
  };
  const answers=`
    <div style="display:flex;gap:6px;flex-wrap:wrap;margin-top:4px">${solo.map(btn).join('')}</div>
    ${group.length?`<div style="display:flex;gap:6px;flex-wrap:wrap;margin-top:6px;align-items:center">
       <span class="lbl" style="opacity:.6">family:</span>${group.map(btn).join('')}</div>`:''}
    <div style="display:flex;gap:6px;flex-wrap:wrap;margin-top:8px;align-items:center;border-top:1px solid var(--rule);padding-top:8px">
      ${reidInput('focus-new','someone else…')}
      <button class="gear" onclick="focusAnswerTyped()">Name</button>
      <button class="gear" onclick="focusAnswer(null,{reject:true})" title="you looked and chose not to name it — the nightly pass will leave this visit alone">leave unnamed</button>
      <button class="gear" onclick="focusGo(1)" title="decide later — nothing is written">skip →</button>
      ${d.confirmed_as?`<button class="gear" onclick="focusAnswer(null,{clear:true})" title="remove the existing confirmation">clear</button>`:''}
    </div>`;

  // The same minutes on the other camera.
  const nb=(d.neighbours||[]).map(n=>{
    const off=n.offset_s==null?'':(n.offset_s>=0?`+${Math.round(n.offset_s/60)}`:`${Math.round(n.offset_s/60)}`)+' min';
    return `<div class="panel" style="padding:9px 12px;margin-top:6px;background:rgba(120,160,220,.06)">
      <div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap">
        <b>${esc(n.source)}</b>
        <span class="lbl" style="opacity:.75">${esc((n.started_at||'').slice(11,16))} · ${off} · ${n.n_crops} crop(s)</span>
        ${n.individual_id?`<span class="flag" style="background:rgba(90,200,120,.15);border-color:rgba(90,200,120,.45)">named ${esc(n.individual_id)}</span>`:'<span class="lbl" style="opacity:.6">unnamed</span>'}
      </div>
      ${focusEvidence(n)}
    </div>`;
  }).join('');
  const nbBlock=nb?`<h3 class="sec" style="margin-top:14px;font-size:14px">The same moment, other camera
      <span class="n">a trail-cam visit can never be matched by appearance — this pairing is the only way it gets a name</span></h3>${nb}`:'';

  return `<div class="panel" style="padding:14px 16px;border-color:rgba(212,175,110,.35)">
    <div style="display:flex;justify-content:space-between;align-items:center;gap:10px;flex-wrap:wrap">
      <div><b style="font-size:15px">Who is this?</b>
        <span class="lbl" style="opacity:.7;margin-left:8px">visit ${f.i+1} of ${f.ids.length}</span></div>
      <div style="display:flex;gap:6px">
        <button class="gear" onclick="focusGo(-1)"${f.i? '':' disabled'}>← prev</button>
        <button class="gear" onclick="focusGo(1)"${f.i+1<f.ids.length?'':' disabled'}>next →</button>
        <button class="gear" onclick="focusExit()">✕ back to the list</button>
      </div>
    </div>
    <div class="lbl" style="opacity:.8;margin-top:6px">${esc(day)} · ${esc(when)} · ${esc(d.source)} · ${mins} crop(s)${d.n_embedded!=null?` · ${d.n_embedded} analysed for appearance`:''}</div>
    <div style="display:flex;gap:6px;flex-wrap:wrap;margin-top:8px">${hint}</div>
    ${d.rep_crop?`<div style="margin-top:10px"><img src="/media/${encodeURI(d.rep_crop)}" loading="lazy" style="max-width:260px;border-radius:8px;cursor:zoom-in"></div>`:''}
    ${focusEvidence(d)}
    ${nbBlock}
    <h3 class="sec" style="margin-top:14px;font-size:14px">Your call</h3>
    ${answers}
  </div>`;
}
function focusAnswerTyped(){
  const inp=document.getElementById('focus-new');
  const v=(inp&&inp.value||'').trim();
  if(!v){ if(inp) inp.focus(); return; }
  focusAnswer(v);
}
async function focusAnswer(name,opts){
  opts=opts||{};
  const f=REID_FOCUS, vid=f.ids[f.i];
  const restore=busyBtn();
  try{
    const body={visit_id:vid};
    if(opts.reject){ body.name=''; body.reject=true; }
    else if(opts.clear){ body.name=''; }
    else body.name=name;
    const r=await fetch('/api/reid/confirm',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify(body)}).then(r=>r.json());
    restore();
    if(r&&r.error){ alert(r.error); return; }
    delete f.cache[vid];          // its verdict changed; re-fetch if we come back
    if(f.i+1<f.ids.length) focusGo(1); else focusRender();
  }catch(e){ restore(); connFail(); }
}

/* THE WINDOW. The queue opens on the last 48 h because a list of every visit ever is an archive,
   not a work list. Says out loud what it is hiding and offers the way out -- a filter you cannot
   see is indistinguishable from missing data. The window is anchored on the newest visit, not on
   now, so a quiet night (or a rig that was down) never renders an empty queue that reads as
   "nothing left to review". */
let REID_SINCE_H=null;              // null = server default (48)
function reidWindowHTML(q){
  const hidden=(q.n_all||0)-(q.n_in_window||0);
  const cur=q.since_h;
  if(!cur) {
    return `<div class="lbl" style="opacity:.7;margin:2px 0 8px">Showing <b>every</b> visit on record (${q.n_all||0}).
      <button class="gear" onclick="reidSetWindow(48)">back to the last 48h</button></div>`;
  }
  const from=q.window_from?fmtClock(q.window_from):'';
  return `<div class="lbl" style="opacity:.75;margin:2px 0 8px">
    Last <b>${cur}h</b> of visits${from?` (since ${esc(from)})`:''} — ${q.n_in_window||0} shown${hidden>0?`, ${hidden} older hidden`:''}.
    ${hidden>0?`<button class="gear" onclick="reidSetWindow(${cur*2})">show ${cur*2}h</button>
                <button class="gear" onclick="reidSetWindow(0)">show all ${q.n_all}</button>`:''}</div>`;
}
function reidSetWindow(h){ REID_SINCE_H=h; loadIndividuals(); }

function reidQueueHTML(q){
  if(!q) return '';
  if(q.__pending) return `<h2 class="sec">Who Is This? <span class="n">confirm or correct — each answer sharpens the next guess</span></h2>
    <p class="empty">Matching the recent visits against the cast — this can take a moment after a busy night…</p>`;
  q={...q, queue:q.queue||[], bootstrap:q.bootstrap||[]};
  const hasFunnel=!!(q.funnel&&q.funnel.visits);
  if(!q.queue.length&&!q.bootstrap.length&&!q.refit&&!hasFunnel) return '';
  REID_VISIT_CLIPS={};
  let html=`<h2 class="sec">Who Is This? <span class="n">confirm or correct — each answer sharpens the next guess</span></h2>`;
  html+=reidFunnelHTML(q.funnel);
  if(q.unembedded>0) html+=`<p class="lbl" style="opacity:.7;margin:2px 0 10px">⚠ ${q.unembedded} recent crops aren't analysed for appearance yet — naming suggestions sharpen once the re-ID step has run (see the README's “Individual re-identification”).</p>`;
  if(q.bootstrap.length){
    html+=`<p class="lbl" style="opacity:.75;margin:4px 0 10px">Nothing confirmed yet, so here are the corpus' look-alike <b>visit groups</b> (each is probably one animal — your eye decides; skip the 2+-animal ones first pass). Naming a group confirms every visit in it.</p>`;
    html+=`<div style="display:flex;flex-direction:column;gap:8px;margin-bottom:14px">${q.bootstrap.map(reidBootCard).join('')}</div>`;
  }
  html+=reidFreshnessHTML(q.cast,q.stale_days);
  html+=reidRefitHTML(q.refit);
  const cast=(q.cast||[]).map(c=>`<span class="flag"${c.n_auto?` title="${c.n_auto} recent visit${c.n_auto>1?'s':''} auto-named ${esc(c.name)} by the nightly pass — ✓/✗ them on the cards below"`:''}${c.status==='departed'?' style="opacity:.7"':''}>${esc(c.name)} · ${c.n_visits} visit${c.n_visits>1?'s':''}${c.n_auto?` <span style="opacity:.65">+${c.n_auto} auto</span>`:''}${c.status==='departed'?` <span style="opacity:.65">· moved on${c.departed_on?' '+esc(c.departed_on):''}</span>`:''}</span>`).join(' ');
  if(cast||q.queue.length){
    const [label]=REID_MODE_LABELS[q.mode]||['recent'];
    html+=`<h2 class="sec">Visit-by-Visit <span class="n">${esc(label)}</span></h2>`;
    if(cast) html+=`<div class="lede" style="margin-bottom:8px">The cast so far: ${cast}</div>`;
    html+=reidModesHTML(q);
    html+=reidWindowHTML(q);
    // ONE AT A TIME. Offered on the visits that still need a verdict (an already-confirmed one
    // doesn't need asking about), because a stack of 30 cards is a wall rather than a question.
    const askable=q.queue.filter(v=>!v.confirmed_as).map(v=>v.visit_id);
    if(askable.length){
      html+=`<div style="margin:6px 0 10px;display:flex;gap:8px;align-items:center;flex-wrap:wrap">
        <button class="gear" onclick="focusStart(${jarg(askable)})" style="padding:7px 12px;border-color:rgba(212,175,110,.5)">🔍 Review one at a time (${askable.length})</button>
        <span class="lbl" style="opacity:.65">one visit, every crop and clip it has, plus the same minutes on the other camera — then pick a name</span>
      </div>`;
    }
    html+=`<div id="reid-focus"></div>`;
    html+=q.queue.length
      ? `<div style="display:flex;flex-direction:column;gap:8px">${q.queue.map(reidCard).join('')}</div>`
      : `<p class="empty">Nothing in this slice${q.offset?' — try the newer page':''}. That is a real answer: it means there is nothing of this kind left to review.</p>`;
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
  const sp=sel.value;
  if(sp==='__other__'){ otherSpeciesInline(sel, v=>postVisitLabel(target,{species:v},after)); return; }
  if(!sp) return;
  postVisitLabel(target, {species:sp}, after);
}
async function reidConfirmMany(visitIds,name,btn){
  let done=0;
  if(btn) btn.disabled=true;
  try{
    for(const vid of visitIds){
      await fetch('/api/reid/confirm',{method:'POST',headers:{'Content-Type':'application/json'},
        body:JSON.stringify({visit_id:vid,name})});
      done++;
      if(btn) btn.textContent=`saving ${done}/${visitIds.length}…`;   // N sequential POSTs deserve a pulse
    }
    loadIndividuals();
  }catch(e){
    alert(`Could not save: ${e}${done?` — ${done} of ${visitIds.length} were saved before the failure`:''}`);
    loadIndividuals();   // re-render so the saved part shows honestly
  }
}
async function reidConfirmFit(name,btn){
  const b=REID_REFIT&&REID_REFIT.fits&&REID_REFIT.fits[name]; if(!b) return;
  const ids=b.visits.filter(x=>(x.similarity||0)>=BULK_FLOOR).map(x=>x.visit_id);
  if(!ids.length) return;
  // Two taps instead of a system confirm(): the first arms the button in place (styleable,
  // works in the in-app browsers that suppress window.confirm), the second commits.
  if(btn && btn.dataset.armed!=='1'){
    btn.dataset.armed='1'; btn.dataset.label=btn.textContent;
    btn.textContent=`really stamp ${ids.length} as ${name}? tap again`;
    setTimeout(()=>{ if(btn.isConnected&&btn.dataset.armed==='1'){ btn.dataset.armed=''; btn.textContent=btn.dataset.label; } },6000);
    return;
  }
  reidConfirmMany(ids,name,btn);
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
/* Undo a "✗ not them". Clearing WITHOUT reject wipes the tombstone (individual_source goes back
   to NULL), which is what puts the visit back in play for the nightly pass. */
async function reidUnreject(vid){
  const restore=busyBtn();
  try{
    const r=await fetch('/api/reid/confirm',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({visit_id:vid,name:null,reject:false})}).then(r=>r.json());
    if(r.error){ restore(); alert(r.error); return; }
    loadIndividuals();
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
/* THE BADGE FACE. Chosen automatically (stats._avatar_for: human-vouched, sharp, confident,
   well-exposed, squarely framed) but pinnable, because the one thing the picker cannot judge is
   POSE -- facial landmarks are a measured dead end on this cast, since a raccoon's eyes sit
   inside a black mask with almost no local contrast. So it gets you a legible animal and you
   decide which one is iconic. */
async function pinAvatar(name, crop){
  const restore=busyBtn();
  try{
    const r=await fetch('/api/individual/avatar',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({name, crop})}).then(r=>r.json());
    restore();
    if(r&&r.error){ alert(r.error); return; }
    loadIndividuals();
  }catch(e){ restore(); connFail(); }
}
function avatarHTML(g){
  const a=g&&g.avatar;
  if(!a||!a.crop){
    return `<div title="no crop clears the badge bar yet" style="width:78px;height:78px;border-radius:50%;background:rgba(255,255,255,.05);border:1px solid var(--rule);display:flex;align-items:center;justify-content:center;font-size:22px;opacity:.5">?</div>`;
  }
  const tip=a.pinned?`pinned by you${a.at?' — '+esc(fmtClock(a.at)):''}`
                    :`picked automatically${a.at?' — '+esc(fmtClock(a.at)):''}. Open the profile to pin a better one.`;
  return `<img src="/media/${encodeURI(a.crop)}" loading="lazy" title="${tip}"
    onclick="openProfile(${jarg(g.id)})"
    style="width:78px;height:78px;border-radius:50%;object-fit:cover;cursor:pointer;
           border:2px solid ${a.pinned?'var(--gilt)':'var(--rule)'}">`;
}

/* WHEN THIS ONE TURNS UP. A 24-bin sparkline of visit ARRIVALS plus, only when the shape is
   strong enough to mean something, the hours it favours. Arrivals not crops: one animal that sat
   in front of the camera for 40 minutes would otherwise outweigh ten separate visits. */
function clockHTML(ck){
  if(!ck||!ck.n_arrivals) return '';
  const hrs=ck.hours||[], top=Math.max(...hrs,1);
  const peak=new Set(ck.peak_hours||[]);
  const bars=hrs.map((c,h)=>{
    const pct=Math.round(100*c/top);
    return `<span title="${h}:00 — ${c} arrival${c===1?'':'s'}" style="display:inline-block;width:4px;margin-right:1px;
      height:${Math.max(2,Math.round(pct*0.18))}px;vertical-align:bottom;
      background:${peak.has(h)?'var(--gilt)':'rgba(255,255,255,.30)'}"></span>`;
  }).join('');
  const label=(ck.peak_hours||[]).length
    ? `usually around ${ck.peak_hours.map(h=>((h%12)||12)+(h<12?'am':'pm')).join(' & ')}`
    : `${ck.n_arrivals} arrival${ck.n_arrivals===1?'':'s'} — no favoured hour yet`;
  return `<div style="margin-top:5px" title="arrival times across ${ck.n_arrivals} visit(s); midnight at the left">
    <div style="height:20px;line-height:0">${bars}</div>
    <div class="lbl" style="opacity:.7;font-size:11px">${esc(label)}</div></div>`;
}

function indivRow(g){
  const span=(g.first_seen&&g.last_seen)?`${g.first_seen.slice(5,10)} → ${g.last_seen.slice(5,10)}`:'';
  // Each thumb is a one-click "make this the badge" — the pose judgement the picker can't make.
  const pinned=(g.avatar&&g.avatar.crop)||'';
  const thumbs=(g.crops||[]).map(c=>
    `<img src="/media/${encodeURI(c)}" loading="lazy" onclick="pinAvatar(${jarg(g.id)},${jarg(c)})"
      title="use this as ${esc(g.id)}'s badge" style="width:64px;height:64px;object-fit:cover;border-radius:4px;
      cursor:pointer;${c===pinned?'outline:2px solid var(--gilt);outline-offset:1px':''}">`).join('');
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
      ${avatarHTML(g)}
      <div style="min-width:150px"><div style="font-weight:600">${g.placeholder?'<span style="opacity:.65">'+esc(g.id)+'</span>':esc(g.id)}</div>
        <div class="lbl" style="opacity:.72">${esc(nameOf(g.species||''))} · ${g.n_crops} crops · ${esc(span)}</div>
        ${clockHTML(g.clock)}
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
  if(!poses.length){ box.innerHTML='<p class="lbl" style="opacity:.7">Not enough analysed crops to cluster poses yet.</p>'; return; }
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

/* ---------- ignore zones (drawn on a snapshot; the rig applies edits on its next frame) ----------
   The editor lives in the Instrument Panel: a fresh /snapshot.jpg (a STILL, so the scene holds
   still under the pointer), existing zones overlaid on it, drag-to-draw for a new one. The
   snapshot is the full frame at native size, so a drag maps to frame pixels by plain
   proportion; /api/zones' frame dims (the capture thread's own measurement) are preferred over
   the image's naturalWidth only because the latter is 0 until the JPEG decodes. */
let ZONES={ list:[], frame:null, src:null };
function zoneFrameDims(){
  if(ZONES.frame && ZONES.frame.w && ZONES.frame.h) return [ZONES.frame.w, ZONES.frame.h];
  const img=$('#zone-img');
  if(img && img.naturalWidth) return [img.naturalWidth, img.naturalHeight];
  return null;
}
async function loadZones(){
  const src=LIVE.sel; if(!src || !$('#zones-sec')) return;
  ZONES.src=src;
  const img=$('#zone-img'), stage=$('#zone-stage');
  img.onload =()=>{ stage.classList.remove('dead'); $('#zone-nofeed').hidden=true;  renderZones(); };
  img.onerror=()=>{ stage.classList.add('dead');    $('#zone-nofeed').hidden=false; };
  img.src='/snapshot.jpg?source='+encodeURIComponent(src)+'&t='+Date.now();
  let d;
  try{
    const r=await fetch('/api/zones?source='+encodeURIComponent(src));
    if(r.status===404){ zoneMsg('The rig is running an older build — restart it to enable zone editing.'); return; }
    d=await r.json();
  }catch(e){ zoneMsg('Could not reach the rig.'); return; }
  ZONES.list=d.zones||[]; ZONES.frame=d.frame||null;
  renderZones();
}
function zoneMsg(t){ const el=$('#zone-msg'); if(!el) return; el.textContent=t||''; el.hidden=!t; }
function renderZones(){
  const stage=$('#zone-stage'), list=$('#zone-list'); if(!stage||!list) return;
  zoneMsg('');
  $('#zones-n').textContent = ZONES.list.length ? String(ZONES.list.length) : '';
  stage.querySelectorAll('.zone-rect').forEach(el=>el.remove());
  const fd=zoneFrameDims();
  if(fd){
    const [fw,fh]=fd;
    ZONES.list.forEach(z=>{
      const el=document.createElement('div');
      el.className='zone-rect'+(z.stale?' stale':'');
      el.style.left  =(100*z.x1/fw)+'%';        el.style.top   =(100*z.y1/fh)+'%';
      el.style.width =(100*(z.x2-z.x1)/fw)+'%'; el.style.height=(100*(z.y2-z.y1)/fh)+'%';
      el.innerHTML=`<button class="zone-x" type="button" title="Stop ignoring this spot" onclick="zoneDel(${z.id|0})">&#10005;</button>`;
      stage.appendChild(el);
    });
  }
  list.innerHTML = ZONES.list.map(z=>`
    <div class="zone-row">
      <span class="dim">${(z.x2-z.x1)}&times;${(z.y2-z.y1)}</span>
      <span class="znote">${esc(z.note||(z.created_by==='config'?'from config':''))}</span>
      ${z.stale?'<span class="zone-stale-b" title="The camera has been seen to move since this spot was drawn — it may no longer cover the right scenery. Remove it and draw it again.">&#9888; cam moved</span>':''}
      <span class="zage">${esc((z.created_at||'').slice(0,10))}</span>
      <button class="gear" type="button" onclick="zoneDel(${z.id|0})">remove</button>
    </div>`).join('') || '<p class="zone-empty">No ignored spots on this camera.</p>';
}
async function zoneDel(id){
  try{
    const r=await fetch('/api/zones/delete?source='+encodeURIComponent(ZONES.src||''),
      {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({id})});
    if(!r.ok){ const d=await r.json().catch(()=>({})); zoneMsg(d.error||'Could not remove that spot.'); return; }
  }catch(e){ zoneMsg('Could not reach the rig.'); return; }
  loadZones();
}
async function zoneAdd(x1,y1,x2,y2){
  const note=($('#zone-note')&&$('#zone-note').value.trim())||null;
  try{
    const r=await fetch('/api/zones?source='+encodeURIComponent(ZONES.src||''),
      {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({x1,y1,x2,y2,note})});
    const d=await r.json().catch(()=>({}));
    if(!r.ok){ zoneMsg(d.error||'Could not add the spot.'); return; }
    if($('#zone-note')) $('#zone-note').value='';
  }catch(e){ zoneMsg('Could not reach the rig.'); return; }
  loadZones();
}
/* Drag-to-draw. Pointer events cover mouse + touch; capture keeps the band tracking even when
   the pointer leaves the stage mid-drag. A sub-6-px (frame-space) box is a slip, not a zone. */
(function(){
  const stage=$('#zone-stage'); if(!stage) return;
  const band=$('#zone-band');
  let drag=null;
  const pos=e=>{ const r=stage.getBoundingClientRect();
    return [Math.min(Math.max(e.clientX-r.left,0),r.width), Math.min(Math.max(e.clientY-r.top,0),r.height), r]; };
  stage.addEventListener('pointerdown',e=>{
    if(e.target.closest('.zone-x')) return;              // that press is a delete, not a draw
    if(!zoneFrameDims()) return;                         // nothing decoded to draw on yet
    const [x,y]=pos(e); drag=[x,y];
    try{ stage.setPointerCapture(e.pointerId); }catch(_){/* capture is a nicety; the drag still works */}
    band.hidden=false; band.style.left=x+'px'; band.style.top=y+'px'; band.style.width='0'; band.style.height='0';
    e.preventDefault();
  });
  stage.addEventListener('pointermove',e=>{
    if(!drag) return;
    const [x,y]=pos(e);
    band.style.left  =Math.min(drag[0],x)+'px'; band.style.top   =Math.min(drag[1],y)+'px';
    band.style.width =Math.abs(x-drag[0])+'px'; band.style.height=Math.abs(y-drag[1])+'px';
  });
  stage.addEventListener('pointerup',e=>{
    if(!drag) return;
    const [x,y,r]=pos(e), [x0,y0]=drag; drag=null; band.hidden=true;
    const fd=zoneFrameDims(); if(!fd || !r.width || !r.height) return;
    const [fw,fh]=fd, sx=fw/r.width, sy=fh/r.height;
    const x1=Math.round(Math.min(x0,x)*sx), x2=Math.round(Math.max(x0,x)*sx);
    const y1=Math.round(Math.min(y0,y)*sy), y2=Math.round(Math.max(y0,y)*sy);
    if((x2-x1)<6 || (y2-y1)<6) return;
    zoneAdd(x1,y1,x2,y2);
  });
  stage.addEventListener('pointercancel',()=>{ drag=null; band.hidden=true; });
})();

/* ---------- settings popout (scoped to the selected camera) ---------- */
/* ---------- camera management -------------------------------------------------------
   The camera list lives in the `cameras` DB table since 2026-08-22; config_local.py only seeds
   it. Two properties this UI exists to communicate, both of which are easy to get wrong:

     * A SAVE IS NOT LIVE. The rig reads the list once at startup and gives each camera its own
       capture thread, so an edit lands on the next restart. Ignore zones apply immediately and
       cameras do not, and someone who assumes otherwise re-types a password that was already
       right. Hence the banner, and "saved — restart to apply" rather than "done".
     * THE SHORT NAME IS PERMANENT. It is stamped on every detection, visit and clip folder, so
       the field is disabled on an edit rather than merely ignored by the server.

   The password is WRITE-ONLY: the server never sends one back (nothing selects the column), so
   the field is always blank on open and an empty submit means "leave it alone". Setting one is
   loopback-only, and can_set_credentials tells us whether THIS browser may. */

function openCameras(){ const m=$('#cameras'); if(!m) return; m.hidden=false; camCancel(); loadCamsAdmin(); }
function closeCameras(){ const m=$('#cameras'); if(m) m.hidden=true; }
function camsMsg(t,bad){ const el=$('#cams-msg'); if(!el) return;
  el.textContent=t||''; el.hidden=!t; el.classList.toggle('bad',!!bad); }

async function loadCamsAdmin(){
  let d; try{ d=await fetch('/api/cameras').then(r=>r.json()); }catch(e){
    camsMsg('Could not reach the rig.',true); return; }
  CAMS.rows=d.rows||[]; CAMS.pending=!!d.pending_restart;
  CAMS.manageable=!!d.manageable; CAMS.canSecret=!!d.can_set_credentials;
  if(!CAMS.manageable){
    // The server withholds `rows` from a viewer, so an empty list here is a permissions answer,
    // not "there are no cameras". Say which.
    camsMsg('Editing cameras needs the operator token — enter it from the footer link.');
    const add=$('#cams-add'); if(add) add.hidden=true;
  }
  renderCams();
}

function renderCams(){
  const list=$('#cams-list'); if(!list) return;
  const running=new Set((LIVE.cams||[]).map(c=>c.source));
  const banner=$('#cams-restart');
  if(banner){
    banner.hidden=!CAMS.pending;
    banner.textContent='Saved. These changes reach the rig the next time it restarts — '
      + 'a camera needs its own capture thread, so unlike ignored spots it cannot be added to a running rig.';
  }
  list.innerHTML = CAMS.rows.map(c=>{
    const addr = c.kind==='local' ? ('USB index '+(c.device_index!=null?c.device_index:'?'))
      : `${esc(c.url_scheme||'rtsp')}://${esc(c.url_host||'')}${c.url_port?':'+c.url_port:''}/${esc(c.url_path||'')}`;
    const flags=[];
    if(!c.enabled) flags.push('<span class="cam-flag off">disabled</span>');
    if(!running.has(c.source)) flags.push('<span class="cam-flag off">not running</span>');
    if(c.kind==='network') flags.push(c.has_password
      ? '<span class="cam-flag">password set</span>'
      : '<span class="cam-flag warn">no password</span>');
    if(c.created_by==='config') flags.push('<span class="cam-flag">from config</span>');
    return `<div class="cam-row">
      <div class="cam-main">
        <span class="cam-name">${esc(c.name||c.source)}</span>
        <span class="cam-src">${esc(c.source)}</span>
        <span class="cam-addr">${addr}</span>
      </div>
      <div class="cam-flags">${flags.join('')}</div>
      <div class="cam-buttons">
        <button class="gear" type="button" onclick="camEdit(${c.id|0})">edit</button>
        <button class="gear" type="button" onclick="camDelete(${c.id|0})">remove</button>
      </div>
    </div>`;
  }).join('') || '<p class="zone-empty">No cameras yet.</p>';
}

function camKindChanged(){
  const net = $('#cam-kind').value === 'network';
  $('#cam-net-fields').hidden = !net;
  $('#cam-local-fields').hidden = net;
  const note=$('#cam-pw-note');
  if(note){
    note.innerHTML = CAMS.canSecret
      ? 'Stored on the rig and never sent back to a browser &mdash; this box stays blank even when a password <em>is</em> set. Leave it blank to keep the current one.'
      : '<strong>You cannot set a password from here.</strong> Camera passwords can only be entered on the rig machine itself, because this dashboard is served over the network without encryption. Everything else on this form saves normally.';
    note.classList.toggle('bad', !CAMS.canSecret);
  }
  const pw=$('#cam-password'); if(pw){ pw.disabled=!CAMS.canSecret; }
}

function camForm(show){ const f=$('#cams-form'); if(f) f.hidden=!show;
  const b=$('#cams-add'); if(b) b.hidden=show; }

function camNew(){
  CAMS.editing=null;
  $('#cams-form-title').textContent='New camera';
  ['cam-id','cam-name','cam-source','cam-host','cam-port','cam-path','cam-username',
   'cam-password','cam-device-index','cam-fw','cam-fh','cam-area'].forEach(id=>{ const el=$('#'+id); if(el) el.value=''; });
  $('#cam-source').disabled=false;
  $('#cam-source-row').hidden=false;
  $('#cam-source-note').hidden=false;
  $('#cam-kind').value='network';
  $('#cam-clips').value=''; $('#cam-enabled').checked=true;
  camKindChanged(); camsMsg(''); camForm(true);
}

function camEdit(id){
  const c=CAMS.rows.find(r=>r.id===id); if(!c) return;
  CAMS.editing=c;
  $('#cams-form-title').textContent='Edit '+(c.name||c.source);
  $('#cam-id').value=c.id;
  $('#cam-name').value=c.name||'';
  $('#cam-source').value=c.source;
  // Permanent, so it is disabled rather than merely rejected by the server on submit.
  $('#cam-source').disabled=true;
  $('#cam-source-note').hidden=false;
  $('#cam-kind').value=c.kind||'network';
  $('#cam-device-index').value=(c.device_index!=null?c.device_index:'');
  $('#cam-host').value=c.url_host||''; $('#cam-port').value=(c.url_port!=null?c.url_port:'');
  $('#cam-path').value=c.url_path||''; $('#cam-username').value=c.username||'';
  $('#cam-password').value='';                    // never prefilled: the server does not send it
  $('#cam-fw').value=(c.frame_width!=null?c.frame_width:'');
  $('#cam-fh').value=(c.frame_height!=null?c.frame_height:'');
  $('#cam-area').value=(c.motion_min_area!=null?c.motion_min_area:'');
  // '' = inherit, and null really does mean inherit -- not "off".
  $('#cam-clips').value = c.record_clips===null||c.record_clips===undefined ? ''
                        : (c.record_clips ? '1' : '0');
  $('#cam-enabled').checked = c.enabled!==false;
  camKindChanged(); camsMsg(''); camForm(true);
}

function camCancel(){ camForm(false); camsMsg(''); CAMS.editing=null; }

function camSave(ev){
  if(ev) ev.preventDefault();
  const num=id=>{ const v=($('#'+id).value||'').trim(); return v===''?null:Number(v); };
  const body={
    kind: $('#cam-kind').value,
    name: ($('#cam-name').value||'').trim() || null,
    frame_width: num('cam-fw'), frame_height: num('cam-fh'),
    motion_min_area: num('cam-area'),
    record_clips: ($('#cam-clips').value===''? null : $('#cam-clips').value==='1'),
    enabled: $('#cam-enabled').checked,
  };
  if(CAMS.editing) body.id=CAMS.editing.id;
  else body.source=($('#cam-source').value||'').trim();
  if(body.kind==='local'){ body.device_index=num('cam-device-index'); }
  else{
    body.url_scheme='rtsp';
    body.url_host=($('#cam-host').value||'').trim();
    body.url_port=num('cam-port');
    body.url_path=($('#cam-path').value||'').trim() || null;
    body.username=($('#cam-username').value||'').trim() || null;
    const pw=$('#cam-password').value||'';
    if(pw) body.password=pw;                      // absent = keep whatever is stored
  }
  fetch('/api/cameras/save',{method:'POST',headers:{'Content-Type':'application/json'},
                             body:JSON.stringify(body)})
    .then(r=>r.json().then(d=>({ok:r.ok,d})))
    .then(({ok,d})=>{
      if(!ok){ camsMsg(d.error||'Could not save that camera.',true); return; }
      $('#cam-password').value='';                // don't leave a secret sitting in the DOM
      camCancel();
      loadCamsAdmin();
    })
    .catch(()=>camsMsg('Could not reach the rig.',true));
  return false;
}

function camDelete(id){
  const c=CAMS.rows.find(r=>r.id===id); if(!c) return;
  if(!confirm(`Remove ${c.name||c.source}?\n\nThe photos, visits and clips it already recorded are kept — they stay filed under "${c.source}". Adding a camera with that same short name later reattaches to them.`)) return;
  fetch('/api/cameras/delete',{method:'POST',headers:{'Content-Type':'application/json'},
                               body:JSON.stringify({id})})
    .then(r=>r.json().then(d=>({ok:r.ok,d})))
    .then(({ok,d})=>{ if(!ok){ camsMsg(d.error||'Could not remove that camera.',true); return; }
                      // Close an edit form still bound to the row we just removed: saving it
                      // afterwards would recreate the camera by undelete.
                      if(CAMS.editing && CAMS.editing.id===id) camCancel();
                      loadCamsAdmin(); })
    .catch(()=>camsMsg('Could not reach the rig.',true));
}

function openSettings(source){ if(source) selectCamera(source); const m=$('#settings'); if(m) m.hidden=false; refreshControls(); loadZones(); }
function closeSettings(){ const m=$('#settings'); if(m) m.hidden=true; }
document.addEventListener('keydown',e=>{ if(e.key==='Escape'){ closeSettings(); closeCameras(); } });

/* ---------- boot ---------- */
buildControls();
async function refreshNaming(){
  let n; try{ n=await fetch('/api/naming').then(r=>r.json()); connOK(); }catch(e){ connFail(); return; }
  const el=document.getElementById('naming'); if(!el) return;
  const map={ loading:['#d9a23b','Identifier: warming up…'],
              ready:['#5b8c5a','Identifier: on'],
              stopped:['#b4503f','Identifier: stopped'] };
  window.__frNaming=n.state;          // first-run checklist reads this
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
function _frChecklist(){
  /* The newcomer's most fragile hour is "is it working?" -- answer it with live state the page
     already polls, instead of leaving them to README archaeology. Each row re-renders on the
     stats poll, so a fixed camera or a finished model download ticks itself green. */
  const cam=window.__frCamera;                      // stashed by refreshLive's checkFeeds pass
  const naming=window.__frNaming;                   // stashed by refreshNaming
  const geo=window.__frGeo;                         // stashed by refreshHeader (lat/lon set?)
  const row=(state,txt)=>'<div class="fr-row">'
    +'<span class="fr-dot" style="color:'+(state==='ok'?'#5b8c5a':state==='warn'?'#d9a23b':'#8a7d63')+'">'
    +(state==='ok'?'●':state==='warn'?'●':'○')+'</span> '+txt+'</div>';
  let rows='';
  rows+=row(cam===true?'ok':cam===false?'warn':'wait',
    cam===true?'Camera connected — frames are coming in.'
    :cam===false?'No camera frames yet. Still plugging in? <code>python backyard_cam.py --list-cameras</code> finds the right index; set <code>camera_index</code> in config_local.py.'
    :'Checking the camera…');
  rows+=row(naming==='ready'?'ok':naming==='loading'?'warn':'wait',
    naming==='ready'?'Species identifier is on — new visitors get named automatically.'
    :naming==='loading'?'Species identifier is downloading/loading its models (first run can take minutes on the model download — gigabytes, one time only).'
    :'Species identifier: not running (it starts with the rig by default).');
  rows+=row(geo?'ok':'warn',
    geo?'Location set — day/night editions and sun-aware features are on.'
    :'No latitude/longitude yet — day/night features are off. Two lines in config_local.py turn them on.');
  return rows;
}
function maybeFirstRun(s){
  const el=document.getElementById('firstrun'); if(!el) return;
  const empty=!(s&&s.total_crops);
  if(empty && localStorage.getItem('cc-introDismissed')!=='1'){
    // A brand-new, empty rig should see the welcome card (it lives in the Live view), not the
    // empty Dispatch we land on by default. Redirect once; afterwards the user can navigate freely.
    if(!__firstRunLanded){ __firstRunLanded=true; show('live'); }
    el.innerHTML='<div class="fr-card"><div class="fr-title">Welcome to your Backyard Observatory</div>'
      +'<p>The camera watches the yard by itself &mdash; as animals visit, this log fills in: their photographs, the species name, and over days, who comes and when.</p>'
      +_frChecklist()
      +'<p class="fr-soft">Try it now: wave at the camera. You&rsquo;ll be drawn on the live view but not saved &mdash; the rig only keeps animals. Then leave it running and check back later. '
      +'Curious how any of it works? <a href="/making-of/" target="_blank" rel="noopener">The making-of site</a> walks the whole pipeline on real data.</p>'
      +'<button class="fr-x" type="button" onclick="dismissIntro()">Got it</button></div>';
    el.hidden=false;
  } else { el.hidden=true; }
}
function dismissIntro(){ localStorage.setItem('cc-introDismissed','1'); const el=document.getElementById('firstrun'); if(el) el.hidden=true; }

loadCameras(); refreshLive(); refreshHeader(); refreshNaming(); refreshWhoshere(); refreshEvalStatus(); refreshRole(); refreshLabeler();
// Land wherever the URL hash points — a tab, a profile, a day, a species sheet, a dated
// dispatch (deep links and refresh keep their place); else the Visit Log — the
// scroll-around-and-see-what-happened surface. maybeFirstRun still redirects a brand-new
// empty rig to Live.
if(!applyHash(location.hash)) show('visits', true);
/* Pollers pause while the page is hidden. A phone left on the dashboard in a background tab was
   hitting three endpoints every few seconds ALL NIGHT — battery there, steady wakeups on the
   same box that runs the detector here. The MJPEG streams already detach off-tab
   (syncLiveStreams) and whoshere/checkFeeds early-return off their tab; this completes the
   pattern for the base pollers. On becoming visible again everything refreshes at once, so the
   page feels instantly fresh instead of up-to-6-seconds stale. */
function vispoll(fn,ms){ setInterval(()=>{ if(!document.hidden) fn(); },ms); }
vispoll(refreshLive,6000);
vispoll(refreshHeader,4000);
vispoll(refreshNaming,4000);
vispoll(checkFeeds,8000);
vispoll(refreshWhoshere,6000);
vispoll(()=>{ const m=$('#settings'); if(m && !m.hidden) refreshControls(); },2000);   // live controls while the panel's open
vispoll(refreshEvalStatus,30*60*1000);   // the eval artifact changes once a day (~2pm batch)
document.addEventListener('visibilitychange',()=>{
  if(document.hidden) return;
  refreshLive(); refreshHeader(); refreshNaming(); refreshWhoshere(); checkFeeds();
});
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
    // ← → walk THE STRIP YOU CLICKED FROM. That used to mean "every /media/ image in the
    // document", which was the same thing back when one grid was ever rendered -- but every tab
    // keeps its DOM (views are toggled, not emptied), so arrowing off the end of a visit's
    // photographs would silently walk into the Favourites album or a species sheet. Scope to the
    // enclosing grid, then to the tab, and only then to the page.
    const strip = el.closest('.crops, .refstrip, .profile-refs') || el.closest('.view') || document;
    gallery=[...strip.querySelectorAll('img[src*="/media/"]')].filter(x=>!lb.contains(x));
    const at=gallery.indexOf(el);
    // ...unless the clicked image is no longer IN the page. That happens when something re-renders
    // the grid out from under the click, and the old code fell back to gallery[0] -- opening a
    // photo the user never clicked, from whatever tab happened to be first in document order.
    // Showing the clicked image alone is the only honest answer: never open something else.
    if(at<0){ gallery=[el]; showIdx(0); }
    else showIdx(at);
    lb.classList.add('open');
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
    if(!(t&&t.tagName==='IMG'&&!lb.contains(t)&&/\/media\//.test(t.getAttribute('src')||''))) return;
    // An image carrying its OWN click action means that action, not "enlarge": the cast badge
    // opens a profile, an individual's thumb pins it as the badge. Leave those alone -- before
    // this, they did BOTH, so one tap pinned an avatar and opened a lightbox on top of it.
    if(t.onclick) return;
    // Otherwise: enlarge, and let nothing else happen. CAPTURE + stopPropagation is what makes
    // that true -- from the bubble phase this ran LAST, so an enclosing card (a crop tile in the
    // photo grid, which drills into its species) had already navigated away, replacing the grid
    // and leaving the lightbox to open on a detached image. That is the "I clicked a photo and it
    // took me somewhere else" bug. stopPropagation halts the walk to other NODES, not the other
    // document-level listeners, so the ⓘ popover still closes on the same click.
    e.preventDefault(); e.stopPropagation();
    openFrom(t);
  }, true);
})();

/* ---------- info popovers: the tappable ⓘ ----------
   The load-bearing explanations used to live ONLY in title= tooltips, which do not exist on a
   phone — so the most-explained tab in the codebase read as pure jargon on the device the
   family actually uses. infoDot() renders a small ⓘ whose text opens in a real popover on
   tap/click/Enter; a title= on the host element stays as the desktop hover shortcut. */
function infoDot(text){ return ` <span class="infodot" role="button" tabindex="0" aria-label="explain" data-info="${esc(text)}">ⓘ</span>`; }
(function(){
  const pop=document.createElement('div');
  pop.className='infopop'; pop.hidden=true;
  document.body.appendChild(pop);
  let openFor=null;
  function closePop(){ pop.hidden=true; openFor=null; }
  function openPop(dot){
    pop.textContent=dot.dataset.info||'';
    pop.hidden=false; openFor=dot;
    pop.style.maxWidth=Math.min(340, window.innerWidth-24)+'px';
    const r=dot.getBoundingClientRect(), ph=pop.offsetHeight, pw=pop.offsetWidth;
    const x=Math.min(Math.max(8, r.left), window.innerWidth-pw-8);
    let y=r.bottom+6; if(y+ph>window.innerHeight-8) y=Math.max(8, r.top-ph-6);
    pop.style.left=x+'px'; pop.style.top=y+'px';
  }
  document.addEventListener('click',e=>{
    const dot=e.target.closest&&e.target.closest('.infodot');
    if(dot){ e.stopPropagation(); e.preventDefault(); (openFor===dot&&!pop.hidden)?closePop():openPop(dot); return; }
    if(!pop.hidden) closePop();
  }, true);   // capture: the dot often sits inside cards with their own click handlers
  document.addEventListener('keydown',e=>{ if(e.key==='Escape') closePop(); });
  window.addEventListener('scroll',()=>{ if(!pop.hidden) closePop(); }, {passive:true});
})();

/* ---------- keyboard floor ----------
   The click-only cards become real tab stops. A MutationObserver stamps role/tabindex onto the
   recurring card selectors as they render (many render paths, one wiring point), and one
   delegated keydown makes Enter/Space activate whatever a click would. */
(function(){
  const SEL='.vcard,.daycard,.rc-card,.crop,.card[data-sp],.fs,.cal-cell[data-day],.tally[onclick]';
  const stampOne=el=>{ if(!el.hasAttribute('tabindex')){ el.setAttribute('tabindex','0'); el.setAttribute('role','button'); } };
  const stamp=root=>{ if(root.querySelectorAll) root.querySelectorAll(SEL).forEach(stampOne); };
  stamp(document);
  new MutationObserver(muts=>muts.forEach(m=>m.addedNodes.forEach(n=>{
    if(n.nodeType!==1) return;
    if(n.matches&&n.matches(SEL)) stampOne(n);
    stamp(n);
  }))).observe(document.body,{childList:true,subtree:true});
  document.addEventListener('keydown',e=>{
    if(e.key!=='Enter'&&e.key!==' ') return;
    const t=e.target;
    if(!t.closest||['BUTTON','A','INPUT','SELECT','TEXTAREA'].includes(t.tagName)) return;
    const el=t.closest(SEL+',.infodot');
    if(!el||el!==t) return;              // only when the stamped element itself holds focus
    e.preventDefault();
    if(typeof el.onclick==='function'||el.hasAttribute('onclick')||el.classList.contains('infodot')){ el.click(); return; }
    const img=el.querySelector('img[src*="/media/"]'); if(img){ img.click(); return; }
    el.click();
  });
})();
