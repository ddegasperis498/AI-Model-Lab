window.AIML = window.AIML || {};
(() => {
  'use strict';
  const A = window.AIML;
  const state = { depth: 0, retry: null, started: 0 };

  const esc = s => String(s ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));

  function ensureUI(){
    if(document.getElementById('aimlLoadingOverlay')) return;
    document.body.insertAdjacentHTML('beforeend', `
      <div id="aimlLoadingOverlay" class="aiml-overlay hidden" aria-live="polite" aria-busy="true">
        <div class="aiml-loading-card glass">
          <div class="aiml-spinner" aria-hidden="true"></div>
          <div class="min-w-0 flex-1">
            <div id="aimlLoadingTitle" class="text-lg font-semibold text-slate-100">Operazione in corso</div>
            <div id="aimlLoadingMessage" class="text-sm text-slate-400 mt-1">Attendere…</div>
            <div id="aimlLoadingMeta" class="text-xs font-mono text-blue-300 mt-3"></div>
            <div class="aiml-progress mt-3"><span id="aimlProgressBar"></span></div>
            <div id="aimlElapsed" class="text-[11px] text-slate-600 mt-2">0.0 s</div>
          </div>
        </div>
      </div>
      <div id="aimlErrorOverlay" class="aiml-overlay hidden">
        <div class="aiml-error-card glass" role="dialog" aria-modal="true" aria-labelledby="aimlErrorTitle">
          <div class="flex items-start gap-3">
            <div class="aiml-error-icon">!</div>
            <div class="min-w-0 flex-1">
              <div class="small-label text-rose-300">Operazione non completata</div>
              <h3 id="aimlErrorTitle" class="text-xl font-semibold mt-1">Errore</h3>
              <p id="aimlErrorMessage" class="text-sm text-slate-300 mt-2"></p>
            </div>
          </div>
          <div class="mt-4 rounded-2xl border border-slate-800 bg-black/20 p-3 text-xs font-mono space-y-1">
            <div><span class="text-slate-600">HTTP:</span> <span id="aimlErrorStatus">—</span></div>
            <div><span class="text-slate-600">Endpoint:</span> <span id="aimlErrorEndpoint">—</span></div>
            <div><span class="text-slate-600">ID:</span> <span id="aimlErrorCorrelation">—</span></div>
            <div><span class="text-slate-600">Ora:</span> <span id="aimlErrorTime">—</span></div>
          </div>
          <details id="aimlErrorDetailsBox" class="mt-3 rounded-xl border border-slate-800 bg-slate-950/40 p-3">
            <summary class="cursor-pointer text-sm text-slate-400">Mostra dettagli tecnici</summary>
            <pre id="aimlErrorDetails" class="scrollbar overflow-auto mt-3 max-h-56 text-xs whitespace-pre-wrap text-rose-200"></pre>
          </details>
          <div class="flex flex-wrap justify-end gap-2 mt-5">
            <button id="aimlErrorCopy" class="rounded-xl bg-slate-800 border border-slate-700 px-3 py-2 text-sm">Copia dettagli</button>
            <button id="aimlErrorRetry" class="rounded-xl bg-purple-600 px-3 py-2 text-sm hidden">Riprova</button>
            <button id="aimlErrorClose" class="rounded-xl bg-blue-600 px-4 py-2 text-sm">Chiudi</button>
          </div>
        </div>
      </div>
      <div id="aimlLiveBadge" class="aiml-live-badge hidden"><span class="aiml-live-dot"></span><span id="aimlLiveText">Training live</span></div>
    `);
    document.getElementById('aimlErrorClose').onclick = hideError;
    document.getElementById('aimlErrorOverlay').addEventListener('click',e=>{ if(e.target.id==='aimlErrorOverlay') hideError(); });
    document.getElementById('aimlErrorCopy').onclick = async()=>{
      const text = buildErrorText(lastError);
      try{ await navigator.clipboard.writeText(text); }catch{}
    };
    document.getElementById('aimlErrorRetry').onclick = ()=>{ const r=state.retry; hideError(); if(r) r(); };
  }

  let ticker=null, lastError=null;
  function showLoading(meta={}){
    ensureUI(); state.depth++; state.started=performance.now();
    const o=document.getElementById('aimlLoadingOverlay'); o.classList.remove('hidden');
    document.getElementById('aimlLoadingTitle').textContent=meta.title||'Operazione in corso';
    document.getElementById('aimlLoadingMessage').textContent=meta.message||'Attendere…';
    document.getElementById('aimlLoadingMeta').textContent=meta.meta||'';
    document.getElementById('aimlProgressBar').style.width=(meta.progress??12)+'%';
    clearInterval(ticker); ticker=setInterval(()=>{
      const s=(performance.now()-state.started)/1000;
      const el=document.getElementById('aimlElapsed'); if(el) el.textContent=`${s.toFixed(1)} s`;
    },100);
  }
  function updateLoading(meta={}){
    if(meta.title) document.getElementById('aimlLoadingTitle').textContent=meta.title;
    if(meta.message) document.getElementById('aimlLoadingMessage').textContent=meta.message;
    if(meta.meta!==undefined) document.getElementById('aimlLoadingMeta').textContent=meta.meta;
    if(meta.progress!==undefined) document.getElementById('aimlProgressBar').style.width=Math.max(0,Math.min(100,meta.progress))+'%';
  }
  function hideLoading(force=false){
    state.depth=force?0:Math.max(0,state.depth-1); if(state.depth>0)return;
    clearInterval(ticker);ticker=null;document.getElementById('aimlLoadingOverlay')?.classList.add('hidden');
  }
  function buildErrorText(e){
    if(!e)return '';
    return [`AI Model Lab V4.1`,`Operazione: ${e.operation||'—'}`,`HTTP: ${e.status||'—'}`,`Endpoint: ${e.endpoint||'—'}`,`Correlation ID: ${e.correlationId||'—'}`,`Ora: ${e.time||new Date().toISOString()}`,`Messaggio: ${e.message||'Errore'}`,`Dettaglio: ${e.detail||''}`].join('\n');
  }
  function showError(err={}, retry=null){
    ensureUI(); hideLoading(true);
    lastError={...err,time:err.time||new Date().toISOString()}; state.retry=retry;
    document.getElementById('aimlErrorTitle').textContent=err.operation||'Errore';
    document.getElementById('aimlErrorMessage').textContent=err.message||'Il backend ha restituito un errore.';
    document.getElementById('aimlErrorStatus').textContent=err.status||'—';
    document.getElementById('aimlErrorEndpoint').textContent=err.endpoint||'—';
    document.getElementById('aimlErrorCorrelation').textContent=err.correlationId||'—';
    document.getElementById('aimlErrorTime').textContent=new Date(lastError.time).toLocaleString('it-IT');
    document.getElementById('aimlErrorDetails').textContent=err.detail||err.stack||err.message||'';
    document.getElementById('aimlErrorRetry').classList.toggle('hidden',!retry);
    document.getElementById('aimlErrorOverlay').classList.remove('hidden');
  }
  function hideError(){ document.getElementById('aimlErrorOverlay')?.classList.add('hidden'); state.retry=null; }
  function setButtonBusy(button,busy,label='Elaborazione…'){
    if(!button)return;
    if(busy){button.dataset.originalHtml=button.innerHTML;button.disabled=true;button.classList.add('opacity-70','cursor-wait');button.innerHTML=`<span class="aiml-mini-spinner"></span>${esc(label)}`;}
    else{button.disabled=false;button.classList.remove('opacity-70','cursor-wait');if(button.dataset.originalHtml)button.innerHTML=button.dataset.originalHtml;delete button.dataset.originalHtml;}
  }
  function setLive(active,text='Training live'){
    ensureUI(); const b=document.getElementById('aimlLiveBadge'); b.classList.toggle('hidden',!active);document.getElementById('aimlLiveText').textContent=text;
  }

  class ApiError extends Error{
    constructor(message,meta={}){super(message);Object.assign(this,meta);this.name='ApiError';}
  }

  async function request(url, options={}, meta={}){
    ensureUI();
    const silent=!!meta.silent; const button=meta.button||null; const timeoutMs=meta.timeoutMs||60000;
    if(!silent) showLoading(meta);
    if(button) setButtonBusy(button,true,meta.busyLabel||'Attendere…');
    const controller=new AbortController();const timer=setTimeout(()=>controller.abort(),timeoutMs);
    const headers={'Content-Type':'application/json',...(options.headers||{})};
    const correlationId=(crypto.randomUUID?crypto.randomUUID():Math.random().toString(16).slice(2));headers['X-Correlation-ID']=correlationId;
    try{
      const res=await fetch(url,{...options,headers,signal:controller.signal});
      const text=await res.text(); let data={};
      try{data=text?JSON.parse(text):{};}catch{data={raw:text};}
      if(!res.ok){
        const be=data?.error||{}; const detail=be.detail ?? data?.detail ?? data?.raw ?? res.statusText;
        throw new ApiError(be.message||data?.detail||`HTTP ${res.status}`,{status:res.status,endpoint:url,operation:meta.operation||`${options.method||'GET'} ${url}`,detail:typeof detail==='string'?detail:JSON.stringify(detail,null,2),correlationId:be.correlation_id||res.headers.get('X-Correlation-ID')||correlationId});
      }
      return data;
    }catch(e){
      const err=e instanceof ApiError?e:new ApiError(e.name==='AbortError'?'Timeout della richiesta':(e.message||'Errore di rete'),{status:e.name==='AbortError'?'TIMEOUT':'NETWORK',endpoint:url,operation:meta.operation||`${options.method||'GET'} ${url}`,detail:e.stack||e.message,correlationId});
      if(meta.showError!==false) showError(err,meta.retry||null);
      throw err;
    }finally{
      clearTimeout(timer); if(button)setButtonBusy(button,false); if(!silent)hideLoading();
    }
  }

  A.UI={ensureUI,showLoading,updateLoading,hideLoading,showError,hideError,setButtonBusy,setLive,request,ApiError};
  document.addEventListener('DOMContentLoaded',ensureUI);
})();
