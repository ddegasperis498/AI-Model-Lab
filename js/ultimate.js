window.AIML = window.AIML || {};
(() => {
  'use strict';
  const $ = id => document.getElementById(id);
  const fmt = (v,d=5) => Number.isFinite(Number(v)) ? Number(v).toFixed(d) : '—';
  const esc = s => String(s ?? '').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const BASE = (location.protocol === 'http:' || location.protocol === 'https:') ? '' : 'http://127.0.0.1:8765';
  const API=BASE+'/api/v4';
  const S={state:null, chart:null, poll:null, lastHistoryStep:0};

  async function api(path, options={}, meta={}){
    return window.AIML.UI.request(API+path, options, meta);
  }

  function log(msg, type='info'){
    const el=$('v4EventLog'); if(!el)return;
    const cls={info:'text-slate-400',ok:'text-emerald-300',warn:'text-amber-300',error:'text-rose-300',torch:'text-blue-300'}[type]||'text-slate-400';
    const row=document.createElement('div'); row.className=cls; row.textContent=`[${new Date().toLocaleTimeString('it-IT')}] ${msg}`;el.appendChild(row);el.scrollTop=el.scrollHeight;
    while(el.children.length>180) el.firstChild.remove();
  }

  function metricCard(label,value,cls=''){
    return `<div class="metric rounded-2xl p-3"><div class="small-label">${esc(label)}</div><div class="text-sm mt-2 ${cls}">${esc(value ?? '—')}</div></div>`;
  }

  function initChart(){
    const c=$('v4LossChart'); if(!c || !window.Chart)return;
    S.chart=new Chart(c.getContext('2d'),{type:'line',data:{labels:[],datasets:[
      {label:'Train loss',data:[],borderColor:'rgba(96,165,250,.95)',backgroundColor:'transparent',pointRadius:0,tension:.22,borderWidth:2},
      {label:'Validation loss',data:[],borderColor:'rgba(251,191,36,.95)',backgroundColor:'transparent',pointRadius:2,tension:.22,borderWidth:2,spanGaps:true},
    ]},options:{responsive:true,maintainAspectRatio:false,animation:false,scales:{x:{ticks:{color:'#64748b'},grid:{color:'rgba(148,163,184,.05)'}},y:{beginAtZero:true,ticks:{color:'#64748b'},grid:{color:'rgba(148,163,184,.05)'}}},plugins:{legend:{labels:{color:'#cbd5e1'}}}}});
  }

  async function health({interactive=false}={}){
    try{
      const h=await api('/health',{}, {silent:!interactive,showError:interactive,title:'Verifica Framework V4',message:'Controllo backend, PyTorch e hardware…',operation:'Health V4'});
      $('v4HealthBadge').textContent=`V4 online · torch ${h.torch}`;$('v4HealthBadge').className='pill rounded-xl px-3 py-2 text-sm text-emerald-300';
      const sys=h.system||{};const gpu=sys.gpu;
      $('v4SystemCards').innerHTML=[
        metricCard('CPU',`${sys.cpu_count??'—'} core`),metricCard('Torch threads',sys.torch_threads),metricCard('RAM',sys.ram_total_human||'—'),
        metricCard('CUDA',sys.cuda_available?'Sì':'No',sys.cuda_available?'text-emerald-300':'text-slate-500'),metricCard('GPU',gpu?.name||'CPU'),metricCard('VRAM',gpu?.total_vram_human||'—')
      ].join('');
      return true;
    }catch(e){
      $('v4HealthBadge').textContent='Backend V4 offline';$('v4HealthBadge').className='pill rounded-xl px-3 py-2 text-sm text-rose-300';
      log(e.message,'error'); return false;
    }
  }

  function config(){
    return {
      d_model:+$('v4DModel').value,n_heads:+$('v4Heads').value,n_kv_heads:+$('v4KvHeads').value,n_layers:+$('v4Layers').value,
      context_length:+$('v4Context').value,mlp_ratio:+$('v4Ratio').value,dropout:+$('v4Dropout').value,rope_theta:+$('v4Rope').value,
      bias:false,tie_embeddings:true,use_sdpa:$('v4Sdpa').checked,gradient_checkpointing:$('v4GradCkpt').checked,compile_model:$('v4Compile').checked,
      tokenizer:$('v4Tokenizer').value,bpe_vocab_size:+$('v4BpeVocab').value,
      lora_rank:+$('v4LoraRank').value,lora_alpha:+$('v4LoraAlpha').value,lora_dropout:0,lora_freeze_base:$('v4FreezeBase').checked,
      optimizer:$('v4Optimizer').value,lr:+$('v4Lr').value,min_lr:+$('v4MinLr').value,warmup_steps:+$('v4Warmup').value,lr_decay_steps:+$('v4Decay').value,
      weight_decay:+$('v4Wd').value,grad_clip:+$('v4Clip').value,grad_accum_steps:+$('v4Accum').value,seed:+$('v4Seed').value,
      device:$('v4Device').value,amp_dtype:$('v4Amp').value,
    };
  }

  async function create(){
    try{
      if(!(await health())) throw new Error('Avvia START_ULTIMATE_V4.bat');
      const req={config:config(),corpus:$('v4Corpus').value,validation_fraction:+$('v4ValPct').value/100};
      log('Creazione Modern Transformer…','torch');
      const st=await api('/lm/create',{method:'POST',body:JSON.stringify(req)},{title:'Creazione Ultimate LLM',message:'Tokenizer, Modern Transformer, optimizer e dataset split…',operation:'Crea Ultimate LLM',button:$('v4Create'),busyLabel:'Creazione…',retry:create});S.state=st;S.lastHistoryStep=0;resetChart();renderState(st);log(`Creato: ${st.params.total.toLocaleString('it-IT')} parametri · trainable ${st.params.trainable.toLocaleString('it-IT')} · tokenizer ${st.tokenizer.kind}/${st.tokenizer.vocab_size}`,'ok');
    }catch(e){log(e.message,'error'); if(e?.name!=='ApiError') window.AIML.UI.showError({operation:'Creazione Ultimate LLM',message:e.message,detail:e.stack,status:'CLIENT',endpoint:'browser'},create);}
  }

  function resetChart(){if(!S.chart)return;S.chart.data.labels=[];S.chart.data.datasets.forEach(d=>d.data=[]);S.chart.update('none');}

  function pushHistory(history=[]){
    if(!S.chart)return;
    for(const h of history){
      if(!h.step || h.step<=S.lastHistoryStep)continue;
      S.chart.data.labels.push(h.step);S.chart.data.datasets[0].data.push(h.train_loss);S.chart.data.datasets[1].data.push(h.val_loss ?? null);S.lastHistoryStep=Math.max(S.lastHistoryStep,h.step);
    }
    if(S.chart.data.labels.length>500){S.chart.data.labels=S.chart.data.labels.slice(-500);S.chart.data.datasets.forEach(d=>d.data=d.data.slice(-500));}
    S.chart.update('none');
  }

  function renderState(st){
    S.state=st;const last=st.last||{};
    $('v4StepMetric').textContent=st.step??0;$('v4ParamMetric').textContent=st.params?.total?.toLocaleString('it-IT')??'—';$('v4TrainableMetric').textContent=st.params?.trainable?.toLocaleString('it-IT')??'—';
    $('v4TrainLossMetric').textContent=fmt(last.train_loss,6);$('v4ValLossMetric').textContent=fmt(last.val_loss,6);$('v4TokSecMetric').textContent=fmt(last.tokens_per_second,0);$('v4LrMetric').textContent=last.lr?Number(last.lr).toExponential(2):'—';$('v4GradMetric').textContent=fmt(last.grad_l2,4);
    pushHistory(st.history||[]);
  }

  async function trainStep(){
    try{const st=await api('/lm/train',{method:'POST',body:JSON.stringify({steps:1,batch_size:+$('v4Batch').value,eval_interval:+$('v4Eval').value})},{title:'Ultimate LLM · training step',message:'Forward → cross entropy → backward → AdamW…',operation:'Ultimate LLM train step',button:$('v4TrainStep'),busyLabel:'Training…',retry:trainStep});renderState(st);log(`step ${st.step} · loss ${fmt(st.last.train_loss,6)} · lr ${Number(st.last.lr).toExponential(2)}`,'torch');}
    catch(e){log(e.message,'error');}
  }

  async function startJob(){
    try{const r=await api('/lm/job/start',{method:'POST',body:JSON.stringify({steps:+$('v4JobSteps').value,batch_size:+$('v4Batch').value,eval_interval:+$('v4Eval').value})},{title:'Avvio training job',message:'Creazione job in background…',operation:'Avvia training job',button:$('v4StartJob'),busyLabel:'Avvio…'});log(`Job avviato: ${r.requested_steps} step`,'ok');startPolling();}
    catch(e){log(e.message,'error');}
  }
  async function stopJob(){try{await api('/lm/job/stop',{method:'POST',body:'{}'},{title:'Arresto training job',message:'Richiesta di stop sicuro…',operation:'Stop training job',button:$('v4StopJob'),busyLabel:'Stop…'});log('Richiesto stop del job.','warn');}catch(e){log(e.message,'error');}}

  function startPolling(){
    if(S.poll)clearInterval(S.poll);
    S.poll=setInterval(async()=>{
      try{
        const j=await api('/lm/job/status',{}, {silent:true,showError:false});const pct=Math.max(0,Math.min(100,(j.progress||0)*100));$('v4Progress').style.width=pct+'%';$('v4JobBadge').textContent=j.running?`training ${j.completed_steps}/${j.requested_steps}`:`${j.completed_steps}/${j.requested_steps} · idle`;
        const st=await api('/lm/state',{}, {silent:true,showError:false});renderState(st);window.AIML.UI.setLive(!!j.running,j.running?`Ultimate LLM · ${j.completed_steps}/${j.requested_steps}`:'');
        if(!j.running){clearInterval(S.poll);S.poll=null;window.AIML.UI.setLive(false);if(j.error){log(j.error,'error');window.AIML.UI.showError({operation:'Training job',message:'Il job è terminato con errore',detail:j.error,status:'JOB',endpoint:'/api/v4/lm/job/status'});}else log('Job terminato.','ok');}
      }catch(e){clearInterval(S.poll);S.poll=null;window.AIML.UI.setLive(false);log(e.message,'error');}
    },600);
  }

  async function generate(){
    try{
      $('v4Generated').textContent='Generazione…';
      const req={prompt:$('v4Prompt').value,max_new_tokens:+$('v4GenTokens').value,temperature:+$('v4Temp').value,top_k:+$('v4TopK').value,top_p:+$('v4TopP').value,repetition_penalty:+$('v4Rep').value,seed:+$('v4Seed').value};
      const r=await api('/lm/generate',{method:'POST',body:JSON.stringify(req)},{title:'Generazione autoregressiva',message:'Sampling token-by-token…',operation:'Ultimate LLM generate',button:$('v4Generate'),busyLabel:'Generazione…',retry:generate});$('v4Generated').textContent=r.text;log(`Generati ${r.new_token_count} token.`,'ok');
    }catch(e){$('v4Generated').textContent='';log(e.message,'error');}
  }

  async function tokenPreview(){
    try{const r=await api('/tokenizer/preview',{method:'POST',body:JSON.stringify({text:$('v4TokenText').value,limit:80})},{title:'Tokenizer microscope',message:'Tokenizzazione e preview del vocabolario…',operation:'Tokenizer preview'});$('v4TokenPreview').innerHTML=`<div class="text-xs text-slate-500 mb-2">${r.kind} · vocab ${r.vocab_size}</div><div class="flex flex-wrap gap-2">${r.tokens.map(t=>`<span class="pill rounded-lg px-2 py-1 text-xs" title="id ${t.id} · ${esc(t.hex)}">${esc(t.text)||'∅'} <span class="text-slate-600">#${t.id}</span></span>`).join('')}</div>`;}catch(e){log(e.message,'error');}
  }

  async function embed(){
    try{const r=await api('/embedding/similarity',{method:'POST',body:JSON.stringify({text:$('v4TokenText').value,limit:16})},{title:'Embedding microscope',message:'Calcolo embedding e cosine similarity…',operation:'Embedding similarity'});const n=r.tokens.length;if(!n){return;}
      let h=`<div style="display:grid;grid-template-columns:70px repeat(${n},44px);gap:2px;width:max-content"><div></div>${r.tokens.map(t=>`<div class="text-[9px] text-slate-500 truncate text-center">${esc(t)}</div>`).join('')}`;
      for(let i=0;i<n;i++){h+=`<div class="text-[9px] text-slate-500 flex items-center truncate">${esc(r.tokens[i])}</div>`;for(let j=0;j<n;j++){const v=r.matrix[i][j];const a=.06+.72*((v+1)/2);h+=`<div class="matrix-cell" title="cos=${v.toFixed(4)}" style="background:rgba(59,130,246,${a})">${v.toFixed(2)}</div>`;}}h+='</div>';$('v4EmbeddingMatrix').innerHTML=h;
    }catch(e){log(e.message,'error');}
  }

  async function saveCheckpoint(){try{const r=await api('/checkpoint/save',{method:'POST',body:JSON.stringify({name:$('v4SaveName').value})},{title:'Salvataggio checkpoint V4',message:'Serializzazione model/optimizer/tokenizer…',operation:'Salva checkpoint V4'});log(`Checkpoint ${r.filename} · ${(r.size_bytes/1024/1024).toFixed(2)} MB`,'ok');}catch(e){log(e.message,'error');}}
  async function saveExperiment(){try{const r=await api('/experiment/save',{method:'POST',body:JSON.stringify({name:$('v4SaveName').value,notes:$('v4Notes').value})},{title:'Salvataggio esperimento',message:'Persistenza configurazione e metriche…',operation:'Salva esperimento'});log(`Esperimento #${r.id} salvato.`,'ok');await listExperiments();}catch(e){log(e.message,'error');}}

  // ---------------- Scalar Autograd ----------------
  class Value{
    constructor(data,label='',children=[],op=''){this.data=Number(data);this.grad=0;this.label=label;this.children=children;this.op=op;this._backward=()=>{};this.local='—';}
    add(other,label=''){other=other instanceof Value?other:new Value(other);const out=new Value(this.data+other.data,label,[this,other],'+');out.local='∂out/∂a=1, ∂out/∂b=1';out._backward=()=>{this.grad+=out.grad;other.grad+=out.grad};return out;}
    mul(other,label=''){other=other instanceof Value?other:new Value(other);const out=new Value(this.data*other.data,label,[this,other],'×');out.local=`∂out/∂a=${other.data.toFixed(4)}, ∂out/∂b=${this.data.toFixed(4)}`;out._backward=()=>{this.grad+=other.data*out.grad;other.grad+=this.data*out.grad};return out;}
    tanh(label=''){const t=Math.tanh(this.data);const out=new Value(t,label,[this],'tanh');out.local=`1-tanh²=${(1-t*t).toFixed(4)}`;out._backward=()=>{this.grad+=(1-t*t)*out.grad};return out;}
    pow2(label=''){const out=new Value(this.data*this.data,label,[this],'^2');out.local=`2x=${(2*this.data).toFixed(4)}`;out._backward=()=>{this.grad+=2*this.data*out.grad};return out;}
  }
  let AG={nodes:[],loss:null,x:null,w:null,b:null,target:null};
  function topo(v,visited=new Set(),out=[]){if(!visited.has(v)){visited.add(v);v.children.forEach(c=>topo(c,visited,out));out.push(v);}return out;}
  function agForward(){const x=new Value(+$('agX').value,'x'),w=new Value(+$('agW').value,'w'),b=new Value(+$('agB').value,'b'),target=new Value(+$('agTarget').value,'target');const wx=x.mul(w,'x·w');const z=wx.add(b,'z');const y=z.tanh('prediction');const negT=new Value(-target.data,'-target');const err=y.add(negT,'error');const sq=err.pow2('error²');const half=new Value(.5,'0.5');const loss=sq.mul(half,'loss');AG={nodes:topo(loss),loss,x,w,b,target};renderAG(false);}
  function agBackward(){if(!AG.loss)agForward();AG.nodes.forEach(n=>n.grad=0);AG.loss.grad=1;const order=[...AG.nodes].reverse();let lines=[];for(const n of order){n._backward();lines.push(`${n.label||n.op}: grad=${n.grad.toFixed(6)} · local=${n.local}`);}renderAG(true);$('agTrace').innerHTML=lines.map(x=>`<div>${esc(x)}</div>`).join('');}
  function agUpdate(){if(!AG.loss)agBackward();const lr=+$('agLr').value;$('agW').value=(AG.w.data-lr*AG.w.grad).toFixed(6);$('agB').value=(AG.b.data-lr*AG.b.grad).toFixed(6);agForward();}
  function renderAG(showGrad){$('agGraph').innerHTML=AG.nodes.map(n=>`<div class="ag-node ${showGrad?'grad':''}"><div class="flex justify-between"><b class="text-blue-300">${esc(n.label||'node')}</b><span class="text-slate-600">${esc(n.op)}</span></div><div class="font-mono text-lg mt-2">value = ${n.data.toFixed(6)}</div><div class="font-mono text-purple-300 text-sm">grad = ${showGrad?n.grad.toFixed(6):'—'}</div><div class="text-[11px] text-slate-500 mt-2">local: ${esc(n.local)}</div></div>`).join('');}

  // ---------------- Scale Lab ----------------
  const PRESETS={tiny:{v:16000,d:256,h:4,kv:2,l:6,c:512},small:{v:32000,d:768,h:12,kv:4,l:12,c:2048},medium:{v:50000,d:2048,h:16,kv:4,l:24,c:4096},large:{v:100000,d:4096,h:32,kv:8,l:32,c:8192}};
  function applyPreset(){const p=PRESETS[$('scalePreset').value];if(!p)return;$('scaleVocab').value=p.v;$('scaleD').value=p.d;$('scaleHeads').value=p.h;$('scaleKv').value=p.kv;$('scaleLayers').value=p.l;$('scaleContext').value=p.c;}
  async function scaleCalc(silent=false){try{const req={vocab_size:+$('scaleVocab').value,d_model:+$('scaleD').value,n_heads:+$('scaleHeads').value,n_kv_heads:+$('scaleKv').value,n_layers:+$('scaleLayers').value,context_length:+$('scaleContext').value,mlp_ratio:+$('scaleRatio').value,batch_size:+$('scaleBatch').value,precision_bytes:+$('scalePrecision').value,optimizer:$('scaleOptimizer').value};const r=await api('/scale/estimate',{method:'POST',body:JSON.stringify(req)},{silent,showError:!silent,title:'Scale & Memory',message:'Stima parametri, memoria e FLOPs…',operation:'Scale estimate'});const M=r.memory;$('scaleCards').innerHTML=[metricCard('Parametri',r.parameter_count_estimate.toLocaleString('it-IT'),'text-purple-300'),metricCard('Pesi',M.parameters_human),metricCard('Optimizer',M.optimizer_human),metricCard('Activation rough',M.activations_rough_human),metricCard('Train total rough',M.training_total_rough_human,'text-amber-300'),metricCard('Inference rough',M.inference_total_rough_human,'text-emerald-300'),metricCard('FLOPs/train step',Number(r.compute.train_flops_per_step_rough).toExponential(3)),metricCard('FLOPs/token',Number(r.compute.inference_flops_per_token_rough).toExponential(3))].join('');$('scaleBreakdown').innerHTML=Object.entries(r.breakdown).map(([k,v])=>`<div class="flex justify-between border-b border-slate-800/60 py-2"><span class="text-slate-500">${esc(k)}</span><span class="font-mono">${Number(v).toLocaleString('it-IT')}</span></div>`).join('');$('scaleNotes').innerHTML=r.notes.map(n=>`<div>• ${esc(n)}</div>`).join('');}catch(e){log(e.message,'error'); if(e?.name!=='ApiError') window.AIML.UI.showError({operation:'Scale & Memory',message:e.message,detail:e.stack,status:'CLIENT',endpoint:'browser'},scaleCalc);}}

  async function listExperiments(silent=false){try{const r=await api('/experiment/list',{}, {silent,showError:!silent,operation:'Elenco esperimenti'});$('expRows').innerHTML=`<table><thead><tr class="text-xs text-slate-500"><th>ID</th><th>Nome</th><th>Step</th><th>Loss</th><th>Parametri</th><th>Note</th></tr></thead><tbody>${r.items.map(x=>`<tr><td>${x.id}</td><td>${esc(x.name)}</td><td>${x.metrics?.step??'—'}</td><td>${fmt(x.metrics?.last?.train_loss,6)}</td><td>${Number(x.metrics?.params?.total||0).toLocaleString('it-IT')}</td><td class="text-left text-slate-500">${esc(x.notes)}</td></tr>`).join('')}</tbody></table>`;}catch(e){$('expRows').innerHTML=`<div class="text-rose-300">${esc(e.message)}</div>`;}}

  function bind(){
    $('v4HealthBtn')?.addEventListener('click',health);$('v4CreateBtn')?.addEventListener('click',create);$('v4StepBtn')?.addEventListener('click',trainStep);$('v4StartJobBtn')?.addEventListener('click',startJob);$('v4StopJobBtn')?.addEventListener('click',stopJob);$('v4GenerateBtn')?.addEventListener('click',generate);$('v4TokenBtn')?.addEventListener('click',tokenPreview);$('v4EmbedBtn')?.addEventListener('click',embed);$('v4CheckpointBtn')?.addEventListener('click',saveCheckpoint);$('v4ExperimentBtn')?.addEventListener('click',saveExperiment);
    $('agForwardBtn')?.addEventListener('click',agForward);$('agBackwardBtn')?.addEventListener('click',agBackward);$('agUpdateBtn')?.addEventListener('click',agUpdate);
    $('scalePreset')?.addEventListener('change',applyPreset);$('scaleCalcBtn')?.addEventListener('click',scaleCalc);$('expRefreshBtn')?.addEventListener('click',listExperiments);
  }

  document.addEventListener('DOMContentLoaded',()=>{initChart();bind();health({interactive:false});agForward();scaleCalc(true);listExperiments(true);});
})();
