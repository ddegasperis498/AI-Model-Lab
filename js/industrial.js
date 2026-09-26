window.AIML = window.AIML || {};
(() => {
  'use strict';
  const A = window.AIML;
  const $ = id => document.getElementById(id);
  const API = location.protocol === 'http:' || location.protocol === 'https:' ? '' : 'http://127.0.0.1:8765';

  const S = {
    mode: localStorage.getItem('aiml-mode') || 'didactic',
    backend: null,
    mlpState: null,
    llmState: null,
    currentState: null,
    mlpTimer: null,
    llmTimer: null,
    mlpRunning: false,
    llmRunning: false,
    mlpChart: null,
    llmChart: null,
  };

  const fmt = (v, d=5) => Number.isFinite(Number(v)) ? Number(v).toFixed(d) : '—';
  const esc = s => String(s ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));

  async function api(path, options={}, meta={}) {
    return A.UI.request(API + path, options, meta);
  }

  function localError(err, operation, retry=null) {
    if (err?.name !== 'ApiError') {
      A.UI.showError({
        operation,
        message: err?.message || 'Operazione non completata',
        detail: err?.stack || err?.message || '',
        status: 'CLIENT',
        endpoint: 'browser',
      }, retry);
    }
  }

  function logAdvanced(message, type='info') {
    const el = $('torchLog');
    if (!el) return;
    const colors = {info:'text-slate-400', ok:'text-emerald-300', warn:'text-amber-300', error:'text-rose-300', torch:'text-blue-300'};
    const d = document.createElement('div');
    d.className = colors[type] || colors.info;
    d.textContent = `[${new Date().toLocaleTimeString('it-IT')}] ${message}`;
    el.appendChild(d); el.scrollTop = el.scrollHeight;
    while (el.children.length > 160) el.firstChild.remove();
  }

  function setTopMetricLabels(advanced) {
    const labels = advanced
      ? {epoch:'Step', loss:'Loss', params:'Parametri', arch:'Architettura', lr:'Learning Rate', phase:'Device'}
      : {epoch:'Epoca', loss:'Loss', params:'Parametri', arch:'Architettura', lr:'Learning Rate', phase:'Fase'};
    if ($('epochMetricLabel')) $('epochMetricLabel').textContent = labels.epoch;
    if ($('lossMetricLabel')) $('lossMetricLabel').textContent = labels.loss;
    if ($('paramMetricLabel')) $('paramMetricLabel').textContent = labels.params;
    if ($('archMetricLabel')) $('archMetricLabel').textContent = labels.arch;
    if ($('lrMetricLabel')) $('lrMetricLabel').textContent = labels.lr;
    if ($('phaseMetricLabel')) $('phaseMetricLabel').textContent = labels.phase;
  }

  function advancedArchitecture(st) {
    const cfg = st?.config || {};
    if (st?.kind === 'mlp') {
      return [cfg.input_size, ...(cfg.hidden_sizes || []), cfg.output_size]
        .filter(v => v !== undefined && v !== null).join('→') || '—';
    }
    if (st?.kind === 'transformer_lm') {
      return `${cfg.n_layers ?? '—'}L · d${cfg.d_model ?? '—'} · h${cfg.n_heads ?? '—'}`;
    }
    return '—';
  }

  function resetAdvancedTopMetrics() {
    setTopMetricLabels(true);
    if ($('epochMetric')) $('epochMetric').textContent = '0';
    if ($('lossMetric')) $('lossMetric').textContent = '—';
    if ($('paramMetric')) $('paramMetric').textContent = '—';
    if ($('archMetric')) $('archMetric').textContent = '—';
    if ($('lrMetric')) $('lrMetric').textContent = '—';
    if ($('phaseMetric')) $('phaseMetric').textContent = S.backend?.cuda_available ? 'cuda' : (S.backend ? 'cpu' : '—');
    if ($('runState')) $('runState').textContent = 'Pronto';
  }

  function renderAdvancedTopMetrics(st, status='Pronto') {
    if (!st || S.mode !== 'advanced') return;
    setTopMetricLabels(true);
    if ($('epochMetric')) $('epochMetric').textContent = st.step ?? 0;
    if ($('lossMetric')) $('lossMetric').textContent = fmt(st.last?.loss, 7);
    if ($('paramMetric')) $('paramMetric').textContent = Number(st.params?.total || 0).toLocaleString('it-IT');
    if ($('archMetric')) $('archMetric').textContent = advancedArchitecture(st);
    if ($('lrMetric')) $('lrMetric').textContent = st.config?.lr != null ? String(st.config.lr) : '—';
    if ($('phaseMetric')) $('phaseMetric').textContent = st.device || '—';
    if ($('runState')) $('runState').textContent = status;
  }

  async function health({interactive=false}={}) {
    try {
      const h = await api('/api/advanced/health', {}, {
        silent: !interactive,
        showError: interactive,
        title: 'Verifica backend PyTorch',
        message: 'Controllo FastAPI, PyTorch e CUDA…',
        meta: 'AI Model Lab V4.1',
        operation: 'Verifica backend PyTorch',
        button: interactive ? $('advRefreshHealth') : null,
        busyLabel: 'Verifica…',
      });
      S.backend = h;
      $('advBackendStatus').textContent = 'ONLINE';
      $('advBackendStatus').className = 'text-emerald-300 font-semibold';
      $('advTorchVersion').textContent = h.pytorch;
      $('advDeviceName').textContent = h.device_name;
      $('advCuda').textContent = h.cuda_available ? `Sì · CUDA ${h.cuda_version || ''}` : 'No';
      $('advThreads').textContent = h.cpu_threads;
      $('engineModeBadge').textContent = S.mode === 'advanced' ? `PyTorch ${h.pytorch}` : 'MLP didattico';
      const hint=$('advancedBackendHint');
      if(hint){hint.className='mt-4 rounded-2xl border border-emerald-500/20 bg-emerald-500/5 p-4 text-sm text-emerald-100/80';hint.innerHTML=`✓ Backend Ultimate V4.1.3 online. PyTorch <b>${h.pytorch}</b> · ${h.cuda_available?`CUDA ${h.cuda_version||''} · ${h.device_name}`:'CPU'}. Pronto per Framework Pro.`;}
      if (S.mode === 'advanced' && !S.currentState) resetAdvancedTopMetrics();
      return true;
    } catch (err) {
      S.backend = null;
      $('advBackendStatus').textContent = 'OFFLINE';
      $('advBackendStatus').className = 'text-rose-300 font-semibold';
      $('advTorchVersion').textContent = '—';
      $('advDeviceName').textContent = '—';
      $('advCuda').textContent = '—';
      $('advThreads').textContent = '—';
      if (S.mode === 'advanced') $('engineModeBadge').textContent = 'Backend PyTorch offline';
      const hint=$('advancedBackendHint');
      if(hint){hint.className='mt-4 rounded-2xl border border-amber-500/20 bg-amber-500/5 p-4 text-sm text-amber-100/80';hint.innerHTML='Backend PyTorch non disponibile. Avvia <b>START_ULTIMATE_V4.bat</b> e riprova.';}
      return false;
    }
  }

  function setMode(mode) {
    S.mode = mode;
    localStorage.setItem('aiml-mode', mode);
    document.body.dataset.labMode = mode;
    $('modeDidacticBtn')?.classList.toggle('mode-active', mode === 'didactic');
    $('modeAdvancedBtn')?.classList.toggle('mode-active', mode === 'advanced');

    document.querySelectorAll('.nav-btn').forEach(b => {
      if (b.classList.contains('advanced-nav')) b.classList.toggle('hidden', mode !== 'advanced');
      else b.classList.toggle('hidden', mode === 'advanced');
    });

    $('openPythonBtn')?.classList.toggle('hidden', mode === 'advanced');
    $('exportModelBtn')?.classList.toggle('hidden', mode === 'advanced');
    $('importModelBtn')?.classList.toggle('hidden', mode === 'advanced');
    $('exportDataBtn')?.classList.toggle('hidden', mode === 'advanced');

    if (mode === 'advanced') {
      A.appAPI?.pause?.();
      A.appAPI?.nav?.('pytorch');
      setTopMetricLabels(true);
      if (S.currentState) renderAdvancedTopMetrics(S.currentState, 'Pronto');
      else resetAdvancedTopMetrics();
      health();
      $('engineModeBadge').textContent = S.backend ? `PyTorch ${S.backend.pytorch}` : 'PyTorch Advanced';
    } else {
      pauseMLP(); pauseLLM();
      setTopMetricLabels(false);
      A.appAPI?.nav?.('dashboard');
      A.appAPI?.refreshAll?.();
      $('engineModeBadge').textContent = 'MLP didattico';
    }
  }

  function initCharts() {
    if ($('torchMlpChart')) {
      S.mlpChart = new Chart($('torchMlpChart').getContext('2d'), {
        type:'line', data:{labels:[], datasets:[{label:'PyTorch MLP Loss',data:[],borderColor:'#60a5fa',pointRadius:0,tension:.2}]},
        options:{responsive:true,maintainAspectRatio:false,animation:false,scales:{x:{ticks:{color:'#64748b'},grid:{color:'rgba(148,163,184,.05)'}},y:{beginAtZero:true,ticks:{color:'#64748b'},grid:{color:'rgba(148,163,184,.05)'}}},plugins:{legend:{labels:{color:'#cbd5e1'}}}}
      });
    }
    if ($('torchLlmChart')) {
      S.llmChart = new Chart($('torchLlmChart').getContext('2d'), {
        type:'line', data:{labels:[], datasets:[{label:'Transformer Cross Entropy',data:[],borderColor:'#c084fc',pointRadius:0,tension:.2}]},
        options:{responsive:true,maintainAspectRatio:false,animation:false,scales:{x:{ticks:{color:'#64748b'},grid:{color:'rgba(148,163,184,.05)'}},y:{beginAtZero:true,ticks:{color:'#64748b'},grid:{color:'rgba(148,163,184,.05)'}}},plugins:{legend:{labels:{color:'#cbd5e1'}}}}
      });
    }
  }

  function mlpConfig() {
    const hidden = $('torchMlpHidden').value.split(',').map(x=>parseInt(x.trim())).filter(n=>Number.isFinite(n)&&n>0);
    return {
      input_size: parseInt($('torchMlpInput').value), hidden_sizes: hidden.length ? hidden : [32,32],
      output_size: parseInt($('torchMlpOutput').value), task: $('torchMlpTask').value,
      activation: $('torchMlpActivation').value, dropout: parseFloat($('torchMlpDropout').value),
      optimizer: $('torchMlpOptimizer').value, lr: parseFloat($('torchMlpLr').value),
      weight_decay: parseFloat($('torchMlpWd').value), grad_clip: parseFloat($('torchMlpClip').value),
      seed: parseInt($('torchMlpSeed').value), device: $('torchDevice').value,
      amp: $('torchAmp').checked,
    };
  }

  function generateMLPData() {
    const preset = $('torchMlpDataset').value;
    const task = $('torchMlpTask').value;
    const inputSize = parseInt($('torchMlpInput').value);
    const data = [];
    if (preset === 'linear') {
      for (let i=0;i<160;i++) { const x=-3+6*i/159; data.push({x:[x],y:[2.5*x+1]}); }
    } else if (preset === 'sine') {
      for (let i=0;i<180;i++) { const x=-Math.PI+2*Math.PI*i/179; data.push({x:[x],y:[Math.sin(x)]}); }
    } else if (preset === 'xor') {
      [[0,0,0],[0,1,1],[1,0,1],[1,1,0]].forEach(r=>{for(let k=0;k<40;k++)data.push({x:[r[0],r[1]],y:[r[2]]});});
    } else if (preset === 'circles') {
      for(let i=0;i<240;i++){const cls=i%2, r=cls?.8:.35, a=Math.random()*Math.PI*2;data.push({x:[r*Math.cos(a),r*Math.sin(a)],y:[cls]});}
    }
    if (data[0]?.x.length !== inputSize) throw new Error(`Il preset ${preset} richiede input_size=${data[0]?.x.length}. Imposta il valore corretto.`);
    return data;
  }

  async function createMLP() {
    try {
      const ok = await health({interactive:false});
      if (!ok) throw new Error('Backend PyTorch offline. Avvia START_ULTIMATE_V4.bat');
      const st = await api('/api/advanced/mlp/create',{method:'POST',body:JSON.stringify(mlpConfig())},{
        title:'Creazione rete PyTorch', message:'Costruzione torch.nn.Module e trasferimento sul device…',
        meta:$('torchDevice').value==='cuda'?'CUDA':'Device automatico', operation:'Crea MLP PyTorch',
        button:$('torchCreateMlp'), busyLabel:'Creazione…', retry:createMLP,
      });
      S.mlpState = S.currentState = st;
      S.mlpChart.data.labels=[];S.mlpChart.data.datasets[0].data=[];S.mlpChart.update('none');
      renderTorchState(st); renderXRay(st);
      logAdvanced(`✓ MLP creato · ${st.params.total.toLocaleString('it-IT')} parametri · ${st.device}`,'ok');
      return st;
    } catch(err) { logAdvanced(`✕ Creazione MLP: ${err.message}`,'error'); localError(err,'Creazione MLP PyTorch',createMLP); return null; }
  }

  async function trainMLP(steps=1,{realtime=false}={}) {
    try {
      if (!S.mlpState) { const created=await createMLP(); if(!created) return null; }
      const data = generateMLPData();
      const btn = realtime ? null : (steps===1 ? $('torchMlpStep') : $('torchMlp50'));
      const st = await api('/api/advanced/mlp/train',{method:'POST',body:JSON.stringify({data,batch_size:parseInt($('torchMlpBatch').value),steps})},{
        silent: realtime,
        title: steps===1?'Training PyTorch · 1 step':`Training PyTorch · ${steps} step`,
        message: 'Forward → loss → backward → gradient clip → optimizer.step()',
        meta:`${S.backend?.device_name||'device'} · ${$('torchMlpOptimizer').value}`,
        operation:`Training MLP (${steps} step)`, button:btn, busyLabel:'Training…',
        retry: realtime?null:()=>trainMLP(steps),
      });
      S.mlpState = S.currentState = st;
      (st.loss_curve || []).forEach(loss=>{S.mlpChart.data.labels.push(S.mlpChart.data.labels.length+1);S.mlpChart.data.datasets[0].data.push(loss);});
      if(S.mlpChart.data.labels.length>400){S.mlpChart.data.labels=S.mlpChart.data.labels.slice(-400);S.mlpChart.data.datasets[0].data=S.mlpChart.data.datasets[0].data.slice(-400);}
      S.mlpChart.update('none'); renderTorchState(st); renderXRay(st);
      logAdvanced(`✓ step=${st.step} · loss=${fmt(st.last.loss,7)} · eval=${fmt(st.last.eval_loss,7)} · grad=${fmt(st.gradients.l2,5)}`,'torch');
      if(realtime) A.UI.setLive(true,`MLP live · step ${st.step} · loss ${fmt(st.last.loss,5)}`);
      return st;
    } catch(err){ pauseMLP(); logAdvanced(`✕ Training fallito · ${err.message}`,'error'); localError(err,'Training MLP PyTorch',()=>trainMLP(steps)); return null; }
  }

  function startMLP(){ if(S.mlpRunning)return;S.mlpRunning=true;if($('runState'))$('runState').textContent='Realtime';A.UI.setLive(true,'MLP live · avvio…');const run=async()=>{if(!S.mlpRunning)return;const result=await trainMLP(1,{realtime:true});if(!result){pauseMLP();return;}S.mlpTimer=setTimeout(run,Math.max(40,parseInt($('torchMlpSpeed').value)||180));};run(); }
  function pauseMLP(){S.mlpRunning=false;if(S.mlpTimer)clearTimeout(S.mlpTimer);S.mlpTimer=null;A.UI.setLive(false);if(S.mode==='advanced'&&$('runState'))$('runState').textContent='Pausa';}


  function llmConfig() {
    return {
      d_model:parseInt($('llmDModel').value), n_heads:parseInt($('llmHeads').value), n_layers:parseInt($('llmLayers').value),
      context_length:parseInt($('llmContext').value), mlp_ratio:parseFloat($('llmRatio').value), dropout:parseFloat($('llmDropout').value),
      optimizer:$('llmOptimizer').value, lr:parseFloat($('llmLr').value), weight_decay:parseFloat($('llmWd').value),
      grad_clip:parseFloat($('llmClip').value), seed:parseInt($('llmSeed').value), device:$('torchDevice').value, amp:$('torchAmp').checked,
      tie_embeddings:$('llmTie').checked,
    };
  }

  async function createLLM() {
    try {
      const ok = await health({interactive:false}); if(!ok) throw new Error('Backend PyTorch offline. Avvia START_ULTIMATE_V4.bat');
      const st=await api('/api/advanced/llm/create',{method:'POST',body:JSON.stringify(llmConfig())},{title:'Creazione Transformer LM',message:'Allocazione embedding, attention block e optimizer…',operation:'Crea Transformer LM',button:$('torchCreateLlm'),busyLabel:'Creazione…',retry:createLLM});
      S.llmState=S.currentState=st;S.llmChart.data.labels=[];S.llmChart.data.datasets[0].data=[];S.llmChart.update('none');
      renderLLMState(st);renderXRay(st);logAdvanced(`✓ Transformer LM creato · ${st.params.total.toLocaleString('it-IT')} parametri · ${st.config.n_layers} block · ${st.config.n_heads} head.`,'ok');return st;
    } catch(err){logAdvanced(`✕ Creazione Transformer: ${err.message}`,'error');localError(err,'Creazione Transformer LM',createLLM);return null;}
  }

  async function trainLLM(steps=1,{realtime=false}={}) {
    try {
      if(!S.llmState){const created=await createLLM();if(!created)return null;}
      const text=$('llmTrainingText').value; const btn=realtime?null:(steps===1?$('llmTrainStepBtn'):$('llm20'));
      const st=await api('/api/advanced/llm/train',{method:'POST',body:JSON.stringify({text,batch_size:parseInt($('llmBatch').value),steps})},{silent:realtime,title:`Transformer training · ${steps} step`,message:'Cross entropy → autograd → optimizer step',operation:'Training Transformer LM',button:btn,busyLabel:'Training…',retry:realtime?null:()=>trainLLM(steps)});
      S.llmState=S.currentState=st;
      (st.loss_curve||[]).forEach(loss=>{S.llmChart.data.labels.push(S.llmChart.data.labels.length+1);S.llmChart.data.datasets[0].data.push(loss);});
      if(S.llmChart.data.labels.length>400){S.llmChart.data.labels=S.llmChart.data.labels.slice(-400);S.llmChart.data.datasets[0].data=S.llmChart.data.datasets[0].data.slice(-400);}
      S.llmChart.update('none');renderLLMState(st);renderXRay(st);renderAttention(st);
      logAdvanced(`✓ Transformer step=${st.step} · loss=${fmt(st.last.loss,5)} · ppl=${fmt(st.last.perplexity,2)} · tok/s=${fmt(st.last.tokens_per_second,0)}`,'torch');if(realtime)A.UI.setLive(true,`Transformer live · step ${st.step}`);return st;
    } catch(err){pauseLLM();logAdvanced(`✕ Transformer training: ${err.message}`,'error');localError(err,'Training Transformer LM',()=>trainLLM(steps));return null;}
  }

  function startLLM(){if(S.llmRunning)return;S.llmRunning=true;if($('runState'))$('runState').textContent='Realtime';A.UI.setLive(true,'Transformer live · avvio…');const run=async()=>{if(!S.llmRunning)return;const result=await trainLLM(1,{realtime:true});if(!result){pauseLLM();return;}S.llmTimer=setTimeout(run,Math.max(20,parseInt($('llmSpeed').value)||80));};run();}
  function pauseLLM(){S.llmRunning=false;if(S.llmTimer)clearTimeout(S.llmTimer);S.llmTimer=null;A.UI.setLive(false);if(S.mode==='advanced'&&$('runState'))$('runState').textContent='Pausa';}


  async function generateLLM(){
    try{
      const result=await api('/api/advanced/llm/generate',{method:'POST',body:JSON.stringify({prompt:$('llmPrompt').value,max_new_tokens:parseInt($('llmGenTokens').value),temperature:parseFloat($('llmTemperature').value),top_k:parseInt($('llmTopK').value)})},{title:'Generazione Transformer',message:'Generazione autoregressiva token-by-token…',operation:'Generazione Transformer',button:$('llmGenerate'),busyLabel:'Generazione…',retry:generateLLM});
      $('llmGenerated').textContent=result.text;logAdvanced(`✓ Generazione completata: ${result.token_ids.length} byte-token.`,'ok');
    }catch(err){logAdvanced(`✕ Generazione: ${err.message}`,'error');localError(err,'Generazione Transformer',generateLLM);}
  }

  function renderTorchState(st){
    $('torchStep').textContent=st.step; $('torchParams').textContent=st.params.total.toLocaleString('it-IT'); $('torchDeviceMetric').textContent=st.device;
    $('torchLoss').textContent=fmt(st.last?.loss,7); $('torchGrad').textContent=fmt(st.gradients?.l2,5); $('torchWeight').textContent=fmt(st.weights?.l2,5);
    $('torchActivationStats').innerHTML=Object.entries(st.activations||{}).slice(0,18).map(([name,v])=>`<div class="rounded-xl border border-slate-800 p-3"><div class="font-mono text-xs text-blue-300">${esc(name)}</div><div class="text-xs text-slate-500 mt-1">μ=${fmt(v.mean,4)} · σ=${fmt(v.std,4)} · zero=${fmt(100*v.zero_fraction,1)}%</div></div>`).join('') || '<div class="text-slate-500">Esegui un training step per raccogliere le activation.</div>';
    renderAdvancedTopMetrics(st, S.mlpRunning ? 'Realtime' : 'Pronto');
  }

  function renderLLMState(st){
    $('llmStepMetric').textContent=st.step;$('llmParams').textContent=st.params.total.toLocaleString('it-IT');$('llmLoss').textContent=fmt(st.last?.loss,5);$('llmPpl').textContent=fmt(st.last?.perplexity,2);$('llmTokSec').textContent=fmt(st.last?.tokens_per_second,0);$('llmDeviceMetric').textContent=st.device;
    renderAdvancedTopMetrics(st, S.llmRunning ? 'Realtime' : 'Pronto');
  }

  function renderXRay(st){
    if(!st)return;
    $('torchModuleTree').innerHTML=(st.modules||[]).map(m=>`<div class="flex justify-between gap-3 border-b border-slate-800/60 py-2"><div><span class="font-mono text-blue-300">${esc(m.name)}</span><span class="text-slate-600 ml-2">${esc(m.type)}</span></div><span class="font-mono text-slate-400">${Number(m.direct_params).toLocaleString('it-IT')}</span></div>`).join('');
    $('torchParamRows').innerHTML=(st.parameter_snapshot||[]).map(p=>`<tr><td class="font-mono text-slate-300">${esc(p.name)}</td><td class="font-mono text-slate-500">${esc(JSON.stringify(p.shape))}</td><td class="font-mono">${fmt(p.mean,5)}</td><td class="font-mono">${fmt(p.std,5)}</td><td class="font-mono text-purple-300">${fmt(p.grad_max_abs,5)}</td><td class="font-mono text-slate-500">${Number(p.numel).toLocaleString('it-IT')}</td></tr>`).join('');
    $('torchXraySummary').textContent=`${st.kind} · ${st.params.total.toLocaleString('it-IT')} parametri · device ${st.device} · autograd PyTorch reale`;
    renderAdvancedTopMetrics(st, (S.mlpRunning || S.llmRunning) ? 'Realtime' : 'Pronto');
    if (st.kind === 'mlp') {
      if ($('torchAttentionTitle')) $('torchAttentionTitle').textContent = 'Attention';
      if ($('torchAttention')) $('torchAttention').innerHTML = '<div class="text-slate-500">Non applicabile: un MLP non usa self-attention. Questa area si attiverà quando ispezionerai un Transformer.</div>';
    } else {
      if ($('torchAttentionTitle')) $('torchAttentionTitle').textContent = 'Attention Head 0 · ultimo block';
    }
  }

  function renderAttention(st){
    const all=st.last_attention;if(!all||!all.length){$('torchAttention').innerHTML='<div class="text-slate-500">La matrice apparirà dopo un training/generation forward.</div>';return;}
    const a=all[0];const n=Math.min(24,a.length);let max=0;for(let i=0;i<n;i++)for(let j=0;j<n;j++)max=Math.max(max,a[i][j]||0);
    let h=`<div class="overflow-auto"><div style="display:grid;grid-template-columns:38px repeat(${n},28px);gap:1px;width:max-content"><div></div>${Array.from({length:n},(_,j)=>`<div class="text-[9px] text-slate-600 text-center">${j}</div>`).join('')}`;
    for(let i=0;i<n;i++){h+=`<div class="text-[9px] text-slate-600 flex items-center">${i}</div>`;for(let j=0;j<n;j++){const v=a[i][j]||0;h+=`<div title="attn[${i},${j}]=${v.toFixed(5)}" style="height:28px;background:rgba(168,85,247,${.04+.86*(v/(max||1))})"></div>`;}}
    h+='</div></div>';$('torchAttention').innerHTML=h;
  }

  async function saveCheckpoint(){try{const name=$('checkpointName').value||'checkpoint';const r=await api('/api/advanced/checkpoint/save',{method:'POST',body:JSON.stringify({name})},{title:'Salvataggio checkpoint',message:'Serializzazione modello e optimizer…',operation:'Salva checkpoint',button:$('saveCheckpoint'),busyLabel:'Salvataggio…',retry:saveCheckpoint});logAdvanced(`✓ Checkpoint salvato: ${r.filename}`,'ok');await listCheckpoints(true);}catch(e){logAdvanced(`✕ Checkpoint: ${e.message}`,'error');localError(e,'Salvataggio checkpoint',saveCheckpoint);}}
  async function listCheckpoints(silent=false){try{const r=await api('/api/advanced/checkpoint/list',{}, {silent,showError:!silent,operation:'Elenco checkpoint'});$('checkpointList').innerHTML=(r.items||[]).map(x=>`<button class="checkpoint-load w-full flex justify-between gap-2 rounded-xl border border-slate-800 p-2 hover:bg-slate-800/50" data-file="${esc(x.filename)}"><span>${esc(x.filename)}</span><span class="text-slate-500">${(x.size_bytes/1024/1024).toFixed(2)} MB</span></button>`).join('')||'<div class="text-slate-500">Nessun checkpoint.</div>';document.querySelectorAll('.checkpoint-load').forEach(b=>b.onclick=()=>loadCheckpoint(b.dataset.file));}catch(e){logAdvanced(e.message,'error');}}
  async function loadCheckpoint(filename){try{const st=await api('/api/advanced/checkpoint/load',{method:'POST',body:JSON.stringify({filename})},{title:'Caricamento checkpoint',message:filename,operation:'Carica checkpoint',retry:()=>loadCheckpoint(filename)});S.currentState=st;if(st.kind==='mlp'){S.mlpState=st;renderTorchState(st);}else{S.llmState=st;renderLLMState(st);renderAttention(st);}renderXRay(st);logAdvanced(`Checkpoint caricato: ${filename}`,'ok');}catch(e){logAdvanced(e.message,'error');}}

  function bind(){
    $('modeDidacticBtn')?.addEventListener('click',()=>setMode('didactic'));
    $('modeAdvancedBtn')?.addEventListener('click',()=>setMode('advanced'));
    $('advRefreshHealth')?.addEventListener('click',()=>health({interactive:true}));
    $('torchCreateMlp')?.addEventListener('click',createMLP);$('torchMlpStep')?.addEventListener('click',()=>trainMLP(1));$('torchMlp50')?.addEventListener('click',()=>trainMLP(50));$('torchMlpStart')?.addEventListener('click',startMLP);$('torchMlpPause')?.addEventListener('click',pauseMLP);
    $('torchCreateLlm')?.addEventListener('click',createLLM);$('llmTrainStepBtn')?.addEventListener('click',()=>trainLLM(1));$('llm20')?.addEventListener('click',()=>trainLLM(20));$('llmStart')?.addEventListener('click',startLLM);$('llmPause')?.addEventListener('click',pauseLLM);$('llmGenerate')?.addEventListener('click',generateLLM);
    $('saveCheckpoint')?.addEventListener('click',saveCheckpoint);$('refreshCheckpoints')?.addEventListener('click',listCheckpoints);
    $('torchRefreshXray')?.addEventListener('click',async()=>{try{const path=S.currentState?.kind==='transformer_lm'?'/api/advanced/llm/state':'/api/advanced/mlp/state';const st=await api(path,{}, {title:'Aggiornamento Torch X-Ray',message:'Lettura moduli, parametri e gradienti…',operation:'Aggiorna Torch X-Ray',button:$('torchRefreshXray'),busyLabel:'Aggiorna…'});S.currentState=st;renderXRay(st);if(st.kind==='transformer_lm')renderAttention(st);}catch(e){logAdvanced(e.message,'error');}});
    $('torchMlpDataset')?.addEventListener('change',e=>{const two=['xor','circles'].includes(e.target.value);$('torchMlpInput').value=two?2:1;$('torchMlpTask').value=two?'classification':'regression';});
  }

  document.addEventListener('DOMContentLoaded', async()=>{
    document.querySelectorAll('.nav-btn:not(.advanced-nav)').forEach(b=>b.classList.add('didactic-nav'));
    initCharts();bind();await health({interactive:false});setMode(S.mode);listCheckpoints(true);
  });
})();
