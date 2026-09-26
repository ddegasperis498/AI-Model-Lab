window.AIML = window.AIML || {};
(() => {
  const A=window.AIML;
  const $=id=>document.getElementById(id);

  const state={
    task:"regression",datasetName:"regressionLinear",dataset:[],trainData:null,valData:null,testData:null,
    net:null,running:false,timer:null,epoch:0,phase:"idle",
    lr:.03,optimizer:"adam",speed:250,hidden:[4,4],activation:"tanh",
    last:null,lossHistory:[], teacherIndex:0, pythonWin:null
  };
  A.appState=state;

  function log(msg,type="system"){
    const el=$("terminal"); if(!el)return;
    const colors={system:"text-slate-400",forward:"text-blue-300",loss:"text-rose-300",backprop:"text-purple-300",update:"text-emerald-300",teacher:"text-yellow-300"};
    const d=document.createElement("div");d.className=`log-line ${colors[type]||colors.system}`;d.textContent=`[${new Date().toLocaleTimeString("it-IT")}] ${msg}`;
    el.appendChild(d);el.scrollTop=el.scrollHeight;while(el.children.length>180)el.firstChild.remove();
  }

  function parseHidden(){
    const parts=$("hiddenSizes").value.split(",").map(s=>parseInt(s.trim())).filter(n=>Number.isFinite(n)&&n>0&&n<=16);
    return parts.length?parts:[3];
  }

  function loadDataset(name=state.datasetName){
    state.datasetName=name;
    if(name==="regressionLinear"){state.task="regression";state.dataset=A.datasets.regressionLinear();}
    else if(name==="regressionSine"){state.task="regression";state.dataset=A.datasets.regressionSine();}
    else if(name==="xor"){state.task="classification";state.dataset=A.datasets.xor();}
    else {state.task="classification";state.dataset=A.datasets.circles();}
    $("taskMode").value=state.task;
    rebuildNetwork();
  }

  function rebuildNetwork(){
    state.hidden=parseHidden();
    state.activation=$("activationSelect").value;
    const inputSize=state.dataset[0]?.x.length || (state.task==="classification"?2:1);
    const outputSize=1;
    state.net=new A.MLP([inputSize,...state.hidden,outputSize],state.activation,state.task);
    state.epoch=0;state.lossHistory=[];state.last=null;
    if(window.lossChart){window.lossChart.data.labels=[];window.lossChart.data.datasets[0].data=[];window.lossChart.update();}
    refreshAll();
    if(A.Advanced?.onModelRebuilt) A.Advanced.onModelRebuilt();
    log(`Nuovo modello creato: ${state.net.sizes.join(" → ")} | parametri=${state.net.paramCount()}`,"system");
  }

  function batchTrain(){
    if(!state.net||!state.dataset.length)return;
    const activeTrain=(state.trainData&&state.trainData.length)?state.trainData:state.dataset;
    const n=Math.min(parseInt($("batchSize").value)||8,activeTrain.length);
    let loss=0,last=null;
    for(let i=0;i<n;i++){
      const s=activeTrain[Math.floor(A.random()*activeTrain.length)];
      last=state.net.trainSample(s.x,s.y,state.lr,state.optimizer,true);
      loss+=last.loss;
    }
    loss/=n;state.epoch++;state.last={...last,loss};
    state.lossHistory.push(loss);if(state.lossHistory.length>400)state.lossHistory.shift();
    window.lossChart.data.labels.push(state.epoch);window.lossChart.data.datasets[0].data.push(loss);
    if(window.lossChart.data.labels.length>250){window.lossChart.data.labels.shift();window.lossChart.data.datasets[0].data.shift();}
    window.lossChart.update("none");
    state.phase="update";
    refreshAll();
    if(state.epoch%5===0) log(`Epoca ${state.epoch}: loss media=${A.fmt(loss,7)} | optimizer=${state.optimizer}`,"update");
    sendPythonState();
  }

  function start(){ if(state.running)return;state.running=true; $("runState").textContent="Training"; loop(); }
  function pause(){state.running=false;if(state.timer)clearTimeout(state.timer);$("runState").textContent="Pausa";}
  function loop(){if(!state.running)return;batchTrain();state.timer=setTimeout(loop,state.speed);}

  function stepPhase(){
    if(!state.net||!state.dataset.length)return;
    const s=state.dataset[state.epoch%state.dataset.length];
    const phases=["forward","loss","backprop","update"];
    const idx=Math.max(0,phases.indexOf(state.phase));
    const next=state.phase==="idle"?"forward":phases[(idx+1)%phases.length];
    state.phase=next;
    if(next==="forward"){
      const out=state.net.forward(s.x);state.last={output:out,target:s.y,...state.net.lossAndGrad(out,s.y),traces:state.net.last.traces};
      log(`FORWARD: x=[${s.x.map(v=>A.fmt(v,3)).join(", ")}] → ŷ=${A.fmt(out[0])}`,"forward");
    }else if(next==="loss"){
      log(`LOSS: target=${A.fmt(state.last.target[0])}, errore=${A.fmt(state.last.error)}, loss=${A.fmt(state.last.loss,8)}`,"loss");
    }else if(next==="backprop"){
      let grad=state.last.grad;
      for(let i=state.net.layers.length-1;i>=0;i--)grad=state.net.layers[i].backward(grad);
      log(`BACKPROP: gradienti calcolati per ${state.net.paramCount()} parametri.`,"backprop");
    }else{
      state.net.step++;state.net.layers.forEach(l=>l.apply(state.lr,state.optimizer,state.net.step));
      state.epoch++;log(`UPDATE: parametri aggiornati con η=${state.lr}.`,"update");
      state.phase="idle";
    }
    refreshAll();sendPythonState();
  }

  function refreshMetrics(){
    // In Framework Pro the top bar is owned by industrial.js and must reflect
    // the real backend model, never the stale didactic MLP state.
    if (document.body.dataset.labMode === 'advanced') return;
    $("epochMetric").textContent=state.epoch;
    $("lossMetric").textContent=state.last?A.fmt(state.last.loss,7):"—";
    $("paramMetric").textContent=state.net?state.net.paramCount():"—";
    $("archMetric").textContent=state.net?state.net.sizes.join("→"):"—";
    $("lrMetric").textContent=A.fmt(state.lr,3);
    $("phaseMetric").textContent=state.phase;
  }

  function refreshParameters(){
    if(A.Advanced?.refreshParameters){ A.Advanced.refreshParameters(); return; }
    const body=$("paramBody"); if(!body||!state.net)return;
    const params=state.net.paramsFlat().slice(0,120);
    body.innerHTML=params.map(p=>`<tr>
      <td class="font-mono text-slate-300">${p.name}</td>
      <td class="${p.value>=0?"text-emerald-300":"text-rose-300"} font-mono">${A.fmt(p.value,6)}</td>
      <td class="text-purple-300 font-mono">${A.fmt(p.grad,6)}</td>
      <td class="text-slate-500">${p.kind}</td>
    </tr>`).join("");
  }

  function refreshXray(){
    if(!state.net)return;
    $("xrayTree").innerHTML=state.net.layers.map((l,i)=>`
      <div class="rounded-xl border border-slate-800 bg-slate-950/40 p-3">
        <div class="flex justify-between"><b>Layer ${i+1}</b><span class="text-blue-300">${l.inSize} → ${l.outSize}</span></div>
        <div class="text-xs text-slate-500 mt-2">Activation: <span class="text-slate-300">${l.activation}</span></div>
        <div class="text-xs text-slate-500">Weights: ${l.inSize*l.outSize} · Bias: ${l.outSize} · Parametri: ${l.paramCount()}</div>
      </div>`).join("");
    $("xraySummary").textContent=`${state.net.sizes.join(" → ")} · ${state.net.layers.length} layer trainabili · ${state.net.paramCount()} parametri`;
  }

  function refreshGraph(){
    if(!state.last?.traces){$("graphArea").innerHTML='<div class="text-slate-500">Esegui almeno un Forward Pass.</div>';return;}
    const parts=[];
    state.last.traces.forEach((t,i)=>{
      if(i===0)parts.push(`<div class="pill rounded-xl p-3 text-center"><div class="small-label">Input</div><div class="font-mono mt-1">[${t.a.map(v=>A.fmt(v,3)).join(", ")}]</div></div>`);
      else parts.push(`<div class="text-slate-700 text-2xl">→</div><div class="pill rounded-xl p-3 text-center"><div class="small-label">${t.name} · ${t.activation}</div><div class="font-mono mt-1">[${t.a.map(v=>A.fmt(v,3)).join(", ")}]</div><div class="text-[10px] text-slate-600 mt-1">z=[${t.z.map(v=>A.fmt(v,3)).join(", ")}]</div></div>`);
    });
    parts.push(`<div class="text-slate-700 text-2xl">→</div><div class="rounded-xl p-3 border border-rose-500/20 bg-rose-500/5 text-center"><div class="small-label">Loss</div><div class="font-mono text-rose-300">${A.fmt(state.last.loss,7)}</div></div>`);
    $("graphArea").innerHTML=parts.join("");
  }

  function refreshBuilder(){
    if(!state.net)return;
    $("builderLayers").innerHTML=state.net.sizes.map((n,i)=>`<div class="rounded-xl p-3 border border-slate-800 bg-slate-950/40 text-center min-w-[100px]">
      <div class="small-label">${i===0?"Input":i===state.net.sizes.length-1?"Output":"Hidden "+i}</div>
      <div class="text-2xl font-semibold mt-1">${n}</div>
      <div class="text-xs text-slate-500">neuroni</div>
    </div>${i<state.net.sizes.length-1?'<div class="text-slate-700 text-2xl">→</div>':""}`).join("");
  }

  function refreshCanvas(){
    A.drawNetwork($("networkCanvas"),state.net,state.phase);
    A.drawDataset($("datasetCanvas"),state.dataset,state.net,state.task);
  }

  function refreshAll(){refreshMetrics();refreshParameters();refreshXray();refreshGraph();refreshBuilder();refreshCanvas();}

  // NEURON LAB
  function neuronLab(){
    const vals=["nx1","nx2","nw1","nw2","nb"].map(id=>parseFloat($(id).value));
    const [x1,x2,w1,w2,b]=vals;
    const z=x1*w1+x2*w2+b; const act=$("nact").value; const y=A.activations[act].f(z);
    $("neuronFormula").innerHTML=`z = (${A.fmt(x1,2)}×${A.fmt(w1,2)}) + (${A.fmt(x2,2)}×${A.fmt(w2,2)}) + ${A.fmt(b,2)} = <b class="text-blue-300">${A.fmt(z,5)}</b><br>${act}(z) = <b class="text-emerald-300">${A.fmt(y,5)}</b>`;
    ["nx1","nx2","nw1","nw2","nb"].forEach(id=>$(id+"Val").textContent=A.fmt(parseFloat($(id).value),2));
  }

  // TEACHER/STUDENT
  function teacherStep(){
    if(state.net.sizes[0]!==1||state.task!=="regression"){
      state.task="regression";state.dataset=A.datasets.regressionLinear();$("taskMode").value="regression";$("datasetSelect").value="regressionLinear";rebuildNetwork();
    }
    const x=-3+A.random()*6, y=2.5*x+1, pred=state.net.predict([x]);
    const result=state.net.trainSample([x],[y],state.lr,state.optimizer,true);state.epoch++;state.last=result;
    const chat=$("teacherChat");
    chat.innerHTML+=`<div class="rounded-xl p-3 bg-yellow-500/5 border border-yellow-500/10"><b class="text-yellow-300">Agente A:</b> Se x=${A.fmt(x,2)}, quale sarà y?</div>`;
    chat.innerHTML+=`<div class="rounded-xl p-3 bg-blue-500/5 border border-blue-500/10"><b class="text-blue-300">Agente B:</b> Predico ${A.fmt(pred,4)}</div>`;
    chat.innerHTML+=`<div class="rounded-xl p-3 bg-purple-500/5 border border-purple-500/10"><b class="text-purple-300">Feedback:</b> target=${A.fmt(y,4)}, loss=${A.fmt(result.loss,6)}. Backprop + update completati.</div>`;
    chat.scrollTop=chat.scrollHeight;refreshAll();sendPythonState();
  }

  // CUSTOM DATA
  function loadCustomData(){
    const lines=$("customData").value.trim().split(/\n+/);
    const data=[];
    for(const line of lines){
      const nums=line.split(/[;,\s]+/).map(Number).filter(Number.isFinite);
      if(nums.length>=2)data.push({x:[nums[0]],y:[nums[1]]});
    }
    if(data.length<2){alert("Inserisci almeno due righe nel formato x,y");return;}
    state.task="regression";state.dataset=data;state.datasetName="custom";$("taskMode").value="regression";rebuildNetwork();log(`Dataset personalizzato caricato: ${data.length} esempi.`,"system");
  }

  // TRANSFORMER
  function runAttention(){
    const r=A.attentionDemo($("attentionText").value);
    $("tokenView").innerHTML=r.tokens.map((t,i)=>`<span class="pill rounded-lg px-2.5 py-1.5 text-sm"><span class="text-slate-500">${i}</span> ${t}</span>`).join("");
    const max=Math.max(...r.attn.flat());
    $("attentionHeatmap").innerHTML=`<div style="display:grid;grid-template-columns:80px repeat(${r.tokens.length},minmax(46px,1fr));gap:2px">
      <div></div>${r.tokens.map(t=>`<div class="text-[10px] text-slate-500 text-center truncate">${t}</div>`).join("")}
      ${r.attn.map((row,i)=>`<div class="text-[10px] text-slate-500 flex items-center">${r.tokens[i]}</div>${row.map(v=>`<div class="heat-cell rounded" style="background:rgba(59,130,246,${.06+.75*(v/max)})">${A.fmt(v,2)}</div>`).join("")}`).join("")}
    </div>`;
    $("attentionExplain").textContent=`Sono stati creati ${r.tokens.length} token. Ogni riga mostra quanto un token "guarda" gli altri dopo softmax(QKᵀ/√d).`;
  }

  // EXPORT
  function download(name,text,type="application/json"){
    const blob=new Blob([text],{type});const url=URL.createObjectURL(blob);const a=document.createElement("a");a.href=url;a.download=name;a.click();URL.revokeObjectURL(url);
  }

  // Python Live sync
  function sendPythonState(){
    if(state.pythonWin&&!state.pythonWin.closed){
      try{state.pythonWin.postMessage({type:"aiml-state",epoch:state.epoch,phase:state.phase,lr:state.lr,optimizer:state.optimizer,
        architecture:state.net?.sizes,last:state.last,params:state.net?.paramsFlat().slice(0,80),
        model:state.net ? JSON.parse(state.net.exportJSON()) : null},"*");}catch{}
    }
  }

  function initChart(){
    const c=$("lossChart").getContext("2d");
    window.lossChart=new Chart(c,{type:"line",data:{labels:[],datasets:[{label:"Loss",data:[],borderColor:"rgba(96,165,250,.95)",backgroundColor:"rgba(59,130,246,.08)",fill:true,tension:.25,pointRadius:0,borderWidth:2}]},
      options:{responsive:true,maintainAspectRatio:false,animation:false,scales:{x:{ticks:{color:"#64748b"},grid:{color:"rgba(148,163,184,.05)"}},y:{beginAtZero:true,ticks:{color:"#64748b"},grid:{color:"rgba(148,163,184,.05)"}}},plugins:{legend:{labels:{color:"#cbd5e1"}}}}});
  }

  function nav(section){
    document.querySelectorAll(".section").forEach(s=>s.classList.toggle("active",s.id===section));
    document.querySelectorAll(".nav-btn").forEach(b=>b.classList.toggle("active",b.dataset.section===section));
    $("pageTitle").textContent=document.querySelector(`.nav-btn[data-section="${section}"]`)?.dataset.title||"AI Model Lab";
    setTimeout(refreshCanvas,30);
  }

  function bind(){
    document.querySelectorAll(".nav-btn").forEach(b=>b.addEventListener("click",()=>nav(b.dataset.section)));
    $("startBtn").onclick=start;$("pauseBtn").onclick=pause;$("stepEpochBtn").onclick=batchTrain;$("stepPhaseBtn").onclick=stepPhase;
    $("resetBtn").onclick=()=>{pause();rebuildNetwork();};
    $("clearLog").onclick=()=>$("terminal").innerHTML="";
    $("rebuildBtn").onclick=rebuildNetwork;
    $("datasetSelect").onchange=e=>loadDataset(e.target.value);
    $("taskMode").onchange=e=>{
      const wanted=e.target.value;
      $("datasetSelect").value=wanted==="regression"?"regressionLinear":"xor";loadDataset($("datasetSelect").value);
    };
    $("lrSlider").oninput=e=>{state.lr=parseFloat(e.target.value);$("lrText").textContent=A.fmt(state.lr,3);refreshMetrics();};
    $("speedSlider").oninput=e=>{state.speed=parseInt(e.target.value);$("speedText").textContent=state.speed+" ms";};
    $("optimizerSelect").onchange=e=>state.optimizer=e.target.value;
    ["nx1","nx2","nw1","nw2","nb"].forEach(id=>$(id).oninput=neuronLab);$("nact").onchange=neuronLab;
    $("teacherStepBtn").onclick=teacherStep;$("teacherAutoBtn").onclick=()=>{for(let i=0;i<10;i++)teacherStep();};
    $("loadCustomBtn").onclick=loadCustomData;
    $("trainCustomBtn").onclick=()=>{for(let i=0;i<100;i++)batchTrain();};
    $("attentionRunBtn").onclick=runAttention;
    $("exportModelBtn").onclick=()=>download("ai-model-lab-model.json",state.net.exportJSON());
    $("exportDataBtn").onclick=()=>download("dataset.json",JSON.stringify(state.dataset,null,2));
    $("openPythonBtn").onclick=()=>{state.pythonWin=window.open("python-live.html","AIMLPythonLive");setTimeout(sendPythonState,300);};
    window.addEventListener("message",e=>{if(e.data?.type==="aiml-python-ready"){state.pythonWin=e.source;sendPythonState();}});
    window.addEventListener("resize",refreshCanvas);
  }

  A.appAPI={rebuildNetwork,loadDataset,batchTrain,refreshAll,nav,pause,start,sendPythonState,log};

  document.addEventListener("DOMContentLoaded",()=>{
    initChart();bind();loadDataset("regressionLinear");neuronLab();runAttention();nav("dashboard");
    setInterval(()=>{if(document.visibilityState==="visible")refreshCanvas();},180);
  });
})();
