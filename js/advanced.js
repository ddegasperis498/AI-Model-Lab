window.AIML = window.AIML || {};
(() => {
  const A=window.AIML;
  const $=id=>document.getElementById(id);
  const Adv=A.Advanced={};
  A.setSeed(42);
  let genChart=null,optChart=null,splitVersion=0,lastGenEpoch=null,lastParamMap=new Map(),currentParamMap=new Map(),helpEnabled=false,tourIndex=0;

  const help={
    taskMode:"Sceglie il tipo di problema. Regressione = valore numerico continuo; classificazione = probabilità/classe.",
    datasetSelect:"Sceglie gli esempi su cui la rete si allena. Cambiare dataset ricrea il modello.",
    hiddenSizes:"Numero di neuroni per ciascun hidden layer. '8,6,4' crea tre layer nascosti.",
    activationSelect:"Funzione non lineare dei neuroni nascosti. Tanh, ReLU e Sigmoid modificano il flusso del gradiente.",
    optimizerSelect:"Algoritmo che usa i gradienti per modificare i parametri. SGD è diretto; Adam adatta il passo per parametro.",
    lrSlider:"Learning rate η: moltiplica il gradiente nell'update. Troppo basso = lento; troppo alto = instabile.",
    speedSlider:"Ritardo tra gli aggiornamenti visuali. Non cambia la matematica, solo la velocità della demo.",
    batchSize:"Quanti esempi casuali vengono usati per ogni epoca visualizzata.",
    startBtn:"Avvia il training continuo.",pauseBtn:"Ferma il training conservando pesi e stato.",stepEpochBtn:"Esegue un singolo batch/epoca visualizzata.",stepPhaseBtn:"Avanza una micro-fase: Forward → Loss → Backprop → Update.",
    resetBtn:"Ricrea i pesi del modello e azzera la cronologia.",rebuildBtn:"Ricostruisce la rete usando architettura e activation correnti.",
    exportModelBtn:"Scarica architettura, pesi e bias in JSON.",importModelBtn:"Carica un JSON precedentemente esportato e ripristina la rete.",exportDataBtn:"Scarica il dataset corrente in JSON.",openPythonBtn:"Apre la pagina Python Live sincronizzata.",
    seedInput:"Seed del generatore pseudocasuale. Consente di ripetere lo stesso esperimento.",trainPct:"Percentuale dati usata per aggiornare i pesi.",valPct:"Percentuale dati non usata per update, ma usata per controllare generalizzazione.",
    runOptimizerArena:"Crea due copie identiche del modello e confronta SGD con Adam.",attentionRunBtn:"Calcola una self-attention didattica sui token inseriti."
  };

  function state(){return A.appState;}
  function fmt(v,d=5){return A.fmt(v,d);}

  function shuffledCopy(arr){
    const out=[...arr];
    for(let i=out.length-1;i>0;i--){const j=Math.floor(A.random()*(i+1));[out[i],out[j]]=[out[j],out[i]];}
    return out;
  }

  function createSplit(){
    const s=state();if(!s?.dataset?.length)return;
    let tr=parseInt($("trainPct")?.value||70),va=parseInt($("valPct")?.value||15);
    if(tr+va>95){va=95-tr;if($("valPct"))$("valPct").value=va;}
    const te=100-tr-va;
    $("trainPctText").textContent=tr+"%";$("valPctText").textContent=va+"%";$("testPctText").textContent=te+"%";
    const data=shuffledCopy(s.dataset),n=data.length,nTr=Math.max(1,Math.floor(n*tr/100)),nVa=Math.max(1,Math.floor(n*va/100));
    s.trainData=data.slice(0,nTr);s.valData=data.slice(nTr,nTr+nVa);s.testData=data.slice(nTr+nVa);
    if(!s.testData.length){s.testData=s.valData.slice(-1);}
    splitVersion++; lastGenEpoch=null;
    if(genChart){genChart.data.labels=[];genChart.data.datasets.forEach(d=>d.data=[]);genChart.update();}
    evaluateAndRender();
  }

  function evaluate(data,model){
    const s=state();if(!model||!data?.length)return {loss:NaN};
    let loss=0,mae=0,ssRes=0,ys=[],preds=[],tp=0,tn=0,fp=0,fn=0,correct=0;
    for(const sample of data){
      const out=model.forward(sample.x);const lg=model.lossAndGrad(out,sample.y);loss+=lg.loss;
      const p=out[0],y=sample.y[0];preds.push(p);ys.push(y);
      if(s.task==="classification"){
        const pc=p>=.5?1:0;if(pc===y)correct++;if(pc===1&&y===1)tp++;else if(pc===0&&y===0)tn++;else if(pc===1&&y===0)fp++;else fn++;
      } else {mae+=Math.abs(p-y);ssRes+=(p-y)*(p-y);}
    }
    const out={loss:loss/data.length};
    if(s.task==="classification") Object.assign(out,{accuracy:correct/data.length,tp,tn,fp,fn});
    else {
      const mean=ys.reduce((a,b)=>a+b,0)/ys.length;const ssTot=ys.reduce((a,y)=>a+(y-mean)*(y-mean),0);Object.assign(out,{mae:mae/data.length,r2:ssTot>1e-12?1-ssRes/ssTot:0});
    }
    return out;
  }

  function evaluateAndRender(){
    const s=state();if(!s?.net||!s.trainData)return;
    // Evaluate on a clone so validation/test do not overwrite the training caches used by step-by-step backprop.
    const evalNet=A.MLP.fromJSON(s.net.exportJSON());
    const tr=evaluate(s.trainData,evalNet),va=evaluate(s.valData,evalNet),te=evaluate(s.testData,evalNet);
    $("trainLossAdv").textContent=fmt(tr.loss,6);$("valLossAdv").textContent=fmt(va.loss,6);$("testLossAdv").textContent=fmt(te.loss,6);
    const gap=va.loss-tr.loss;$("gapAdv").textContent=fmt(gap,6);$("gapAdv").className="text-xl mt-1 "+(gap>Math.max(.05,tr.loss*.8)?"text-rose-300":"text-emerald-300");
    if(s.task==="classification"){
      $("scoreAdv").textContent=`Acc ${fmt(te.accuracy*100,1)}%`;$("confusionBox").classList.remove("hidden");
      $("confusionMatrix").innerHTML=`<div class="metric rounded-xl p-4"><div class="small-label">True Positive</div><div class="text-2xl text-emerald-300">${te.tp}</div></div><div class="metric rounded-xl p-4"><div class="small-label">False Positive</div><div class="text-2xl text-rose-300">${te.fp}</div></div><div class="metric rounded-xl p-4"><div class="small-label">False Negative</div><div class="text-2xl text-rose-300">${te.fn}</div></div><div class="metric rounded-xl p-4"><div class="small-label">True Negative</div><div class="text-2xl text-emerald-300">${te.tn}</div></div>`;
    }else{$("scoreAdv").textContent=`R² ${fmt(te.r2,3)}`;$("confusionBox").classList.add("hidden");}
    let status="Stabile",cls="warn-ok";
    if(!Number.isFinite(va.loss)){status="Dati insufficienti";cls="warn-mid";}
    else if(gap>Math.max(.1,tr.loss*1.2)){status="Possibile overfitting";cls="warn-bad";}
    else if(tr.loss>1 && va.loss>1){status="Underfitting / training iniziale";cls="warn-mid";}
    $("overfitStatus").textContent=status;$("overfitStatus").className=`text-sm mt-2 ${cls}`;
    if(genChart && lastGenEpoch!==s.epoch){lastGenEpoch=s.epoch;genChart.data.labels.push(s.epoch);genChart.data.datasets[0].data.push(tr.loss);genChart.data.datasets[1].data.push(va.loss);genChart.data.datasets[2].data.push(te.loss);if(genChart.data.labels.length>160){genChart.data.labels.shift();genChart.data.datasets.forEach(d=>d.data.shift());}genChart.update("none");}
  }

  function diagnostics(){
    const s=state();if(!s?.net)return;
    const params=s.net.paramsFlat(),grads=params.map(p=>p.grad||0),weights=params.filter(p=>p.kind==="weight").map(p=>p.value||0);
    const gradNorm=Math.sqrt(grads.reduce((a,g)=>a+g*g,0)),gradMax=Math.max(0,...grads.map(Math.abs)),weightNorm=Math.sqrt(weights.reduce((a,w)=>a+w*w,0));
    $("gradNorm").textContent=fmt(gradNorm,5);$("gradMax").textContent=fmt(gradMax,5);$("weightNorm").textContent=fmt(weightNorm,5);
    let dead=0,totalRelu=0;
    s.net.layers.forEach((l,i)=>{if(l.activation==="relu"&&i<s.net.layers.length-1){const a=l.cache?.out||[];dead+=a.filter(v=>v===0).length;totalRelu+=a.length;}});
    $("deadRelu").textContent=totalRelu?`${dead}/${totalRelu}`:"N/A";
    const checks=[];
    if(gradMax>10)checks.push(["Exploding gradient","Il gradiente massimo è molto alto: riduci Learning Rate o cambia inizializzazione.","warn-bad"]);
    else if(gradNorm>0 && gradNorm<1e-6)checks.push(["Vanishing gradient","I gradienti sono quasi nulli: layer profondi o activation satura possono rallentare l'apprendimento.","warn-mid"]);
    else checks.push(["Gradienti","Il flusso dei gradienti è numericamente ragionevole.","warn-ok"]);
    if(totalRelu&&dead/Math.max(1,totalRelu)>.6)checks.push(["Dead ReLU","Molti neuroni ReLU producono zero. Prova LR minore o tanh.","warn-bad"]);
    else if(totalRelu)checks.push(["ReLU","La maggior parte dei neuroni ReLU è attiva.","warn-ok"]);
    if(weightNorm>50)checks.push(["Pesi grandi","La norma dei pesi è elevata: possibile instabilità.","warn-mid"]);
    $("healthMonitor").innerHTML=checks.map(([t,m,c])=>`<div class="rounded-xl border border-slate-800 p-3"><b class="${c}">${t}</b><div class="text-slate-500 mt-1">${m}</div></div>`).join("");
    $("gradientFlow").innerHTML=s.net.layers.map((l,i)=>{const gs=[];l.grads?.dW?.forEach(r=>r.forEach(g=>gs.push(Math.abs(g))));const mean=gs.length?gs.reduce((a,b)=>a+b,0)/gs.length:0;const pct=Math.min(100,Math.max(1,Math.log10(mean+1e-9)*18+160));return `<div><div class="flex justify-between text-xs"><span>Layer ${i+1} · ${l.activation}</span><span class="font-mono text-purple-300">mean |grad| ${fmt(mean,6)}</span></div><div class="statbar mt-1"><span style="width:${pct}%"></span></div></div>`;}).join("");
  }

  function parameterDeltas(){
    const s=state();if(!s?.net)return;
    currentParamMap=new Map(s.net.paramsFlat().map(p=>[p.name,{...p}]));
    const body=$("paramBody");if(!body)return;
    body.innerHTML=[...currentParamMap.values()].slice(0,160).map(p=>{const prev=lastParamMap.get(p.name);const before=prev?.value ?? p.value;const delta=p.value-before;return `<tr><td class="font-mono text-slate-300">${p.name}</td><td class="font-mono text-slate-400">${fmt(before,6)}</td><td class="font-mono text-purple-300">${fmt(p.grad,6)}</td><td class="font-mono ${delta>=0?"text-emerald-300":"text-rose-300"}">${fmt(delta,6)}</td><td class="font-mono ${p.value>=0?"text-emerald-300":"text-rose-300"}">${fmt(p.value,6)}</td><td class="text-slate-500">${p.kind}</td></tr>`;}).join("");
    lastParamMap=currentParamMap;
  }

  function initCharts(){
    if($("generalizationChart"))genChart=new Chart($("generalizationChart").getContext("2d"),{type:"line",data:{labels:[],datasets:[{label:"Train",data:[],borderColor:"#60a5fa",pointRadius:0},{label:"Validation",data:[],borderColor:"#c084fc",pointRadius:0},{label:"Test",data:[],borderColor:"#34d399",pointRadius:0}]},options:{responsive:true,maintainAspectRatio:false,animation:false,scales:{x:{ticks:{color:"#64748b"},grid:{color:"rgba(148,163,184,.05)"}},y:{beginAtZero:true,ticks:{color:"#64748b"},grid:{color:"rgba(148,163,184,.05)"}}},plugins:{legend:{labels:{color:"#cbd5e1"}}}}});
    if($("optimizerChart"))optChart=new Chart($("optimizerChart").getContext("2d"),{type:"line",data:{labels:[],datasets:[{label:"SGD",data:[],borderColor:"#60a5fa",pointRadius:0},{label:"Adam",data:[],borderColor:"#c084fc",pointRadius:0}]},options:{responsive:true,maintainAspectRatio:false,animation:false,scales:{x:{ticks:{color:"#64748b"},grid:{color:"rgba(148,163,184,.05)"}},y:{beginAtZero:true,ticks:{color:"#64748b"},grid:{color:"rgba(148,163,184,.05)"}}},plugins:{legend:{labels:{color:"#cbd5e1"}}}}});
  }

  function optimizerArena(){
    const s=state();if(!s?.net)return;
    const epochs=Math.max(10,Math.min(500,parseInt($("arenaEpochs").value)||80)),sgdLr=parseFloat($("arenaSgdLr").value)||.03,adamLr=parseFloat($("arenaAdamLr").value)||.01;
    const base=s.net.exportJSON(),mS=A.MLP.fromJSON(base),mA=A.MLP.fromJSON(base),data=s.trainData?.length?s.trainData:s.dataset;
    const sgd=[],adam=[];
    for(let e=0;e<epochs;e++){
      let ls=0,la=0;for(let i=0;i<data.length;i++){const sample=data[i];ls+=mS.trainSample(sample.x,sample.y,sgdLr,"sgd",true).loss;la+=mA.trainSample(sample.x,sample.y,adamLr,"adam",true).loss;}sgd.push(ls/data.length);adam.push(la/data.length);
    }
    optChart.data.labels=Array.from({length:epochs},(_,i)=>i+1);optChart.data.datasets[0].data=sgd;optChart.data.datasets[1].data=adam;optChart.update();
    $("optimizerSummary").innerHTML=`Stessi pesi iniziali, stesso ordine degli esempi. Loss finale SGD: <b class="text-blue-300">${fmt(sgd.at(-1),7)}</b> · Adam: <b class="text-purple-300">${fmt(adam.at(-1),7)}</b>. Non esiste un optimizer universalmente migliore: il comportamento dipende da problema e iperparametri.`;
  }

  function importModel(file){
    const r=new FileReader();r.onload=()=>{try{const net=A.MLP.fromJSON(JSON.parse(r.result));const s=state();s.net=net;s.task=net.task;s.hidden=net.sizes.slice(1,-1);$("hiddenSizes").value=s.hidden.join(",");$("activationSelect").value=net.hiddenActivation;$("taskMode").value=s.task;A.appAPI.refreshAll();createSplit();A.appAPI.log("Modello importato correttamente.","system");}catch(err){alert("File modello non valido: "+err.message);}};r.readAsText(file);
  }

  function installHelp(){
    Object.entries(help).forEach(([id,text])=>{const el=$(id);if(!el)return;el.title=text;const label=el.closest("div")?.querySelector("label");if(label&&!label.querySelector(".auto-help")){const q=document.createElement("span");q.className="help-dot auto-help ml-1";q.textContent="?";q.title=text;label.appendChild(q);}});
  }

  const tour=[
    ["dashboard","taskMode","Task e Dataset","Scegli se vuoi regressione o classificazione e quali esempi userà la rete."],
    ["dashboard","hiddenSizes","Architettura","Qui decidi quanti hidden layer e quanti neuroni contiene ciascun layer."],
    ["dashboard","networkCanvas","Rete neurale","Le connessioni sono pesi reali: colore = segno, spessore = intensità."],
    ["parameters","paramBody","Parameter Inspector","Qui osservi valore precedente, gradiente, delta e nuovo valore di ogni parametro."],
    ["generalization","generalizationChart","Generalizzazione","Train, validation e test mostrano se il modello apprende una regola generalizzabile."],
    ["diagnostics","healthMonitor","Diagnostica","Questa pagina segnala gradienti esplosivi/svaniti e neuroni ReLU morti."],
    ["optimizer","optimizerChart","Optimizer Arena","SGD e Adam vengono confrontati partendo dagli stessi pesi."],
    ["transformer","attentionHeatmap","Mini Transformer","Osserva una self-attention didattica token per token."]
  ];

  function closeTour(){document.querySelectorAll(".tour-overlay,.tour-card").forEach(e=>e.remove());document.querySelectorAll(".tour-focus").forEach(e=>e.classList.remove("tour-focus"));}
  function showTour(i){closeTour();if(i<0||i>=tour.length)return; tourIndex=i;const [section,id,title,text]=tour[i];A.appAPI.nav(section);setTimeout(()=>{const target=$(id);if(!target)return;target.classList.add("tour-focus");const overlay=document.createElement("div");overlay.className="tour-overlay";const card=document.createElement("div");card.className="tour-card";card.style.left="50%";card.style.bottom="28px";card.style.transform="translateX(-50%)";card.innerHTML=`<div class="small-label">Passo ${i+1} / ${tour.length}</div><h3 class="text-lg font-semibold mt-1">${title}</h3><p class="text-sm text-slate-400 mt-2">${text}</p><div class="flex justify-between mt-4"><button id="tourClose" class="rounded-lg bg-slate-800 px-3 py-2">Chiudi</button><div class="flex gap-2"><button id="tourPrev" class="rounded-lg bg-slate-800 px-3 py-2" ${i===0?"disabled":""}>Indietro</button><button id="tourNext" class="rounded-lg bg-blue-600 px-3 py-2">${i===tour.length-1?"Fine":"Avanti"}</button></div></div>`;document.body.append(overlay,card);$("tourClose").onclick=closeTour;$("tourPrev").onclick=()=>showTour(i-1);$("tourNext").onclick=()=>i===tour.length-1?closeTour():showTour(i+1);},100);}

  function bind(){
    $("trainPct")?.addEventListener("input",createSplit);$("valPct")?.addEventListener("input",createSplit);$("resplitBtn")?.addEventListener("click",createSplit);
    $("runOptimizerArena")?.addEventListener("click",optimizerArena);
    $("applySeedBtn")?.addEventListener("click",()=>{A.setSeed(parseInt($("seedInput").value)||42);A.appAPI.loadDataset($("datasetSelect").value);createSplit();A.appAPI.log(`Seed applicato: ${$("seedInput").value}. Dataset e pesi ricreati.`,"system");});
    document.querySelectorAll(".lrPreset").forEach(b=>b.onclick=()=>{$("lrSlider").value=b.dataset.lr;$("lrSlider").dispatchEvent(new Event("input"));});
    $("importModelBtn")?.addEventListener("click",()=>$("importModelFile").click());$("importModelFile")?.addEventListener("change",e=>e.target.files?.[0]&&importModel(e.target.files[0]));
    $("tourBtn")?.addEventListener("click",()=>showTour(0));$("helpModeBtn")?.addEventListener("click",()=>{helpEnabled=!helpEnabled;$("helpModeBtn").classList.toggle("bg-purple-500/15",helpEnabled);alert(helpEnabled?"Help contestuale attivo: passa il mouse sui controlli e sui ? per leggere la spiegazione.":"Help contestuale disattivato. I tooltip restano disponibili sui simboli ?.");});
  }

  Adv.refreshParameters=parameterDeltas;
  Adv.onModelRebuilt=()=>{setTimeout(()=>{lastParamMap=new Map();createSplit();parameterDeltas();diagnostics();},0);};

  document.addEventListener("DOMContentLoaded",()=>{
    initCharts();bind();installHelp();
    setTimeout(()=>{createSplit();parameterDeltas();diagnostics();},150);
    setInterval(()=>{const s=state();if(!s?.net)return;evaluateAndRender();diagnostics();parameterDeltas();},900);
  });
})();
