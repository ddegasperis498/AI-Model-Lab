window.AIML = window.AIML || {};
(() => {
  const A=window.AIML;

  function resizeCanvas(canvas,ctx) {
    const dpr=Math.max(1,Math.min(2,window.devicePixelRatio||1));
    const r=canvas.getBoundingClientRect();
    const w=Math.max(1,Math.floor(r.width*dpr)),h=Math.max(1,Math.floor(r.height*dpr));
    if(canvas.width!==w||canvas.height!==h){canvas.width=w;canvas.height=h;}
    ctx.setTransform(dpr,0,0,dpr,0,0);
    return {w:r.width,h:r.height,dpr};
  }

  A.drawNetwork=(canvas,net,phase="idle")=>{
    if(!canvas||!net) return;
    const ctx=canvas.getContext("2d"),{w,h}=resizeCanvas(canvas,ctx);
    ctx.clearRect(0,0,w,h);
    ctx.strokeStyle="rgba(148,163,184,.035)";
    for(let x=20;x<w;x+=36){ctx.beginPath();ctx.moveTo(x,0);ctx.lineTo(x,h);ctx.stroke();}
    for(let y=20;y<h;y+=36){ctx.beginPath();ctx.moveTo(0,y);ctx.lineTo(w,y);ctx.stroke();}

    const sizes=net.sizes;
    const padX=70,usableW=Math.max(100,w-padX*2);
    const nodes=sizes.map((count,li)=>{
      const x=padX+(sizes.length===1?0:usableW*li/(sizes.length-1));
      const gap=Math.min(76,(h-100)/Math.max(1,count));
      const total=(count-1)*gap;
      return Array.from({length:count},(_,j)=>({x,y:h/2-total/2+j*gap}));
    });

    net.layers.forEach((layer,li)=>{
      layer.W.forEach((row,j)=>row.forEach((weight,i)=>{
        const a=nodes[li][i],b=nodes[li+1][j];
        const positive=weight>=0;
        ctx.save();
        ctx.strokeStyle=positive?"rgba(52,211,153,.70)":"rgba(251,113,133,.70)";
        ctx.lineWidth=Math.max(.6,Math.min(7,.7+Math.abs(weight)*2.6));
        if(phase!=="idle"){ctx.shadowBlur=10;ctx.shadowColor=positive?"rgba(52,211,153,.65)":"rgba(251,113,133,.65)";}
        ctx.beginPath();ctx.moveTo(a.x,a.y);ctx.lineTo(b.x,b.y);ctx.stroke();ctx.restore();
      }));
    });

    nodes.forEach((layerNodes,li)=>layerNodes.forEach((p,j)=>{
      const isInput=li===0,isOutput=li===nodes.length-1;
      const val=isInput?null:net.layers[li-1].cache?.out?.[j];
      ctx.save();
      ctx.shadowBlur=phase!=="idle"?22:8;
      ctx.shadowColor=isOutput?"rgba(52,211,153,.8)":isInput?"rgba(96,165,250,.8)":"rgba(192,132,252,.8)";
      ctx.beginPath();
      ctx.fillStyle=isOutput?"rgba(16,185,129,.18)":isInput?"rgba(59,130,246,.18)":"rgba(168,85,247,.17)";
      ctx.strokeStyle=isOutput?"rgb(52,211,153)":isInput?"rgb(96,165,250)":"rgb(192,132,252)";
      ctx.lineWidth=2;ctx.arc(p.x,p.y,24,0,Math.PI*2);ctx.fill();ctx.stroke();
      ctx.shadowBlur=0;ctx.fillStyle="#e2e8f0";ctx.font="600 11px Inter";ctx.textAlign="center";
      ctx.fillText(isInput?`x${j+1}`:isOutput?`y${j+1}`:`h${li}.${j+1}`,p.x,p.y-2);
      if(val!=null){ctx.fillStyle="#94a3b8";ctx.font="10px ui-monospace";ctx.fillText(A.fmt(val,2),p.x,p.y+12);}
      ctx.restore();
    }));
  };

  A.drawDataset=(canvas,dataset,net,task)=>{
    if(!canvas||!dataset) return;
    const ctx=canvas.getContext("2d"),{w,h}=resizeCanvas(canvas,ctx);
    ctx.clearRect(0,0,w,h);
    ctx.fillStyle="rgba(2,6,23,.55)";ctx.fillRect(0,0,w,h);

    if(task==="classification" && dataset[0]?.x?.length===2) {
      // decision surface
      if(net){
        const step=9;
        for(let px=0;px<w;px+=step) for(let py=0;py<h;py+=step){
          const x1=(px/w)*2-1,x2=1-(py/h)*2;
          const p=net.predict([x1,x2]);
          ctx.fillStyle=p>=.5?`rgba(59,130,246,${.04+.18*p})`:`rgba(244,63,94,${.04+.18*(1-p)})`;
          ctx.fillRect(px,py,step+1,step+1);
        }
      }
      dataset.forEach(s=>{
        const px=(s.x[0]+1)/2*w,py=(1-(s.x[1]+1)/2)*h;
        ctx.beginPath();ctx.fillStyle=s.y[0]>.5?"#60a5fa":"#fb7185";ctx.arc(px,py,4,0,Math.PI*2);ctx.fill();
      });
    } else {
      const xs=dataset.map(s=>s.x[0]),ys=dataset.map(s=>s.y[0]);
      const xmin=Math.min(...xs),xmax=Math.max(...xs),ymin=Math.min(...ys),ymax=Math.max(...ys);
      const mx=x=>20+(x-xmin)/(xmax-xmin||1)*(w-40);
      const my=y=>h-20-(y-ymin)/(ymax-ymin||1)*(h-40);
      ctx.fillStyle="#60a5fa";
      dataset.forEach(s=>{ctx.beginPath();ctx.arc(mx(s.x[0]),my(s.y[0]),3,0,Math.PI*2);ctx.fill();});
      if(net){
        ctx.strokeStyle="#34d399";ctx.lineWidth=2;ctx.beginPath();
        for(let k=0;k<120;k++){
          const x=xmin+(xmax-xmin)*k/119,y=net.predict([x]);
          if(k===0)ctx.moveTo(mx(x),my(y));else ctx.lineTo(mx(x),my(y));
        }
        ctx.stroke();
      }
    }
  };
})();
