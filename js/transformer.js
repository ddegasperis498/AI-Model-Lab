window.AIML = window.AIML || {};
(() => {
  const A=window.AIML;
  function hashWord(w){
    let h=2166136261>>>0;
    for(let i=0;i<w.length;i++){h^=w.charCodeAt(i);h=Math.imul(h,16777619);}
    return h>>>0;
  }
  function seeded(seed){
    let s=seed>>>0;
    return ()=>{s=(Math.imul(1664525,s)+1013904223)>>>0;return s/4294967296;};
  }
  function vec(word,d=6){
    const r=seeded(hashWord(word));
    return Array.from({length:d},()=>r()*2-1);
  }
  const dot=(a,b)=>a.reduce((s,v,i)=>s+v*b[i],0);
  const softmax=arr=>{
    const m=Math.max(...arr),e=arr.map(x=>Math.exp(x-m)),s=e.reduce((a,b)=>a+b,0);
    return e.map(x=>x/s);
  };
  A.attentionDemo=(text)=>{
    const tokens=text.trim().split(/\s+/).filter(Boolean).slice(0,10);
    const E=tokens.map(t=>vec(t,6));
    // Educational simplified self-attention: Q,K,V derived via deterministic linear-ish transforms.
    const Q=E.map(v=>v.map((x,i)=>x*(.7+i*.05)));
    const K=E.map(v=>v.map((x,i)=>x*(.55+(5-i)*.04)));
    const V=E.map(v=>v.map((x,i)=>x*(.8+(i%2)*.1)));
    const scores=Q.map(q=>K.map(k=>dot(q,k)/Math.sqrt(q.length)));
    const attn=scores.map(softmax);
    const out=attn.map(row=>V[0].map((_,d)=>row.reduce((s,a,j)=>s+a*V[j][d],0)));
    return {tokens,E,Q,K,V,scores,attn,out};
  };
})();
