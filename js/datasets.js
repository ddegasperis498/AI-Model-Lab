window.AIML = window.AIML || {};
(() => {
  const A=window.AIML;
  function regressionLinear(n=80) {
    return Array.from({length:n},(_,i)=>{
      const x=-2+4*i/(n-1), noise=(A.random()*2-1)*.12;
      return {x:[x],y:[2.5*x+1+noise]};
    });
  }
  function regressionSine(n=100) {
    return Array.from({length:n},(_,i)=>{
      const x=-Math.PI+2*Math.PI*i/(n-1);
      return {x:[x],y:[Math.sin(x)]};
    });
  }
  function xor(n=120) {
    const arr=[];
    for(let i=0;i<n;i++) {
      const x1=A.random()*2-1, x2=A.random()*2-1;
      const cls=((x1>0)!==(x2>0))?1:0;
      arr.push({x:[x1,x2],y:[cls]});
    }
    return arr;
  }
  function circles(n=160) {
    const arr=[];
    for(let i=0;i<n;i++) {
      const r=A.random()<.5 ? .35+A.random()*.18 : .72+A.random()*.18;
      const a=A.random()*Math.PI*2;
      const x1=r*Math.cos(a)+(A.random()*2-1)*.05;
      const x2=r*Math.sin(a)+(A.random()*2-1)*.05;
      arr.push({x:[x1,x2],y:[r>.55?1:0]});
    }
    return arr;
  }
  A.datasets={regressionLinear,regressionSine,xor,circles};
})();
