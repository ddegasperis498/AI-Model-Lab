window.AIML = window.AIML || {};

(() => {
  const A = window.AIML;

  A._rng = Math.random;
  A.random = () => A._rng();
  A.setSeed = (seed=12345) => {
    let s = (Number(seed) >>> 0) || 1;
    A._rng = () => {
      s = (Math.imul(1664525, s) + 1013904223) >>> 0;
      return s / 4294967296;
    };
    return s;
  };
  A.rand = (scale=1) => (A.random()*2-1)*scale;
  A.clamp = (x,a,b) => Math.max(a,Math.min(b,x));
  A.fmt = (v,d=5) => Number.isFinite(v) ? (Math.abs(v)>0 && Math.abs(v)<1e-4 ? v.toExponential(3) : v.toFixed(d)) : "—";

  A.activations = {
    linear: {
      f: x => x,
      d: (_z,a) => 1
    },
    tanh: {
      f: x => Math.tanh(x),
      d: (_z,a) => 1-a*a
    },
    relu: {
      f: x => Math.max(0,x),
      d: z => z>0 ? 1 : 0
    },
    sigmoid: {
      f: x => 1/(1+Math.exp(-A.clamp(x,-40,40))),
      d: (_z,a) => a*(1-a)
    }
  };

  class DenseLayer {
    constructor(inSize,outSize,activation="tanh") {
      this.inSize=inSize; this.outSize=outSize; this.activation=activation;
      const scale = Math.sqrt(2/Math.max(1,inSize));
      this.W = Array.from({length:outSize},()=>Array.from({length:inSize},()=>A.rand(scale)));
      this.b = Array.from({length:outSize},()=>A.rand(.08));
      this.mW = this.W.map(r=>r.map(()=>0)); this.vW = this.W.map(r=>r.map(()=>0));
      this.mb = this.b.map(()=>0); this.vb=this.b.map(()=>0);
      this.cache = null; this.grads = null;
    }

    forward(input) {
      const act=A.activations[this.activation];
      const z=this.W.map((row,j)=>row.reduce((s,w,i)=>s+w*input[i],this.b[j]));
      const out=z.map(v=>act.f(v));
      this.cache={input:[...input],z:[...z],out:[...out]};
      return out;
    }

    backward(gradOut) {
      const {input,z,out}=this.cache;
      const act=A.activations[this.activation];
      const dz=gradOut.map((g,j)=>g*act.d(z[j],out[j]));
      const dW=this.W.map((row,j)=>row.map((_,i)=>dz[j]*input[i]));
      const db=[...dz];
      const dInput=Array(this.inSize).fill(0);
      for(let j=0;j<this.outSize;j++) for(let i=0;i<this.inSize;i++) dInput[i]+=this.W[j][i]*dz[j];
      this.grads={dW,db,dz};
      return dInput;
    }

    apply(lr,opt="sgd",step=1) {
      if(!this.grads) return;
      const clip=g=>A.clamp(g,-20,20);
      if(opt==="adam") {
        const b1=.9,b2=.999,eps=1e-8;
        for(let j=0;j<this.outSize;j++) {
          for(let i=0;i<this.inSize;i++) {
            const g=clip(this.grads.dW[j][i]);
            this.mW[j][i]=b1*this.mW[j][i]+(1-b1)*g;
            this.vW[j][i]=b2*this.vW[j][i]+(1-b2)*g*g;
            const mh=this.mW[j][i]/(1-Math.pow(b1,step));
            const vh=this.vW[j][i]/(1-Math.pow(b2,step));
            this.W[j][i]-=lr*mh/(Math.sqrt(vh)+eps);
          }
          const g=clip(this.grads.db[j]);
          this.mb[j]=b1*this.mb[j]+(1-b1)*g;
          this.vb[j]=b2*this.vb[j]+(1-b2)*g*g;
          const mh=this.mb[j]/(1-Math.pow(b1,step));
          const vh=this.vb[j]/(1-Math.pow(b2,step));
          this.b[j]-=lr*mh/(Math.sqrt(vh)+eps);
        }
      } else {
        for(let j=0;j<this.outSize;j++) {
          for(let i=0;i<this.inSize;i++) this.W[j][i]-=lr*clip(this.grads.dW[j][i]);
          this.b[j]-=lr*clip(this.grads.db[j]);
        }
      }
    }

    paramCount(){ return this.inSize*this.outSize+this.outSize; }
  }

  class MLP {
    constructor(sizes,hiddenActivation="tanh",task="regression") {
      this.sizes=[...sizes]; this.task=task; this.hiddenActivation=hiddenActivation; this.step=0;
      this.layers=[];
      for(let i=0;i<sizes.length-1;i++) {
        const isLast=i===sizes.length-2;
        const activation=isLast ? (task==="classification" ? "sigmoid":"linear") : hiddenActivation;
        this.layers.push(new DenseLayer(sizes[i],sizes[i+1],activation));
      }
      this.last=null;
    }

    forward(x) {
      let a=[...x];
      const traces=[{name:"input",a:[...a]}];
      this.layers.forEach((layer,idx)=>{
        a=layer.forward(a);
        traces.push({name:`layer_${idx+1}`,z:[...layer.cache.z],a:[...a],activation:layer.activation});
      });
      this.last={x:[...x],output:[...a],traces};
      return a;
    }

    lossAndGrad(output,target) {
      if(this.task==="classification") {
        const p=A.clamp(output[0],1e-7,1-1e-7);
        const y=target[0];
        const loss=-(y*Math.log(p)+(1-y)*Math.log(1-p));
        // because last activation is sigmoid and layer.backward multiplies sigmoid derivative,
        // dL/dp is used here.
        const grad=[-(y/p)+(1-y)/(1-p)];
        return {loss,grad,error:p-y};
      }
      const diff=output[0]-target[0];
      return {loss:.5*diff*diff,grad:[diff],error:diff};
    }

    trainSample(x,target,lr=.03,opt="sgd",apply=true) {
      const out=this.forward(x);
      const lg=this.lossAndGrad(out,target);
      let grad=lg.grad;
      for(let i=this.layers.length-1;i>=0;i--) grad=this.layers[i].backward(grad);
      this.step++;
      if(apply) this.layers.forEach(l=>l.apply(lr,opt,this.step));
      return {output:out,target:[...target],...lg,traces:this.last.traces};
    }

    predict(x){ return this.forward(x)[0]; }
    paramCount(){ return this.layers.reduce((s,l)=>s+l.paramCount(),0); }

    paramsFlat() {
      const out=[];
      this.layers.forEach((l,li)=>{
        for(let j=0;j<l.outSize;j++) {
          for(let i=0;i<l.inSize;i++) out.push({
            name:`L${li+1}.W[${j},${i}]`,value:l.W[j][i],
            grad:l.grads?.dW?.[j]?.[i] ?? 0, layer:li, kind:"weight",from:i,to:j
          });
          out.push({name:`L${li+1}.b[${j}]`,value:l.b[j],grad:l.grads?.db?.[j] ?? 0,layer:li,kind:"bias",to:j});
        }
      });
      return out;
    }

    exportJSON() {
      return JSON.stringify({
        version:1,sizes:this.sizes,task:this.task,hiddenActivation:this.hiddenActivation,
        layers:this.layers.map(l=>({inSize:l.inSize,outSize:l.outSize,activation:l.activation,W:l.W,b:l.b}))
      },null,2);
    }

    static fromJSON(obj) {
      if(typeof obj==="string") obj=JSON.parse(obj);
      const net=new MLP(obj.sizes,obj.hiddenActivation,obj.task);
      obj.layers.forEach((src,i)=>{net.layers[i].W=src.W.map(r=>[...r]);net.layers[i].b=[...src.b];});
      return net;
    }
  }

  A.DenseLayer=DenseLayer;
  A.MLP=MLP;
})();
