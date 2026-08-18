import re, numpy as np, onnx
from onnx import numpy_helper
from numpy.lib.stride_tricks import sliding_window_view
base="benchmarks/vnncomp2026_benchmarks/benchmarks/yolo_2023/2.0"
m=onnx.load(f"{base}/onnx/TinyYOLO.onnx")
init={i.name:numpy_helper.to_array(i).astype(np.float64) for i in m.graph.initializer}
txt=open(f"{base}/vnnlib/TinyYOLO_prop_000001_eps_1_255.vnnlib").read()
L=np.zeros((1,3,52,52)); U=np.zeros((1,3,52,52))
for mm in re.finditer(r"\(assert \((<=|>=) X\[0,(\d+),(\d+),(\d+)\] ([-0-9.eE+]+)\)\)", txt):
    op,c,h,w,v=mm.group(1),int(mm.group(2)),int(mm.group(3)),int(mm.group(4)),float(mm.group(5))
    (U if op=="<=" else L)[0,c,h,w]=v
def attr(n,name,default=None):
    for a in n.attribute:
        if a.name==name:
            return list(a.ints) if a.ints else (a.i if a.type==2 else a.f)
    return default
def conv_ab(c,r,W,pad,st):
    def go(x,Wx):
        xp=np.pad(x,((0,0),(0,0),(pad[0],pad[2]),(pad[1],pad[3])))
        O,I,kh,kw=Wx.shape
        win=sliding_window_view(xp,(kh,kw),axis=(2,3))[:,:,::st[0],::st[1],:,:]
        return np.einsum('bihwkl,oikl->bohw',win,Wx)
    return go(c,W), go(r,np.abs(W))
vals={m.graph.input[0].name:((L+U)/2,(U-L)/2)}
relu_w=[]
for n in m.graph.node:
    t=n.op_type
    if t=="Conv":
        c,r=vals[n.input[0]]; W=init[n.input[1]]
        pad=attr(n,"pads",[0,0,0,0]); st=attr(n,"strides",[1,1])
        cc,rr=conv_ab(c,r,W,pad,st)
        if len(n.input)>2: cc=cc+init[n.input[2]].reshape(1,-1,1,1)
        vals[n.output[0]]=(cc,rr)
    elif t=="Relu":
        c,r=vals[n.input[0]]; lo,hi=c-r,c+r
        relu_w.append((n.input[0], float((hi-lo).max()), int(((lo<0)&(hi>0)).sum()), lo.size))
        lo2,hi2=np.maximum(lo,0),np.maximum(hi,0); vals[n.output[0]]=((lo2+hi2)/2,(hi2-lo2)/2)
    elif t=="Add":
        a=vals[n.input[0]]; b=vals[n.input[1]] if n.input[1] in vals else (init[n.input[1]],0*init[n.input[1]])
        vals[n.output[0]]=(a[0]+b[0], a[1]+b[1])
    elif t=="Pad":
        c,r=vals[n.input[0]]; p=init[n.input[1]].astype(int) if len(n.input)>1 else attr(n,"pads")
        pw=[(p[i],p[i+4]) for i in range(4)]
        vals[n.output[0]]=(np.pad(c,pw),np.pad(r,pw))
    elif t=="AveragePool":
        c,r=vals[n.input[0]]; k=attr(n,"kernel_shape",[2,2]); st=attr(n,"strides",k); pad=attr(n,"pads",[0,0,0,0])
        W=np.zeros((c.shape[1],c.shape[1],k[0],k[1]))
        for i in range(c.shape[1]): W[i,i]=1.0/(k[0]*k[1])
        cc,rr=conv_ab(c,r,W,pad,st); vals[n.output[0]]=(cc,rr)
    elif t=="Flatten":
        c,r=vals[n.input[0]]; vals[n.output[0]]=(c.reshape(1,-1),r.reshape(1,-1))
    else:
        raise SystemExit("unhandled "+t)
print(f"{'relu input':>16} {'EXACT-IBP max_w':>16} {'unstable':>10} {'of':>8}")
for name,w,u,tot in relu_w: print(f"{name:>16} {w:>16.5f} {u:>10} {tot:>8}")
c,r=vals[m.graph.output[0].name]; lo,hi=c-r,c+r
print(f"\nIBP Y[269] = [{lo[0,269]:.4f}, {hi[0,269]:.4f}]   width {hi[0,269]-lo[0,269]:.4f}")
print(f"NY alpha-CROWN Y[269] lower = -42784.91   (true min +1.675)")
