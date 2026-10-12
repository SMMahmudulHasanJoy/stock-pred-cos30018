"""Refit every KAN configuration (experiments A-C), saving the full fitted
curve for each so every result can be plotted and tabulated. ~3 min."""
import contextlib, io, json, os, warnings; warnings.filterwarnings("ignore")
import numpy as np, torch
from kan import KAN
HERE=os.path.dirname(os.path.abspath(__file__))
T=np.round(np.arange(0,10.0001,0.01),2)
S={"S1":np.loadtxt(HERE+"/S1.txt",delimiter=","),"S2":np.loadtxt(HERE+"/S2.txt",delimiter=",")}
def fit(y,width,grid,k,steps=200,seed=0):
    torch.manual_seed(seed); np.random.seed(seed)
    x=torch.tensor(T,dtype=torch.float32)[:,None]; s=float(np.abs(y).max()); yy=torch.tensor(y/s,dtype=torch.float32)[:,None]
    ds={"train_input":x,"train_label":yy,"test_input":x,"test_label":yy}
    net=KAN(width=width,grid=grid,k=k,grid_range=[0,10],seed=seed,device="cpu",auto_save=False)
    with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
        net.fit(ds,opt="LBFGS",steps=steps,update_grid=False)
    with torch.no_grad(): p=net(x).numpy()[:,0]*s
    return p, sum(q.numel() for q in net.parameters() if q.requires_grad)
runs=[]
for k in (1,3):
    for grid in (5,10,20,50,100): runs.append(("A",[1,1],grid,k))
for h in (2,3):
    for grid in (5,10): runs.append(("C",[1,h,1],grid,1))
out={"T":T.tolist(),"S1":S["S1"].tolist(),"S2":S["S2"].tolist(),"fits":[]}
for exp,width,grid,k in runs:
    for fn in ("S1","S2"):
        p,npar=fit(S[fn],width,grid,k); e=p-S[fn]
        out["fits"].append(dict(exp=exp,width=width,grid=grid,k=k,fn=fn,params=npar,pred=p.tolist(),
            max_err=float(np.abs(e).max()),rmse=float(np.sqrt((e**2).mean())),mean_abs=float(np.abs(e).mean()),
            argmax_t=float(T[int(np.abs(e).argmax())])))
        print(exp,width,grid,k,fn,"max %.4f"%out["fits"][-1]["max_err"],flush=True)
json.dump(out,open(HERE+"/all_fits.json","w"))
print("saved all_fits.json")
