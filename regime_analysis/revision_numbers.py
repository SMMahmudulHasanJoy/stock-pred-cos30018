"""All numbers for the revised manuscript, from results_rev1/ (leak-free)."""
import json, glob, os, numpy as np
R="results_rev1"; SEEDS=[0,100,200]; H=[1,2,100,200]; REG=["Normal","Volatile","Trending"]
def rm(a,b): return float(np.sqrt(np.mean((a-b)**2)))
def load(sub,pat,h,s):
    P=[];T=[];L=[];tr=[];n=[]
    for f in range(5):
        p=f"{R}/{sub}/{pat}".replace("{h}",str(h)).replace("{s}",str(s)).replace("{f}",str(f))
        with np.load(p,allow_pickle=False) as d:
            P.append(d["prediction"].astype(float));T.append(d["truth"].astype(float));L.append(d["labels"])
            m=json.loads(str(d["meta"])); tr.append(m.get("train_rmse")); n.append(m["n_train"])
    return np.concatenate(P),np.concatenate(T),np.concatenate(L),tr,n
def seeds(sub,pat,h,ss=SEEDS):
    te=[];trn=[]
    for s in ss:
        P,T,L,tr,n=load(sub,pat,h,s); te.append(rm(P,T))
        if all(t is not None for t in tr): n=np.array(n,float); trn.append(float((n*np.array(tr)).sum()/n.sum()))
    te=np.array(te); out={"test_mean":te.mean(),"test_std":te.std(ddof=1) if len(te)>1 else 0,"test_min":te.min()}
    if trn: trn=np.array(trn); out["train_mean"]=trn.mean(); out["train_std"]=trn.std(ddof=1) if len(trn)>1 else 0
    return out
out={}
# Table 4: LSTM grid h=1
out["table4"]={c:seeds("preds_lstmgrid","lstmcfg__%s__h{h}__s{s}__f{f}.npz"%c,1) for c in ["L4-100-linear","L5-100-linear","L6-100-linear","L6-50-linear","L6-20-linear","L6-20-tanh","L3-20-tanh","L2-20-tanh","L2-10-tanh"]}
# Table 5: best LSTM per horizon
out["table5"]={h:seeds("preds_lstmgrid","lstmcfg__L2-20-tanh__h{h}__s{s}__f{f}.npz",h) for h in H}
# Table 6: KAN grid h=1
out["table6"]={c:seeds("preds_kangrid","kancfg__%s__h{h}__s{s}__f{f}.npz"%c,1) for c in ["C1","C2","C3","C4","C5","C6"]}
# naive per horizon (and at h=1 for the screening tables)
out["naive"]={h:seeds("preds","naive__-__h{h}__s0__f{f}.npz",h,[0]) for h in H}
# Table 7: regime x horizon for LSTM best, KAN pykan, naive
def regime_stats(sub,pat,h,ss=SEEDS):
    per={r:[] for r in REG+["All"]}; win={}
    for s in ss:
        P,T,L,_,_=load(sub,pat,h,s)
        for r in REG+["All"]:
            m=np.ones(len(L),bool) if r=="All" else (L==r); per[r].append(rm(P[m],T[m])); win[r]=int(m.sum())
    return {r:{"mean":float(np.mean(v)),"std":float(np.std(v,ddof=1)) if len(v)>1 else 0.0,"n":win[r]} for r,v in per.items()}
out["table7"]={h:{"LSTM":regime_stats("preds_lstmgrid","lstmcfg__L2-20-tanh__h{h}__s{s}__f{f}.npz",h),
                  "KAN":regime_stats("preds_pykan","kan__-__h{h}__s{s}__f{f}.npz",h),
                  "Naive":regime_stats("preds","naive__-__h{h}__s0__f{f}.npz",h,[0])} for h in H}
# KAN excluding fold 0 at long horizons
def excl0(h):
    v=[]
    for s in SEEDS:
        P=[];T=[]
        for f in range(1,5):
            with np.load(f"{R}/preds_pykan/kan__-__h{h}__s{s}__f{f}.npz",allow_pickle=False) as d: P.append(d["prediction"].astype(float));T.append(d["truth"].astype(float))
        v.append(rm(np.concatenate(P),np.concatenate(T)))
    return {"mean":float(np.mean(v)),"std":float(np.std(v,ddof=1))}
out["kan_excl_fold0"]={h:excl0(h) for h in (100,200)}
# fold-0 divergence check
out["kan_fold0_range"]={h:[ [float(np.load(f"{R}/preds_pykan/kan__-__h{h}__s{s}__f0.npz")["prediction"].min()),float(np.load(f"{R}/preds_pykan/kan__-__h{h}__s{s}__f0.npz")["prediction"].max())] for s in SEEDS] for h in (100,200)}
# param counts
def params(sub,pat,h): 
    with np.load(f"{R}/{sub}/{pat}".replace("{h}",str(h)).replace("{s}","0").replace("{f}","4"),allow_pickle=False) as d: return json.loads(str(d["meta"]))["parameters"]
out["params"]={h:{"LSTM":params("preds_lstmgrid","lstmcfg__L2-20-tanh__h{h}__s{s}__f{f}.npz",h),"KAN":params("preds_pykan","kan__-__h{h}__s{s}__f{f}.npz",h)} for h in H}
# Figure 4 series: last fold, seed 0, h=1, step 1
P_l,T_,_,_,_=load("preds_lstmgrid","lstmcfg__L2-20-tanh__h{h}__s{s}__f{f}.npz",1,0)
with np.load(f"{R}/preds_lstmgrid/lstmcfg__L2-20-tanh__h1__s0__f4.npz",allow_pickle=False) as d: lf=d["prediction"][:,0].astype(float); tf=d["truth"][:,0].astype(float); dates=d["dates"].tolist()
with np.load(f"{R}/preds_pykan/kan__-__h1__s0__f4.npz",allow_pickle=False) as d: kf=d["prediction"][:,0].astype(float)
with np.load(f"{R}/preds/naive__-__h1__s0__f4.npz",allow_pickle=False) as d: nf=d["prediction"][:,0].astype(float)
out["fig4"]={"dates":dates,"actual":tf.tolist(),"lstm":lf.tolist(),"kan":kf.tolist(),"naive":nf.tolist(),"rmse":{"lstm":rm(lf,tf),"kan":rm(kf,tf),"naive":rm(nf,tf)}}
json.dump(out,open(f"{R}/revision_numbers.json","w"),indent=1,default=float)
# print the key tables
print("TABLE 4 (train / test, h=1):")
for c,v in out["table4"].items(): print(f"  {c:14s} {v['train_mean']:.4f}+-{v['train_std']:.4f}  {v['test_mean']:.4f}+-{v['test_std']:.4f}")
print("  naive h=1 test %.4f"%out["naive"][1]["test_mean"])
print("TABLE 5 (best LSTM per horizon):"); [print(f"  h={h}: {v['test_mean']:.4f}+-{v['test_std']:.4f} best {v['test_min']:.4f}   naive {out['naive'][h]['test_mean']:.4f}") for h,v in out["table5"].items()]
print("TABLE 6 (KAN grid):"); [print(f"  {c}: train {v['train_mean']:.4f}+-{v['train_std']:.4f} test {v['test_mean']:.4f}+-{v['test_std']:.4f}") for c,v in out["table6"].items()]
print("TABLE 7:")
for h in H:
    for r in REG+["All"]:
        t=out["table7"][h]; print(f"  h={h:3d} {r:9s} n={t['LSTM'][r]['n']:3d}  LSTM {t['LSTM'][r]['mean']:.3f}+-{t['LSTM'][r]['std']:.3f}  KAN {t['KAN'][r]['mean']:.3f}+-{t['KAN'][r]['std']:.3f}  Naive {t['Naive'][r]['mean']:.3f}")
print("KAN excl fold0:",{h:"%.3f+-%.3f"%(v["mean"],v["std"]) for h,v in out["kan_excl_fold0"].items()})
print("fold0 ranges:",out["kan_fold0_range"]); print("params:",out["params"]); print("fig4 rmse:",out["fig4"]["rmse"], "n points",len(dates))
