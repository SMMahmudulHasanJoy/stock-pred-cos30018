import json, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.size":9,"axes.spines.top":False,"axes.spines.right":False})
D=json.load(open("all_fits.json")); T=np.array(D["T"]); S={"S1":np.array(D["S1"]),"S2":np.array(D["S2"])}
KN={"S1":[0,2,4,6,8],"S2":[0,3,5,6,7]}; BLUE,ORANGE,GREY="#2f6f9f","#c46a3f","#999999"
F=D["fits"]
def get(exp,width,grid,k,fn): return next(f for f in F if f["exp"]==exp and w(f["width"])==w(width) and f["grid"]==grid and f["k"]==k and f["fn"]==fn)
def w(x): return [a[0] if isinstance(a,list) else a for a in x]
# one page-figure per configuration: fit + error, both functions
for exp,width,grid,k in [(f["exp"],f["width"],f["grid"],f["k"]) for f in F if f["fn"]=="S1"]:
    fig,ax=plt.subplots(2,2,figsize=(11,5.2),gridspec_kw={"height_ratios":[2,1]})
    for j,fn in enumerate(("S1","S2")):
        r=get(exp,width,grid,k,fn); p=np.array(r["pred"]); e=p-S[fn]
        ax[0,j].plot(T,S[fn],color="k",lw=2,label=fn+" (target)"); ax[0,j].plot(T,p,color=ORANGE,lw=1.2,ls="--",label="KAN fit")
        for kn in KN[fn]: ax[0,j].axvline(kn,color=GREY,lw=.6,ls=":")
        for gl in np.linspace(0,10,grid+1): ax[0,j].axvline(gl,color="#3a8f5c",lw=.5,alpha=.35)
        ax[0,j].set_title(f"{fn}:  max err {r['max_err']:.4f}   rmse {r['rmse']:.4f}   ({r['params']} params)"); ax[0,j].legend(fontsize=7,loc="upper left")
        ax[1,j].plot(T,e,color=BLUE,lw=1); ax[1,j].axhline(0,color="k",lw=.5); ax[1,j].fill_between(T,e,0,color=BLUE,alpha=.15)
        for kn in KN[fn]: ax[1,j].axvline(kn,color=GREY,lw=.6,ls=":")
        ax[1,j].set_ylabel("KAN - target"); ax[1,j].set_xlabel("t")
        ax[1,j].set_ylim(-1.6,1.6)          # same scale on every error panel
        if r["max_err"]<1e-4:
            ax[1,j].text(5,1.1,"exact fit: error below 0.0001 everywhere\n(what remains is floating-point noise, about 1e-6)",ha="center",fontsize=8,color="#3a8f5c",bbox=dict(fc="white",ec="#3a8f5c",lw=.6))
        elif r["max_err"]<0.1:
            ax[1,j].text(5,1.1,f"small error: worst {r['max_err']:.4f} at t={r['argmax_t']:.2f}\n(too small to see at this scale)",ha="center",fontsize=8,color="#555555",bbox=dict(fc="white",ec="#999999",lw=.6))
    kind="straight-line pieces (k=1)" if k==1 else "smooth cubic pieces (k=3)"
    fig.suptitle(f"KAN width {w(width)},  grid = {grid} intervals,  {kind}     (dotted grey = target's knots; faint green = KAN's grid lines)",fontsize=10)
    fig.tight_layout(); fig.savefig(f"results_figs/fit_{exp}_w{'-'.join(map(str,w(width)))}_g{grid}_k{k}.png",dpi=140); plt.close(fig)
# summary curves
fig,ax=plt.subplots(1,2,figsize=(11,3.6))
for i,metric in enumerate(("max_err","rmse")):
    for fn,m,c in (("S1","o",BLUE),("S2","s",ORANGE)):
        for k,ls in ((1,"-"),(3,"--")):
            rows=sorted([f for f in F if f["exp"]=="A" and f["fn"]==fn and f["k"]==k],key=lambda f:f["grid"])
            ax[i].plot([f["grid"] for f in rows],[max(f[metric],1e-5) for f in rows],marker=m,ls=ls,color=c,label=f"{fn}, k={k}")
    ax[i].set_xscale("log"); ax[i].set_yscale("log"); ax[i].set_xticks([5,10,20,50,100]); ax[i].set_xticklabels([5,10,20,50,100]); ax[i].grid(alpha=.3,which="both")
    ax[i].set_xlabel("grid intervals"); ax[i].set_ylabel({"max_err":"worst error","rmse":"RMSE"}[metric]); ax[i].set_title({"max_err":"Experiment A/B: worst error vs grid","rmse":"Experiment A/B: RMSE vs grid"}[metric]); ax[i].legend(fontsize=7)
fig.tight_layout(); fig.savefig("results_figs/summary_AB.png",dpi=150); plt.close(fig)
# experiment C bar chart
fig,ax=plt.subplots(figsize=(8,3.4)); rows=[f for f in F if f["exp"]=="C"]; labels=[f"{w(f['width'])}\ngrid {f['grid']}" for f in rows if f["fn"]=="S1"]
x=np.arange(len(labels)); s1=[f["max_err"] for f in rows if f["fn"]=="S1"]; s2=[f["max_err"] for f in rows if f["fn"]=="S2"]
ax.bar(x-0.18,s1,0.36,color=BLUE,label="S1"); ax.bar(x+0.18,s2,0.36,color=ORANGE,label="S2")
ax.axhline(get("A",[1,1],5,1,"S2")["max_err"],color=ORANGE,ls=":",lw=1,label="S2 with single edge, grid 5 (1.48)")
ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_ylabel("worst error"); ax.set_title("Experiment C: one hidden layer, straight-line pieces"); ax.legend(fontsize=7)
for i,(a,b) in enumerate(zip(s1,s2)): ax.text(i-0.18,a+0.02,f"{a:.3f}",ha="center",fontsize=7); ax.text(i+0.18,b+0.02,f"{b:.3f}",ha="center",fontsize=7)
fig.tight_layout(); fig.savefig("results_figs/summary_C.png",dpi=150); plt.close(fig)
# the input data itself
fig,ax=plt.subplots(1,2,figsize=(11,3.2))
for a_,fn in zip(ax,("S1","S2")):
    a_.plot(T,S[fn],color=BLUE,lw=1.8); [a_.axvline(kn,color=GREY,ls=":",lw=.8) for kn in KN[fn]]
    a_.set_title(f"{fn}: knots at {KN[fn]},  1001 points on t = 0 : 0.01 : 10,  range 0 to {S[fn].max():.0f}"); a_.set_xlabel("t")
fig.tight_layout(); fig.savefig("results_figs/data.png",dpi=150); plt.close(fig)
print(len([f for f in F if f["fn"]=="S1"]),"configs plotted")
