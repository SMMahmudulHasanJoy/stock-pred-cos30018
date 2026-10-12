import numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False})
BLUE,ORANGE,GREY,GREEN="#2f6f9f","#c46a3f","#888888","#3a8f5c"
T=np.round(np.arange(0,10.0001,0.01),2); A=[2,-1,4,-3,-1]
def S(th): return sum(a*np.maximum(0,T-t) for a,t in zip(A,th))
S1,S2=S([0,2,4,6,8]),S([0,3,5,6,7])
def save(fig,n): fig.tight_layout(); fig.savefig(n,dpi=160); plt.close(fig)

# 1. hinge function: one building block
fig,ax=plt.subplots(1,3,figsize=(10,2.8))
for i,(th,a,ttl) in enumerate([(3,1,"max(0, t - 3): flat, then rises at slope 1"),(3,2,"2 x max(0, t - 3): same hinge, steeper"),(3,-1,"-1 x max(0, t - 3): same hinge, going down")]):
    ax[i].plot(T,a*np.maximum(0,T-th),color=BLUE,lw=2); ax[i].axvline(th,color=GREY,ls=":"); ax[i].annotate("knot\n(hinge point)",(th,0),xytext=(th+0.8,a*3 if a>0 else -1.5),arrowprops=dict(arrowstyle="->",color=GREY),fontsize=8,color=GREY)
    ax[i].set_title(ttl,fontsize=9); ax[i].set_xlabel("t"); ax[i].set_xlim(0,10)
save(fig,"f1_hinge.png")

# 2. building S1 step by step
fig,ax=plt.subplots(1,5,figsize=(13,2.6),sharey=True); th=[0,2,4,6,8]; acc=np.zeros_like(T)
for j in range(5):
    piece=A[j]*np.maximum(0,T-th[j]); acc=acc+piece
    ax[j].plot(T,piece,color=ORANGE,lw=1,ls="--",label="this hinge"); ax[j].plot(T,acc,color=BLUE,lw=2,label="running total")
    ax[j].set_title(f"+ ({A[j]}) x max(0, t - {th[j]})",fontsize=9); ax[j].axvline(th[j],color=GREY,ls=":"); ax[j].set_xlabel("t")
ax[0].legend(fontsize=7); ax[4].set_title("+ (-1) x max(0, t - 8)   = S1",fontsize=9)
save(fig,"f2_build_s1.png")

# 3. S1 vs S2 with knots marked and slopes labelled
fig,ax=plt.subplots(1,2,figsize=(11,3.4))
for a_,(y,th,name) in zip(ax,[(S1,[0,2,4,6,8],"S1  (knots evenly spaced)"),(S2,[0,3,5,6,7],"S2  (knots NOT evenly spaced)")]):
    a_.plot(T,y,color=BLUE,lw=2.2)
    for k in th: a_.axvline(k,color=GREY,ls=":",lw=1); a_.plot(k,np.interp(k,T,y),"o",color=ORANGE,ms=6)
    cum=np.cumsum(A); segs=th+[10]
    for i in range(5):
        m=(segs[i]+segs[i+1])/2; a_.text(m,np.interp(m,T,y)+1.2,f"slope {cum[i]}",ha="center",fontsize=8,color=GREEN)
    a_.set_title(name); a_.set_xlabel("t"); a_.set_ylabel("S(t)"); a_.set_xticks(range(0,11))
save(fig,"f3_s1_s2.png")

# 4. what a grid is: the KAN's fixed knots vs the function's knots
fig,ax=plt.subplots(2,1,figsize=(10,4.2),sharex=True)
for a_,(y,th,name) in zip(ax,[(S1,[0,2,4,6,8],"S1"),(S2,[0,3,5,6,7],"S2")]):
    a_.plot(T,y,color=BLUE,lw=2,label=name)
    for g in range(0,11,2): a_.axvline(g,color=GREEN,lw=1.4,alpha=.7)
    for k in th: a_.plot(k,np.interp(k,T,y),"o",color=ORANGE,ms=8,zorder=5)
    hit=all(k in range(0,11,2) for k in th)
    a_.set_title(f"{name}: orange dots = where the function bends;  green lines = where a 5-interval grid lets the KAN bend   ->  {'every bend is on a grid line' if hit else 'bends at 3, 5 and 7 fall between grid lines'}",fontsize=9)
    a_.set_ylabel("S(t)")
ax[1].set_xlabel("t"); ax[1].set_xticks(range(0,11))
save(fig,"f4_grid.png")

# 5. piecewise linear spline from basis "hats"
fig,ax=plt.subplots(1,2,figsize=(11,3.2))
knots=np.arange(0,11,2)
for k in knots:
    hat=np.clip(1-np.abs(T-k)/2,0,None); ax[0].plot(T,hat,lw=1.2)
ax[0].set_title("Linear spline building blocks on a 5-interval grid:\none 'hat' per grid line; each is zero except near its own line",fontsize=9); ax[0].set_xlabel("t")
w=[0,4,6,16,20,22]; y=sum(wi*np.clip(1-np.abs(T-k)/2,0,None) for wi,k in zip(w,knots))
ax[1].plot(T,y,color=BLUE,lw=2.4,label="hats x heights, added up"); ax[1].plot(T,S1,color=ORANGE,lw=1,ls="--",label="S1")
for k,wi in zip(knots,w): ax[1].plot(k,wi,"o",color=GREEN)
ax[1].set_title("Choose the height of each hat and add them:\nwith heights 0,4,6,16,20,22 you get S1 exactly",fontsize=9); ax[1].legend(fontsize=8); ax[1].set_xlabel("t")
save(fig,"f5_hats.png")

# 6. cubic can't make a kink
fig,ax=plt.subplots(1,2,figsize=(11,3.2))
tt=np.linspace(1,5,400); kink=np.maximum(0,tt-3)
ax[0].plot(tt,kink,color=BLUE,lw=2.4,label="a kink (linear pieces)"); 
for s,c in ((0.6,ORANGE),(0.25,GREEN)):
    ax[0].plot(tt,np.logaddexp(0,(tt-3)/s)*s,color=c,lw=1.4,ls="--",label=f"a smooth curve trying to copy it (sharper -> closer)")
ax[0].legend(fontsize=7); ax[0].set_title("A smooth (cubic) curve can only round a corner off,\nnever reproduce it exactly",fontsize=9)
x=np.linspace(0,10,500); 
for g,c in ((5,ORANGE),(20,GREEN)):
    kn=np.linspace(0,10,g+1); ax[1].plot(x,np.interp(x,kn,np.interp(kn,T,S2)),color=c,lw=1.3,ls="--",label=f"straight lines between {g} grid points")
ax[1].plot(T,S2,color=BLUE,lw=2,label="S2"); ax[1].legend(fontsize=7); ax[1].set_title("Finer grid = more places to bend = closer fit\n(error shrinks as the grid gets finer)",fontsize=9); ax[1].set_xlabel("t")
save(fig,"f6_cubic_vs_fine.png")

# 7. KAN diagram: [1,1] edge and [1,3,1]
fig,ax=plt.subplots(1,2,figsize=(11,3.4)); 
for a_ in ax: a_.axis("off"); a_.set_xlim(0,10); a_.set_ylim(0,6)
def node(a_,x,y,txt): a_.add_patch(plt.Circle((x,y),0.45,fc="white",ec=BLUE,lw=2)); a_.text(x,y,txt,ha="center",va="center",fontsize=9)
def edge(a_,x0,y0,x1,y1,lab): a_.annotate("",(x1-0.45,y1),(x0+0.45,y0),arrowprops=dict(arrowstyle="->",color=ORANGE,lw=2)); a_.text((x0+x1)/2,(y0+y1)/2+0.45,lab,ha="center",fontsize=8,color=ORANGE)
node(ax[0],2,3,"t"); node(ax[0],8,3,"out"); edge(ax[0],2,3,8,3,"one learned curve f(t)")
ax[0].text(5,1.2,"KAN width [1,1]: the whole model is a single curve.\nThe KAN learns the shape of that curve.\nIf the curve can be S1, the fit is perfect.",ha="center",fontsize=9)
node(ax[1],1.5,3,"t"); 
for i,y in enumerate((5,3,1)): node(ax[1],5,y,f"h{i+1}"); edge(ax[1],1.5,3,5,y,"curve"); edge(ax[1],5,y,8.5,3,"curve")
node(ax[1],8.5,3,"out"); ax[1].text(5,-0.2,"KAN width [1,3,1]: curves feed into curves. Bending a bent curve\ncan put a corner where no single grid line is.",ha="center",fontsize=9)
save(fig,"f7_kan_diagrams.png")

# 8. training as hill-descent cartoon
fig,ax=plt.subplots(figsize=(6,3)); p=np.linspace(-3,3,300); loss=(p**2)*0.6+0.3*np.sin(3*p)+1
ax.plot(p,loss,color=BLUE,lw=2); pts=[-2.6,-1.9,-1.2,-0.5,-0.05]
for i,q in enumerate(pts): ax.plot(q,np.interp(q,p,loss),"o",color=ORANGE); ax.text(q,np.interp(q,p,loss)+0.35,f"step {i+1}",ha="center",fontsize=8)
ax.set_xlabel("a model setting (one of many)"); ax.set_ylabel("total error on the 1001 points"); ax.set_title("Training: adjust the settings a little, check if the error fell, repeat",fontsize=9)
save(fig,"f8_training.png")

# 9. the key result (reuse the fitted lines by recomputing quickly with pykan is slow; draw from saved numbers via interpolation proxy)
import json; R=json.load(open("/sessions/adoring-bold-cannon/mnt/stock-pred-cos30018/orf_example/results.json"))
fig,ax=plt.subplots(figsize=(7,3.6))
for name,m,c in (("S1","o",BLUE),("S2","s",ORANGE)):
    for k,ls,lab in ((1,"-","straight-line pieces (k=1)"),(3,"--","smooth cubic pieces (k=3)")):
        rows=sorted([r for r in R if r["exp"].startswith("A") and r["fn"]==name and r["k"]==k],key=lambda r:r["grid"])
        ax.plot([r["grid"] for r in rows],[max(r["max_err"],1e-5) for r in rows],marker=m,ls=ls,color=c,label=f"{name}, {lab}")
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xticks([5,10,20,50,100]); ax.set_xticklabels([5,10,20,50,100])
ax.set_xlabel("number of grid intervals the KAN is allowed"); ax.set_ylabel("worst error anywhere on 0..10"); ax.axhline(1e-4,color=GREY,ls=":",lw=.8); ax.text(5.2,1.3e-4,"effectively exact",fontsize=8,color=GREY)
ax.legend(fontsize=7); ax.set_title("The result: worst error vs grid size",fontsize=10); ax.grid(alpha=.3,which="both")
save(fig,"f9_result.png")
print("figures done")
