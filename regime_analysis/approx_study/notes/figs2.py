import json,glob,os,numpy as np,sys
sys.path.insert(0,'/sessions/adoring-bold-cannon/mnt/stock-pred-cos30018/regime_analysis/approx_study')
import approx_study as A
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
import torch
t,y=A.load('CBA.AX'); T=len(t)
FITS='/sessions/adoring-bold-cannon/mnt/stock-pred-cos30018/regime_analysis/approx_study/fits'
SUM=json.load(open('/sessions/adoring-bold-cannon/mnt/stock-pred-cos30018/regime_analysis/approx_study/approx_summary.json'))
def best(tk,m,b):
    rs=[json.load(open(p)) for p in glob.glob(os.path.join(FITS,'%s__%s__b%d__s*.json'%(tk,m,b)))]
    return sorted(rs,key=lambda r:r['rmse'])[0]
# ---- train both nets (seed 0) keeping snapshots and hidden activations
def train(width,fourier,seed=0,epochs=20000,lr=1e-2,snap=(200,2000,19999)):
    torch.manual_seed(seed); np.random.seed(seed)
    X=torch.tensor(t[:,None],dtype=torch.float32); Y=torch.tensor(y[:,None],dtype=torch.float32)
    if fourier:
        nf,s=fourier; Bm=torch.tensor(np.random.default_rng(seed).normal(0,s,(1,nf)),dtype=torch.float32)
        F=torch.cat([torch.sin(2*np.pi*X@Bm),torch.cos(2*np.pi*X@Bm)],1); d=2*nf
    else: F=X; d=1; Bm=None
    net=torch.nn.Sequential(torch.nn.Linear(d,width),torch.nn.Tanh(),torch.nn.Linear(width,width),torch.nn.Tanh(),torch.nn.Linear(width,1))
    opt=torch.optim.Adam(net.parameters(),lr=lr); sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,epochs)
    curve=[]; snaps={}
    for ep in range(epochs):
        opt.zero_grad(); loss=torch.mean((net(F)-Y)**2); loss.backward(); opt.step(); sch.step(); curve.append(loss.item()**0.5)
        if ep in snap: snaps[ep]=net(F).detach().numpy().ravel()
    with torch.no_grad():
        a1=torch.tanh(net[0](F)).numpy(); pred=net(F).numpy().ravel()
    return np.array(curve),snaps,a1,pred,(Bm.numpy().ravel() if Bm is not None else None),F.numpy()
c1,s1,a1,p1,_,_=train(6,None); c2,s2,a2,p2,B,F2=train(4,(4,6.0))
print('final',c1[-1],c2[-1])
# ---- Fig A: hidden-unit curves (the basis each network builds)
fig,ax=plt.subplots(1,3,figsize=(10,3.1))
for j in range(6): ax[0].plot(t,a1[:,j],lw=1)
ax[0].set_title('Plain MLP, w=6: layer-1 units tanh(w_i1 t + b_i1)',fontsize=8); ax[0].set_xlabel('t')
for j in range(F2.shape[1]): ax[1].plot(t,F2[:,j],lw=.8)
ax[1].set_title('Fourier MLP: the 2m=8 input features γ(t)',fontsize=8); ax[1].set_xlabel('t')
for j in range(4): ax[2].plot(t,a2[:,j],lw=1)
ax[2].set_title('Fourier MLP, w=4: layer-1 units tanh(W1 γ(t)+b1)',fontsize=8); ax[2].set_xlabel('t')
for a in ax: a.grid(alpha=.3,lw=.4)
fig.tight_layout(); fig.savefig('fig_units.png',dpi=180); plt.close(fig)
# ---- Fig B: snapshots during training, both nets
fig,ax=plt.subplots(1,2,figsize=(10,3.3),sharey=True)
for a,s,name in zip(ax,(s1,s2),('MLP of t (w=6)','Fourier MLP (m=4, w=4)')):
    a.plot(t,y,color='#aaa',lw=.8,label='target')
    for ep,c in zip((200,2000,19999),('#c6dbef','#6baed6','#08519c')): a.plot(t,s[ep],color=c,lw=1,label='after %d steps'%(ep+1))
    a.set_title(name,fontsize=9); a.legend(fontsize=7,frameon=False); a.grid(alpha=.3,lw=.4); a.set_xlabel('t')
fig.tight_layout(); fig.savefig('fig_snaps.png',dpi=180); plt.close(fig)
# ---- Fig C: training curves
fig,ax=plt.subplots(figsize=(5,3.1))
ax.plot(c1,label='MLP of t (w=6, N=61)'); ax.plot(c2,label='Fourier MLP (m=4, w=4, N=61)'); ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel('Adam step'); ax.set_ylabel('training RMSE'); ax.legend(fontsize=7,frameon=False); ax.grid(alpha=.3,lw=.4,which='both')
fig.tight_layout(); fig.savefig('fig_curves.png',dpi=180); plt.close(fig)
# ---- Fig D: residual spectra at budget 60 (best seed) for mlp, fmlp, kan11
fig,ax=plt.subplots(figsize=(7,3.2))
freqs=np.fft.rfftfreq(T,d=1.0/(T-1))  # cycles per unit t
for m,name,c in [('mlp','MLP of t','#2f6f9f'),('fmlp','Fourier MLP','#5fb0e0'),('kan11','KAN [1,1]','#c46a3f')]:
    r=best('CBA.AX',m,60); res=y-np.array(r['pred']); P=np.abs(np.fft.rfft(res))/T
    ax.plot(freqs[1:200],P[1:200],lw=.9,label='%s residual, RMSE %.4f'%(name,r['rmse']),color=c)
Py=np.abs(np.fft.rfft(y-y.mean()))/T; ax.plot(freqs[1:200],Py[1:200],color='#999',lw=.8,label='target (mean removed)')
ax.set_yscale('log'); ax.set_xlabel('frequency (cycles over the whole series)'); ax.set_ylabel('amplitude'); ax.legend(fontsize=7,frameon=False); ax.grid(alpha=.3,lw=.4,which='both')
ax.set_title('Which frequencies are left unfitted at budget 60 (CBA)',fontsize=9)
fig.tight_layout(); fig.savefig('fig_spectrum.png',dpi=180); plt.close(fig)
# ---- Fig E: random Fourier features approximate a Gaussian kernel
fig,ax=plt.subplots(figsize=(6,3.1)); d=np.linspace(0,0.3,400); sig=6.0
ax.plot(d,np.exp(-2*np.pi**2*sig**2*d**2),'k',lw=1.5,label='exp(-2π²σ²Δ²), σ=6')
for m,c in [(2,'#fdae6b'),(8,'#e6550d'),(200,'#a63603')]:
    b=np.random.default_rng(0).normal(0,sig,m); ax.plot(d,np.mean(np.cos(2*np.pi*np.outer(d,b)),1),lw=1,color=c,label='(1/m) Σ cos(2π b_j Δ), m=%d'%m)
ax.axvline(1/(2*np.pi*sig),color='#777',ls='--',lw=.8); ax.text(1/(2*np.pi*sig)+0.004,0.8,'ℓ = 1/(2πσ) ≈ 0.0265\n≈ 24 trading days',fontsize=7)
ax.set_xlabel('Δ = t − t′ (fraction of the series)'); ax.set_ylabel('γ(t)·γ(t′) / m'); ax.legend(fontsize=7,frameon=False); ax.grid(alpha=.3,lw=.4)
fig.tight_layout(); fig.savefig('fig_kernel.png',dpi=180); plt.close(fig)
# ---- Fig F: RMSE vs N log-log with fitted slopes, 3 tickers
MOD=[('kan11','KAN [1,1]','#c46a3f'),('kan1h1','KAN [1,h,1]','#e0a070'),('mlp','MLP of t','#2f6f9f'),('fmlp','Fourier MLP','#5fb0e0')]
fig,axs=plt.subplots(1,3,figsize=(11,3.4),sharey=True); slopes={}
for ax,tk in zip(axs,['CBA.AX','TSLA','LLY']):
    for m,name,c in MOD:
        xs=np.array([SUM['%s|%s|%d'%(tk,m,b)]['params'] for b in A.BUDGETS],float); ys=np.array([SUM['%s|%s|%d'%(tk,m,b)]['rmse_mean'] for b in A.BUDGETS])
        sl,ic=np.polyfit(np.log(xs),np.log(ys),1); slopes.setdefault(m,[]).append(sl)
        ax.plot(xs,ys,'o-',ms=3,lw=1,color=c,label='%s (slope %.2f)'%(name,sl))
        ax.plot(xs,np.exp(ic)*xs**sl,':',lw=.8,color=c)
    xx=np.array([15,240]); ax.plot(xx,0.3*xx**-0.5,'k--',lw=.8,label='∝ N^(−1/2)')
    ax.set_xscale('log'); ax.set_yscale('log'); ax.set_title(tk); ax.set_xlabel('N (parameters)'); ax.grid(alpha=.3,lw=.4,which='both'); ax.legend(fontsize=6,frameon=False)
axs[0].set_ylabel('RMSE (scaled units)')
fig.tight_layout(); fig.savefig('fig_slopes.png',dpi=180); plt.close(fig)
print({m:np.round(np.mean(v),2) for m,v in slopes.items()})
# ---- Fig G: achievable parameter counts vs budget
fig,ax=plt.subplots(figsize=(6,3.1))
w=np.arange(1,16); ax.plot(w**2+4*w+1,w,'o-',ms=3,label='MLP of t: N=w²+4w+1')
for m in (2,4,6,8): ax.plot(w**2+(2*m+3)*w+1,w,'s--',ms=2.5,lw=.8,label='Fourier MLP, m=%d: N=w²+(2m+3)w+1'%m)
for b in A.BUDGETS: ax.axvline(b,color='#bbb',lw=.7)
ax.set_xscale('log'); ax.set_xlabel('N (parameters); vertical lines = budgets'); ax.set_ylabel('width w'); ax.legend(fontsize=6,frameon=False); ax.grid(alpha=.3,lw=.4,which='both')
ax.set_title('Why budgets are only met approximately: N is an integer function of (m, w)',fontsize=8)
fig.tight_layout(); fig.savefig('fig_counts.png',dpi=180); plt.close(fig)
# ---- Fig H: KAN[1,1] vs LSQ spline identity + B-spline basis
from scipy.interpolate import BSpline
fig,ax=plt.subplots(1,2,figsize=(10,3.1))
G=12; kn=np.r_[[0]*3,np.linspace(0,1,G+1),[1]*3]
for i in range(G+3): ax[0].plot(t,BSpline(kn,np.eye(G+3)[i],3)(t),lw=.9)
ax[0].set_title('Cubic B-spline basis on G=12 equal intervals (G+3=15 functions): local bumps',fontsize=8); ax[0].set_xlabel('t')
rk=best('CBA.AX','kan11',60); rl=[json.load(open(p)) for p in glob.glob(os.path.join(FITS,'CBA.AX__lsq__b60__s0.json'))][0]
ax[1].plot(t,y,color='#aaa',lw=.8,label='target'); ax[1].plot(t,rk['pred'],lw=1.1,label='KAN [1,1], G=51, RMSE %.4f'%rk['rmse']); ax[1].plot(t,rl['pred'],'--',lw=1,label='LSQ cubic spline, G=57, RMSE %.4f'%rl['rmse'])
ax[1].set_title('A [1,1] KAN is a least-squares spline (CBA, budget 60)',fontsize=8); ax[1].legend(fontsize=7,frameon=False); ax[1].set_xlabel('t')
for a in ax: a.grid(alpha=.3,lw=.4)
fig.tight_layout(); fig.savefig('fig_spline.png',dpi=180); plt.close(fig)
print('max |KAN11 - LSQ| at matched G would need same G; here G differ (51 vs 57).')
