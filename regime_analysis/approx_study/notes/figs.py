import json,glob,os,numpy as np,sys
sys.path.insert(0,'/sessions/adoring-bold-cannon/mnt/stock-pred-cos30018/regime_analysis/approx_study')
import approx_study as A
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
t,y=A.load('CBA.AX')
# Fig 1: Fourier features of t
rng=np.random.default_rng(0); B=rng.normal(0,6,4)
fig,ax=plt.subplots(2,1,figsize=(7,4.2),sharex=True)
ax[0].plot(t,t,'k'); ax[0].set_ylabel('input t'); ax[0].set_title('What the plain MLP sees: one number, t')
for b in B: ax[1].plot(t,np.sin(2*np.pi*b*t),lw=0.9,label='sin(2π·%.1f·t)'%b)
ax[1].set_ylabel('features'); ax[1].set_xlabel('t (0 = 2 Jan 2020, 1 = 31 Jul 2023)'); ax[1].legend(fontsize=6,ncol=2,frameon=False)
ax[1].set_title('What the Fourier MLP sees: sines (and cosines) of t at random frequencies b_j ~ N(0, 6²)')
fig.tight_layout(); fig.savefig('fig_features.png',dpi=180); plt.close(fig)
# Fig 2: fits at budget 60 CBA, best seed each
FITS='/sessions/adoring-bold-cannon/mnt/stock-pred-cos30018/regime_analysis/approx_study/fits'
def best(m,b):
    rs=[json.load(open(p)) for p in glob.glob(os.path.join(FITS,'CBA.AX__%s__b%d__s*.json'%(m,b)))]
    return sorted(rs,key=lambda r:r['rmse'])[0]
fig,axs=plt.subplots(3,1,figsize=(7,6.5),sharex=True)
for ax,(m,name) in zip(axs,[('mlp','MLP of t, w=6'),('fmlp','Fourier MLP, m=4, w=4'),('kan11','KAN [1,1], G=51')]):
    r=best(m,60); ax.plot(t,y,color='#999',lw=0.8,label='CBA close (scaled)'); ax.plot(t,r['pred'],lw=1.2,label='%s: N=%d, RMSE %.4f'%(name,r['params'],r['rmse']))
    ax.legend(fontsize=7,frameon=False,loc='upper left'); ax.grid(alpha=.3,lw=.4)
axs[-1].set_xlabel('t'); fig.suptitle('Budget 60 on CBA: the same two-hidden-layer tanh network, fed t (top) or γ(t) (middle); KAN [1,1] for reference',fontsize=9)
fig.tight_layout(); fig.savefig('fig_fits.png',dpi=180); plt.close(fig)
# Fig 3: spectral bias: training curves, w=6 plain vs (4,4) fourier, 20000 steps, seed 0
import torch
def curve(width,fourier,seed=0,epochs=20000,lr=1e-2):
    torch.manual_seed(seed); np.random.seed(seed)
    X=torch.tensor(t[:,None],dtype=torch.float32); Y=torch.tensor(y[:,None],dtype=torch.float32)
    if fourier:
        nf,s=fourier; Bm=torch.tensor(np.random.default_rng(seed).normal(0,s,(1,nf)),dtype=torch.float32)
        F=torch.cat([torch.sin(2*np.pi*X@Bm),torch.cos(2*np.pi*X@Bm)],1); d=2*nf
    else: F=X; d=1
    net=torch.nn.Sequential(torch.nn.Linear(d,width),torch.nn.Tanh(),torch.nn.Linear(width,width),torch.nn.Tanh(),torch.nn.Linear(width,1))
    opt=torch.optim.Adam(net.parameters(),lr=lr); sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,epochs)
    out=[]; snaps={}
    for ep in range(epochs):
        opt.zero_grad(); loss=torch.mean((net(F)-Y)**2); loss.backward(); opt.step(); sch.step()
        out.append(loss.item()**0.5)
        if ep in (200,2000,19999): snaps[ep]=net(F).detach().numpy().ravel()
    return np.array(out),snaps
c1,s1=curve(6,None); c2,s2=curve(4,(4,6.0))
fig,ax=plt.subplots(1,2,figsize=(8,3.2))
ax[0].plot(c1,label='MLP of t (w=6, N=61)'); ax[0].plot(c2,label='Fourier MLP (m=4, w=4, N=61)'); ax[0].set_xscale('log'); ax[0].set_yscale('log')
ax[0].set_xlabel('Adam step'); ax[0].set_ylabel('training RMSE'); ax[0].legend(fontsize=7,frameon=False); ax[0].grid(alpha=.3,lw=.4,which='both'); ax[0].set_title('Training curves, CBA, seed 0',fontsize=9)
ax[1].plot(t,y,color='#999',lw=.8)
for ep,c in zip((200,2000,19999),('#c6dbef','#6baed6','#08519c')): ax[1].plot(t,s1[ep],color=c,lw=1,label='MLP of t after %d steps'%(ep+1))
ax[1].legend(fontsize=7,frameon=False); ax[1].set_title('Spectral bias: the plain MLP fits the trend first, detail (slowly) later',fontsize=8); ax[1].set_xlabel('t'); ax[1].grid(alpha=.3,lw=.4)
fig.tight_layout(); fig.savefig('fig_bias.png',dpi=180); plt.close(fig)
print('fin',c1[-1],c2[-1])
