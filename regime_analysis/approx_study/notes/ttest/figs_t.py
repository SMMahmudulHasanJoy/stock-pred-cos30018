import csv,json,numpy as np
from scipy import stats
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
R='/sessions/adoring-bold-cannon/mnt/results/'
S=list(csv.DictReader(open(R+'differences_summary_KAN_vs_MLP.csv')))
P=list(csv.DictReader(open(R+'differences_KAN_vs_MLP.csv')))
T=list(csv.DictReader(open(R+'differences_ttests_KAN_vs_MLP.csv')))
COMP=["KAN [1,1] - MLP of t","KAN [1,1] - Fourier MLP","KAN [1,h,1] - KAN [1,1]","Fourier MLP - MLP of t"]
TK=["CBA.AX","TSLA","LLY"]; BUD=[15,30,60,120,240]; f=float
out={}
def cells(comp,col='d_rmse_mean'):
    return [(d['ticker'],int(d['budget']),f(d[col]),f(d['rmse_A']),f(d['rmse_B'])) for d in S if d['comparison']==comp]
# ---- worked example numbers
c=cells(COMP[0]); d=np.array([x[2] for x in c]); n=len(d); m=d.mean(); s=d.std(ddof=1); se=s/np.sqrt(n); t=m/se; p=2*stats.t.sf(abs(t),n-1)
out['ex']=dict(d=d.tolist(),n=n,mean=m,sd=s,se=se,t=t,p=p,ci=stats.t.ppf(.975,n-1)*se,dcohen=m/s)
tt=[r for r in T if r['comparison']==COMP[0] and r['metric']=='RMSE' and r['scope']=='all 15 cells'][0]
print('check vs csv: t %.4f vs %.4f ; p %.3g vs %.3g'%(t,f(tt['t_statistic']),p,f(tt['p_value'])))
# variance reduction
A=np.array([x[3] for x in c]); B=np.array([x[4] for x in c])
out['varA']=A.var(ddof=1); out['varB']=B.var(ddof=1); out['cov']=np.cov(A,B)[0,1]; out['varD']=d.var(ddof=1); out['corr']=np.corrcoef(A,B)[0,1]
# Welch unpaired for contrast
tw,pw=stats.ttest_ind(A,B,equal_var=False); out['welch']=(tw,pw)
# all comparisons, RMSE and max err: t, p, wilcoxon, sign test, cohen d, bonferroni
rows=[]
for comp in COMP:
    for col,lab in [('d_rmse_mean','RMSE'),('d_maxerr_mean','max|err|'),('d_sec_mean','seconds')]:
        dd=np.array([x[2] for x in cells(comp,col)]); n=len(dd); m=dd.mean(); s=dd.std(ddof=1); t=m/(s/np.sqrt(n)); p=2*stats.t.sf(abs(t),n-1)
        try: wp=stats.wilcoxon(dd).pvalue
        except Exception: wp=float('nan')
        k=int((dd<0).sum()); sp=stats.binomtest(k,n,0.5).pvalue
        rows.append(dict(comp=comp,metric=lab,n=n,mean=m,sd=s,t=t,p=p,wil=wp,neg=k,sign=sp,cohen=m/s,ci=stats.t.ppf(.975,n-1)*s/np.sqrt(n)))
out['rows']=rows
# per-seed pseudo-replication: 45 differences
ps=np.array([f(r['d_rmse']) for r in P if r['comparison']==COMP[0]]); tp=stats.ttest_1samp(ps,0); out['pseudo']=(len(ps),ps.mean(),tp.statistic,tp.pvalue)
json.dump(out,open('numbers.json','w'),indent=1,default=float)
# ===== figures
# Fig 1: design grid: A and B RMSE per cell, paired
fig,ax=plt.subplots(1,2,figsize=(10,3.4))
cols={'CBA.AX':'#2f6f9f','TSLA':'#c46a3f','LLY':'#4a9a4a'}
for tk in TK:
    xs=[x[1] for x in c if x[0]==tk]; a=[x[3] for x in c if x[0]==tk]; b=[x[4] for x in c if x[0]==tk]
    ax[0].plot(xs,a,'o-',color=cols[tk],label='%s: KAN [1,1]'%tk); ax[0].plot(xs,b,'s--',color=cols[tk],label='%s: MLP of t'%tk)
    for x_,a_,b_ in zip(xs,a,b): ax[0].plot([x_,x_],[a_,b_],color=cols[tk],alpha=.35,lw=3)
ax[0].set_xscale('log'); ax[0].set_xlabel('budget N*'); ax[0].set_ylabel('RMSE (mean of 3 seeds)'); ax[0].legend(fontsize=6,frameon=False,ncol=2); ax[0].set_title('The 15 cells: each thick bar is one paired difference d_i',fontsize=8); ax[0].grid(alpha=.3,lw=.4)
ax[1].scatter(B,A,c=[cols[x[0]] for x in c],s=28); lim=[0,0.1]; ax[1].plot(lim,lim,'k--',lw=.8,label='A = B'); 
ax[1].set_xlabel('RMSE of B (MLP of t)'); ax[1].set_ylabel('RMSE of A (KAN [1,1])'); ax[1].set_title('Same 15 cells as a scatter; below the line = KAN better. r = %.2f'%out['corr'],fontsize=8); ax[1].legend(fontsize=7,frameon=False); ax[1].grid(alpha=.3,lw=.4)
fig.tight_layout(); fig.savefig('fig_cells.png',dpi=180); plt.close(fig)
# Fig 2: the 15 differences and the t distribution
fig,ax=plt.subplots(1,2,figsize=(10,3.3))
lab=['%s %d'%(x[0].replace('.AX',''),x[1]) for x in c]
ax[0].bar(range(n),d,color=[cols[x[0]] for x in c]); ax[0].axhline(0,color='k',lw=.8); ax[0].axhline(m,color='r',ls='--',lw=1,label='mean = %.4f'%m)
ax[0].set_xticks(range(n)); ax[0].set_xticklabels(lab,rotation=90,fontsize=6); ax[0].set_ylabel('d_i = RMSE_A − RMSE_B'); ax[0].legend(fontsize=7,frameon=False); ax[0].set_title('The 15 paired differences (KAN [1,1] − MLP of t)',fontsize=8)
x=np.linspace(-8,8,800); ax[1].plot(x,stats.t.pdf(x,n-1),'k',label='t density, %d d.f.'%(n-1)); ax[1].fill_between(x,0,stats.t.pdf(x,n-1),where=np.abs(x)>=abs(t),color='r',alpha=.4,label='two-sided p = %.2g'%p)
ax[1].axvline(t,color='r',lw=1); ax[1].text(t+0.2,0.2,'t = %.2f'%t,fontsize=8,color='r'); ax[1].set_xlabel('t'); ax[1].legend(fontsize=7,frameon=False); ax[1].set_title('Where the observed t falls under H0',fontsize=8)
fig.tight_layout(); fig.savefig('fig_t.png',dpi=180); plt.close(fig)
# Fig 3: forest plot of all comparisons, RMSE and max err
fig,axs=plt.subplots(1,2,figsize=(10,3.4))
for ax,met in zip(axs,['RMSE','max|err|']):
    rr=[r for r in rows if r['metric']==met]
    for i,r in enumerate(rr):
        ax.errorbar(r['mean'],i,xerr=r['ci'],fmt='D',color='k',capsize=4)
        dd=np.array([x[2] for x in cells(r['comp'],'d_rmse_mean' if met=='RMSE' else 'd_maxerr_mean')]); ax.scatter(dd,[i+0.25]*len(dd),s=12,color='#888',alpha=.7)
        ax.text(ax.get_xlim()[1] if False else 0, i-0.33,'t=%.2f, p=%.2g'%(r['t'],r['p']),fontsize=6.5,ha='left',va='center')
    ax.axvline(0,color='k',ls=':'); ax.set_yticks(range(len(rr))); ax.set_yticklabels([r['comp'] for r in rr],fontsize=7); ax.set_xlabel('difference A − B in %s (mean, 95%% CI; dots = 15 cells)'%met,fontsize=8); ax.grid(alpha=.3,lw=.4,axis='x'); ax.invert_yaxis()
fig.tight_layout(); fig.savefig('fig_forest.png',dpi=180); plt.close(fig)
# Fig 4: paired vs unpaired variance picture
fig,ax=plt.subplots(figsize=(6,3.2))
ax.boxplot([A,B,d],labels=['RMSE_A\n(KAN [1,1])','RMSE_B\n(MLP of t)','d = A − B'],widths=.5)
for i,v in enumerate([A,B,d],1): ax.scatter(np.random.default_rng(1).normal(i,0.04,len(v)),v,s=12,color='#2f6f9f',alpha=.7)
ax.axhline(0,color='k',ls=':',lw=.8); ax.set_title('Pairing removes the cell-to-cell spread: sd(A)=%.3f, sd(B)=%.3f, sd(d)=%.3f'%(A.std(ddof=1),B.std(ddof=1),s),fontsize=8); ax.grid(alpha=.3,lw=.4,axis='y')
fig.tight_layout(); fig.savefig('fig_pairing.png',dpi=180); plt.close(fig)
# Fig 5: dependence structure: differences vs budget per ticker (nested), duplicates
fig,ax=plt.subplots(1,2,figsize=(10,3.2))
for comp,a in zip([COMP[0],COMP[2]],ax):
    cc=cells(comp)
    for tk in TK:
        a.plot([x[1] for x in cc if x[0]==tk],[x[2] for x in cc if x[0]==tk],'o-',color=cols[tk],label=tk)
    a.axhline(0,color='k',ls=':'); a.set_xscale('log'); a.set_xlabel('budget N*'); a.set_ylabel('d_i'); a.set_title(comp+': differences are smooth in N (not independent draws)',fontsize=7.5); a.legend(fontsize=7,frameon=False); a.grid(alpha=.3,lw=.4)
fig.tight_layout(); fig.savefig('fig_depend.png',dpi=180); plt.close(fig)
# Fig 6: sampling distribution via permutation (sign flips) for the worked example
rng=np.random.default_rng(0); flips=np.array([ (d*rng.choice([-1,1],n)).mean() for _ in range(20000)])
pperm=(np.abs(flips)>=abs(m)).mean()
fig,ax=plt.subplots(figsize=(6,3.1)); ax.hist(flips,bins=80,color='#bbb'); ax.axvline(m,color='r',lw=1.2,label='observed mean %.4f'%m); ax.axvline(-m,color='r',lw=1.2,ls='--')
ax.set_title('Sign-flip permutation: 20,000 random re-signings of the 15 d_i; p ≈ %.2g'%max(pperm,1/20000),fontsize=8); ax.set_xlabel('mean of re-signed differences'); ax.legend(fontsize=7,frameon=False)
fig.tight_layout(); fig.savefig('fig_perm.png',dpi=180); plt.close(fig)
out['perm_p']=pperm; json.dump(out,open('numbers.json','w'),indent=1,default=float)
print(json.dumps({k:v for k,v in out.items() if k!='rows'},indent=1,default=float))
for r in rows: print('%-26s %-8s mean %.4f sd %.4f t %.2f p %.2g wil %.2g neg %d/15 sign %.2g d %.2f'%(r['comp'],r['metric'],r['mean'],r['sd'],r['t'],r['p'],r['wil'],r['neg'],r['sign'],r['cohen']))
