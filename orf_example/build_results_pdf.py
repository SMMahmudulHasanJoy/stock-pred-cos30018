from reportlab.lib.pagesizes import A4, landscape
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak, Table, TableStyle, KeepTogether
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import cm
import json, numpy as np
from PIL import Image as PI
D=json.load(open("all_fits.json")); F=D["fits"]; T=np.array(D["T"])
def w(x): return [a[0] if isinstance(a,list) else a for a in x]
ss=getSampleStyleSheet()
H1=ParagraphStyle("H1",parent=ss["Heading1"],fontSize=15,spaceBefore=10,spaceAfter=6,textColor=colors.HexColor("#1f3f5f"))
H2=ParagraphStyle("H2",parent=ss["Heading2"],fontSize=11.5,spaceBefore=8,spaceAfter=4,textColor=colors.HexColor("#2f6f9f"))
B=ParagraphStyle("B",parent=ss["Normal"],fontSize=9.6,leading=13,spaceAfter=6)
CAP=ParagraphStyle("Cap",parent=B,fontSize=8.4,leading=11,textColor=colors.HexColor("#555555"),spaceAfter=8)
MONO=ParagraphStyle("M",parent=B,fontName="Courier",fontSize=8)
def P(t,s=B): return Paragraph(t,s)
def fig(name,wcm,cap=None):
    iw,ih=PI.open(name).size; out=[Image(name,width=wcm*cm,height=wcm*ih/iw*cm)]
    if cap: out.append(P(cap,CAP))
    return KeepTogether(out)
def tbl(rows,colw=None,fs=8.5):
    t=Table(rows,colWidths=colw,hAlign="LEFT",repeatRows=1)
    t.setStyle(TableStyle([("FONT",(0,0),(-1,0),"Helvetica-Bold",fs),("FONT",(0,1),(-1,-1),"Helvetica",fs),("BACKGROUND",(0,0),(-1,0),colors.HexColor("#dde8f0")),
        ("GRID",(0,0),(-1,-1),0.3,colors.HexColor("#999999")),("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,colors.HexColor("#f6f8fa")]),("TOPPADDING",(0,0),(-1,-1),2),("BOTTOMPADDING",(0,0),(-1,-1),2)]))
    return t
def e(x): return "0 (exact)" if x<1e-4 else f"{x:.4f}"
S=[]
S+=[P("KAN approximation of S1 and S2: complete results",ParagraphStyle("T",parent=ss["Title"],fontSize=19,spaceAfter=4)),
    P("Experiments A (single edge, linear splines), B (single edge, cubic splines) and C (one hidden layer). Every fitted configuration is shown with its full curve, its error at every point, and its summary numbers. Data and generator from Nadia's ExampleForORF.m.",ParagraphStyle("st",parent=B,textColor=colors.HexColor("#555555"),spaceAfter=10)),
    P("Setup",H2),
    tbl([["Item","Value"],["Targets","S1: knots [0,2,4,6,8];  S2: knots [0,3,5,6,7];  slopes [2,-1,4,-3,-1] for both;  S(t) = sum_j A_j * max(0, t - theta_j)"],
         ["Sample","1001 points, t = 0, 0.01, ..., 10 (files S1.txt, S2.txt; regenerated from the script, match to 4e-15)"],
         ["KAN library","PyKAN 0.2.8 (PyTorch 2.2.2), grid_range [0, 10], grid updating off, seed 0"],
         ["Training","L-BFGS, 200 steps, full batch, target scaled to [0,1] for optimisation and rescaled after"],
         ["Errors","computed on all 1001 points in original units; worst error = max |KAN - S|; RMSE = root mean square; mean abs = mean |KAN - S|"],
         ["Reproduce","python dump_fits.py  (writes all_fits.json);  python make_result_figs.py;  python build_results_pdf.py"]],[2.6*cm,15*cm],fs=8),
    Spacer(1,8), fig("results_figs/data.png",17,"The two target functions. Dotted lines mark the knots.")]

# master table
S+=[PageBreak(),P("1. All results in one table",H1),
    P("Sorted by experiment, then grid. 'Worst at t' is where the largest error occurs; for S2 on coarse grids it sits on a knot the grid cannot reach.")]
rows=[["Exp","KAN width","Grid","Spline","Params","Fn","Worst error","RMSE","Mean abs","Worst at t"]]
for f in F:
    rows.append([f["exp"] if f["exp"]=="C" or f["k"]==1 else "B", str(w(f["width"])),str(f["grid"]),"linear k=1" if f["k"]==1 else "cubic k=3",str(f["params"]),f["fn"],e(f["max_err"]),e(f["rmse"]),e(f["mean_abs"]),f"{f['argmax_t']:.2f}"])
S.append(tbl(rows,[1*cm,2*cm,1.1*cm,2*cm,1.4*cm,0.9*cm,2.2*cm,2*cm,2*cm,1.8*cm],fs=7.8))
S+=[Spacer(1,10),P("Summary charts",H2),fig("results_figs/summary_AB.png",17,"Experiments A and B: worst error (left) and RMSE (right) against grid size, log scales. Solid = linear splines, dashed = cubic."),
    fig("results_figs/summary_C.png",13,"Experiment C: worst error by configuration. Dotted line: the single-edge grid-5 result on S2, for comparison.")]

# side-by-side S1/S2 comparison table per experiment
S+=[PageBreak(),P("2. S1 versus S2, side by side",H1)]
for exp,title,sel in (("A","Experiment A: single edge [1,1], linear splines (k=1)",lambda f:f["exp"]=="A" and f["k"]==1),
                      ("B","Experiment B: single edge [1,1], cubic splines (k=3)",lambda f:f["exp"]=="A" and f["k"]==3),
                      ("C","Experiment C: one hidden layer, linear splines (k=1)",lambda f:f["exp"]=="C")):
    S.append(P(title,H2))
    rws=[["KAN width","Grid","Params","S1 worst","S1 RMSE","S2 worst","S2 RMSE","S2 / S1 worst"]]
    seen=[]
    for f in F:
        if not sel(f) or f["fn"]!="S1": continue
        g=next(x for x in F if sel(x) and x["fn"]=="S2" and w(x["width"])==w(f["width"]) and x["grid"]==f["grid"])
        ratio="n/a" if f["max_err"]<1e-4 else f"{g['max_err']/f['max_err']:.1f}x"
        rws.append([str(w(f["width"])),str(f["grid"]),str(f["params"]),e(f["max_err"]),e(f["rmse"]),e(g["max_err"]),e(g["rmse"]),ratio])
    S.append(tbl(rws,[2*cm,1.2*cm,1.5*cm,2.2*cm,2*cm,2.2*cm,2*cm,2.6*cm])); S.append(Spacer(1,6))

# per-configuration pages
S+=[PageBreak(),P("3. Every fit, one configuration per figure",H1),
    P("Top row: target (black) and KAN fit (orange dashed), with the target's knots (dotted grey) and the KAN's grid lines (faint green). Bottom row: the error KAN - target at every point. Every error panel uses the same vertical scale, -1.6 to +1.6, so figures can be compared directly: a flat line at zero is a perfect fit, a wave is a real error. Where the error is too small to show at this scale, the panel says so and gives the number.")]
order=[f for f in F if f["fn"]=="S1"]
for f in order:
    exp="B" if (f["exp"]=="A" and f["k"]==3) else f["exp"]
    name=f"results_figs/fit_{f['exp']}_w{'-'.join(map(str,w(f['width'])))}_g{f['grid']}_k{f['k']}.png"
    g=next(x for x in F if x["fn"]=="S2" and w(x["width"])==w(f["width"]) and x["grid"]==f["grid"] and x["k"]==f["k"] and x["exp"]==f["exp"])
    S.append(fig(name,17,f"Experiment {exp}. Width {w(f['width'])}, grid {f['grid']}, k={f['k']}, {f['params']} parameters. S1: worst {e(f['max_err'])}, RMSE {e(f['rmse'])}, worst at t={f['argmax_t']:.2f}. S2: worst {e(g['max_err'])}, RMSE {e(g['rmse'])}, worst at t={g['argmax_t']:.2f}."))
doc=SimpleDocTemplate("KAN_S1_S2_results.pdf",pagesize=A4,leftMargin=1.8*cm,rightMargin=1.8*cm,topMargin=1.6*cm,bottomMargin=1.6*cm,title="KAN approximation of S1 and S2: complete results",author="S. M. Mahmudul Hasan Joy")
def footer(c,d): c.setFont("Helvetica",8); c.setFillColor(colors.grey); c.drawRightString(A4[0]-1.8*cm,1*cm,f"Page {d.page}"); c.drawString(1.8*cm,1*cm,"KAN S1/S2 results (experiments A-C)")
doc.build(S,onFirstPage=footer,onLaterPages=footer)
from pypdf import PdfReader; print("pages:",len(PdfReader("KAN_S1_S2_results.pdf").pages))
