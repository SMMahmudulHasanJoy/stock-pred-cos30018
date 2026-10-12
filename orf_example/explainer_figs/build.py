from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak, Table, TableStyle, KeepTogether
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import cm
import json
ORF="/sessions/adoring-bold-cannon/mnt/stock-pred-cos30018/orf_example/"
ss=getSampleStyleSheet()
H1=ParagraphStyle("H1",parent=ss["Heading1"],fontSize=17,spaceBefore=14,spaceAfter=8,textColor=colors.HexColor("#1f3f5f"))
H2=ParagraphStyle("H2",parent=ss["Heading2"],fontSize=12.5,spaceBefore=10,spaceAfter=5,textColor=colors.HexColor("#2f6f9f"))
B=ParagraphStyle("B",parent=ss["Normal"],fontSize=10.3,leading=14.5,spaceAfter=7)
CAP=ParagraphStyle("Cap",parent=B,fontSize=8.8,leading=11.5,textColor=colors.HexColor("#555555"),spaceAfter=10)
BOX=ParagraphStyle("Box",parent=B,backColor=colors.HexColor("#eef4f9"),borderPadding=(6,8,6,8),leftIndent=4,rightIndent=4,spaceBefore=4,spaceAfter=12)
DEF=ParagraphStyle("Def",parent=B,leftIndent=14,spaceAfter=5)
CODE=ParagraphStyle("Code",parent=B,fontName="Courier",fontSize=8.6,leading=11,backColor=colors.HexColor("#f4f4f4"),borderPadding=(5,6,5,6),leftIndent=4,rightIndent=4,spaceAfter=10)
def P(t,s=B): return Paragraph(t,s)
def fig(name,w=16.5,cap=None):
    from PIL import Image as PI
    iw,ih=PI.open(name).size; h=w*ih/iw
    out=[Image(name,width=w*cm,height=h*cm)]
    if cap: out.append(P(cap,CAP))
    return KeepTogether(out)
def tbl(rows,colw=None):
    t=Table(rows,colWidths=colw,hAlign="LEFT")
    t.setStyle(TableStyle([("FONT",(0,0),(-1,0),"Helvetica-Bold",9),("FONT",(0,1),(-1,-1),"Helvetica",9),
        ("BACKGROUND",(0,0),(-1,0),colors.HexColor("#dde8f0")),("GRID",(0,0),(-1,-1),0.4,colors.HexColor("#999999")),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,colors.HexColor("#f6f8fa")]),("TOPPADDING",(0,0),(-1,-1),3),("BOTTOMPADDING",(0,0),(-1,-1),3)]))
    return t
R=json.load(open(ORF+"results.json"))
def g(exp,fn,k,grid):
    for r in R:
        if r["exp"].startswith(exp) and r["fn"]==fn and r["k"]==k and r["grid"]==grid: return r["max_err"]
def f(x): return "0 (exact)" if x<1e-4 else f"{x:.3f}"

S=[]
S+= [P("Teaching a KAN to Copy a Bent Line",ParagraphStyle("T",parent=ss["Title"],fontSize=22,spaceAfter=4)),
     P("A plain-language guide to Nadia's example for the Operations Research Forum revision",ParagraphStyle("st",parent=B,fontSize=11.5,textColor=colors.HexColor("#555555"),spaceAfter=18)),
     P("<b>Who this is for.</b> Anyone who wants to understand what the example does and what it shows, without assuming any background in mathematics or machine learning. Technical words are explained the first time they appear and collected in a glossary at the end. Every claim is backed by a number in Section 7.",BOX),
     P("Contents",H2)]
for i,t in enumerate(["The question the example answers","Two bent lines: what S1 and S2 are","How Nadia's code builds them, step by step","What a KAN is, and what its 'grid' means","How a KAN is trained","What we tested","The results","What the results mean for the paper","Glossary"],1):
    S.append(P(f"{i}.&nbsp;&nbsp;{t}",DEF))
S.append(PageBreak())

# 1
S+=[P("1. The question the example answers",H1),
P("The paper compares two kinds of learning model, called KAN and LSTM, on stock prices. One reviewer, a mathematician who works on approximation, raised a point about how KANs are built on the inside. Nadia's answer is a small, clean example that has nothing to do with stock prices. It asks a very simple question:"),
P("<b>If you know exactly what a function looks like, and it is as simple as a few straight lines joined together, can a small KAN copy it? And does it matter <i>where</i> the lines join?</b>",BOX),
P("Why ask this? Because the answer exposes something about how a KAN works that is invisible on noisy stock data: a KAN can only bend its curves at certain fixed positions. If the thing it is copying bends somewhere else, it struggles. The example makes that concrete with two functions that are identical in every respect except where they bend."),
P("A 'function' here just means a rule that turns one number into another. Give it a value of <i>t</i> between 0 and 10, it hands back a value <i>S(t)</i>. Draw all those pairs and you get a line on a graph. The two functions are called S1 and S2.")]

# 2
S+=[P("2. Two bent lines: what S1 and S2 are",H1),
P("Both S1 and S2 are <b>piecewise linear</b>: made of straight pieces joined end to end, like a bent wire. Each place where the wire bends is called a <b>knot</b> (mathematicians' word) or a <b>kink</b> (the everyday word). Between knots the line is perfectly straight; at a knot it changes direction."),
fig("f3_s1_s2.png",cap="Figure 1. S1 (left) and S2 (right). Orange dots are the knots, the places where each line bends. The green labels show the slope of each straight piece, how steeply it rises or falls. The slopes are the same five numbers in both pictures, in the same order. The only difference is where the bends happen: S1 bends at 0, 2, 4, 6, 8 (evenly spaced), S2 at 0, 3, 5, 6, 7 (bunched up in the middle)."),
P("Look at the two pictures and notice: the same five steepnesses, in the same order. First a gentle rise, then gentler, then steep, then gentle, then almost flat. S2 is simply S1 with the bends shifted. This is deliberate. Any difference in how well a KAN copies them cannot be blamed on one function being 'harder' in the usual sense. The only variable is knot placement."),
P("Each function is given to the computer as a list of 1,001 points: the value of <i>S(t)</i> at <i>t</i> = 0, 0.01, 0.02, and so on up to 10. That is what the files S1.txt and S2.txt contain.")]

# 3
S+=[P("3. How Nadia's code builds them, step by step",H1),
P("Nadia's Matlab script is short. Here it is in full, and then what each part does."),
P("A = [2 -1 4 -3 -1];           % five slopes<br/>Theta1 = [0, 2, 4, 6, 8];     % knots of S1 (evenly spaced)<br/>Theta2 = [0, 3, 5, 6, 7];     % knots of S2 (not evenly spaced)<br/>for each t from 0 to 10 in steps of 0.01:<br/>&nbsp;&nbsp;&nbsp;&nbsp;S1(t) = sum over j of  A(j) * max(0, t - Theta1(j))<br/>&nbsp;&nbsp;&nbsp;&nbsp;S2(t) = sum over j of  A(j) * max(0, t - Theta2(j))",CODE),
P("<b>The building block: a hinge.</b> Everything rests on one tiny piece: <b>max(0, t - 3)</b>. Read it as 'take <i>t</i> minus 3, but if that would be negative, use 0 instead'. Before <i>t</i> reaches 3 the result is 0, flat. After 3 it rises in a straight line. The shape is a hinge, or a hockey stick, and the point where it starts rising is its knot. Multiply it by a number and the stick gets steeper, or flips to point downward."),
fig("f1_hinge.png",cap="Figure 2. The hinge max(0, t - 3) and two multiples of it. Flat, then straight; the knot is where it changes."),
P("<b>Adding hinges.</b> Now add five hinges together, each starting at a different knot and each multiplied by one of the five slopes. Every time <i>t</i> passes a new knot, one more hinge 'switches on' and the overall steepness changes by that hinge's slope. The result is a bent line with a bend at every knot. Figure 3 shows S1 being assembled one hinge at a time."),
fig("f2_build_s1.png",cap="Figure 3. Building S1 by adding one hinge at a time (dashed orange = the hinge being added, blue = the running total). After five hinges, the total is S1. S2 is built the same way with the knots moved."),
P("That is the whole of the script. The <b>for</b> loop just walks through the 1,001 values of <i>t</i>, and for each one adds up the five hinges. The last two lines save the results to the text files. There is nothing hidden: the functions are exactly the sum of five hinges, which is why we know precisely where every bend is.")]

# 4
S+=[P("4. What a KAN is, and what its 'grid' means",H1),
P("A <b>KAN</b> (Kolmogorov-Arnold Network) is a learning model. Most such models have fixed shapes inside and learn only numbers. A KAN is the other way round: the thing it learns is the <i>shape of a curve</i>. The smallest KAN, written [1,1], has one input, one output, and a single learned curve connecting them (Figure 4, left). Give it <i>t</i>, it looks up its curve at <i>t</i>, and hands back the height. Training a [1,1] KAN on S1 means: find the curve shape that best matches S1. If it can take exactly the shape of S1, the copy will be perfect."),
fig("f7_kan_diagrams.png",w=16.5,cap="Figure 4. Left: the smallest KAN, one learned curve. Right: a KAN with a hidden layer of three units; the output is a curve of a sum of curves of the input."),
P("<b>How the curve is stored: the grid.</b> A computer cannot store 'any curve'. A KAN stores its curve as a <b>spline</b>: a line made of small pieces, with the join points fixed in advance at evenly spaced positions. Those fixed positions are the <b>grid</b>. A '5-interval grid' on 0 to 10 puts join points at 0, 2, 4, 6, 8, 10. The KAN is free to choose how high the curve is at each grid point, but it is <i>not</i> free to move the grid points. So a KAN's curve can bend at grid points and nowhere else."),
P("This is the whole crux of the example. Compare where S1 and S2 bend with where a 5-interval grid allows bending:"),
fig("f4_grid.png",cap="Figure 5. Green vertical lines: where a 5-interval grid lets the KAN bend. Orange dots: where the function actually bends. For S1 every bend sits on a green line. For S2, the bends at 3, 5 and 7 fall between them, where the KAN cannot bend."),
P("<b>The simplest spline: straight pieces.</b> With 'linear' splines (written <i>k</i> = 1), the curve is straight between grid points. Technically it is a sum of little tent shapes, one centred on each grid point (Figure 6, left); choosing the height of each tent and adding them gives any straight-piece line with bends at the grid points. With the right six heights you get S1 exactly (Figure 6, right)."),
fig("f5_hats.png",cap="Figure 6. Left: the six tent-shaped building blocks of a linear spline on a 5-interval grid. Right: scale each tent and add them, and S1 appears exactly, because S1's bends are at the grid points."),
P("<b>Smooth splines.</b> The paper's KANs use <i>cubic</i> splines (<i>k</i> = 3): the pieces are smooth curves rather than straight segments, and they join without any visible corner. That smoothness is normally a virtue, but a smooth curve <i>cannot</i> make a sharp corner at all. It can only round one off (Figure 7, left). So with cubic splines neither S1 nor S2 can ever be copied exactly, whatever the grid.")]
S+=[fig("f6_cubic_vs_fine.png",cap="Figure 7. Left: a smooth curve can approach a corner but never reproduce it. Right: making the grid finer gives more places to bend, so the copy of S2 improves, at the cost of more numbers to learn.")]

# 5
S+=[P("5. How a KAN is trained",H1),
P("'Training' means finding the heights at the grid points (and a few other internal numbers) that make the KAN's curve match the target as closely as possible. The computer does not know the answer in advance; it searches for it."),
P("The procedure is the same for every learning model:"),
P("1. Start with the heights set to small random values. The KAN's curve is nonsense.<br/>2. Feed in all 1,001 values of <i>t</i>, and compare the KAN's output with the true <i>S(t)</i> at each one. Add up how wrong it is. That total is called the <b>loss</b> or <b>error</b>.<br/>3. Work out, for each height, which direction would reduce the error, and nudge every height a little that way.<br/>4. Repeat steps 2 and 3. Each repeat is one <b>step</b>. We used 200 steps.",DEF),
fig("f8_training.png",w=11,cap="Figure 8. Training as walking downhill: each step adjusts the model's settings so the total error falls. The method used here, L-BFGS, takes big well-aimed steps and is standard for small problems like this."),
P("Two details worth knowing. First, we told the KAN that <i>t</i> lives between 0 and 10, so its grid covers that range. Second, we did not let the KAN move its grid during training (a PyKAN option called 'grid updating'), because the point of the example is to see what a <i>fixed</i> grid can and cannot do."),
P("<b>How we measure success.</b> After training, we compute the KAN's output at all 1,001 points and look at the single <b>worst</b> error anywhere, the largest gap between the KAN's line and the true line. Nadia's field, approximation theory, measures quality this way: a copy is only as good as its worst point. We also report the root-mean-square error (RMSE), the paper's usual measure, which is more like an average.")]

# 6
S+=[P("6. What we tested",H1),
P("Four experiments, each on both S1 and S2:"),
P("<b>A. One curve, straight pieces, grid size varied.</b> The smallest KAN ([1,1]) with linear splines, trying grids of 5, 10, 20, 50 and 100 intervals. This is the cleanest test of 'can it bend in the right places'.",DEF),
P("<b>B. One curve, smooth pieces, grid size varied.</b> The same, but with cubic splines, as in the paper.",DEF),
P("<b>C. A hidden layer.</b> KANs of shape [1,2,1] and [1,3,1] (Figure 4, right) on coarse grids, to see whether stacking curves can compensate for a grid that does not line up with the bends. This is the 'one or two layers, very few units' setting Nadia suggested.",DEF),
P("<b>D. A reference that is allowed to move its knots.</b> Not a KAN: a plain bent line whose five knot positions <i>and</i> five slopes are both learned. If the search finds the right knots, this should copy both functions perfectly with only 11 numbers. It is the 'free-knot spline' the reviewer mentions, included to show what the KAN is being compared against.",DEF)]

# 7
S+=[P("7. The results",H1),
P("All numbers are the worst error over the 1,001 points. For scale, S1 rises to 22 and S2 to 18, so an error of 1.5 is about 7% of the function's height, clearly visible on a graph; an error of 0.02 would be invisible."),
P("<b>Experiment A: one curve, straight pieces</b>",H2),
tbl([["Grid intervals","Where the KAN can bend","S1 worst error","S2 worst error","Numbers learned"],
     ["5","0, 2, 4, 6, 8, 10",f(g("A","S1",1,5)),f(g("A","S2",1,5)),"12"],
     ["10","every whole number",f(g("A","S1",1,10)),f(g("A","S2",1,10)),"17"],
     ["20","every 0.5",f(g("A","S1",1,20)),f(g("A","S2",1,20)),"27"],
     ["50","every 0.2",f(g("A","S1",1,50)),f(g("A","S2",1,50)),"57"],
     ["100","every 0.1",f(g("A","S1",1,100)),f(g("A","S2",1,100)),"107"]],[2.6*cm,4.2*cm,2.8*cm,2.8*cm,3*cm]),
Spacer(1,6),
P("<b>The headline is the first row.</b> With a 5-interval grid, the KAN copies S1 <i>perfectly</i> and misses S2 by 1.48. Same KAN, same number of settings, same training. The only difference is that S1's bends sit on the grid and S2's do not. Figure 9 shows what that 1.48 looks like: the KAN's line cuts across S2's corners because it is not allowed to bend at 3, 5 or 7."),
fig(ORF+"figure_fits.png",cap="Figure 9. The actual fitted lines. Left, S1: a 5-interval KAN (orange dashed) sits exactly on top of the target. Right, S2: the same KAN cuts the corners; a 20-interval KAN (blue dashed) nearly fixes it."),
P("<b>Row two is a lucky coincidence worth understanding.</b> With 10 intervals the grid lines are at every whole number, and S2's knots (3, 5, 6, 7) happen to be whole numbers, so they land on the grid and S2 becomes exact as well. Had Nadia chosen a knot at, say, 3.3, no uniform grid of any reasonable size would hit it. This is why the suggestion at the end of this document is to move S2's knots off the whole numbers."),
P("<b>Rows three to five</b> show small errors creeping back in for both functions. That is not a representability problem (a finer grid can always bend where a coarser one could); it is that more grid points means more numbers to learn, and 200 training steps no longer finds the perfect answer. The error keeps falling with grid size all the same."),
P("<b>Experiment B: one curve, smooth pieces</b>",H2),
tbl([["Grid intervals","S1 worst error","S2 worst error"]]+[[str(gr),f(g("A","S1",3,gr)),f(g("A","S2",3,gr))] for gr in (5,10,20,50,100)],[3*cm,3*cm,3*cm]),
Spacer(1,6),
P("With smooth (cubic) splines neither function is ever exact, as predicted in Section 4: a smooth curve cannot make a corner. The error halves roughly every time the grid doubles, and crucially S1 and S2 now behave almost identically. Once the KAN cannot make corners anyway, it no longer matters much where the corners are. This is the setting the paper uses, so it is the more relevant row for the stock-price work."),
fig("f9_result.png",w=12.5,cap="Figure 10. Experiments A and B on one chart (log scales). Solid lines: straight pieces; dashed: smooth pieces. The straight-piece S1 line sits at zero (exact) from the start; S2 joins it only once the grid reaches 10."),
P("<b>Experiment C: a hidden layer</b>",H2),
tbl([["KAN shape","Grid","S1 worst error","S2 worst error","Numbers learned"],
     ["[1,2,1]","5",f(g("B: hidden layer [1,2,1]","S1",1,5)),f(g("B: hidden layer [1,2,1]","S2",1,5)),"48"],
     ["[1,2,1]","10",f(g("B: hidden layer [1,2,1]","S1",1,10)),f(g("B: hidden layer [1,2,1]","S2",1,10)),"68"],
     ["[1,3,1]","5",f(g("B: hidden layer [1,3,1]","S1",1,5)),f(g("B: hidden layer [1,3,1]","S2",1,5)),"72"],
     ["[1,3,1]","10",f(g("B: hidden layer [1,3,1]","S1",1,10)),f(g("B: hidden layer [1,3,1]","S2",1,10)),"102"]],[2.6*cm,1.8*cm,2.8*cm,2.8*cm,3*cm]),
Spacer(1,6),
P("Stacking curves helps S2 a great deal on the coarse grid: [1,3,1] on 5 intervals brings the worst error from 1.48 down to 0.03. Feeding a bent curve into another bent curve can produce an effective bend at a position that is not on the grid. The price is six times as many numbers to learn, and the search gets harder: [1,3,1] on the finer 10-interval grid is actually <i>worse</i> than on 5, because training got stuck in a poor solution. More capacity is not automatically better."),
P("<b>Experiment D: the free-knot reference</b>",H2),
tbl([["Function","Worst error","Knots the search found","True knots"],["S1","0 (exact)","0, 2, 4, 6, 8","0, 2, 4, 6, 8"],["S2","0 (exact)","0, 3, 5, 6, 7","0, 3, 5, 6, 7"]],[2.2*cm,2.6*cm,4.2*cm,4.2*cm]),
Spacer(1,6),
P("When the knots themselves are allowed to move, both functions are recovered exactly with just 11 numbers, and the search finds the true knot positions. But this came with a catch that is the reviewer's point in miniature: the search only succeeded for S2 when run from 14 different random starting positions and the best kept. With 6 starts it settled on wrong knots (3, 4.3, 5.0, 6.3) and an error of 0.48. Finding knot positions is a hard search with many false bottoms. Fixing the grid in advance, as KANs do, removes that difficulty, and the cost is exactly what Experiment A showed.")]

# 8
S+=[P("8. What the results mean for the paper",H1),
P("<b>For the reviewer's question.</b> The example demonstrates, in the cleanest possible setting, that a KAN's accuracy depends on whether the structure it is copying lines up with its fixed grid. Two functions that differ only in knot placement get a perfect copy and a 7% error from the same KAN. That is the honest answer to a mathematician who works on free-knot splines: yes, the fixed grid is a real limitation, and here is exactly how it shows up."),
P("<b>For the stock-price results.</b> Stock prices are not bent lines with known knots, and the paper uses smooth cubic splines, where Experiment B showed knot placement matters far less because no corner can be made anyway. So the example does not say the paper's KAN results are wrong. It says what kind of thing a KAN is: a sum of fixed-grid splines, exact when the data's structure happens to match the grid, approximate otherwise, and improving as the grid is refined at the cost of more parameters. That is a fair description to put in the paper."),
P("<b>Three honest limitations of the example itself.</b> First, S2's knots being whole numbers means a grid of 10 catches them by chance (Table A, row 2); moving one knot to a non-integer value would make the point hold for every uniform grid. Second, the small residual errors on fine grids are a training-budget effect, not a representability one; more steps would remove them. Third, Experiment D used a simple search; better free-knot algorithms exist and that is Nadia's area, so she may want to replace it."),
P("<b>Reproducing everything.</b> Run <font face='Courier'>python fit_kan.py</font> in the <font face='Courier'>orf_example</font> folder of the repository. It regenerates all numbers and figures in about three minutes. The data files are regenerated from Nadia's script and were checked against her originals to within 4 x 10<super>-15</super>.")]

# 9 glossary
S+=[P("9. Glossary",H1)]
for term,d in [
 ("Function","A rule that turns one number into another. Here, t in, S(t) out. Drawn as a line on a graph."),
 ("Piecewise linear","Made of straight segments joined end to end. A bent wire."),
 ("Knot / kink","A place where a piecewise-linear line changes direction. 'Knot' is the mathematical term, 'kink' the everyday one; in this document they mean the same thing."),
 ("Slope","How steeply a straight segment rises (positive) or falls (negative). S1 and S2 share the slopes 2, 1, 5, 2, 1 across their five pieces."),
 ("Hinge","The function max(0, t - c): flat until t reaches c, then rising in a straight line. Also called a ramp or hockey stick. Adding hinges builds any piecewise-linear line."),
 ("Spline","A curve stored as pieces joined at fixed points, with the heights (and in smooth splines, the curvature) chosen to match a target."),
 ("Grid","The fixed, evenly spaced positions at which a KAN's spline is allowed to bend. A 5-interval grid on 0 to 10 has join points at 0, 2, 4, 6, 8, 10. The model chooses heights at these points but cannot move the points."),
 ("Linear vs cubic spline (k = 1, k = 3)","k = 1: straight pieces, can make sharp corners at grid points. k = 3: smooth curved pieces, no visible joins, cannot make a sharp corner anywhere."),
 ("Free-knot spline","A spline whose join points are themselves chosen to fit the data, rather than fixed in advance. More flexible, but finding the best positions is a difficult search."),
 ("KAN","Kolmogorov-Arnold Network. A learning model whose adjustable parts are curve shapes (splines) rather than plain numbers. Written by the number of units in each layer: [1,1] is one curve; [1,3,1] feeds the input through three curves whose outputs are summed and passed through one more."),
 ("Hidden layer","Units between the input and output. Curves feeding into curves. Lets the model form bends at positions not on any single grid."),
 ("Training","Searching for the model's settings (here, spline heights) that minimise the error against the target. Done by repeatedly nudging the settings in the direction that reduces error."),
 ("Step","One round of that nudging. We used 200."),
 ("L-BFGS","The particular search method used. Takes large, well-aimed steps; standard for small problems."),
 ("Worst error (max error)","The largest gap between the model's output and the true value at any of the 1,001 points. The measure used in approximation theory."),
 ("RMSE","Root-mean-square error. An average-sized error, the measure used in the main paper. Less sensitive to a single bad point than the worst error."),
 ("Parameters / numbers learned","How many adjustable settings the model has. More parameters means more flexibility but a harder search."),
 ("Representability","Whether a model <i>could</i> copy the target exactly if its settings were perfect. S1 is representable by a 5-interval linear spline; S2 is not. Separate from whether training actually finds those settings.")]:
    S.append(P(f"<b>{term}.</b> {d}",DEF))

doc=SimpleDocTemplate("KAN_example_explained.pdf",pagesize=A4,leftMargin=2*cm,rightMargin=2*cm,topMargin=1.8*cm,bottomMargin=1.8*cm,title="Teaching a KAN to Copy a Bent Line",author="S. M. Mahmudul Hasan Joy")
def footer(c,d): c.setFont("Helvetica",8); c.setFillColor(colors.grey); c.drawRightString(A4[0]-2*cm,1.1*cm,f"Page {d.page}"); c.drawString(2*cm,1.1*cm,"KAN example for the ORF revision")
doc.build(S,onFirstPage=footer,onLaterPages=footer)
from pypdf import PdfReader; print("pages:",len(PdfReader("KAN_example_explained.pdf").pages))
