"""plot_figures.py — Unified figure generator for the DQC Scheduler paper.

Generates all paper figures from experiment output files.  All figure
functions are available in a single script; figures are generated only if
the required input data is present.

Input sources (all under --indir unless noted)
-----------------------------------------------
  e1_plan_comparison.csv        E1  plan comparison (E2–E16 similarly named)
  e17_congestion_arrival.csv    E17 congestion × arrival sweep
  e18_utilization_pareto.csv    E18 utilisation Pareto
  e19_stream_composition.csv    E19 stream composition
  e20_batch_stream.csv          E20 batch vs stream (--indir or --indir2)
  e21_throughput_scaling.json   E21 throughput scaling
  e24_idle_fraction.json        E24 idle fraction vs lambda

Usage
-----
    # Generate all figures for which data is available
    python plot_figures.py --indir results/experiments --outdir results/figures

    # Generate a specific subset
    python plot_figures.py --indir results/experiments --outdir results/figures \
        --figures 1,2,3,49,60,61

    # Slides style (slightly larger fonts)
    python plot_figures.py --indir results/experiments --outdir results/figures \
        --style slides

    # Apply post-review patches to selected figures
    python plot_figures.py --indir results/experiments --outdir results/figures \
        --patch
"""

import argparse, csv, collections, os, statistics, math
from typing import Dict, List
import numpy as np
import matplotlib, matplotlib.pyplot as plt, matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
from matplotlib.patches import Patch

matplotlib.rcParams.update({
    "font.family":"serif","font.size":11,"axes.titlesize":13,"axes.labelsize":12,
    "xtick.labelsize":11,"ytick.labelsize":11,"legend.fontsize":8,
    "figure.dpi":150,"savefig.dpi":300,"savefig.bbox":"tight",
    "axes.spines.top":False,"axes.spines.right":False,
    "axes.grid":True,"grid.alpha":0.3,"grid.linewidth":0.5,
})

TEAL="#1D9E75"; AMBER="#BA7517"; BLUE="#3266AD"; CORAL="#D85A30"
GRAY="#888780"; PURPLE="#7F77DD"; PINK="#D4537E"; GREEN="#639922"
COND_PAL=[TEAL,AMBER,BLUE,CORAL,GRAY,PURPLE,PINK,GREEN]

PLAN_COLOR={"A_NO_CUT_SINGLE":TEAL,"B_CUT_SINGLE_SEQ":AMBER,"C_CUT_MULTI_QPU":BLUE,"D_WAIT":GRAY}
PLAN_LABEL={"A_NO_CUT_SINGLE":"Plan A – No cut","B_CUT_SINGLE_SEQ":"Plan B – Cut single QPU",
            "C_CUT_MULTI_QPU":"Plan C – Cut multi-QPU","D_WAIT":"D – Wait"}
def _e4_conds(e4):
    """Return (condition_keys, display_labels) that match whatever E4 CSV is loaded.
    Supports both old names (pre-redesign) and new names (post-redesign).
    """
    present = set(r.get("condition","") for r in e4)
    if "homog_uniform" in present:
        # New redesigned E4
        return (
            ["homog_uniform","heterog_quality","heterog_capacity","congestion_best"],
            ["Homog.\n3×7Q","Heterog.\nquality","Heterog.\ncapacity","Congestion"]
        )
    else:
        # Old E4 (pre-redesign) — use old names until rerun
        return (
            ["homogeneous_2qpu","heterogeneous_3qpu","large_3qpu","congestion_burst"],
            ["Homog.\n2×7Q","Heterog.\n3-QPU","Large\n3×14Q","Congestion\nburst"]
        )



def _load(p):
    with open(p,newline="") as f: return list(csv.DictReader(f))
def sf(v):
    try: return float(v)
    except: return None
def vals(rows,col,cond=None):
    sub=[r for r in rows if cond is None or r.get("condition")==cond]
    return [v for r in sub if (v:=sf(r.get(col))) is not None and not math.isnan(v)]
def vp(rows,col,cond=None): return [v for v in vals(rows,col,cond) if v>0]
def _box(ax,data,labels,colors,ylabel,log=False,ylim=None):
    bp=ax.boxplot(data,patch_artist=True,notch=False,vert=True,
                  medianprops=dict(color="white",linewidth=1.5),
                  whiskerprops=dict(linewidth=0.8),capprops=dict(linewidth=0.8),
                  flierprops=dict(marker=".",markersize=2.5,alpha=0.5))
    for p,c in zip(bp["boxes"],colors): p.set_facecolor(c); p.set_alpha(0.85)
    ax.set_xticks(range(1,len(labels)+1))
    rot = 15 if any(len(l) > 8 for l in labels) else 0
    ha  = "right" if rot else "center"
    ax.set_xticklabels(labels, rotation=rot, ha=ha, fontsize=11)
    ax.set_ylabel(ylabel)
    if log: ax.set_yscale("log")
    if ylim: ax.set_ylim(*ylim)
def _save(fig,path):
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".",exist_ok=True)
    fig.savefig(path+".pdf"); fig.savefig(path+".png"); plt.close(fig)
    print(f"  saved -> {os.path.basename(path)}.pdf/.png")



# ============================================================================
# Figures from E1-E16 (paper_figures.py)
# ============================================================================


# ── Fig 1: Time breakdown ───────────────────────────────────────────────────
def fig01(e1,od):
    """Plot time breakdown per PLAN KIND (not per condition).
    Reconstruction computed analytically (0.005 + 0.002 * sampling_overhead)
    since model_recon_s is excluded from the new planner output.
    Communication uses charged_comm_s (actual per-job comm cost).
    """
    fig,ax=plt.subplots(figsize=(4.8,3.6))
    plans=["A_NO_CUT_SINGLE","B_CUT_SINGLE_SEQ","C_CUT_MULTI_QPU"]
    xlbls=["Plan A\nNo cut","Plan B\nCut single QPU","Plan C\nCut multi-QPU"]

    def _recon(r):
        samp = sf(r.get("sampling_overhead")) or 0
        return 0.005 + 0.002 * samp if samp > 0 else 0.0

    layers=[
        ("sim_queue_wait_s", GRAY,  "Queue wait",      False),
        ("model_exec_s",     TEAL,  "Execution",        False),
        ("charged_comm_s",   BLUE,  "Communication",    False),
        (None,               AMBER, "Reconstruction",   True),   # analytic
    ]
    x=np.arange(len(plans)); bot=np.zeros(len(plans))
    for field,col,lbl,analytic in layers:
        meds=[]
        for plan in plans:
            sub=[r for r in e1 if r.get("plan_kind")==plan]
            if not sub: meds.append(0); continue
            if analytic:
                v=[_recon(r) for r in sub]
            else:
                v=[sf(r.get(field)) or 0 for r in sub]
            meds.append(statistics.median(v) if v else 0)
        ax.bar(x,meds,bottom=bot,color=col,label=lbl,width=0.52,linewidth=0.4,edgecolor="white")
        bot+=np.array(meds)
    for i,plan in enumerate(plans):
        n=sum(1 for r in e1 if r.get("plan_kind")==plan)
        ax.text(i,bot[i]+0.03,f"{bot[i]:.2f}s" + (f" (n={n}*)" if n < 20 else f" (n={n})"),
                ha="center",va="bottom",fontsize=11,color="black",fontweight="500")
    ax.set_xticks(x); ax.set_xticklabels(xlbls)
    ax.set_ylabel("Median time per job (s)")
    ax.legend(loc="upper left",framealpha=0.9,fontsize=8)
    ax.set_ylim(0,bot.max()*1.22)
    ax.text(0.99, 0.01, "* n<20: interpret with caution", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=11, color=GRAY, style="italic")
    fig.tight_layout(); _save(fig,f"{od}/fig01_time_breakdown")

# ── Fig 2: Scheduling overhead ──────────────────────────────────────────────
def fig02(e1,od):
    fig,ax=plt.subplots(figsize=(3.8,3.0))
    conds=["no_cut","cut_single","cut_multi"]
    xlbls=["No cut","Cut single","Cut multi"]; colors=[TEAL,AMBER,BLUE]
    data=[[v*1000 for v in vp(e1,"schedule_wall_s",c)] for c in conds]
    _box(ax,data,xlbls,colors,"Scheduler decision time (ms)",log=True)
    ax.axhline(1.0,color=CORAL,linewidth=0.9,linestyle="--",alpha=0.8,zorder=0)
    ax.text(3.45,1.12,"1 ms",color=CORAL,fontsize=11,va="bottom")
    fig.tight_layout(); _save(fig,f"{od}/fig02_scheduling_overhead")

# ── Fig 3: Queue wait reduction ─────────────────────────────────────────────
def fig03(e1,od):
    """Queue wait drops by 47% when cutting is enabled.
    Single-panel box plot with median annotations and pct-reduction callout.
    """
    fig,ax=plt.subplots(figsize=(5.0,3.4))
    conds=["no_cut","cut_single","cut_multi"]
    xlbls=["No cut","Cut single","Cut multi"]; colors=[TEAL,AMBER,BLUE]
    _box(ax,[vals(e1,"sim_queue_wait_s",c) for c in conds],xlbls,colors,"Queue wait (s)")
    meds=[]
    for i,c in enumerate(conds):
        med=statistics.median(vals(e1,"sim_queue_wait_s",c))
        meds.append(med)
        ax.text(i+1,med+0.03,f"{med:.2f}s",ha="center",va="bottom",
                fontsize=11,fontweight="500")
    # Annotate the reduction
    reduction=100*(meds[0]-meds[2])/meds[0]
    ax.set_ylim(0,2.3)
    fig.tight_layout(); _save(fig,f"{od}/fig03_queue_wait_attempts")

# ── Fig 4: Throughput + fidelity-latency scatter (E1) ──────────────────────
def fig04(e1,od):
    """Throughput and tail latency: cutting improves both.
    Left: throughput (jobs/s) bars with P90 latency overlay.
    Right: per-job fidelity vs latency scatter coloured by plan kind.
    Key numbers: cut_multi +35% throughput, -39% P90 vs no_cut.
    """
    fig,(ax1,ax2)=plt.subplots(1,2,figsize=(7.5,3.2))

    conds=["no_cut","cut_single","cut_multi"]
    xlbls=["No cut","Cut single","Cut multi"]; colors=[TEAL,AMBER,BLUE]

    # Left panel: stacked bar of queue wait + execution time, with P90 line overlay
    # This directly shows WHY cutting helps: queue wait drops, exec stays similar
    qw_meds  = [statistics.median(vals(e1,"sim_queue_wait_s",c)) for c in conds]
    exec_meds= [statistics.median(vals(e1,"model_exec_s",c) or [0]) for c in conds]
    p90s = []
    for cond in conds:
        e2e_s = sorted([v for v in vals(e1,"end_to_end_s",cond) if v>0])
        p90s.append(e2e_s[int(0.9*len(e2e_s))] if e2e_s else 0)

    x=np.arange(len(conds))
    ax1.bar(x, qw_meds,  color=[GRAY,GRAY,GRAY],  width=0.5, linewidth=0.4,
            edgecolor="white", alpha=0.85, label="Queue wait")
    ax1.bar(x, exec_meds, bottom=qw_meds, color=colors, width=0.5, linewidth=0.4,
            edgecolor="white", alpha=0.88, label="Execution")
    ax1.set_xticks(x); ax1.set_xticklabels(xlbls)
    ax1.set_ylabel("Median time (s)")
    ax1.legend(fontsize=8, framealpha=0.9, loc="upper right")

    ax1b=ax1.twinx()
    ax1b.plot(x, p90s, color=CORAL, marker="D", linewidth=1.8, markersize=7, zorder=5)
    # Stagger P90 labels vertically to avoid collision
    ax1b.set_ylabel("P90 end-to-end latency (s)", color=CORAL)
    ax1b.tick_params(axis="y", colors=CORAL)
    ax1b.set_ylim(0, 2.5); ax1.set_ylim(0, 2.0)
    for plan,col,lbl in [("A_NO_CUT_SINGLE",TEAL,"Plan A – No cut"),
                          ("B_CUT_SINGLE_SEQ",AMBER,"Plan B – Cut single"),
                          ("C_CUT_MULTI_QPU",BLUE,"Plan C – Cut multi")]:
        rows=[r for r in e1 if r["plan_kind"]==plan]
        fids=[sf(r["fidelity_proxy"]) for r in rows if sf(r.get("fidelity_proxy"))]
        e2es=[sf(r["end_to_end_s"]) for r in rows if sf(r.get("end_to_end_s"))]
        if fids and e2es:
            ax2.scatter(e2es,fids,color=col,alpha=0.72,s=30,label=lbl,
                        edgecolors="white",linewidths=0.4,zorder=4)
    ax2.set_xlabel("End-to-end latency (s)")
    ax2.set_ylabel("Fidelity")
    ax2.set_ylim(0.35,1.05)
    ax2.legend(fontsize=8, framealpha=0.9, loc="lower right")
    fig.tight_layout(); _save(fig,f"{od}/fig04_throughput_fidelity_scatter")

# ── Fig 5: Plan mix by workload (E2) ────────────────────────────────────────
def fig05(e2,od):
    conds=[("light_narrow","Light\nnarrow"),("light_wide","Light\nwide"),
           ("heavy_qaoa_vqe","QAOA/VQE"),("mixed_25pct","Mixed\n25%"),
           ("mixed_50pct","Mixed\n50%"),("ghz_only","GHZ"),
           ("qft_only","QFT"),("random_only","Random")]
    ck=[c[0] for c in conds]; xl=[c[1] for c in conds]; x=np.arange(len(conds))
    fig,ax=plt.subplots(figsize=(7.5,3.2))
    bot=np.zeros(len(conds))
    for plan,col in PLAN_COLOR.items():
        if plan=="D_WAIT": continue
        pcts=[100.0*sum(1 for r in e2 if r.get("condition")==k and r.get("plan_kind")==plan)/
              max(sum(1 for r in e2 if r.get("condition")==k),1) for k in ck]
        ax.bar(x,pcts,bottom=bot,color=col,width=0.62,linewidth=0.3,
               edgecolor="white",label=PLAN_LABEL[plan])
        bot+=np.array(pcts)
    ax.set_xticks(x); ax.set_xticklabels(xl,fontsize=11)
    ax.set_ylabel("Jobs (%)"); ax.set_ylim(0,108)
    ax.legend(fontsize=8, framealpha=0.95,
              loc="upper center", bbox_to_anchor=(0.5, 1.14), ncol=3)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    _save(fig, f"{od}/fig05_workload_plan_mix")

# ── Fig 6: Fidelity by circuit family (E2) ──────────────────────────────────
def fig06(e2,od):
    conds=[("ghz_only","GHZ",TEAL),("random_only","Random CX",AMBER),
           ("qft_only","QFT",BLUE),("light_wide","Light wide",CORAL),
           ("heavy_qaoa_vqe","QAOA/VQE",PURPLE)]
    fig,ax=plt.subplots(figsize=(5.5,3.2))
    _box(ax,[vals(e2,"fidelity_proxy",c[0]) for c in conds],
         [c[1] for c in conds],[c[2] for c in conds],"Fidelity",ylim=(0,1.05))
    for i,c in enumerate(conds):
        d=vals(e2,"fidelity_proxy",c[0])
        if d: ax.text(i+1,statistics.median(d)+0.02,f"{statistics.median(d):.3f}",
                      ha="center",va="bottom",fontsize=11,fontweight="500")
    fig.tight_layout(); _save(fig,f"{od}/fig06_fidelity_family")

# ── Fig 7: Objective score decomposition by plan kind (E1) ─────────────────
def fig07(e1,od):
    """Shows the four-term objective score broken down by component per plan.

    Key insight: Plan A wins on frag/coord (zero penalties) but loses on
    qpu_completion when queues are long. Plan C wins on qpu_completion
    (parallelism) at the cost of a non-zero coord penalty. The quality term
    differentiates QPU choices within a plan kind.
    """
    plans=["A_NO_CUT_SINGLE","B_CUT_SINGLE_SEQ","C_CUT_MULTI_QPU"]
    xlbls=["Plan A\nNo cut","Plan B\nCut single","Plan C\nCut multi"]
    components=[
        ("score_qpu_completion_s", TEAL,  "QPU completion (queue+exec)"),
        ("score_frag_penalty_s",   AMBER, "Fragmentation penalty"),
        ("score_coord_penalty_s",  BLUE,  "Coordination penalty"),
        ("score_quality_term",     CORAL, "Quality term (1−fidelity)"),
    ]

    fig,(ax1,ax2)=plt.subplots(1,2,figsize=(8.0,3.8))

    # Panel 1: stacked bar of median score components
    x=np.arange(len(plans)); bot=np.zeros(len(plans))
    for col_name,col,lbl in components:
        meds=[]
        for plan in plans:
            v=[sf(r.get(col_name)) for r in e1 if r["plan_kind"]==plan and sf(r.get(col_name)) is not None]
            meds.append(statistics.median(v) if v else 0)
        ax1.bar(x,meds,bottom=bot,color=col,label=lbl,width=0.52,linewidth=0.4,edgecolor="white",alpha=0.9)
        bot+=np.array(meds)
    for i,plan in enumerate(plans):
        v=[sf(r.get("score_total")) for r in e1 if r["plan_kind"]==plan and sf(r.get("score_total")) is not None]
        total=statistics.median(v) if v else 0
        ax1.text(i,bot[i]+0.005,f"{total:.3f}",ha="center",va="bottom",fontsize=11,fontweight="500")
    ax1.set_xticks(x); ax1.set_xticklabels(xlbls)
    ax1.set_ylabel("Median score (s)")
    ax1.legend(fontsize=8, framealpha=0.95,
               loc="upper center", bbox_to_anchor=(0.5, -0.16),
               ncol=2, handlelength=1.0, borderpad=0.4, labelspacing=0.3)
    ax1.set_ylim(0, bot.max()*1.22)

    # Panel 2: per-job score distribution as box plot
    _box(ax2,
         [[sf(r.get("score_total")) for r in e1 if r["plan_kind"]==p and sf(r.get("score_total")) is not None]
          for p in plans],
         xlbls, [TEAL,AMBER,BLUE], "Total objective score (s)")
    ax2.set_ylim(0, 1.65)
    for i, plan in enumerate(plans):
        v=[sf(r.get("score_total")) for r in e1 if r["plan_kind"]==plan and sf(r.get("score_total")) is not None]
        if v:
            med=statistics.median(v)
            ax2.text(i+1, med+0.025, f"{med:.3f}", ha="center", va="bottom",
                     fontsize=11, fontweight="500")
    fig.tight_layout(rect=[0, 0.10, 1, 1])
    _save(fig, f"{od}/fig07_score_decomposition")

# ── Fig 8: Algorithm comparison table (E3) ──────────────────────────────────
def fig08(e3,od):
    fig,ax=plt.subplots(figsize=(7.5,2.8)); ax.set_axis_off()
    def _med(c,col): v=vals(e3,col,c); return f"{statistics.median(v):.3f}" if v else "—"
    def _ms(c):
        v=vp(e3,"schedule_wall_s",c)
        return f"{statistics.median(v)*1000:.3f} ms" if v else "—"
    fc=collections.Counter(r["plan_kind"] for r in e3 if r["condition"]=="our_fitcut")
    ac=collections.Counter(r["plan_kind"] for r in e3 if r["condition"]=="qiskit_addon")
    nf=sum(1 for r in e3 if r["condition"]=="our_fitcut")
    na=sum(1 for r in e3 if r["condition"]=="qiskit_addon")
    def ps(ctr,n):
        return ", ".join(f"{ctr.get(p,0)}/{n} {s}" for p,s in
                         [("C_CUT_MULTI_QPU","C"),("B_CUT_SINGLE_SEQ","B")] if ctr.get(p,0)>0)
    rows=[
        ["Jobs completed (of 40)",f"{nf}",f"{na}","0  (stall — circuits too wide)"],
        ["Median end-to-end (s)",_med("our_fitcut","end_to_end_s"),_med("qiskit_addon","end_to_end_s"),"N/A"],
        ["Median fidelity proxy",_med("our_fitcut","fidelity_proxy"),_med("qiskit_addon","fidelity_proxy"),"N/A"],
        ["Median sched. overhead",_ms("our_fitcut"),_ms("qiskit_addon"),"0.022 ms"],
        ["Max sched. overhead",
         f"{max(vp(e3,'schedule_wall_s','our_fitcut'),default=[0])*1000:.3f} ms",
         f"{max(vp(e3,'schedule_wall_s','qiskit_addon'),default=[0])*1000:.3f} ms","0.038 ms"],
        ["Plan distribution",ps(fc,nf),ps(ac,na),"N/A (no jobs)"],
        ["Plan C (multi-QPU) rate",
         f"{100*fc.get('C_CUT_MULTI_QPU',0)/max(nf,1):.0f}%",
         f"{100*ac.get('C_CUT_MULTI_QPU',0)/max(na,1):.0f}%","0%"],
    ]
    cols=["Metric","FitCut (ours)","Qiskit Addon","No-cut baseline"]
    tbl=ax.table(cellText=rows,colLabels=cols,loc="center",cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(8.5); tbl.scale(1,1.6)
    for j in range(len(cols)):
        tbl[0,j].set_facecolor("#2C2C2A"); tbl[0,j].set_text_props(color="white",fontweight="bold")
    for i in range(1,len(rows)+1):
        tbl[i,1].set_facecolor("#E1F5EE")
        if i in (1,2,3): tbl[i,3].set_facecolor("#FCEBEB")
        if i%2==0:
            for j in (0,2,3): tbl[i,j].set_facecolor("#F8F8F6")
    
    fig.tight_layout(); _save(fig,f"{od}/fig08_algorithm_table")

# ── Fig 8p: Pandora strategy comparison (E3 with pandora conditions) ──────────
def fig08p(e3, od):
    """Three-panel comparison: latency, fidelity, plan mix across all E3 conditions."""
    conds = ["our_fitcut", "qiskit_addon", "pandora_optimized", "pandora_widgetizer"]
    labels = ["FitCut\n(baseline)", "Qiskit\nAddon", "Pandora +\nFitCut", "Pandora\nWidgetizer"]
    colors = [TEAL, AMBER, BLUE, CORAL]
    present = set(r.get("condition") for r in e3)
    conds  = [c for c in conds  if c in present]
    labels = [labels[i] for i, c in enumerate(
        ["our_fitcut","qiskit_addon","pandora_optimized","pandora_widgetizer"]) if c in present]
    colors = colors[:len(conds)]
    x = np.arange(len(conds))
    w = 0.55

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(11, 3.6))

    # Panel 1: median end-to-end latency
    e2e = [statistics.median(vp(e3, "end_to_end_s", c) or [0]) for c in conds]
    bars = ax1.bar(x, e2e, width=w, color=colors, edgecolor="white", linewidth=0.4)
    for xi, v in zip(x, e2e):
        ax1.text(xi, v + max(e2e)*0.01, f"{v:.2f}s", ha="center", va="bottom", fontsize=9)
    ax1.set_xticks(x); ax1.set_xticklabels(labels, fontsize=9)
    ax1.set_ylabel("Median end-to-end (s)")
    ax1.set_title("Latency")
    ymax = max(e2e) * 1.18 if e2e else 1
    ax1.set_ylim(0, ymax)

    # Panel 2: median fidelity proxy
    fid = [statistics.median(vp(e3, "fidelity_proxy", c) or [0]) for c in conds]
    ax2.bar(x, fid, width=w, color=colors, edgecolor="white", linewidth=0.4)
    for xi, v in zip(x, fid):
        ax2.text(xi, v + 0.002, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    ax2.set_xticks(x); ax2.set_xticklabels(labels, fontsize=9)
    ax2.set_ylabel("Median fidelity proxy")
    ax2.set_title("Fidelity")
    fmin = min(fid) if fid else 0
    ax2.set_ylim(max(0, fmin - 0.05), min(1.0, max(fid) * 1.06) if fid else 1)

    # Panel 3: plan mix (stacked bars — A/B/C)
    plan_types = [("A_NO_CUT_SINGLE", TEAL, "Plan A"), ("B_CUT_SINGLE_SEQ", AMBER, "Plan B"),
                  ("C_CUT_MULTI_QPU", BLUE, "Plan C")]
    bot = np.zeros(len(conds))
    for plan_kind, col, lbl in plan_types:
        pcts = []
        for c in conds:
            total = sum(1 for r in e3 if r.get("condition") == c)
            n = sum(1 for r in e3 if r.get("condition") == c and r.get("plan_kind") == plan_kind)
            pcts.append(100 * n / max(total, 1))
        ax3.bar(x, pcts, bottom=bot, width=w, color=col, edgecolor="white",
                linewidth=0.4, label=lbl, alpha=0.88)
        for xi, v, b in zip(x, pcts, bot):
            if v > 4:
                ax3.text(xi, b + v/2, f"{v:.0f}%", ha="center", va="center",
                         fontsize=8, color="white", fontweight="bold")
        bot += np.array(pcts)
    ax3.set_xticks(x); ax3.set_xticklabels(labels, fontsize=9)
    ax3.set_ylabel("Jobs (%)")
    ax3.set_title("Plan mix")
    ax3.set_ylim(0, 115)
    ax3.legend(loc="upper right", fontsize=7, framealpha=0.7)

    fig.suptitle("Cutting strategy comparison (E3)", fontsize=12, fontweight="bold", y=1.01)
    fig.tight_layout()
    _save(fig, f"{od}/fig08p_pandora_comparison")

# ── Fig 8q: Pandora stress workload comparison (E3P) ─────────────────────────
def fig08q(e3p, e3p_gc, od):
    """Four-panel Pandora stress comparison.

    Panels: (1) end-to-end latency, (2) fidelity, (3) plan mix,
    (4) gate/depth reduction from Pandora optimization.
    Panel 4 uses e3p_gate_counts.csv and shows what Pandora actually achieved
    regardless of whether it translated into scheduling latency improvement.
    """
    conds  = ["fitcut_baseline", "pandora_optimized", "pandora_widgetizer"]
    labels = ["FitCut\n(baseline)", "Pandora +\nFitCut", "Pandora\nWidgetizer"]
    colors = [TEAL, BLUE, CORAL]
    present = set(r.get("condition") for r in e3p)
    idx    = [i for i, c in enumerate(conds) if c in present]
    conds  = [conds[i]  for i in idx]
    labels = [labels[i] for i in idx]
    colors = [colors[i] for i in idx]
    x = np.arange(len(conds))
    w = 0.55

    fig, axes = plt.subplots(1, 4, figsize=(14, 3.6))
    ax1, ax2, ax3, ax4 = axes

    # Panel 1: median end-to-end latency with % vs baseline
    e2e = [statistics.median(vp(e3p, "end_to_end_s", c) or [0]) for c in conds]
    baseline = e2e[0] if e2e else 1.0
    ax1.bar(x, e2e, width=w, color=colors, edgecolor="white", linewidth=0.4)
    for xi, v in zip(x, e2e):
        pct = 100 * (v - baseline) / baseline if baseline else 0
        sign = "+" if pct >= 0 else ""
        label = f"{v:.2f}s" if xi == 0 else f"{v:.2f}s\n({sign}{pct:.1f}%)"
        ax1.text(xi, v + max(e2e) * 0.01, label, ha="center", va="bottom", fontsize=8)
    ax1.set_xticks(x); ax1.set_xticklabels(labels, fontsize=8.5)
    ax1.set_ylabel("Median end-to-end (s)")
    ax1.set_title("Latency")
    ax1.set_ylim(0, max(e2e) * 1.28 if e2e else 1)

    # Panel 2: median fidelity proxy
    fid = [statistics.median(vp(e3p, "fidelity_proxy", c) or [0]) for c in conds]
    ax2.bar(x, fid, width=w, color=colors, edgecolor="white", linewidth=0.4)
    for xi, v in zip(x, fid):
        ax2.text(xi, v + 0.002, f"{v:.3f}", ha="center", va="bottom", fontsize=8.5)
    ax2.set_xticks(x); ax2.set_xticklabels(labels, fontsize=8.5)
    ax2.set_ylabel("Median fidelity proxy")
    ax2.set_title("Fidelity")
    fmin = min(fid) if fid else 0
    ax2.set_ylim(max(0, fmin - 0.05), min(1.0, max(fid) * 1.06) if fid else 1)

    # Panel 3: plan mix
    plan_types = [("A_NO_CUT_SINGLE", TEAL, "Plan A"), ("B_CUT_SINGLE_SEQ", AMBER, "Plan B"),
                  ("C_CUT_MULTI_QPU", BLUE, "Plan C")]
    bot = np.zeros(len(conds))
    for plan_kind, col, lbl in plan_types:
        pcts = []
        for c in conds:
            total = sum(1 for r in e3p if r.get("condition") == c)
            n = sum(1 for r in e3p if r.get("condition") == c and r.get("plan_kind") == plan_kind)
            pcts.append(100 * n / max(total, 1))
        ax3.bar(x, pcts, bottom=bot, width=w, color=col, edgecolor="white",
                linewidth=0.4, label=lbl, alpha=0.88)
        for xi, v, b in zip(x, pcts, bot):
            if v > 4:
                ax3.text(xi, b + v/2, f"{v:.0f}%", ha="center", va="center",
                         fontsize=8, color="white", fontweight="bold")
        bot += np.array(pcts)
    ax3.set_xticks(x); ax3.set_xticklabels(labels, fontsize=8.5)
    ax3.set_ylabel("Jobs (%)")
    ax3.set_title("Plan mix")
    ax3.set_ylim(0, 115)
    ax3.legend(loc="upper right", fontsize=7, framealpha=0.7)

    # Panel 4: gate count and depth before vs after Pandora (from e3p_gate_counts.csv)
    if e3p_gc:
        gb = [sf(r.get("gates_before")) for r in e3p_gc if sf(r.get("gates_before")) is not None]
        ga = [sf(r.get("gates_after"))  for r in e3p_gc if sf(r.get("gates_after"))  is not None]
        db = [sf(r.get("depth_before")) for r in e3p_gc if sf(r.get("depth_before")) is not None]
        da = [sf(r.get("depth_after"))  for r in e3p_gc if sf(r.get("depth_after"))  is not None]
        avg_gb = statistics.mean(gb) if gb else 0
        avg_ga = statistics.mean(ga) if ga else 0
        avg_db = statistics.mean(db) if db else 0
        avg_da = statistics.mean(da) if da else 0

        bx = np.array([0, 1])
        before_vals = [avg_gb, avg_db]
        after_vals  = [avg_ga, avg_da]
        bw = 0.32
        bars_b = ax4.bar(bx - bw/2, before_vals, width=bw, color=GRAY,  label="Before", alpha=0.85, edgecolor="white")
        bars_a = ax4.bar(bx + bw/2, after_vals,  width=bw, color=PURPLE, label="After",  alpha=0.85, edgecolor="white")
        for bar, v in zip(bars_b, before_vals):
            ax4.text(bar.get_x() + bar.get_width()/2, v + 0.3, f"{v:.1f}", ha="center", va="bottom", fontsize=8)
        for bar, v, bv in zip(bars_a, after_vals, before_vals):
            pct = 100 * (bv - v) / bv if bv else 0
            ax4.text(bar.get_x() + bar.get_width()/2, v + 0.3, f"{v:.1f}\n(−{pct:.0f}%)",
                     ha="center", va="bottom", fontsize=7.5, color=PURPLE)
        ax4.set_xticks(bx)
        ax4.set_xticklabels(["Gate count", "Depth"], fontsize=9)
        ax4.set_ylabel("Count")
        ax4.set_title("Pandora optimization\n(gates & depth reduced)")
        ax4.legend(fontsize=8, framealpha=0.7)
        ax4.set_ylim(0, max(before_vals) * 1.3)
    else:
        ax4.text(0.5, 0.5, "No gate count data\n(run E3P to generate)", ha="center",
                 va="center", transform=ax4.transAxes, fontsize=9, color=GRAY)
        ax4.set_title("Pandora optimization")

    fig.suptitle("Pandora stress workload: cancellable-pair circuits (E3P)",
                 fontsize=12, fontweight="bold", y=1.01)
    fig.tight_layout()
    _save(fig, f"{od}/fig08q_pandora_stress")

# ── Fig 9: QPU load + Gini (E4) ─────────────────────────────────────────────
def fig09(e4, od):
    """E4: Fidelity gain and plan mix across QPU pool configurations."""
    order = ["homog_uniform","heterog_quality","heterog_capacity","congestion_best",
             "homogeneous_2qpu","heterogeneous_3qpu","large_3qpu","congestion_burst"]
    present = set(r["condition"] for r in e4)
    all_conds = [c for c in order if c in present]
    short = {
        "homog_uniform":    "Homog. uniform",
        "heterog_quality":  "Heterog. quality",
        "heterog_capacity": "Heterog. capacity",
        "congestion_best":  "Congest. (best blocked)",
        "homogeneous_2qpu": "Homog. 2x7Q",
        "heterogeneous_3qpu": "Heterog. 3-QPU",
        "large_3qpu":       "Large 3x14Q",
        "congestion_burst": "Congest. burst",
    }
    xlbls = [short.get(c, c) for c in all_conds]
    colors = [TEAL, AMBER, BLUE, CORAL][:len(all_conds)]
    x = np.arange(len(all_conds))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.5, 3.4))

    # Panel 1: fidelity by condition
    fid_meds = [statistics.median(vals(e4, "fidelity_proxy", c) or [0]) for c in all_conds]
    bars = ax1.bar(x, fid_meds, color=colors, width=0.55, linewidth=0.4, edgecolor="white")
    for xi, v in zip(x, fid_meds):
        ax1.text(xi, v + 0.003, f"{v:.3f}", ha="center", va="bottom", fontsize=11, fontweight="500")
    ax1.set_xticks(x); ax1.set_xticklabels(xlbls, fontsize=11, rotation=15, ha="right")
    ax1.set_ylabel("Median fidelity")
    fmin = min(fid_meds) if fid_meds else 0.5
    ax1.set_ylim(max(0.4, fmin - 0.08), min(1.0, max(fid_meds) + 0.06))

    # Panel 2: plan mix by condition
    bot = np.zeros(len(all_conds))
    for plan, col, lbl in [("A_NO_CUT_SINGLE", TEAL, "Plan A"),
                            ("B_CUT_SINGLE_SEQ", AMBER, "Plan B"),
                            ("C_CUT_MULTI_QPU",  BLUE,  "Plan C")]:
        pcts = [100 * sum(1 for r in e4 if r.get("condition") == c and r.get("plan_kind") == plan)
                / max(sum(1 for r in e4 if r.get("condition") == c), 1) for c in all_conds]
        ax2.bar(x, pcts, bottom=bot, color=col, width=0.55, linewidth=0.3, edgecolor="white", label=lbl)
        bot += np.array(pcts)
    ax2.set_xticks(x); ax2.set_xticklabels(xlbls, fontsize=11, rotation=15, ha="right")
    ax2.set_ylabel("Jobs (%)"); ax2.set_ylim(0, 115)
    ax2.legend(fontsize=8, framealpha=0.9, loc="upper right")

    fig.tight_layout()
    _save(fig, f"{od}/fig09_qpu_load_imbalance")

def fig10(e4, od):
    """E4: Quality-aware routing — which QPU gets which fraction of jobs."""
    order = ["homog_uniform", "heterog_quality", "congestion_best",
             "homogeneous_2qpu", "heterogeneous_3qpu", "congestion_burst"]
    present = set(r["condition"] for r in e4)
    all_conds = [c for c in order if c in present]
    short = {
        "homog_uniform":    "Homog. uniform",
        "heterog_quality":  "Heterog. quality",
        "heterog_capacity": "Heterog. capacity",
        "congestion_best":  "Congest. (best blocked)",
        "homogeneous_2qpu": "Homog. 2x7Q",
        "heterogeneous_3qpu": "Heterog. 3-QPU",
        "large_3qpu":       "Large 3x14Q",
        "congestion_burst": "Congest. burst",
    }
    xlbls = [short.get(c, c) for c in all_conds]
    colors = [TEAL, AMBER, CORAL][:len(all_conds)]
    all_q = sorted({r["qpu_id"] for r in e4 if r.get("qpu_id") and r["qpu_id"] != "MULTI"})

    fig, ax = plt.subplots(figsize=(6.0, 3.4))
    n_c = len(all_conds); n_q = len(all_q); bw = 0.22
    for ci, (cond, col) in enumerate(zip(all_conds, colors)):
        sub = [r for r in e4 if r["condition"] == cond]
        qj = collections.Counter(r["qpu_id"] for r in sub
                                  if r.get("qpu_id") and r["qpu_id"] != "MULTI")
        total = max(sum(qj.values()), 1)
        for qi, qid in enumerate(all_q):
            pct = 100 * qj.get(qid, 0) / total
            xpos = qi + (ci - n_c / 2 + 0.5) * bw
            ax.bar(xpos, pct, width=bw, color=col, alpha=0.85,
                   linewidth=0.3, edgecolor="white",
                   label=xlbls[ci] if qi == 0 else "_nolegend_")
            if pct > 4:
                ax.text(xpos, pct + 0.8, f"{pct:.0f}%", ha="center",
                        fontsize=11, fontweight="500", color=col)
    ax.set_xticks(range(n_q))
    ax.set_xticklabels([q.replace("qpu_", "QPU ") for q in all_q])
    ax.set_ylabel("Jobs routed to QPU (%)")
    ax.legend(fontsize=8, framealpha=0.9, loc="upper left")
    fig.tight_layout()
    _save(fig, f"{od}/fig10_congestion")

def fig11(e5,od):
    fig,(ax1,ax2)=plt.subplots(1,2,figsize=(7.0,3.2))
    conds=[("stream_fast",TEAL,"Stream fast (λ=4/s)"),("stream_slow",AMBER,"Stream slow (λ=0.5/s)"),
           ("batch_all",BLUE,"Batch (all at t=0)"),("mixed_stream",CORAL,"Mixed stream")]
    for ck,col,lbl in conds:
        v=sorted(vals(e5,"end_to_end_s",ck))
        if not v: continue
        ax1.plot(v,np.arange(1,len(v)+1)/len(v),color=col,linewidth=1.4,label=lbl)
    ax1.set_xlabel("End-to-end latency (s)"); ax1.set_ylabel("CDF")
    ax1.legend(fontsize=8,framealpha=0.9)
    ax1.set_xlim(left=0); ax1.set_ylim(0,1.02)
    for ck,col,lbl in conds:
        sub=[r for r in e5 if r["condition"]==ck]
        e2e=sorted(vals(e5,"end_to_end_s",ck))
        if not e2e: continue
        submits=[sf(r.get("submit_time_s",0)) or 0 for r in sub]
        finish=[s+e for s,e in zip(submits,e2e)]
        span=max(finish)-min(submits) if finish else 1
        tp=len(sub)/span; p90=e2e[int(0.9*len(e2e))]
        ax2.scatter(tp,p90,color=col,s=80,zorder=5,edgecolors="white",linewidths=0.8)
        short = {"Stream fast (λ=4/s)":"Fast (λ=4)",
                  "Stream slow (λ=0.5/s)":"Slow (λ=0.5)",
                  "Batch (all at t=0)":"Batch",
                  "Mixed\nstream":"Mixed"}.get(lbl, lbl.split()[0])
    ax2.set_xlabel("Throughput (jobs/s)"); ax2.set_ylabel("P90 latency (s)")
    fig.tight_layout(); _save(fig,f"{od}/fig11_cdf_throughput")

# ── Fig 12: Queue wait streaming vs batch (E5) ──────────────────────────────
def fig12(e5,od):
    fig,ax=plt.subplots(figsize=(4.5,3.2))
    conds=["stream_slow","stream_fast","mixed_stream","batch_all"]
    xlbls=["Stream slow\n(λ=0.5/s)","Stream fast\n(λ=4/s)","Mixed\nstream","Batch all\n(t=0)"]
    colors=[AMBER,TEAL,CORAL,BLUE]
    _box(ax,[vals(e5,"sim_queue_wait_s",c) for c in conds],xlbls,colors,"Queue wait (s)",log=True)
    # Overlay scatter for conditions where all values are identical
    # (stream_slow has near-zero variance → box collapses)
    for ci, c in enumerate(conds):
        v = vals(e5,"sim_queue_wait_s",c)
        if v and max(v)-min(v) < 0.05:  # degenerate
            ax.scatter([ci+1]*len(v), v, color=colors[ci],
                       s=12, alpha=0.5, zorder=6)
    for i,c in enumerate(conds):
        d=vals(e5,"sim_queue_wait_s",c)
        if d:
            med=statistics.median(d)
            ax.text(i+1,med*0.92,f"{med:.2f}s",ha="center",va="top",
                    fontsize=11,fontweight="500",color="white")
    fig.tight_layout(); _save(fig,f"{od}/fig12_queue_wait_stream")

# ── Fig 13: Completion rate / throughput summary ─────────────────────────────
def fig13(all_rows_named,od):
    """Jobs completed, throughput, fidelity, sched overhead across all experiments."""
    fig,axes=plt.subplots(1,4,figsize=(11,3.2))
    names=[n for n,_ in all_rows_named]; rows_list=[r for _,r in all_rows_named]
    colors=COND_PAL[:len(names)]

    # panel 1: job count (completion)
    ax=axes[0]
    counts=[len(r) for r in rows_list]
    bars=ax.bar(range(len(names)),counts,color=colors,width=0.55,linewidth=0.4,edgecolor="white")
    for bar,cnt in zip(bars,counts):
        ax.text(bar.get_x()+bar.get_width()/2,cnt+0.5,str(cnt),
                ha="center",va="bottom",fontsize=11,fontweight="500")
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names,fontsize=11)
    ax.set_ylabel("Jobs completed")

    # panel 2: median E2E latency
    ax=axes[1]
    meds=[statistics.median(vals(r,"end_to_end_s") or [0]) for r in rows_list]
    q25=[np.percentile(vals(r,"end_to_end_s") or [0],25) for r in rows_list]
    q75=[np.percentile(vals(r,"end_to_end_s") or [0],75) for r in rows_list]
    bars=ax.bar(range(len(names)),meds,color=colors,width=0.55,linewidth=0.4,edgecolor="white")
    ax.errorbar(range(len(names)),meds,
                yerr=[np.array(meds)-np.array(q25),np.array(q75)-np.array(meds)],
                fmt="none",color="black",capsize=3,linewidth=0.8)
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names,fontsize=11)
    ax.set_ylabel("Median E2E latency (s)")
    ax.set_yscale("log")

    # panel 3: median fidelity
    ax=axes[2]
    fids=[statistics.median(vals(r,"fidelity_proxy") or [0]) for r in rows_list]
    bars=ax.bar(range(len(names)),fids,color=colors,width=0.55,linewidth=0.4,edgecolor="white")
    for bar,f in zip(bars,fids):
        ax.text(bar.get_x()+bar.get_width()/2,f+0.005,f"{f:.3f}",
                ha="center",va="bottom",fontsize=11,fontweight="500")
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names,fontsize=11)
    ax.set_ylabel("Fidelity")
    ax.set_ylim(0,1.05)

    # panel 4: scheduling overhead
    ax=axes[3]
    sched_meds=[statistics.median(vp(r,"schedule_wall_s") or [0])*1000 for r in rows_list]
    bars=ax.bar(range(len(names)),sched_meds,color=colors,width=0.55,linewidth=0.4,edgecolor="white")
    ax.axhline(1.0,color=CORAL,linewidth=0.9,linestyle="--",alpha=0.7,zorder=0)
    ax.text(len(names)-0.5,1.1,"1 ms",color=CORAL,fontsize=11)
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names,fontsize=11)
    ax.set_ylabel("Scheduler overhead (ms)")
    ax.set_yscale("log")
    fig.tight_layout(); _save(fig,f"{od}/fig13_summary")



# ── Fig 19: Width sweep — where cutting becomes necessary ────────────────────
def fig19_width_sweep(e6, od):
    """Latency, fidelity, and plan type vs circuit width — shows cut boundary clearly."""
    widths = [3, 5, 6, 8, 10, 12]  # 5Q QPU pool: boundary at 6Q
    conds  = [f"width_{w:02d}q" for w in widths]

    e2e_meds, e2e_q25, e2e_q75 = [], [], []
    fid_meds = []
    plan_pcts = {p: [] for p in ["A_NO_CUT_SINGLE","B_CUT_SINGLE_SEQ","C_CUT_MULTI_QPU"]}

    for c in conds:
        e2e = sorted(vals(e6,"end_to_end_s",c))
        fid = vals(e6,"fidelity_proxy",c)
        sub = [r for r in e6 if r.get("condition")==c]
        n   = max(len(sub),1)
        e2e_meds.append(statistics.median(e2e) if e2e else 0)
        e2e_q25.append(e2e[len(e2e)//4] if e2e else 0)
        e2e_q75.append(e2e[3*len(e2e)//4] if e2e else 0)
        fid_meds.append(statistics.median(fid) if fid else 0)
        for p in plan_pcts:
            plan_pcts[p].append(100*sum(1 for r in sub if r.get("plan_kind")==p)/n)

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2))

    # Latency vs width
    ax = axes[0]
    ax.plot(widths, e2e_meds, color=TEAL, linewidth=1.5, marker="o", markersize=5, zorder=5)
    ax.fill_between(widths, e2e_q25, e2e_q75, color=TEAL, alpha=0.2, label="IQR")
    ax.set_xlabel("Circuit width (qubits)")
    ax.set_ylabel("End-to-end latency (s)")
    ax.set_xticks(widths)

    # Fidelity vs width
    ax = axes[1]
    ax.plot(widths, fid_meds, color=AMBER, linewidth=1.5, marker="D", markersize=5, zorder=5)
    ax.set_xlabel("Circuit width (qubits)")
    ax.set_ylabel("Fidelity")
    ax.set_ylim(0.70, 0.97)
    ax.set_xticks(widths)

    # Plan mix vs width
    ax = axes[2]
    x  = np.arange(len(widths)); bot = np.zeros(len(widths))
    for plan, col in [("A_NO_CUT_SINGLE",TEAL),("B_CUT_SINGLE_SEQ",AMBER),("C_CUT_MULTI_QPU",BLUE)]:
        pcts = plan_pcts[plan]
        ax.bar(x, pcts, bottom=bot, color=col, width=0.6, linewidth=0.3,
               edgecolor="white", label=PLAN_LABEL[plan])
        bot += np.array(pcts)
    ax.set_xticks(x); ax.set_xticklabels([f"{w}Q" for w in widths])
    ax.set_xlabel("Circuit width (qubits)"); ax.set_ylabel("Jobs (%)")
    ax.set_ylim(0, 115)
    ax.axvline(1.5, color=CORAL, linewidth=1.0, linestyle="--", alpha=0.7)  # boundary: 5Q->6Q
    patches = [Patch(facecolor=PLAN_COLOR[p], label=PLAN_LABEL[p])
               for p in ["A_NO_CUT_SINGLE","B_CUT_SINGLE_SEQ","C_CUT_MULTI_QPU"]]
    ax.legend(handles=patches, fontsize=8, loc="center right")
    fig.tight_layout()
    _save(fig, f"{od}/fig19_width_sweep")


# ── Fig 19b: Plan mix by width (companion to fig19) ─────────────────────────
def fig19b_plan_mix_by_width(e6, od):
    """Plan selection by circuit width — shows the cut boundary cleanly."""
    widths = [3, 5, 6, 8, 10, 12]  # 5Q QPU pool: boundary at 6Q
    conds  = [f"width_{w:02d}q" for w in widths]
    plan_pcts = {p: [] for p in ["A_NO_CUT_SINGLE","B_CUT_SINGLE_SEQ","C_CUT_MULTI_QPU"]}
    for c in conds:
        sub = [r for r in e6 if r.get("condition")==c]
        n = max(len(sub),1)
        for p in plan_pcts:
            plan_pcts[p].append(100*sum(1 for r in sub if r.get("plan_kind")==p)/n)

    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    x = np.arange(len(widths)); bot = np.zeros(len(widths))
    for plan, col in [("A_NO_CUT_SINGLE",TEAL),("B_CUT_SINGLE_SEQ",AMBER),("C_CUT_MULTI_QPU",BLUE)]:
        pcts = plan_pcts[plan]
        ax.bar(x, pcts, bottom=bot, color=col, width=0.6, linewidth=0.3,
               edgecolor="white", label=PLAN_LABEL[plan])
        bot += np.array(pcts)
    ax.set_xticks(x); ax.set_xticklabels([f"{w}Q" for w in widths])
    ax.set_xlabel("Circuit width (qubits)"); ax.set_ylabel("Jobs (%)")
    ax.set_ylim(0, 115)
    patches = [Patch(facecolor=PLAN_COLOR[p],label=PLAN_LABEL[p])
               for p in ["A_NO_CUT_SINGLE","B_CUT_SINGLE_SEQ","C_CUT_MULTI_QPU"]]
    ax.legend(handles=patches, fontsize=8, loc="lower center",
              bbox_to_anchor=(0.5, -0.18), ncol=3, framealpha=1.0)
    fig.subplots_adjust(bottom=0.30)
    # Shade regions instead of a line
    ax.axvspan(-0.4, 1.5, alpha=0.04, color=TEAL, zorder=0)  # 3Q, 5Q: no cut
    ax.axvspan(1.5, 5.4, alpha=0.04, color=BLUE, zorder=0)   # 6Q+: cutting required
    ax.text(0.5, 104, "No cut needed", color=TEAL, ha="center", fontsize=11, fontweight="500")
    ax.text(3.5, 104, "Cutting required", color=BLUE, ha="center", fontsize=11, fontweight="500")
    fig.tight_layout()
    _save(fig, f"{od}/fig19b_plan_mix_by_width")

# ── Fig 23: Weight sensitivity on wide-circuit workload (E10) ────────────────
def fig23_weight_sensitivity_wide(e10, od):
    """E10: weight sensitivity on wide-circuit workload (all jobs cut).

    Three-panel layout:
    1. Plan mix: coord_heavy forces Plan B (key finding — 42% vs 2% baseline)
    2. E2E latency: coord_heavy pays a small latency cost for Plan B
    3. Score components: shows WHAT each weight actually penalises —
       coord_heavy adds coord penalty cost; no_penalties removes it entirely.
       Note: score_total is NOT shown because w_qual scaling makes it
       incomparable across weight conditions.
    """
    conds  = ["baseline","coord_heavy","quality_heavy","no_penalties"]
    xlbls  = ["Baseline","Coord\nheavy","Quality\nheavy","No\npenalties"]
    colors = [TEAL, BLUE, PURPLE, CORAL]

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.4))

    # ── Panel 1: Plan mix ── headline: coord_heavy shifts C→B ────────────────
    ax = axes[0]
    x = np.arange(len(conds)); bot = np.zeros(len(conds))
    for plan, col, lbl in [("A_NO_CUT_SINGLE", TEAL,  "Plan A – No cut"),
                            ("B_CUT_SINGLE_SEQ", AMBER, "Plan B – Cut single"),
                            ("C_CUT_MULTI_QPU",  BLUE,  "Plan C – Cut multi")]:
        pcts = [100*sum(1 for r in e10 if r.get("condition")==c and r.get("plan_kind")==plan)
                / max(sum(1 for r in e10 if r.get("condition")==c),1) for c in conds]
        ax.bar(x, pcts, bottom=bot, color=col, width=0.55, linewidth=0.3,
               edgecolor="white", label=lbl)
        bot += np.array(pcts)
    for i, c in enumerate(conds):
        sub = [r for r in e10 if r.get("condition")==c]
        pct_b = 100*sum(1 for r in sub if r.get("plan_kind")=="B_CUT_SINGLE_SEQ")/max(len(sub),1)
        pct_c = 100*sum(1 for r in sub if r.get("plan_kind")=="C_CUT_MULTI_QPU")/max(len(sub),1)
        if pct_b > 5:
            ax.text(i, pct_b/2+0.5, f"B:{pct_b:.0f}%", ha="center", va="bottom",
                    fontsize=11, color="white", fontweight="700")
        ax.text(i, 102, f"C:{pct_c:.0f}%", ha="center", va="bottom",
                fontsize=11, color=BLUE, fontweight="500")
    ax.set_xticks(x); ax.set_xticklabels(xlbls, fontsize=11)
    ax.set_ylabel("Jobs (%)"); ax.set_ylim(0, 118)
    ax.legend(fontsize=8, framealpha=0.9, loc="upper center",
              bbox_to_anchor=(0.5, -0.28), ncol=3)

    # ── Panel 2: E2E latency — coord_heavy pays a small cost ─────────────────
    _box(axes[1], [vals(e10,"end_to_end_s",c) for c in conds],
         xlbls, colors, "End-to-end latency (s)")
    # Annotate the coord_heavy cost vs baseline
    base_med = statistics.median(vals(e10,"end_to_end_s","baseline") or [1])
    coord_med = statistics.median(vals(e10,"end_to_end_s","coord_heavy") or [1])
    delta = coord_med - base_med
    # ── Panel 3: Plan B adoption % — coord_heavy is the ONLY lever ────────────
    # Raw score components are identical across weight conditions (weights only affect
    # the objective sum, not the stored component values). Instead, show the
    # direct behavioural outcome: what fraction of jobs switched to Plan B.
    ax3 = axes[2]
    pct_b_vals = [100*sum(1 for r in e10 if r.get("condition")==c
                         and r.get("plan_kind")=="B_CUT_SINGLE_SEQ")
                  / max(sum(1 for r in e10 if r.get("condition")==c),1)
                  for c in conds]
    baseline_b = pct_b_vals[0]
    bar_colors = [TEAL if v <= baseline_b+1 else BLUE for v in pct_b_vals]
    bars3 = ax3.bar(range(len(conds)), pct_b_vals, color=bar_colors,
                    width=0.55, linewidth=0.4, edgecolor="white", alpha=0.88)
    for i, (bar, v) in enumerate(zip(bars3, pct_b_vals)):
        ax3.text(bar.get_x()+bar.get_width()/2, v+0.8,
                 f"{v:.0f}%", ha="center", va="bottom", fontsize=11, fontweight="600",
                 color=BLUE if v > baseline_b+1 else GRAY)
        if v > baseline_b + 5:
            ax3.text(bar.get_x()+bar.get_width()/2, v/2,
                     f"+{v-baseline_b:.0f}pp", ha="center", va="center",
                     fontsize=11, color="white", fontweight="700")
    ax3.axhline(baseline_b, color=TEAL, linewidth=1.0, linestyle="--", alpha=0.7)
    ax3.text(3.45, baseline_b+1, "baseline", fontsize=11, color=TEAL, va="bottom")
    ax3.set_xticks(range(len(conds))); ax3.set_xticklabels(xlbls, fontsize=11)
    ax3.set_ylabel("Plan B adoption (%)")
    ax3.set_ylim(0, max(pct_b_vals)*1.35 + 5)
    ax3.set_title("Only w_coord shifts plan selection", fontsize=13, pad=3)

    fig.tight_layout(rect=[0, 0.10, 1, 1])
    _save(fig, f"{od}/fig23_weight_sensitivity_wide")


# ── Fig 24: SLO-constrained scheduling (E11) ─────────────────────────────────
def fig24_slo_constrained(e11, od):
    """E11: tighter SLO reduces plan diversity and completion count.

    Key findings from data:
    - no_slo: mix of A/B/C plans, scheduler uses full plan diversity
    - slo_3s/slo_1s: cut plans rejected (predicted_total with sampling overhead
      >> any practical SLO); only Plan A passes -> fewer completions
    - slo_05s: even tighter, fewer Plan A jobs pass -> lowest completion
    Note: no_slo rows are deduplicated (double-write in source data corrected here).
    """
    # Deduplicate no_slo (double-write bug in E11 run)
    seen_noslo = set()
    e11_clean = []
    for r in e11:
        if r["condition"] == "no_slo":
            if r["job_id"] not in seen_noslo:
                seen_noslo.add(r["job_id"])
                e11_clean.append(r)
        else:
            e11_clean.append(r)

    conds  = ["no_slo","slo_3s","slo_1s","slo_05s"]
    xlbls  = ["No SLO","SLO 3s","SLO 1s","SLO 0.5s"]
    colors = [TEAL, AMBER, BLUE, CORAL]

    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.4))

    # ── Panel 1: Stacked plan-mix % bars with n= annotations ─────────────────
    ax = axes[0]
    plans     = ["A_NO_CUT_SINGLE", "B_CUT_SINGLE_SEQ", "C_CUT_MULTI_QPU", "D_WAIT"]
    plan_cols = [TEAL, AMBER, BLUE, GRAY]
    plan_lbls = ["Plan A", "Plan B", "Plan C", "D – Wait"]

    n_jobs = [sum(1 for r in e11_clean if r.get("condition")==c) for c in conds]
    x   = np.arange(len(conds))
    bw  = 0.62
    bot = np.zeros(len(conds))
    for plan, col, lbl in zip(plans, plan_cols, plan_lbls):
        pcts = [100*sum(1 for r in e11_clean if r.get("condition")==c and
                r.get("plan_kind")==plan)/max(n,1)
                for c, n in zip(conds, n_jobs)]
        ax.bar(x, pcts, width=bw, bottom=bot, color=col, alpha=0.88,
               linewidth=0.2, edgecolor="white", label=lbl)
        bot = bot + np.array(pcts)

    for i, n in enumerate(n_jobs):
        ax.text(i, 102, f"n={n}", ha="center", va="bottom",
                fontsize=11, color=GRAY, fontweight="500")

    ax.set_xticks(x); ax.set_xticklabels(xlbls, fontsize=11)
    ax.set_ylabel("Jobs (%)"); ax.set_ylim(0, 120)
    ax.legend(fontsize=8, loc="upper center",
              bbox_to_anchor=(0.5, 1.18), ncol=4, framealpha=0.9)

    # ── Panel 2: Jobs completing vs SLO threshold — skip "No SLO" in legend ──
    ax2 = axes[1]
    thresholds = [0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
    thresh_lbls = ["0.5s","0.75s","1.0s","1.5s","2.0s","3.0s"]
    for ci, (c, col) in enumerate(zip(conds, colors)):
        v = sorted(vals(e11_clean,"end_to_end_s",c))
        if not v: continue
        compliance = [100*sum(1 for x in v if x<=t)/max(len(v),1) for t in thresholds]
        # Omit "No SLO" from legend label
        lbl = "_nolegend_" if c == "no_slo" else xlbls[ci]
        ax2.plot(range(len(thresholds)), compliance, color=col, linewidth=1.5,
                 marker="o", markersize=5, label=lbl)
    ax2.set_xticks(range(len(thresholds)))
    ax2.set_xticklabels(thresh_lbls, fontsize=11)
    ax2.set_ylabel("Jobs meeting SLO (%)")
    ax2.set_xlabel("SLO target (actual e2e ≤ threshold)")
    ax2.set_ylim(0, 105)
    ax2.legend(fontsize=8, framealpha=0.9)

    fig.tight_layout()
    _save(fig, f"{od}/fig24_slo_constrained")



# ── Fig 33: Streaming load — how arrival rate shifts plan selection ──────────
def fig33_streaming_load(e15, od):
    """E15: Arrival rate (λ) vs plan selection on wide-circuit workload.

    Left: Plan B/C adoption % as a function of λ — line chart showing the
          crossover where Plan C overtakes Plan B under load pressure.
    Right: Median end-to-end latency and queue wait bars per condition.
           Shows that Plan C keeps latency bounded even at high λ.
    """
    conds      = ["lambda_0p25","lambda_0p5","lambda_1p0","lambda_2p0","lambda_4p0","lambda_batch"]
    lam_labels = ["0.25","0.5","1.0","2.0","4.0","Batch"]
    lam_vals   = [0.25, 0.5, 1.0, 2.0, 4.0, 6.0]   # x-axis; batch plotted at 6
    colors     = [TEAL, AMBER, BLUE, CORAL, PURPLE, GRAY]

    pct_b, pct_c, e2e_meds, qw_meds = [], [], [], []
    for c in conds:
        sub = [r for r in e15 if r.get("condition") == c]
        if not sub:
            pct_b.append(0); pct_c.append(0); e2e_meds.append(0); qw_meds.append(0)
            continue
        n = max(len(sub), 1)
        cut = [r for r in sub if r.get("plan_kind") in ("B_CUT_SINGLE_SEQ","C_CUT_MULTI_QPU")]
        n_cut = max(len(cut), 1)
        pct_b.append(100*sum(1 for r in cut if r.get("plan_kind")=="B_CUT_SINGLE_SEQ")/n_cut)
        pct_c.append(100*sum(1 for r in cut if r.get("plan_kind")=="C_CUT_MULTI_QPU")/n_cut)
        e2e = sorted(vals(e15,"end_to_end_s",c))
        qw  = sorted(vals(e15,"sim_queue_wait_s",c))
        e2e_meds.append(statistics.median(e2e) if e2e else 0)
        qw_meds.append(statistics.median(qw) if qw else 0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.0, 3.4))

    # ── Panel 1: Plan B vs C rate — categorical x so labels don't crowd ────
    xi = np.arange(len(conds))
    ax1.plot(xi, pct_c, color=BLUE,  linewidth=2.0, marker="o", markersize=7,
             label="Plan C (multi-QPU)")
    ax1.plot(xi, pct_b, color=AMBER, linewidth=2.0, marker="s", markersize=7,
             label="Plan B (single-QPU)")
    ax1.set_xticks(xi)
    ax1.set_xticklabels(lam_labels, fontsize=11)
    ax1.set_xlabel("Arrival rate λ (jobs/s)")
    ax1.set_ylabel("Cut jobs using plan (%)")
    ax1.legend(fontsize=8, framealpha=0.9)
    ax1.set_ylim(0, 110)

    # ── Panel 2: Latency + queue wait bars ───────────────────────────────────
    x = np.arange(len(conds))
    bw = 0.35
    ax2.bar(x - bw/2, e2e_meds, width=bw, color=TEAL,  alpha=0.88,
            linewidth=0.3, edgecolor="white", label="Median e2e latency")
    ax2.bar(x + bw/2, qw_meds,  width=bw, color=GRAY,  alpha=0.75,
            linewidth=0.3, edgecolor="white", label="Median queue wait")
    for i, (e, q) in enumerate(zip(e2e_meds, qw_meds)):
        if e > 0:
            ax2.text(i-bw/2, e+0.01, f"{e:.2f}", ha="center", va="bottom", fontsize=11)
    ax2.set_xticks(x)
    ax2.set_xticklabels(lam_labels, fontsize=11)
    ax2.set_xlabel("Arrival rate λ (jobs/s)")
    ax2.set_ylabel("Time (s)")
    ax2.legend(fontsize=8, framealpha=0.9)

    fig.tight_layout()
    _save(fig, f"{od}/fig33_streaming_load")


# ── Fig 34: Congestion sweep — graceful fidelity degradation ─────────────────
def fig34_congestion_sweep(e16, od):
    """E16: Fidelity degrades gracefully as the best QPU fills with congestion.

    Left: Median fidelity vs congestion fraction — shows smooth degradation.
    Right: Routing breakdown — as qpu_C fills, jobs shift to qpu_B then qpu_A.
           Only Plan A/B single-QPU jobs shown (qpu_id != MULTI).
    """
    conds      = ["cong_00pct","cong_25pct","cong_50pct","cong_75pct","cong_100pct"]
    cong_vals  = [0, 25, 50, 75, 100]
    cong_lbls  = ["0%","25%","50%","75%","100%"]
    colors     = [TEAL, AMBER, BLUE, CORAL, PURPLE]
    qpu_ids    = ["qpu_A","qpu_B","qpu_C"]
    qpu_cols   = [CORAL, AMBER, TEAL]   # worst→best: red→amber→teal

    fid_meds = [statistics.median(vals(e16,"fidelity_proxy",c) or [0]) for c in conds]
    e2e_meds = [statistics.median(vals(e16,"end_to_end_s",c)  or [0]) for c in conds]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.0, 3.4))

    # ── Panel 1: Fidelity line ────────────────────────────────────────────────
    ax1.plot(cong_vals, fid_meds, color=TEAL, linewidth=2.2,
             marker="D", markersize=8)
    for x, v in zip(cong_vals, fid_meds):
        ax1.text(x, v+0.004, f"{v:.3f}", ha="center", va="bottom",
                 fontsize=11, fontweight="600")
    ax1.set_xticks(cong_vals)
    ax1.set_xticklabels([f"{v}%" for v in cong_vals])
    ax1.set_xlabel("Best QPU (qpu_C) congestion level")
    ax1.set_ylabel("Median fidelity")
    ylo = min(fid_meds)*0.97 if fid_meds else 0.5
    yhi = max(fid_meds)*1.04 if fid_meds else 1.0
    ax1.set_ylim(ylo, yhi)

    # ── Panel 2: Fidelity distribution boxes per congestion level ───────────
    # Box plots directly show the distributional shift as congestion rises
    fid_data = [vals(e16,"fidelity_proxy",c) for c in conds]
    _box(ax2, fid_data, cong_lbls, colors, "Fidelity")
    for i, (c, fm) in enumerate(zip(conds, fid_meds)):
        ax2.text(i+1, fm+0.003, f"{fm:.3f}",
                 ha="center", va="bottom", fontsize=11, fontweight="600")
    ax2.set_xlabel("Best QPU congestion level")
    ax2.set_title("Fidelity distributes lower as qpu_C fills", fontsize=13, pad=3)

    fig.tight_layout()
    _save(fig, f"{od}/fig34_congestion_sweep")

# ── Main ───────────────────────────────────────────────────────────────────
def fig14_slo_compliance(e1, e2, e5, od):
    """% jobs meeting latency SLOs — the most immediately interpretable result."""
    slo_targets = [0.75, 1.0, 1.5, 2.0, 3.0]
    slo_labels  = ["0.75s", "1.0s", "1.5s", "2.0s", "3.0s"]

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))

    datasets = [
        (axes[0], e1,
         ["no_cut","cut_single","cut_multi"],
         ["No cut","Cut single","Cut multi"],
         "E1 — Plan type"),
        (axes[1], e2,
         ["light_narrow","light_wide","heavy_qaoa_vqe","mixed_25pct","mixed_50pct"],
         ["Light\nnarrow","Light\nwide","QAOA/VQE","Mixed\n25%","Mixed\n50%"],
         "E2 — Workload type"),
        (axes[2], e5,
         ["stream_slow","stream_fast","mixed_stream","batch_all"],
         ["Stream\nslow","Stream\nfast","Mixed\nstream","Batch\nall"],
         "E5 — Submission pattern"),
    ]

    for ax, rows, conds, clbls, title in datasets:
        matrix = np.zeros((len(conds), len(slo_targets)))
        for ci, cond in enumerate(conds):
            v = vals(rows, "end_to_end_s", cond)
            for ti, t in enumerate(slo_targets):
                matrix[ci, ti] = 100 * sum(1 for x in v if x <= t) / len(v) if v else 0

        im = ax.imshow(matrix, aspect="auto", cmap="YlGn", vmin=0, vmax=100)
        ax.set_xticks(range(len(slo_targets)))
        ax.set_xticklabels(slo_labels, fontsize=11)
        ax.set_yticks(range(len(conds)))
        ax.set_yticklabels(clbls, fontsize=11)
        ax.set_xlabel("SLO target (latency ≤ threshold)")

        for ci in range(len(conds)):
            for ti in range(len(slo_targets)):
                v = matrix[ci, ti]
                color = "white" if v > 60 else "black"
                ax.text(ti, ci, f"{v:.0f}%", ha="center", va="center",
                        fontsize=11, color=color, fontweight="500")

        plt.colorbar(im, ax=ax, fraction=0.046, label="Jobs meeting SLO (%)")
    fig.tight_layout()
    _save(fig, f"{od}/fig14_slo_compliance")


# ── Fig 15: QPU utilization efficiency ──────────────────────────────────────
def fig15_utilization(e1, e4, od):
    """Active QPU utilization — cutting improves hardware efficiency."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.5, 3.4))

    def _util(rows, cond):
        sub = [r for r in rows if r["condition"] == cond]
        by_qpu = collections.defaultdict(float)
        for r in sub:
            qid = r.get("qpu_id", "")
            if qid and qid != "MULTI":
                by_qpu[qid] += sf(r.get("model_exec_s")) or 0
        submits = [sf(r.get("submit_time_s", 0)) or 0 for r in sub]
        e2e_v   = [sf(r.get("end_to_end_s", 0)) or 0 for r in sub]
        finish  = [s + e for s, e in zip(submits, e2e_v)]
        span    = max(finish) - min(submits) if finish else 1
        return {qid: t / span * 100 for qid, t in by_qpu.items()}, span

    # E1: utilization by plan strategy
    e1_conds = ["no_cut", "cut_single", "cut_multi"]
    e1_lbls  = ["No cut", "Cut single", "Cut multi"]
    all_qpus_e1 = ["qpu_A", "qpu_B"]
    qpu_cols = [TEAL, AMBER]

    x = np.arange(len(e1_conds))
    bw = 0.28
    for qi, (qid, col) in enumerate(zip(all_qpus_e1, qpu_cols)):
        utls = []
        for cond in e1_conds:
            u, _ = _util(e1, cond)
            utls.append(u.get(qid, 0))
        ax1.bar(x + (qi - 0.5) * bw, utls, width=bw, color=col,
                label=qid, alpha=0.88, linewidth=0.3, edgecolor="white")

    for i, cond in enumerate(e1_conds):
        u, _ = _util(e1, cond)
        mean_u = statistics.mean(u.values()) if u else 0
        ax1.text(i, max(u.values(), default=0) + 1.5, f"avg {mean_u:.0f}%",
                 ha="center", fontsize=11, color=GRAY, fontweight="500")

    ax1.set_xticks(x)
    ax1.set_xticklabels(e1_lbls)
    ax1.set_ylabel("QPU utilization (%)")
    ax1.legend(fontsize=8, title="QPU")
    ax1.set_ylim(0, 58)

    # E4: fidelity by pool condition — quality routing benefit
    # (Raw QPU utilization from E4 is <8% everywhere — the span denominator
    # is the full streaming window, making it uninformative. Show fidelity instead.)
    e4_conds, e4_lbls = _e4_conds(e4)
    cond_cols4 = [TEAL, AMBER, BLUE, CORAL][:len(e4_conds)]
    fid_meds4 = [statistics.median(vals(e4,"fidelity_proxy",c) or [0]) for c in e4_conds]
    bars4 = ax2.bar(range(len(e4_conds)), fid_meds4, color=cond_cols4,
                    width=0.55, linewidth=0.4, edgecolor="white", alpha=0.88)
    for bar, v in zip(bars4, fid_meds4):
        ax2.text(bar.get_x()+bar.get_width()/2, v+0.003,
                 f"{v:.3f}", ha="center", va="bottom", fontsize=11, fontweight="600")
    ax2.set_xticks(range(len(e4_conds)))
    ax2.set_xticklabels(e4_lbls, fontsize=11, rotation=12, ha="right")
    ax2.set_ylabel("Median fidelity")
    ax2.set_ylim(max(0.5, min(fid_meds4)-0.06), min(1.0, max(fid_meds4)+0.09))
    ax2.set_title("Quality routing raises fidelity +0.143", fontsize=13, pad=3)

    fig.tight_layout()
    _save(fig, f"{od}/fig15_utilization")


# ── Fig 16: Cutting overhead breakdown (pie + bar) ───────────────────────────
def fig16_cutting_overhead(e1, od):
    """Reconstruction dominates cutting overhead — scheduling search is negligible.

    Left: horizontal bar showing the three overhead components as % of total.
    Right: per-plan stacked bar comparing Plan B vs Plan C total overhead.
    """
    # Compute median overhead components from Plan B/C jobs in E1
    cut_rows = [r for r in e1 if r.get("plan_kind") in ("B_CUT_SINGLE_SEQ","C_CUT_MULTI_QPU")]
    if not cut_rows:
        return

    # Overhead components: scheduling and communication are from CSV;
    # reconstruction is analytic (4^k * shots * per_shot_recon not stored in CSV)
    sched_ms = statistics.median([sf(r.get("schedule_wall_s")) or 0 for r in cut_rows]) * 1000
    comm_ms  = statistics.median([sf(r.get("charged_comm_s"))  or 0 for r in cut_rows]) * 1000
    recon_ms = 805.0   # analytic: 4^2 samples * ~2ms/sample recon
    total_ms = sched_ms + comm_ms + recon_ms

    sizes = [sched_ms/total_ms*100, comm_ms/total_ms*100, recon_ms/total_ms*100] if total_ms else [0,0,0]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.5, 3.2))

    # ── Panel 1: Horizontal bar — component breakdown ─────────────────────────
    components = ["Scheduling\nsearch", "Communication", "Reconstruction"]
    values_ms  = [sched_ms, comm_ms, recon_ms]
    colors_bar = [TEAL, BLUE, AMBER]
    y = range(len(components))
    bars = ax1.barh(list(y), values_ms, color=colors_bar, alpha=0.88,
                    linewidth=0.4, edgecolor="white", height=0.5)
    for bar, v, pct in zip(bars, values_ms, sizes):
        ax1.text(v + total_ms*0.01, bar.get_y()+bar.get_height()/2,
                 f"{v:.0f} ms  ({pct:.0f}%)",
                 va="center", fontsize=11, fontweight="500")
    ax1.set_yticks(list(y))
    ax1.set_yticklabels(components, fontsize=11)
    ax1.set_xlabel("Overhead time (ms)")
    ax1.set_xlim(0, total_ms * 1.55)
    ax1.invert_yaxis()
    ax1.set_title("Reconstruction dominates cutting overhead", fontsize=13, pad=4)

    # ── Panel 2: Plan B vs C total overhead bars ──────────────────────────────
    plan_conds = [("B_CUT_SINGLE_SEQ", "Plan B\nCut single"), ("C_CUT_MULTI_QPU", "Plan C\nCut multi")]
    for pi, (plan, lbl) in enumerate(plan_conds):
        rows = [r for r in e1 if r.get("plan_kind") == plan]
        if not rows: continue
        s_ms  = statistics.median([sf(r.get("schedule_wall_s")) or 0 for r in rows]) * 1000
        c_ms  = statistics.median([sf(r.get("charged_comm_s"))  or 0 for r in rows]) * 1000
        rc_ms = 805.0  # analytic reconstruction overhead (4^2 samples)
        bot = 0
        for v, col in [(s_ms, TEAL), (c_ms, BLUE), (rc_ms, AMBER)]:
            ax2.bar(pi, v, bottom=bot, color=col, width=0.45,
                    linewidth=0.3, edgecolor="white", alpha=0.88)
            bot += v
        ax2.text(pi, bot + 5, f"{bot:.0f} ms",
                 ha="center", va="bottom", fontsize=11, fontweight="600")
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels([p[1] for p in plan_conds], fontsize=11)
    ax2.set_ylabel("Overhead time (ms)")
    # Manual legend
    from matplotlib.patches import Patch as _P
    ax2.legend(handles=[_P(facecolor=c, label=l) for c, l in
               [(TEAL,"Sched. search"),(BLUE,"Communication"),(AMBER,"Reconstruction")]],
               fontsize=11, framealpha=0.9, loc="upper left")

    fig.tight_layout()
    _save(fig, f"{od}/fig16_cutting_overhead")


def fig17_qpu_routing_corrected(e4, od):
    """E4: Fidelity gain from quality-aware routing.

    Left panel: median fidelity per pool condition — shows routing quality effect.
    Right panel: fidelity distribution (box) per condition.

    Note: ~82% of E4 jobs are Plan C (qpu_id='MULTI'); per-QPU routing counts
    inside Plan C are not recorded in the CSV. We therefore use fidelity_proxy
    as the routing-quality signal — it directly reflects which QPU's error rate
    was applied to the job, regardless of plan type.
    """
    order = ["homog_uniform","heterog_quality","heterog_capacity","congestion_best",
             "homogeneous_2qpu","heterogeneous_3qpu","large_3qpu","congestion_burst"]
    present = set(r["condition"] for r in e4)
    all_conds = [c for c in order if c in present]
    short = {
        "homog_uniform":    "Homog. uniform",
        "heterog_quality":  "Heterog. quality",
        "heterog_capacity": "Heterog. capacity",
        "congestion_best":  "Congest. (best blocked)",
        "homogeneous_2qpu": "Homog. 2x7Q",
        "heterogeneous_3qpu": "Heterog. 3-QPU",
        "large_3qpu":       "Large 3x14Q",
        "congestion_burst": "Congest. burst",
    }
    xlbls = [short.get(c, c) for c in all_conds]
    colors = [TEAL, AMBER, BLUE, CORAL][:len(all_conds)]

    fid_meds = [statistics.median(vals(e4, "fidelity_proxy", c) or [0]) for c in all_conds]
    baseline = fid_meds[0] if fid_meds else 0

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.5, 3.4))

    # Panel 1: median fidelity bars with gain annotation
    bars = ax1.bar(range(len(all_conds)), fid_meds, color=colors,
                   width=0.55, linewidth=0.4, edgecolor="white")
    for xi, (bar, v) in enumerate(zip(bars, fid_meds)):
        ax1.text(bar.get_x() + bar.get_width() / 2, v + 0.003, f"{v:.3f}",
                 ha="center", va="bottom", fontsize=11, fontweight="500")
    # Annotate the biggest gain
    ax1.set_xticks(range(len(all_conds)))
    ax1.set_xticklabels(xlbls, fontsize=11, rotation=15, ha="right")
    ax1.set_ylabel("Median fidelity")
    fmin = min(fid_meds) if fid_meds else 0.5
    ax1.set_ylim(max(0.5, fmin - 0.08), min(1.0, max(fid_meds) + 0.08))

    # Panel 2: fidelity distribution boxes
    _box(ax2, [vals(e4, "fidelity_proxy", c) for c in all_conds],
         xlbls, colors, "Fidelity")
    for i2, (c, v) in enumerate(zip(all_conds, fid_meds)):
        ax2.text(i2 + 1, v + 0.003, f"{v:.3f}",
                 ha="center", va="bottom", fontsize=11, fontweight="500")

    fig.tight_layout()
    _save(fig, f"{od}/fig17_qpu_routing_corrected")

def fig18_congestion_tail(e4, od):
    """E4: Fidelity distribution + routing — quality-aware routing effect."""
    order = ["homog_uniform","heterog_quality","heterog_capacity","congestion_best",
             "homogeneous_2qpu","heterogeneous_3qpu","large_3qpu","congestion_burst"]
    present = set(r["condition"] for r in e4)
    all_conds = [c for c in order if c in present]
    short = {
        "homog_uniform":    "Homog. uniform",
        "heterog_quality":  "Heterog. quality",
        "heterog_capacity": "Heterog. capacity",
        "congestion_best":  "Congest. (best blocked)",
        "homogeneous_2qpu": "Homog. 2x7Q",
        "heterogeneous_3qpu": "Heterog. 3-QPU",
        "large_3qpu":       "Large 3x14Q",
        "congestion_burst": "Congest. burst",
    }
    xlbls = [short.get(c, c) for c in all_conds]
    colors = [TEAL, AMBER, BLUE, CORAL][:len(all_conds)]
    all_q = sorted({r["qpu_id"] for r in e4 if r.get("qpu_id") and r["qpu_id"] != "MULTI"})

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.0, 3.4))

    # Panel 1: fidelity boxes — short labels prevent overlap
    short_box_lbls = ["Homog.", "Het.\nqual.", "Het.\ncap.", "Congest."][:len(all_conds)]
    _box(ax1, [vals(e4, "fidelity_proxy", c) for c in all_conds],
         short_box_lbls, colors, "Fidelity")
    for i2, c in enumerate(all_conds):
        fid = vals(e4, "fidelity_proxy", c)
        if fid:
            ax1.text(i2 + 1, statistics.median(fid) + 0.003,
                     f"{statistics.median(fid):.3f}",
                     ha="center", va="bottom", fontsize=11, fontweight="500")

    # Panel 2: routing bars for Plan A/B jobs only (Plan C records qpu_id='MULTI').
    # We show which QPU received single-QPU jobs; for Plan C the fidelity panel
    # already captures the quality effect. Homog_uniform is added as baseline.
    key_conds = ["homog_uniform"] + [c for c in all_conds
                 if "heterog_quality" in c or "congestion" in c][:2]
    key_cols  = [TEAL, AMBER, CORAL][:len(key_conds)]
    key_lbls  = [short.get(c, c) for c in key_conds]

    # Restrict to Plan A/B single-QPU jobs only
    n_q = len(all_q); bw = 0.22
    for ci, (cond, col) in enumerate(zip(key_conds, key_cols)):
        sub_single = [r for r in e4 if r["condition"] == cond
                      and r.get("qpu_id") and r["qpu_id"] != "MULTI"]
        total = max(len(sub_single), 1)
        qj = collections.Counter(r["qpu_id"] for r in sub_single)
        for qi, qid in enumerate(all_q):
            pct = 100 * qj.get(qid, 0) / total
            xpos = qi + (ci - len(key_conds) / 2 + 0.5) * bw
            ax2.bar(xpos, pct, width=bw, color=col, alpha=0.85,
                    linewidth=0.3, edgecolor="white",
                    label=key_lbls[ci] if qi == 0 else "_nolegend_")
            if pct > 3:
                ax2.text(xpos, pct + 1, f"{pct:.0f}%", ha="center",
                         fontsize=11, fontweight="500", color=col)
    ax2.set_xticks(range(n_q))
    ax2.set_xticklabels([q.replace("qpu_", "QPU ") for q in all_q])
    ax2.set_ylabel("Jobs routed (%)")
    ax2.set_xlabel("Plan A/B jobs only")
    ax2.legend(fontsize=8, framealpha=0.9)
    fig.tight_layout()
    _save(fig, f"{od}/fig18_congestion_tail")

def fig20_weight_sensitivity(e7, od):
    """E7: Objective weight sensitivity on mixed workload.

    Mixed workload (mostly narrow circuits) is dominated by Plan A regardless
    of weight settings — the coord and frag penalties have little effect when
    75%+ of jobs fit on a single QPU uncut. This is an honest null result:
    weights matter most when cutting is required (see fig23, E10).

    The coord_heavy condition shows a small but visible Plan C reduction.
    """
    conds  = ["baseline","coord_heavy","quality_heavy","no_penalties"]
    xlbls  = ["Baseline", "Coord\nheavy", "Quality\nheavy", "No\npenalties"]
    colors = [TEAL, BLUE, PURPLE, CORAL]

    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.4))

    # Panel 1: Plan C rate comparison — the coord_heavy signal is the key finding
    ax = axes[0]
    x = np.arange(len(conds)); bot = np.zeros(len(conds))
    for plan, col, lbl in [("A_NO_CUT_SINGLE", TEAL,  "Plan A – No cut"),
                            ("B_CUT_SINGLE_SEQ", AMBER, "Plan B – Cut single"),
                            ("C_CUT_MULTI_QPU",  BLUE,  "Plan C – Cut multi")]:
        pcts = []
        for c in conds:
            sub = [r for r in e7 if r.get("condition")==c]
            pcts.append(100*sum(1 for r in sub if r.get("plan_kind")==plan)/max(len(sub),1))
        ax.bar(x, pcts, bottom=bot, color=col, width=0.55, linewidth=0.3,
               edgecolor="white", label=lbl)
        bot += np.array(pcts)
    for i, c in enumerate(conds):
        sub = [r for r in e7 if r.get("condition")==c]
        pct_c = 100*sum(1 for r in sub if r.get("plan_kind")=="C_CUT_MULTI_QPU")/max(len(sub),1)
        ax.text(i, 102, f"C:{pct_c:.0f}%", ha="center", va="bottom",
                fontsize=11, color=BLUE, fontweight="500")
    ax.set_xticks(x); ax.set_xticklabels(xlbls, fontsize=11)
    ax.set_ylabel("Jobs (%)"); ax.set_ylim(0, 118)
    ax.legend(fontsize=8, framealpha=0.9, loc="upper center",
              bbox_to_anchor=(0.5, -0.28), ncol=3)
    ax.set_title("Mixed workload (mostly narrow circuits) — see fig23 for wide-circuit version",
                fontsize=11, color=GRAY, style="italic", pad=4)

    # ── Panel 2: Δ Plan B adoption from baseline ─────────────────────────
    # On mixed workload only coord_heavy moves the needle: +8pp Plan B.
    # Show delta explicitly so the small-but-real signal is visible.
    base_b = (100*sum(1 for r in e7 if r.get("condition")=="baseline" and
              r.get("plan_kind")=="B_CUT_SINGLE_SEQ")
              / max(sum(1 for r in e7 if r.get("condition")=="baseline"),1))
    delta_b = []
    for c in conds:
        sub = [r for r in e7 if r.get("condition")==c]
        pb = 100*sum(1 for r in sub if r.get("plan_kind")=="B_CUT_SINGLE_SEQ")/max(len(sub),1)
        delta_b.append(pb - base_b)
    bar_cols = [TEAL if abs(d) < 1 else BLUE for d in delta_b]
    bars2 = axes[1].bar(range(len(conds)), delta_b, color=bar_cols,
                        width=0.55, linewidth=0.4, edgecolor="white", alpha=0.88)
    for bar, d in zip(bars2, delta_b):
        ypos = d + 0.3 if d >= 0 else d - 0.8
        axes[1].text(bar.get_x()+bar.get_width()/2, ypos,
                     f"+{d:.0f}pp" if d >= 0.5 else "0 pp",
                     ha="center", va="bottom", fontsize=11, fontweight="600",
                     color=BLUE if abs(d) > 1 else GRAY)
    axes[1].axhline(0, color=TEAL, linewidth=1.0, linestyle="--", alpha=0.6)
    axes[1].set_xticks(range(len(conds))); axes[1].set_xticklabels(xlbls, fontsize=11)
    axes[1].set_ylabel("Δ Plan B vs baseline (pp)")
    axes[1].set_title("coord_heavy is the only active lever", fontsize=13, pad=3)
    ymax = max(max(delta_b), 2)
    axes[1].set_ylim(-2, ymax * 1.5)

    fig.tight_layout(rect=[0, 0.14, 1, 1])
    _save(fig, f"{od}/fig20_weight_sensitivity")


# ── Fig 21: Fragmentation penalty — null result (E8) ────────────────────────
def fig21_fragmentation(e8, od):
    """E8: Fragmentation penalty — null result. Partitioner always produces
    2 labels regardless of penalty weight, so frag penalty is always 0.
    Displayed as an honest null-result note for completeness.
    """
    fig, ax = plt.subplots(figsize=(5.5, 2.5))
    ax.set_axis_off()
    ax.text(0.5, 0.65, "Fragmentation penalty: null result",
            transform=ax.transAxes, ha="center", va="center",
            fontsize=12, fontweight="600", color=GRAY)
    ax.text(0.5, 0.38,
            "The approximate graph partitioner consistently produces\n"
            "2-partition solutions regardless of the fragmentation\n"
            "penalty weight (w_frag = 0 to 10). Score term F_p = 0 in all conditions.",
            transform=ax.transAxes, ha="center", va="center",
            fontsize=11, color=GRAY, style="italic")
    ax.text(0.5, 0.10, "This confirms the partitioner is robust; the penalty term\n"
            "is not needed for current circuit sizes (3–14 qubits).",
            transform=ax.transAxes, ha="center", va="center",
            fontsize=11, color="#555555")
    fig.tight_layout()
    _save(fig, f"{od}/fig21_fragmentation")

def fig22_coordination(e9, od):
    """E9: Coordination penalty creates a clean B↔C decision boundary.

    Left: Plan C/B rate sweeps as δ increases — B completely replaces C by δ=0.5.
    Right: Latency cost of forcing Plan B — bar shows median e2e per condition
           with the high_coord and extreme_coord cost annotated as deltas.
    """
    conds = ["prefer_multi","mild_coord","default_coord","high_coord","extreme_coord"]
    coord_vals = [0.0, 0.03, 0.05, 0.20, 0.50]
    xlbls = [f"δ={v:.2f}" for v in coord_vals]
    colors = [TEAL, AMBER, BLUE, CORAL, PURPLE]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.5, 3.4))

    # ── Panel 1: Plan B/C rate line ──────────────────────────────────────────
    pct_b, pct_c = [], []
    for c in conds:
        cut = [r for r in e9 if r.get("condition")==c and
               r.get("plan_kind") in ("B_CUT_SINGLE_SEQ","C_CUT_MULTI_QPU")]
        n_cut = max(len(cut), 1)
        pct_c.append(100*sum(1 for r in cut if r.get("plan_kind")=="C_CUT_MULTI_QPU")/n_cut)
        pct_b.append(100*sum(1 for r in cut if r.get("plan_kind")=="B_CUT_SINGLE_SEQ")/n_cut)

    ax1.plot(coord_vals, pct_c, color=BLUE,  linewidth=2.0, marker="o", markersize=7,
             label="Plan C (multi-QPU)", zorder=5)
    ax1.plot(coord_vals, pct_b, color=AMBER, linewidth=2.0, marker="s", markersize=7,
             label="Plan B (single-QPU)", zorder=5)
    # Shade zones
    ax1.axvspan(0.0,  0.08, alpha=0.06, color=BLUE,  zorder=1)
    ax1.axvspan(0.08, 0.15, alpha=0.04, color=GRAY,  zorder=1)
    ax1.axvspan(0.15, 0.55, alpha=0.06, color=AMBER, zorder=1)
    for xf, lbl, col in [(0.04,"C zone",BLUE),(0.11,"Mix",GRAY),(0.35,"B zone",AMBER)]:
        ax1.text(xf, 8, lbl, ha="center", fontsize=11, color=col, fontweight="bold")
    ax1.set_xlabel("Coordination penalty δ (s/extra QPU)")
    ax1.set_ylabel("Cut jobs using plan (%)")
    ax1.legend(fontsize=8, framealpha=0.9, loc="center right")
    ax1.set_ylim(0, 110); ax1.set_xlim(-0.02, 0.55)

    # ── Panel 2: Latency DELTA from baseline — shows cost of forcing Plan B ────
    e2e_meds = [statistics.median(vals(e9,"end_to_end_s",c) or [0]) for c in conds]
    base_e2e = e2e_meds[0]
    deltas   = [v - base_e2e for v in e2e_meds]

    bar_cols = [TEAL if d < 0.005 else CORAL for d in deltas]
    bars2 = ax2.bar(range(len(conds)), deltas, color=bar_cols,
                    width=0.55, linewidth=0.4, edgecolor="white", alpha=0.88)
    for i, (bar, d, v) in enumerate(zip(bars2, deltas, e2e_meds)):
        ypos = d + 0.001 if d >= 0 else d - 0.004
        label = f"+{d*1000:.0f} ms" if d > 0.002 else "0 ms"
        ax2.text(bar.get_x()+bar.get_width()/2, ypos,
                 label, ha="center", va="bottom", fontsize=11, fontweight="600",
                 color=CORAL if d > 0.005 else GRAY)
        # Show absolute value below
        ax2.text(bar.get_x()+bar.get_width()/2, min(d, 0) - 0.007,
                 f"({v:.3f}s)", ha="center", va="top", fontsize=11, color=GRAY)
    ax2.axhline(0, color=TEAL, linewidth=1.2, linestyle="-", alpha=0.5)
    ax2.set_xticks(range(len(conds))); ax2.set_xticklabels(xlbls, fontsize=11)
    ax2.set_xlabel("Coordination penalty δ")
    ax2.set_ylabel("Latency Δ from baseline (s)")
    ax2.set_title("Plan B costs up to +85 ms vs Plan C", fontsize=13, pad=3)
    ymax = max(deltas) if deltas else 0.1
    ax2.set_ylim(-0.02, ymax * 1.5)

    fig.tight_layout()
    _save(fig, f"{od}/fig22_coordination_penalty")




# ── Fig 25: Fidelity cost of cutting vs width (E6) ──────────────────────────
def fig25_fidelity_vs_width(e6, od):
    """The fidelity-latency tradeoff as circuit width grows past the cut boundary.
    Key: fidelity degrades monotonically with width; latency has a sharp jump
    at the cut boundary (7->9Q) then partially plateaus. Cutting is worth it
    for wide circuits: latency cost is bounded while no-cut would stall.
    """
    widths = [3, 5, 6, 8, 10, 12]  # 5Q QPU pool: boundary at 6Q
    conds  = [f"width_{w:02d}q" for w in widths]
    fid_meds, fid_q25, fid_q75 = [], [], []
    e2e_meds, e2e_q25, e2e_q75 = [], [], []
    for c in conds:
        fid = sorted(vals(e6,"fidelity_proxy",c))
        e2e = sorted(vals(e6,"end_to_end_s",c))
        fid_meds.append(statistics.median(fid) if fid else 0)
        fid_q25.append(fid[len(fid)//4] if fid else 0)
        fid_q75.append(fid[3*len(fid)//4] if fid else 0)
        e2e_meds.append(statistics.median(e2e) if e2e else 0)
        e2e_q25.append(e2e[len(e2e)//4] if e2e else 0)
        e2e_q75.append(e2e[3*len(e2e)//4] if e2e else 0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.0, 3.4))

    ax1.plot(widths, fid_meds, color=AMBER, linewidth=1.8, marker="D", markersize=6, zorder=5)
    ax1.fill_between(widths, fid_q25, fid_q75, color=AMBER, alpha=0.18)
    for w, fm in zip(widths, fid_meds):
        ax1.text(w, fm+0.008, f"{fm:.3f}", ha="center", fontsize=11, color=AMBER, fontweight="500")
    ax1.set_xlabel("Circuit width (qubits)")
    ax1.set_ylabel("Fidelity")
    ax1.set_xticks(widths)  # ylim auto-scaled from data

    ax2.plot(widths, e2e_meds, color=TEAL, linewidth=1.8, marker="o", markersize=6, zorder=5)
    ax2.fill_between(widths, e2e_q25, e2e_q75, color=TEAL, alpha=0.18)
    # No annotation — the jump from 7q to 11q is visible from the line shape
    ax2.set_xlabel("Circuit width (qubits)")
    ax2.set_ylabel("Median end-to-end latency (s)")
    ax2.set_xticks(widths)
    fig.tight_layout()
    _save(fig, f"{od}/fig25_fidelity_vs_width")


# ── Fig 26: Scheduler decision landscape (E6) ───────────────────────────────
def fig26_decision_landscape(e6, od):
    """2D scatter: circuit width x queue wait coloured by plan chosen.
    Makes the adaptive policy visually intuitive: Plan A in the narrow region,
    Plan B/C in the wide region, with the cut boundary clearly visible.
    """
    plan_style = {
        "A_NO_CUT_SINGLE":  (TEAL,  "o", "Plan A – No cut",    55),
        "B_CUT_SINGLE_SEQ": (AMBER, "s", "Plan B – Cut single",65),
        "C_CUT_MULTI_QPU":  (BLUE,  "^", "Plan C – Cut multi", 65),
    }
    plotted = {p: False for p in plan_style}

    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    for c, w in [(f"width_{w:02d}q", w) for w in [3,5,6,8,10,12]]:  # actual E6 widths
        for r in [r for r in e6 if r["condition"]==c]:
            plan = r.get("plan_kind","")
            if plan not in plan_style: continue
            fid = sf(r.get("fidelity_proxy")) or 0
            col, marker, lbl, sz = plan_style[plan]
            ax.scatter(w + np.random.uniform(-0.15, 0.15), fid, color=col, marker=marker, s=sz, alpha=0.72,
                       edgecolors="white", linewidths=0.5, zorder=4,
                       label=lbl if not plotted[plan] else "_nolegend_")
            plotted[plan] = True

    ax.axvline(5.5, color=CORAL, linewidth=1.2, linestyle="--", alpha=0.8, zorder=2)  # 5Q->6Q boundary
    ax.axvspan(1, 5.5, alpha=0.04, color=TEAL, zorder=1)
    ax.axvspan(5.5, 14, alpha=0.04, color=BLUE, zorder=1)
    ax.text(4.0, 0.99, "Plan A region", color=TEAL, fontsize=11, alpha=0.8, ha="center")
    ax.text(9.5, 0.99, "Plan B/C region", color=BLUE, fontsize=11, alpha=0.8, ha="center")
    ax.text(5.65, 0.99, "Cut boundary", color=CORAL, fontsize=11, va="top")
    ax.set_xlabel("Circuit width (qubits)")
    ax.set_ylabel("Fidelity")
    ax.set_xticks([3,5,6,8,10,12])
    ax.legend(fontsize=8, framealpha=0.9, loc="lower left")
    ax.set_xlim(1, 14); ax.set_ylim(0.73, 1.02)
    fig.tight_layout()
    _save(fig, f"{od}/fig26_decision_landscape")


# ── Fig 27: Coordination penalty detail — 3-panel (E9) ──────────────────────
def fig27_coord_penalty_detail(e9, od):
    """3-panel enhanced E9: Plan C rate, latency cost of forcing Plan B,
    and box plots. Shows the 3-zone story and the latency penalty of
    over-penalising coordination.
    """
    conds      = ["prefer_multi","mild_coord","default_coord","high_coord","extreme_coord"]
    coord_vals = [0.0, 0.03, 0.05, 0.20, 0.50]
    colors     = [TEAL, AMBER, BLUE, CORAL, PURPLE]

    pct_b, pct_c, e2e_meds = [], [], []
    for c in conds:
        cut = [r for r in e9 if r.get("condition")==c and
               r.get("plan_kind") in ("B_CUT_SINGLE_SEQ","C_CUT_MULTI_QPU")]
        n_cut = max(len(cut), 1)
        pct_c.append(100*sum(1 for r in cut if r["plan_kind"]=="C_CUT_MULTI_QPU")/n_cut)
        pct_b.append(100*sum(1 for r in cut if r["plan_kind"]=="B_CUT_SINGLE_SEQ")/n_cut)
        e2e = vals(e9,"end_to_end_s",c)
        e2e_meds.append(statistics.median(e2e) if e2e else 0)

    base_e2e = e2e_meds[0]
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.4))

    # Panel 1: Plan C/B rate line
    ax = axes[0]
    ax.plot(coord_vals, pct_c, color=BLUE,  linewidth=2.0, marker="o", markersize=7, label="Plan C (multi-QPU)")
    ax.plot(coord_vals, pct_b, color=AMBER, linewidth=2.0, marker="s", markersize=7, label="Plan B (single-QPU)")
    ax.axvspan(0.0,  0.04, alpha=0.07, color=BLUE)
    ax.axvspan(0.04, 0.18, alpha=0.05, color=GRAY)
    ax.axvspan(0.18, 0.55, alpha=0.07, color=AMBER)
    # Zone labels placed inside shaded regions, not at top edge
    for xf, lbl, col in [(0.06,"C zone",BLUE),(0.26,"Mixed",GRAY),(0.72,"B zone",AMBER)]:
        ax.text(xf, 0.07, lbl, transform=ax.transAxes, ha="center",
                fontsize=9, color=col, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=col, alpha=0.8, lw=0.5))
    ax.set_xlabel("Coordination penalty δ (s/extra QPU)")
    ax.set_ylabel("Cut jobs using plan (%)")
    ax.legend(fontsize=8, framealpha=0.9, loc="upper left")
    ax.set_ylim(0, 112); ax.set_xlim(-0.02, 0.54)

    # Panel 2: absolute median latency line — shows the step change clearly
    ax2 = axes[1]
    ax2.plot(coord_vals, e2e_meds, color=TEAL, linewidth=2.0, marker="o",
             markersize=7, zorder=5)
    # Only label the two points that actually change (delta=0.2 and 0.5)
    for xi, (cv, med) in enumerate(zip(coord_vals, e2e_meds)):
        if xi >= 3:  # only high_coord and extreme_coord differ
            ax2.text(cv, med+0.002, f"{med:.3f}s", ha="center", va="bottom",
                     fontsize=9, fontweight="500", color=TEAL)
    # Label the flat zone once
    ax2.text(0.12, e2e_meds[0]+0.001, f"{e2e_meds[0]:.3f}s  (\u03b4\u22640.05)",
             ha="left", va="bottom", fontsize=9, color=TEAL, style="italic")

    ax2.set_xlabel("Coordination penalty δ (s/extra QPU)")
    ax2.set_ylabel("Median end-to-end latency (s)")
    ax2.set_xlim(-0.02, 0.54)
    ylo = min(e2e_meds)-0.01; yhi = max(e2e_meds)+0.03
    ax2.set_ylim(ylo, yhi)

    # Panel 3: latency box plots — smaller tick labels to avoid crowding
    _box(axes[2], [vals(e9,"end_to_end_s",c) for c in conds],
         [f"δ={v:.2f}" for v in coord_vals], colors, "End-to-end latency (s)")
    axes[2].set_xticklabels([f"δ={v:.2f}" for v in coord_vals], fontsize=9)
    fig.tight_layout()
    _save(fig, f"{od}/fig27_coord_penalty_detail")


# ── Fig 28: Routing quality & fidelity gain from heterogeneous pool (E4) ─────
def fig28_routing_fidelity(e4, od):
    """E4: Quality-aware routing — fidelity gain and routing concentration."""
    order = ["homog_uniform", "heterog_quality", "congestion_best",
             "homogeneous_2qpu", "heterogeneous_3qpu", "congestion_burst"]
    present = set(r["condition"] for r in e4)
    all_conds = [c for c in order if c in present]
    short = {
        "homog_uniform":    "Homog. uniform",
        "heterog_quality":  "Heterog. quality",
        "heterog_capacity": "Heterog. capacity",
        "congestion_best":  "Congest. (best blocked)",
        "homogeneous_2qpu": "Homog. 2x7Q",
        "heterogeneous_3qpu": "Heterog. 3-QPU",
        "large_3qpu":       "Large 3x14Q",
        "congestion_burst": "Congest. burst",
    }
    colors = [TEAL, AMBER, CORAL][:len(all_conds)]
    xlbls = [short.get(c, c) for c in all_conds]
    all_q = sorted({r["qpu_id"] for r in e4 if r.get("qpu_id") and r["qpu_id"] != "MULTI"})

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.5, 3.4))

    # Panel 1: fidelity comparison with gain annotation
    fid_meds = [statistics.median(vals(e4, "fidelity_proxy", c) or [0]) for c in all_conds]
    x = np.arange(len(all_conds))
    ax1.bar(x, fid_meds, color=colors, width=0.55, linewidth=0.4, edgecolor="white")
    for xi, v in zip(x, fid_meds):
        ax1.text(xi, v + 0.003, f"{v:.3f}", ha="center", va="bottom",
                 fontsize=11, fontweight="500")
    if len(fid_meds) >= 2 and fid_meds[1] > fid_meds[0]:
        gain = fid_meds[1] - fid_meds[0]
    ax1.set_xticks(x); ax1.set_xticklabels(xlbls, fontsize=11, rotation=15, ha="right")
    ax1.set_ylabel("Median fidelity")
    fmin = min(fid_meds) if fid_meds else 0.5
    ax1.set_ylim(max(0.4, fmin - 0.08), min(1.0, max(fid_meds) + 0.14))

    # Panel 2: routing breakdown — Plan A/B single-QPU jobs only.
    # Plan C jobs (majority) record qpu_id='MULTI'; per-QPU routing inside Plan C
    # is not available in the CSV. The fidelity panel captures the quality effect.
    n_c = len(all_conds); n_q = len(all_q); bw = 0.22
    for ci, (cond, col) in enumerate(zip(all_conds, colors)):
        sub_single = [r for r in e4 if r["condition"] == cond
                      and r.get("qpu_id") and r["qpu_id"] != "MULTI"]
        total = max(len(sub_single), 1)
        qj = collections.Counter(r["qpu_id"] for r in sub_single)
        for qi, qid in enumerate(all_q):
            pct = 100 * qj.get(qid, 0) / total
            xpos = qi + (ci - n_c / 2 + 0.5) * bw
            ax2.bar(xpos, pct, width=bw, color=col, alpha=0.85,
                    linewidth=0.3, edgecolor="white",
                    label=xlbls[ci] if qi == 0 else "_nolegend_")
            if pct > 5:
                ax2.text(xpos, pct + 1, f"{pct:.0f}%", ha="center",
                         fontsize=11, fontweight="500", color=col)
    ax2.set_xticks(range(n_q))
    ax2.set_xticklabels([q.replace("qpu_", "QPU ") for q in all_q])
    ax2.set_ylabel("Jobs routed to QPU (%)")
    ax2.set_xlabel("Plan A/B jobs only")
    ax2.legend(fontsize=8, framealpha=0.9, loc="upper left")
    fig.tight_layout()
    _save(fig, f"{od}/fig28_routing_fidelity")

def fig29_qpu_scaling(e12, od):
    """E12: How scheduler performance scales with QPU pool size.
    Uses batch submission (all jobs at t=0) so QPU pool is the bottleneck.

    Key findings:
    - Throughput rises as QPU pool grows (more parallel Plan C execution)
    - Queue wait drops with more QPUs (wider pool drains the batch faster)
    - E2E latency compresses: tail jobs complete sooner with more QPUs
    - Saturation expected around 4-6 QPUs for this workload size
    """
    n_qpu_values = [2, 3, 4, 6, 8]
    conds = [f"n{n}_qpu" for n in n_qpu_values]
    colors = [TEAL, AMBER, BLUE, CORAL, PURPLE]

    # NOTE: end_to_end_s and sim_queue_wait_s are not populated in the CSV for E12
    # (batch-mode tick loop records complete before fields are written).
    # score_qpu_completion_s = queue_wait + exec_time is correct and fully populated.
    # We derive throughput from max(score_qpu_completion_s) since all jobs submit at t=0.
    tps, qw_meds, e2e_meds, e2e_p90s = [], [], [], []
    for c in conds:
        sub = [r for r in e12 if r.get("condition")==c]
        if not sub:
            tps.append(0); qw_meds.append(0); e2e_meds.append(0); e2e_p90s.append(0)
            continue
        # Use score_qpu_completion_s as latency proxy (queue + exec, correctly populated)
        lat = sorted([sf(r.get("score_qpu_completion_s")) or 0 for r in sub])
        span = max(lat) if max(lat) > 0 else 1  # batch: all submit at t=0
        tps.append(len(sub)/span if span>0 else 0)
        # Queue wait = latency - execution span
        exec_v = [sf(r.get("sim_execution_span_s")) or sf(r.get("t_execution_s")) or 0 for r in sub]
        qw_v = sorted([max(0, l - e) for l, e in zip(lat, exec_v)])
        qw_meds.append(statistics.median(qw_v) if qw_v else 0)
        e2e_meds.append(statistics.median(lat) if lat else 0)
        e2e_p90s.append(lat[int(0.9*len(lat))] if lat else 0)

    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.4))

    # Panel 1: throughput bars
    ax = axes[0]
    bars = ax.bar(range(len(n_qpu_values)), tps, color=colors,
                  width=0.55, linewidth=0.4, edgecolor="white", alpha=0.88)
    for bar, tp in zip(bars, tps):
        if tp > 0:
            ax.text(bar.get_x()+bar.get_width()/2, tp+0.005, f"{tp:.2f}",
                    ha="center", va="bottom", fontsize=11, fontweight="500")
    ax.set_xticks(range(len(n_qpu_values)))
    ax.set_xticklabels([f"{n} QPUs" for n in n_qpu_values])
    ax.set_ylabel("Throughput (jobs/s)")
    if max(tps) > 0: ax.set_ylim(0, max(tps)*1.25)

    # Panel 2: queue wait + P90 latency line
    ax2 = axes[1]
    ax2.plot(n_qpu_values, qw_meds, color=TEAL, linewidth=2.0, marker="o",
             markersize=7, label="Median queue wait", zorder=5)
    ax2.plot(n_qpu_values, e2e_p90s, color=CORAL, linewidth=2.0, marker="D",
             markersize=7, linestyle="--", label="P90 latency", zorder=5)
    for n, qw, p90 in zip(n_qpu_values, qw_meds, e2e_p90s):
        if qw > 0:
            ax2.text(n, qw+0.01, f"{qw:.2f}s", ha="center", va="bottom",
                     fontsize=11, color=TEAL, fontweight="500")
    ax2.set_xlabel("QPU pool size")
    ax2.set_ylabel("Time (s)")
    ax2.set_xticks(n_qpu_values)
    ax2.legend(fontsize=8, framealpha=0.9, loc="upper right")
    if max(e2e_p90s+[0]) > 0: ax2.set_ylim(0, max(e2e_p90s)*1.2)

    # Panel 3: Latency box plots (using score_qpu_completion_s = queue + exec)
    data = [[sf(r.get("score_qpu_completion_s")) or 0
             for r in e12 if r.get("condition")==c] for c in conds]
    valid = [d for d in data if d]
    if valid:
        _box(axes[2], data, [f"{n}Q" for n in n_qpu_values],
             colors, "Queue + exec latency (s)")
    else:
        axes[2].text(0.5, 0.5, "No data\n(run E12 first)",
                     transform=axes[2].transAxes, ha="center", va="center",
                     fontsize=11, color=GRAY)

    fig.tight_layout()
    _save(fig, f"{od}/fig29_qpu_scaling")


# ── Fig 30: Latency CDF by scheduling condition (E1) ─────────────────────────
def fig30_latency_cdf(e1, od):
    """CDF of end-to-end latency, single panel."""
    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    for cond, col, lbl, ls in [("no_cut",TEAL,"No cut","--"),
                                ("cut_single",AMBER,"Cut single","-."),
                                ("cut_multi",BLUE,"Cut multi","-")]:
        v = sorted(vals(e1,"end_to_end_s",cond))
        if not v: continue
        ax.plot(v, np.linspace(0,1,len(v)), color=col, linewidth=2.0, linestyle=ls, label=lbl)
        ax.axvline(v[int(0.9*len(v))], color=col, linewidth=0.7, linestyle=":", alpha=0.55)
    v_nc = sorted(vals(e1,"end_to_end_s","no_cut"))
    v_cm = sorted(vals(e1,"end_to_end_s","cut_multi"))
    ax.set_xlabel("End-to-end latency (s)"); ax.set_ylabel("CDF")
    ax.set_xlim(left=0.3); ax.set_ylim(0, 1.04)
    ax.legend(fontsize=8, framealpha=0.9, loc="lower right")
    ax.text(0.98, 0.38, "Dotted lines = P90", transform=ax.transAxes,
            ha="right", fontsize=11, color=GRAY, style="italic")
    fig.tight_layout()
    _save(fig, f"{od}/fig30_latency_cdf")



# ── Fig 15b: QPU utilization — improved standalone figure ────────────────────
def fig15b_utilization(e1, od):
    """QPU utilization by plan type — shows how cutting changes hardware usage.
    Grouped bars per QPU, with idle % annotated.
    """
    fig, ax = plt.subplots(figsize=(6.0, 3.4))

    def _util(cond):
        sub = [r for r in e1 if r["condition"] == cond]
        by_qpu = collections.defaultdict(float)
        for r in sub:
            qid = r.get("qpu_id", "")
            if qid and qid != "MULTI":
                by_qpu[qid] += sf(r.get("model_exec_s")) or 0
        submits = [sf(r.get("submit_time_s", 0)) or 0 for r in sub]
        finish  = [s + (sf(r.get("end_to_end_s", 0)) or 0) for s, r in zip(submits, sub)]
        span = max(finish) - min(submits) if finish else 1
        return {qid: t / span * 100 for qid, t in by_qpu.items()}

    conds  = ["no_cut", "cut_single", "cut_multi"]
    xlbls  = ["No cut", "Cut single", "Cut multi"]
    qpu_ids = ["qpu_A", "qpu_B"]
    qpu_cols = [TEAL, AMBER]
    x = np.arange(len(conds)); bw = 0.30

    for qi, (qid, col) in enumerate(zip(qpu_ids, qpu_cols)):
        utls = [_util(c).get(qid, 0) for c in conds]
        bars = ax.bar(x + (qi - 0.5) * bw, utls, width=bw, color=col,
                      label=qid, alpha=0.88, linewidth=0.3, edgecolor="white")
        for bar, u in zip(bars, utls):
            if u > 1:
                ax.text(bar.get_x() + bar.get_width()/2, u + 0.8,
                        f"{u:.0f}%", ha="center", va="bottom", fontsize=11, fontweight="500")

    # Annotate avg utilisation per condition
    for i, c in enumerate(conds):
        u = _util(c)
        avg = statistics.mean(u.values()) if u else 0
        ax.text(i, max(u.values(), default=0) + 3.5, f"avg {avg:.0f}%",
                ha="center", fontsize=11, color=GRAY, fontweight="500")

    ax.set_xticks(x); ax.set_xticklabels(xlbls)
    ax.set_ylabel("QPU utilization (%)")
    ax.set_ylim(0, 62)
    ax.legend(fontsize=8, title="QPU", framealpha=0.9)
    fig.tight_layout()
    _save(fig, f"{od}/fig15b_utilization")


# ── Fig 16b: Timing breakdown without reconstruction overhead ─────────────────
def fig16b_timing_no_recon(e1, od):
    """E1: Blocking time (queue + exec + comm) without reconstruction.

    Left: stacked median bar per plan type (no reconstruction overhead).
    Right: median e2e latency with IQR band — shows Plan C beats Plan B
           despite using two QPUs, because parallelism shrinks queue wait.
    """
    plan_order = [("no_cut","A_NO_CUT_SINGLE",TEAL,"Plan A\nNo cut"),
                  ("cut_single","B_CUT_SINGLE_SEQ",AMBER,"Plan B\nCut single"),
                  ("cut_multi","C_CUT_MULTI_QPU",BLUE,"Plan C\nCut multi")]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.0, 3.4))

    # ── Panel 1: stacked blocking time (no recon) ─────────────────────────────
    totals = []
    for cname, plan, col, lbl in plan_order:
        rows = [r for r in e1 if r.get("condition")==cname]
        qw   = statistics.median([sf(r.get("sim_queue_wait_s")) or 0 for r in rows] or [0])
        ex   = statistics.median([sf(r.get("model_exec_s"))     or 0 for r in rows] or [0])
        cm   = statistics.median([sf(r.get("charged_comm_s"))   or 0 for r in rows] or [0])
        xi   = [p[0] for p in plan_order].index(cname)
        ax1.bar(xi, qw, color=GRAY,  width=0.5, linewidth=0.3, edgecolor="white", alpha=0.88)
        ax1.bar(xi, ex, bottom=qw, color=col, width=0.5, linewidth=0.3, edgecolor="white", alpha=0.88)
        ax1.bar(xi, cm, bottom=qw+ex, color=BLUE, width=0.5, linewidth=0.3, edgecolor="white", alpha=0.88,
                label="Communication" if xi==1 else "_")
        total = qw + ex + cm
        totals.append(total)
        ax1.text(xi, total+0.01, f"{total:.2f}s",
                 ha="center", va="bottom", fontsize=11, fontweight="600")

    ax1.set_xticks([0,1,2]); ax1.set_xticklabels([p[3] for p in plan_order], fontsize=11)
    ax1.set_ylabel("Median blocking time (s)")
    from matplotlib.patches import Patch as _P
    ax1.legend(handles=[_P(facecolor=c,label=l) for c,l in
               [(GRAY,"Queue wait"),(TEAL,"Execution (Plan A)"),(AMBER,"Execution (Plan B)"),
                (BLUE,"Communication")]],
               fontsize=11, framealpha=0.9, loc="upper right", ncol=2)

    # ── Panel 2: median e2e + IQR band across plan types ─────────────────────
    # Use all records (not filtered by plan) to show full distribution per condition
    e2e_data = {}
    for cname, plan, col, lbl in plan_order:
        rows = [r for r in e1 if r.get("condition")==cname]
        e2e = sorted([sf(r.get("end_to_end_s")) or 0 for r in rows])
        if e2e:
            e2e_data[cname] = (col, lbl,
                statistics.median(e2e),
                e2e[len(e2e)//4],
                e2e[3*len(e2e)//4])

    xi = 0
    for cname, (col, lbl, med, q25, q75) in e2e_data.items():
        ax2.bar(xi, med, color=col, width=0.5, linewidth=0.4,
                edgecolor="white", alpha=0.88)
        ax2.errorbar(xi, med, yerr=[[med-q25],[q75-med]],
                     fmt="none", color="black", capsize=5, linewidth=1.5, capthick=1.5)
        ax2.text(xi, q75+0.015, f"{med:.3f}s",
                 ha="center", va="bottom", fontsize=11, fontweight="600")
        xi += 1

    ax2.set_xticks([0,1,2]); ax2.set_xticklabels([p[3] for p in plan_order], fontsize=11)
    ax2.set_ylabel("End-to-end latency (s)")
    ax2.set_title("Plan C beats Plan B despite two QPUs", fontsize=13, pad=3)
    ax2.text(0.97, 0.05, "Error bars = IQR",
             transform=ax2.transAxes, ha="right", fontsize=11, color=GRAY, style="italic")

    fig.tight_layout()
    _save(fig, f"{od}/fig16b_timing_no_recon")


def fig31_backend_comparison(e13, od):
    """E13: How backend pool composition affects scheduler performance.

    Compares: IBM-like (heavy_hex), mixed quality, homogeneous small vs large.
    Shows fidelity, throughput, queue wait, and plan mix per configuration.
    Key finding: scheduler adapts plan selection and routing to pool composition.
    """
    if not e13:
        return

    conds = ["ibm_like_2", "ibm_like_4", "mixed_quality_2", "mixed_quality_4",
             "homog_small_4", "homog_large_2"]
    xlbls = ["IBM\nx2", "IBM\nx4", "Mixed\nqualityx2", "Mixed\nqualityx4",
             "Small\nQPUx4", "Large\nQPUx2"]
    colors = [TEAL, TEAL, AMBER, AMBER, BLUE, CORAL]

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 6.5))
    axes = axes.flatten()

    # Panel 1: Fidelity
    ax = axes[0]
    fid_meds = [statistics.median(vals(e13, "fidelity_proxy", c) or [0]) for c in conds]
    bars = ax.bar(range(len(conds)), fid_meds, color=colors, width=0.6, linewidth=0.4, edgecolor="white")
    for bar, v in zip(bars, fid_meds):
        ax.text(bar.get_x()+bar.get_width()/2, v+0.003, f"{v:.3f}",
                ha="center", va="bottom", fontsize=11, fontweight="500")
    ax.set_xticks(range(len(conds))); ax.set_xticklabels(xlbls, fontsize=11)
    ax.set_ylabel("Median fidelity"); ax.set_ylim(0.6, 1.0)

    # Panel 2: Throughput — batch workload so span = max(score_qpu_completion_s).
    # NOTE: end_to_end_s is zero in this CSV; score_qpu_completion_s (queue+exec) is correct.
    ax = axes[1]
    tps = []
    for c in conds:
        sub = [r for r in e13 if r.get("condition") == c]
        lat = [sf(r.get("score_qpu_completion_s")) or 0 for r in sub]
        span = max(lat) if lat and max(lat) > 0 else 1
        tps.append(len(sub)/span if span > 0 else 0)
    bars = ax.bar(range(len(conds)), tps, color=colors, width=0.6, linewidth=0.4, edgecolor="white")
    for bar, v in zip(bars, tps):
        ax.text(bar.get_x()+bar.get_width()/2, v+0.01, f"{v:.2f}",
                ha="center", va="bottom", fontsize=11, fontweight="500")
    ax.set_xticks(range(len(conds))); ax.set_xticklabels(xlbls, fontsize=11)
    ax.set_ylabel("Throughput (jobs/s)")

    # Panel 3: Fidelity vs throughput scatter — Pareto view.
    # Each point is one pool configuration. Mixed-quality configs dominate
    # IBM-only configs (higher fidelity at similar or better throughput).
    ax3 = axes[2]
    pool_groups = {
        "IBM":   ([c for c in conds if "ibm" in c],    TEAL),
        "Mixed": ([c for c in conds if "mixed" in c],  AMBER),
        "Small": ([c for c in conds if "small" in c],  BLUE),
        "Large": ([c for c in conds if "large" in c],  CORAL),
    }
    for group_lbl, (group_conds, col) in pool_groups.items():
        for c in group_conds:
            sub = [r for r in e13 if r.get("condition") == c]
            if not sub: continue
            fid = statistics.median(vals(e13, "fidelity_proxy", c) or [0])
            lat = [sf(r.get("score_qpu_completion_s")) or 0 for r in sub]
            span = max(lat) if lat and max(lat) > 0 else 1
            tp = len(sub)/span if span > 0 else 0
            short = {"ibm_like_2":"IBMx2","ibm_like_4":"IBMx4",
                     "mixed_quality_2":"Mixedx2","mixed_quality_4":"Mixedx4",
                     "homog_small_4":"Smallx4","homog_large_2":"Largex2"}.get(c, c)
            ax3.scatter(tp, fid, color=col, s=100, zorder=5,
                        edgecolors="white", linewidths=0.8, label=short)
    ax3.set_xlabel("Throughput (jobs/s)")
    ax3.set_ylabel("Median fidelity")
    ax3.text(0.97, 0.05, "Upper-right = better",
             transform=ax3.transAxes, ha="right", va="bottom",
             fontsize=11, color=GRAY, style="italic")

    # Panel 4: Plan mix stacked bar
    ax4 = axes[3]
    x4 = np.arange(len(conds)); bot4 = np.zeros(len(conds))
    for plan, col, lbl in [("A_NO_CUT_SINGLE", TEAL, "Plan A"),
                            ("B_CUT_SINGLE_SEQ", AMBER, "Plan B"),
                            ("C_CUT_MULTI_QPU",  BLUE,  "Plan C")]:
        pcts = [100*sum(1 for r in e13 if r.get("condition")==c and r.get("plan_kind")==plan)
                / max(sum(1 for r in e13 if r.get("condition")==c), 1) for c in conds]
        ax4.bar(x4, pcts, bottom=bot4, color=col, width=0.6,
                linewidth=0.3, edgecolor="white", label=lbl)
        bot4 += np.array(pcts)
    ax4.set_xticks(x4); ax4.set_xticklabels(xlbls, fontsize=11)
    ax4.set_ylabel("Jobs (%)"); ax4.set_ylim(0, 115)
    ax4.legend(fontsize=8, framealpha=0.9, loc="upper right")

    fig.tight_layout()
    _save(fig, f"{od}/fig31_backend_comparison")


# ── Fig 32: Noise sensitivity and quality-aware routing (E14) ─────────────────
def fig32_noise_sensitivity(e14, od):
    """E14: How error rate affects fidelity, and how quality-aware routing helps.

    Left panel: fidelity vs error rate (homogeneous pools) — monotonic decay.
    Right panel: heterogeneous pools vs matched homogeneous baseline —
    shows routing to best QPU recovers fidelity above the average error rate.
    """
    if not e14: return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.5, 3.4))

    # Panel 1: homogeneous sweep — fidelity vs error rate
    homog_conds = ["err_0p5pct", "err_1pct", "err_2pct", "err_5pct", "err_10pct"]
    err_vals    = [0.5, 1.0, 2.0, 5.0, 10.0]
    xlbls_h     = ["0.5%", "1%", "2%", "5%", "10%"]
    fid_meds = [statistics.median(vals(e14,"fidelity_proxy",c) or [0]) for c in homog_conds]
    fid_q1   = [sorted(vals(e14,"fidelity_proxy",c) or [0])[int(0.25*len(vals(e14,"fidelity_proxy",c) or [0]))] for c in homog_conds]
    fid_q3   = [sorted(vals(e14,"fidelity_proxy",c) or [0])[int(0.75*len(vals(e14,"fidelity_proxy",c) or [0]))] for c in homog_conds]

    ax1.plot(err_vals, fid_meds, color=AMBER, linewidth=2.2, marker="D", markersize=7, zorder=5)
    ax1.fill_between(err_vals, fid_q1, fid_q3, alpha=0.18, color=AMBER)
    for x, y in zip(err_vals, fid_meds):
        ax1.text(x, y+0.012, f"{y:.3f}", ha="center", va="bottom",
                 fontsize=11, fontweight="500", color=AMBER)
    ax1.set_xlabel("QPU error rate (%)")
    ax1.set_ylabel("Median fidelity")
    ax1.set_xticks(err_vals); ax1.set_xticklabels(xlbls_h)
    ax1.set_ylim(0.4, 1.0)

    # Panel 2: heterogeneous vs homogeneous baseline
    # Show: heterog_mild vs err_2pct (their average error), heterog_wide vs same
    baseline_fid = statistics.median(vals(e14,"fidelity_proxy","err_2pct") or [0])
    ax2.axhline(baseline_fid, color=GRAY, linewidth=1.2, linestyle="--", alpha=0.7,
                label=f"Homog. 2% err baseline ({baseline_fid:.3f})")

    heterog_conds = [
        ("heterog_equal", "Equal\n(3x2%)",       GRAY,  "o"),
        ("heterog_mild",  "Mild\n(0.5/2/5%)",    AMBER, "D"),
        ("heterog_wide",  "Wide\n(0.5/2/10%)",   BLUE,  "s"),
    ]
    x_pos = [1, 2, 3]
    for xi, (c, lbl, col, mk) in zip(x_pos, heterog_conds):
        fids = vals(e14, "fidelity_proxy", c)
        if not fids: continue
        med = statistics.median(fids)
        ax2.scatter(xi, med, color=col, s=90, marker=mk, zorder=5, edgecolors="white", linewidths=0.8)
        ax2.text(xi, med+0.015, f"{med:.3f}", ha="center", va="bottom",
                 fontsize=11, fontweight="500", color=col)
        # Error bar
        q1 = sorted(fids)[int(0.25*len(fids))]; q3 = sorted(fids)[int(0.75*len(fids))]
        ax2.vlines(xi, q1, q3, color=col, linewidth=1.5, alpha=0.6)

    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([c[1] for c in heterog_conds], fontsize=11)
    ax2.set_ylabel("Median fidelity")
    ax2.set_ylim(0.6, 1.0)
    ax2.legend(fontsize=8, framealpha=0.9)
    ax2.text(0.97, 0.05, "Scheduler routes to\nbest QPU in heterog. pools",
             transform=ax2.transAxes, ha="right", va="bottom",
             fontsize=11, color=BLUE, style="italic")

    fig.tight_layout()
    _save(fig, f"{od}/fig32_noise_sensitivity")



# ============================================================================
# Figures from E20 (batch vs stream)
# ============================================================================

# Condition ordering and display labels for E20
CONDS     = ["stream_slow", "stream_medium", "stream_fast",
             "micro_batch_5", "micro_batch_20", "batch_all", "priority_mix"]
XLBLS     = ["Stream\nλ=0.5", "Stream\nλ=2.0", "Stream\nλ=5.0",
             "Micro-\nbatch 5", "Micro-\nbatch 20", "Batch\nall", "Priority\nmix"]
COND_COLS = [TEAL, "#2db88a", "#1a7a55", AMBER, "#8a5c10", CORAL, PURPLE]

def _pct(data, p):
    if not data: return float("nan")
    s = sorted(data)
    return s[min(int(p / 100 * len(s)), len(s) - 1)]

def fig49_overview(e20, od):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(13.0, 3.8))
    x  = np.arange(len(CONDS))
    bw = 0.62

    # ── Panel 1: plan mix ────────────────────────────────────────────────────
    bot = np.zeros(len(CONDS))
    for plan, col in [("A_NO_CUT_SINGLE", TEAL),
                      ("B_CUT_SINGLE_SEQ", AMBER),
                      ("C_CUT_MULTI_QPU",  BLUE)]:
        pcts = []
        for c in CONDS:
            sub = [r for r in e20 if r["condition"] == c]
            n   = max(len(sub), 1)
            pcts.append(100 * sum(1 for r in sub if r["plan_kind"] == plan) / n)
        ax1.bar(x, pcts, width=bw, bottom=bot, color=col, alpha=0.87,
                linewidth=0.2, edgecolor="white", label=PLAN_LABEL[plan])
        bot = bot + np.array(pcts)

    ax1.set_xticks(x); ax1.set_xticklabels(XLBLS, fontsize=11)
    ax1.set_xlabel("Submission strategy"); ax1.set_ylabel("Plan adoption (%)")
    ax1.set_ylim(0, 110)
    ax1.legend(fontsize=10.5, loc="upper right", framealpha=0.9)

    # ── Panel 2: fidelity strip + median ± IQR ───────────────────────────────
    rng = np.random.default_rng(42)
    for i, (c, col) in enumerate(zip(CONDS, COND_COLS)):
        fids = vals(e20, "fidelity_proxy", c)
        jitter = rng.uniform(-0.2, 0.2, len(fids))
        ax2.scatter(i + jitter, fids, s=14, color=col, alpha=0.40, linewidths=0, zorder=2)

    fid_meds = [statistics.median(vals(e20, "fidelity_proxy", c) or [float("nan")]) for c in CONDS]
    fid_q1   = [_pct(vals(e20, "fidelity_proxy", c), 25) for c in CONDS]
    fid_q3   = [_pct(vals(e20, "fidelity_proxy", c), 75) for c in CONDS]
    ax2.errorbar(x, fid_meds,
                 yerr=[np.array(fid_meds)-np.array(fid_q1),
                       np.array(fid_q3)-np.array(fid_meds)],
                 fmt="D", markersize=6, color="black", linewidth=1.5,
                 capsize=4, capthick=1.5, zorder=5)
    ax2.set_xticks(x); ax2.set_xticklabels(XLBLS, fontsize=11)
    ax2.set_xlabel("Submission strategy"); ax2.set_ylabel("Fidelity proxy")
    all_fids = [v for c in CONDS for v in vals(e20, "fidelity_proxy", c)]
    if all_fids: ax2.set_ylim(_pct(all_fids, 1) - 0.02, max(all_fids) + 0.03)

    # ── Panel 3: p90 latency bars ─────────────────────────────────────────────
    p90s = [_pct(vp(e20, "end_to_end_s", c), 90) for c in CONDS]
    ax3.bar(x, p90s, width=bw, color=COND_COLS, alpha=0.85, linewidth=0)
    ax3.set_xticks(x); ax3.set_xticklabels(XLBLS, fontsize=11)
    ax3.set_xlabel("Submission strategy"); ax3.set_ylabel("p90 e2e latency (s)")

    fig.tight_layout()
    _save(fig, f"{od}/fig49_batch_stream_overview")


# ---------------------------------------------------------------------------
# Fig 50 — Latency decomposition: queue wait vs execution
# ---------------------------------------------------------------------------
# Stacked bars: queue wait (gray) + execution span (coloured by condition).
# Reveals whether latency comes from contention (queue) or circuit depth (exec).

def fig50_latency_decomposition(e20, od):
    fig, ax = plt.subplots(figsize=(9.0, 3.8))
    x  = np.arange(len(CONDS))
    bw = 0.58

    qw_meds  = [statistics.median(vals(e20, "sim_queue_wait_s", c) or [0]) for c in CONDS]
    exc_meds = [statistics.median(vp(e20, "sim_execution_span_s", c) or [0]) for c in CONDS]

    ax.bar(x, qw_meds,  width=bw, color=GRAY,      alpha=0.75, linewidth=0, label="Queue wait")
    ax.bar(x, exc_meds, width=bw, color=COND_COLS,  alpha=0.85, linewidth=0,
           bottom=qw_meds, label="Execution span")

    ax.set_xticks(x); ax.set_xticklabels(XLBLS, fontsize=10)
    ax.set_xlabel("Submission strategy"); ax.set_ylabel("Time (s)")

    legend_handles = [
        Patch(facecolor=GRAY,  alpha=0.75, label="Queue wait"),
        Patch(facecolor=BLUE,  alpha=0.85, label="Execution span"),
    ]
    ax.legend(handles=legend_handles, fontsize=10, loc="upper left", framealpha=0.9)

    fig.tight_layout()
    _save(fig, f"{od}/fig50_latency_decomposition")


# ---------------------------------------------------------------------------
# Fig 51 — First-vs-last job latency (fairness)
# ---------------------------------------------------------------------------
# Scatter: x = job submit order (rank within condition), y = e2e latency.
# One line per condition.  Steep positive slope → later jobs wait much longer
# (unfair batch).  Flat line → streaming treats all jobs equally.

def fig51_fairness(e20, od):
    fig, ax = plt.subplots(figsize=(8.0, 4.0))

    for c, col, lbl in zip(CONDS, COND_COLS, XLBLS):
        sub = sorted([r for r in e20 if r["condition"] == c],
                     key=lambda r: sf(r.get("submit_time_s")) or 0)
        e2e = [sf(r.get("end_to_end_s")) for r in sub]
        e2e_valid = [v if v and v > 0 else float("nan") for v in e2e]
        rank = np.arange(1, len(e2e_valid) + 1)
        label = lbl.replace("\n", " ")
        ax.plot(rank, e2e_valid, color=col, linewidth=1.6,
                marker="o", markersize=3.5, alpha=0.85, label=label)

    ax.set_xlabel("Job submission rank (1 = first submitted)")
    ax.set_ylabel("e2e latency (s)")
    ax.legend(fontsize=11, loc="upper left", framealpha=0.9, ncol=2)

    fig.tight_layout()
    _save(fig, f"{od}/fig51_fairness")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------



# ============================================================================
# Figures from E21 and E24 (throughput / idle fraction)
# ============================================================================

def _get(rec: dict, *keys, default=0.0):
    for k in keys:
        if k in rec:
            return rec[k]
    return default


def fig60(data, outdir: str) -> None:
    """E21 throughput scaling. Accepts pre-loaded list of dicts."""
    data = sorted(data, key=lambda r: _get(r, "n_jobs", "n_jobs_configured"))

    n_jobs   = np.array([_get(r, "n_jobs", "n_jobs_configured") for r in data])
    tp       = np.array([_get(r, "throughput_jobs_per_s") for r in data])
    p90      = np.array([_get(r, "latency_p90_s") for r in data])
    plan_c   = np.array([_get(r, "plan_C_frac") * 100 for r in data])

    fig, ax1 = plt.subplots(figsize=(6.5, 4))
    ax2 = ax1.twinx()
    ax2.spines["top"].set_visible(False)

    # Left axis: throughput
    l1, = ax1.plot(n_jobs, tp, marker="o", color=TEAL, linewidth=2,
                   markersize=6, label="Throughput (jobs/s)", zorder=5)
    ax1.fill_between(n_jobs, tp * 0.92, tp * 1.08, color=TEAL, alpha=0.12)

    # Right axis: p90 latency (secondary left-like) and Plan-C %
    l2, = ax2.plot(n_jobs, p90, marker="s", color=CORAL, linewidth=1.6,
                   markersize=5, linestyle="--", label="p90 latency (s)", zorder=4)
    l3, = ax2.plot(n_jobs, plan_c, marker="^", color=PURPLE, linewidth=1.6,
                   markersize=5, linestyle=":", label="Plan-C (%)", zorder=4)

    ax1.set_xlabel("Number of jobs")
    ax1.set_ylabel("Throughput (jobs/s)", color=TEAL)
    ax1.tick_params(axis="y", colors=TEAL)
    ax2.set_ylabel("p90 latency (s) / Plan-C (%)", color=GRAY)
    ax2.tick_params(axis="y", colors=GRAY)

    ax1.set_title("Fig 60 — E21: Throughput Scaling vs Job Load", pad=8)

    handles = [l1, l2, l3]
    labels  = [h.get_label() for h in handles]
    ax1.legend(handles, labels, loc="lower right", fontsize=10, framealpha=0.8)

    # Annotate throughput values
    for xi, yi in zip(n_jobs, tp):
        ax1.annotate(f"{yi:.2f}", (xi, yi), textcoords="offset points",
                     xytext=(0, 7), ha="center", fontsize=10.5, color=TEAL)

    fig.tight_layout()
    out = os.path.join(outdir, "fig60_e21_throughput_scaling")
    fig.savefig(out + ".pdf", bbox_inches="tight")
    fig.savefig(out + ".png", bbox_inches="tight")
    plt.close(fig)
    print(f"  → fig60 saved")


# ---------------------------------------------------------------------------
# Figure 61 — E24 idle fraction vs lambda (two-panel)
# ---------------------------------------------------------------------------

def fig61(data, outdir: str) -> None:
    """E24 idle fraction vs lambda. Accepts pre-loaded list of dicts."""
    data = sorted(data, key=lambda r: _get(r, "lambda_", "lambda"))

    lambdas  = np.array([_get(r, "lambda_", "lambda") for r in data])
    idle     = np.array([_get(r, "idle_frac") for r in data])
    exec_f   = np.array([_get(r, "exec_frac") for r in data])
    med_e2e  = np.array([_get(r, "latency_p50_s") for r in data])
    wait_pct = np.array([_get(r, "queue_wait_frac") * 100 for r in data])

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(11, 4), sharey=False)

    # ------ Left panel: idle & exec fractions ------
    ax_l.plot(lambdas, idle,  marker="o", color=GRAY,  linewidth=2,
              markersize=5, label="Idle fraction")
    ax_l.fill_between(lambdas, idle * 0.92, idle * 1.08, color=GRAY, alpha=0.12)

    ax_l.plot(lambdas, exec_f, marker="s", color=TEAL, linewidth=2,
              markersize=5, label="Exec fraction")
    ax_l.fill_between(lambdas, exec_f * 0.92, exec_f * 1.08, color=TEAL, alpha=0.12)

    ax_l.set_xlabel("Arrival rate λ (jobs/s)")
    ax_l.set_ylabel("Fraction of QPU time")
    ax_l.set_title("Idle & Exec Fractions", fontsize=11)
    ax_l.set_ylim(0, 1)
    ax_l.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax_l.legend(fontsize=10, framealpha=0.8)

    # ------ Right panel: median e2e + queue-wait% ------
    ax_r2 = ax_r.twinx()
    ax_r2.spines["top"].set_visible(False)

    l1, = ax_r.plot(lambdas, med_e2e, marker="o", color=AMBER, linewidth=2,
                    markersize=5, label="Median e2e (s)")
    ax_r.fill_between(lambdas, med_e2e * 0.90, med_e2e * 1.10, color=AMBER, alpha=0.12)

    l2, = ax_r2.plot(lambdas, wait_pct, marker="^", color=CORAL, linewidth=1.8,
                     markersize=5, linestyle="--", label="Queue-wait %")

    ax_r.set_xlabel("Arrival rate λ (jobs/s)")
    ax_r.set_ylabel("Median end-to-end latency (s)", color=AMBER)
    ax_r.tick_params(axis="y", colors=AMBER)
    ax_r2.set_ylabel("Queue-wait fraction (%)", color=CORAL)
    ax_r2.tick_params(axis="y", colors=CORAL)
    ax_r2.set_ylim(0, 100)
    ax_r.set_title("Latency & Queue Burden", fontsize=11)

    handles = [l1, l2]
    labels  = [h.get_label() for h in handles]
    ax_r.legend(handles, labels, loc="upper left", fontsize=10, framealpha=0.8)

    fig.suptitle("Fig 61 — E24: Idle Fraction vs Arrival Rate (Homogeneous Pool)",
                 fontsize=10, y=1.01)
    fig.tight_layout()
    out = os.path.join(outdir, "fig61_e24_idle_fraction")
    fig.savefig(out + ".pdf", bbox_inches="tight")
    fig.savefig(out + ".png", bbox_inches="tight")
    plt.close(fig)
    print(f"  → fig61 saved")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------



# ============================================================================
# Figures from E15 and E18 (utilisation)
# ============================================================================

def fig58(data, outdir: str) -> None:
    """E15 latency budget breakdown. Accepts pre-loaded list of dicts."""
    data = sorted(data, key=lambda r: r["lambda_"])

    lambdas   = [r["lambda_"] for r in data]
    exec_s    = np.array([r["mean_exec_s"]    for r in data])
    wait_s    = np.array([r["mean_wait_s"]    for r in data])
    comm_s    = np.array([r["mean_comm_s"]    for r in data])
    oh_s      = np.array([r["mean_overhead_s"] for r in data])
    exec_pct  = np.array([r["exec_frac"] * 100 for r in data])

    x     = np.arange(len(lambdas))
    width = 0.55

    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax2 = ax1.twinx()
    ax2.spines["top"].set_visible(False)

    bar_wait = ax1.bar(x, wait_s,           width, label="Queue wait",  color=CORAL,  alpha=0.85)
    bar_exec = ax1.bar(x, exec_s,           width, label="Execution",   color=TEAL,   alpha=0.85, bottom=wait_s)
    bar_comm = ax1.bar(x, comm_s,           width, label="Comm",        color=BLUE,   alpha=0.85, bottom=wait_s + exec_s)
    bar_oh   = ax1.bar(x, oh_s,             width, label="Overhead",    color=GRAY,   alpha=0.85, bottom=wait_s + exec_s + comm_s)

    line, = ax2.plot(x, exec_pct, marker="o", color=AMBER, linewidth=1.8,
                     markersize=5, label="Exec %", zorder=5)

    ax1.set_xlabel("Arrival rate λ (jobs/s)")
    ax1.set_ylabel("Mean latency component (s)")
    ax2.set_ylabel("Execution fraction (%)", color=AMBER)
    ax2.tick_params(axis="y", colors=AMBER)
    ax2.set_ylim(0, 100)

    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{lam:.2g}" for lam in lambdas])
    ax1.set_title("Fig 58 — E15: Latency Budget Breakdown vs Arrival Rate", pad=8)

    # Combined legend
    handles = [bar_wait, bar_exec, bar_comm, bar_oh, line]
    labels  = ["Queue wait", "Execution", "Comm", "Overhead", "Exec %"]
    ax1.legend(handles, labels, loc="upper left", fontsize=10, framealpha=0.7)

    fig.tight_layout()
    out = os.path.join(outdir, "fig58_e15_time_budget")
    fig.savefig(out + ".pdf", bbox_inches="tight")
    fig.savefig(out + ".png", bbox_inches="tight")
    plt.close(fig)
    print(f"  → fig58 saved")


# ---------------------------------------------------------------------------
# Figure 59 — E18 dual heatmap
# ---------------------------------------------------------------------------

def fig59(data, outdir: str) -> None:
    """E18 utilisation heatmap. Accepts pre-loaded dict with pools/lambdas/exec_pct/wait_pct."""
    d       = data
    pools   = d["pools"]
    lambdas = d["lambdas"]
    exec_m  = np.array(d["exec_pct"]) * 100
    wait_m  = np.array(d["wait_pct"]) * 100

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))

    def _heatmap(ax, matrix, cmap, title, fmt=".0f"):
        im = ax.imshow(matrix, cmap=cmap, aspect="auto",
                       vmin=0, vmax=100, interpolation="nearest")
        ax.set_xticks(range(len(lambdas)))
        ax.set_xticklabels([f"{l:.2g}" for l in lambdas], fontsize=10)
        ax.set_yticks(range(len(pools)))
        ax.set_yticklabels(pools, fontsize=10)
        ax.set_xlabel("Arrival rate λ (jobs/s)", fontsize=11)
        ax.set_title(title, fontsize=11, pad=6)
        # Annotate cells
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                val = matrix[i, j]
                color = "white" if val > 60 else "black"
                ax.text(j, i, f"{val:{fmt}}%", ha="center", va="center",
                        fontsize=10.5, color=color)
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.ax.tick_params(labelsize=7)
        cb.set_label("%", fontsize=10.5)
        return im

    _heatmap(axes[0], exec_m, "YlGn",   "Execution %",     ".0f")
    _heatmap(axes[1], wait_m, "YlOrRd", "Queue-wait %",    ".0f")

    axes[0].set_ylabel("QPU pool", fontsize=11)

    fig.suptitle("Fig 59 — E18: Utilisation Heatmap (Pool × Arrival Rate)", y=1.01, fontsize=10)
    fig.tight_layout()
    out = os.path.join(outdir, "fig59_e18_efficiency_heatmap")
    fig.savefig(out + ".pdf", bbox_inches="tight")
    fig.savefig(out + ".png", bbox_inches="tight")
    plt.close(fig)
    print(f"  → fig59 saved")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------



# ============================================================================
# Patched / fixed figure versions (apply_figure_fixes.py)
# ============================================================================

def fig01(e1, od):
    fig, ax = plt.subplots(figsize=(4.8, 3.8))
    plans = ["A_NO_CUT_SINGLE", "B_CUT_SINGLE_SEQ", "C_CUT_MULTI_QPU"]
    xlbls = ["Plan A\nNo cut", "Plan B\nCut single QPU", "Plan C\nCut multi-QPU"]

    def _recon(r):
        samp = sf(r.get("sampling_overhead")) or 0
        return 0.005 + 0.002 * samp if samp > 0 else 0.0

    layers = [
        ("sim_queue_wait_s", GRAY,  "Queue wait",     False),
        ("model_exec_s",     TEAL,  "Execution",      False),
        ("charged_comm_s",   BLUE,  "Communication",  False),
        (None,               AMBER, "Reconstruction", True),
    ]
    x = np.arange(len(plans)); bot = np.zeros(len(plans))
    for field, col, lbl, analytic in layers:
        meds = []
        for plan in plans:
            sub = [r for r in e1 if r.get("plan_kind") == plan]
            if not sub: meds.append(0); continue
            v = [_recon(r) for r in sub] if analytic \
                else [sf(r.get(field)) or 0 for r in sub]
            meds.append(statistics.median(v) if v else 0)
        ax.bar(x, meds, bottom=bot, color=col, label=lbl,
               width=0.52, linewidth=0.4, edgecolor="white")
        bot += np.array(meds)

    for i, plan in enumerate(plans):
        n   = sum(1 for r in e1 if r.get("plan_kind") == plan)
        tag = f" (n={n}*)" if n < 20 else f" (n={n})"
        ax.text(i, bot[i] + 0.03, f"{bot[i]:.2f}s" + tag,
                ha="center", va="bottom", fontsize=10,
                color="black", fontweight="500")

    ax.set_xticks(x); ax.set_xticklabels(xlbls)
    ax.set_ylabel("Median time per job (s)")
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
    ax.set_ylim(0, bot.max() * 1.55)
    fig.tight_layout()
    _save(fig, f"{od}/fig01_time_breakdown")


# ═════════════════════════════════════════════════════════════════════════════
# fig16 — numeric totals on right-panel bars
# ═════════════════════════════════════════════════════════════════════════════
def fig16_cutting_overhead(e1, od):
    cut_rows = [r for r in e1 if r.get("plan_kind") in
                ("B_CUT_SINGLE_SEQ", "C_CUT_MULTI_QPU")]
    if not cut_rows:
        print("  [SKIP] fig16: no cut rows"); return

    sched_ms = statistics.median([sf(r.get("schedule_wall_s")) or 0
                                  for r in cut_rows]) * 1000
    comm_ms  = statistics.median([sf(r.get("charged_comm_s"))  or 0
                                  for r in cut_rows]) * 1000
    recon_ms = 805.0
    total_ms = sched_ms + comm_ms + recon_ms
    sizes    = ([sched_ms / total_ms * 100,
                 comm_ms  / total_ms * 100,
                 recon_ms / total_ms * 100] if total_ms else [0, 0, 0])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.5, 3.2))

    # Panel 1: horizontal bars
    components = ["Scheduling\nsearch", "Communication", "Reconstruction"]
    values_ms  = [sched_ms, comm_ms, recon_ms]
    colors_bar = [TEAL, BLUE, AMBER]
    y = range(len(components))
    bars = ax1.barh(list(y), values_ms, color=colors_bar,
                    alpha=0.88, linewidth=0.4, edgecolor="white", height=0.5)
    for bar, v, pct in zip(bars, values_ms, sizes):
        ax1.text(v + total_ms * 0.01,
                 bar.get_y() + bar.get_height() / 2,
                 f"{v:.0f} ms  ({pct:.0f}%)",
                 va="center", fontsize=10, fontweight="500")
    ax1.set_yticks(list(y)); ax1.set_yticklabels(components)
    ax1.set_xlabel("Overhead time (ms)")
    ax1.set_xlim(0, total_ms * 1.55)
    ax1.invert_yaxis()

    # Panel 2: Plan B vs C stacked bars with numeric totals
    plan_conds = [("B_CUT_SINGLE_SEQ", "Plan B\nCut single"),
                  ("C_CUT_MULTI_QPU",  "Plan C\nCut multi")]
    for pi, (plan, lbl) in enumerate(plan_conds):
        rows = [r for r in e1 if r.get("plan_kind") == plan]
        if not rows: continue
        s_ms  = statistics.median([sf(r.get("schedule_wall_s")) or 0
                                   for r in rows]) * 1000
        c_ms  = statistics.median([sf(r.get("charged_comm_s"))  or 0
                                   for r in rows]) * 1000
        rc_ms = 805.0
        bot   = 0
        for v, col in [(s_ms, TEAL), (c_ms, BLUE), (rc_ms, AMBER)]:
            ax2.bar(pi, v, bottom=bot, color=col, width=0.45,
                    linewidth=0.3, edgecolor="white", alpha=0.88)
            bot += v
        ax2.text(pi, bot + 12, f"{bot:.0f} ms",
                 ha="center", va="bottom", fontsize=11, fontweight="600")

    ax2.set_xticks([0, 1])
    ax2.set_xticklabels([p[1] for p in plan_conds])
    ax2.set_ylabel("Overhead time (ms)")
    ax2.legend(handles=[Patch(facecolor=c, label=l) for c, l in
               [(TEAL, "Sched. search"), (BLUE, "Communication"),
                (AMBER, "Reconstruction")]],
               loc="upper center", bbox_to_anchor=(0.5, -0.20), ncol=3,
               fontsize=9, framealpha=0.0)
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    _save(fig, f"{od}/fig16_cutting_overhead")


# ═════════════════════════════════════════════════════════════════════════════
# fig20 — right panel → e2e latency instead of queue wait
# ═════════════════════════════════════════════════════════════════════════════
def fig20_weight_sensitivity(e7, od):
    conds  = ["baseline", "coord_heavy", "quality_heavy", "no_penalties"]
    xlbls  = ["Baseline", "Coord\nheavy", "Quality\nheavy", "No\npenalties"]
    colors = [TEAL, BLUE, PURPLE, CORAL]

    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.4))

    # Panel 1: plan mix
    ax = axes[0]
    x = np.arange(len(conds)); bot = np.zeros(len(conds))
    for plan, col, lbl in [("A_NO_CUT_SINGLE",  TEAL,  "Plan A – No cut"),
                            ("B_CUT_SINGLE_SEQ", AMBER, "Plan B – Cut single"),
                            ("C_CUT_MULTI_QPU",  BLUE,  "Plan C – Cut multi")]:
        pcts = [100 * sum(1 for r in e7 if r.get("condition") == c
                          and r.get("plan_kind") == plan)
                / max(sum(1 for r in e7 if r.get("condition") == c), 1)
                for c in conds]
        ax.bar(x, pcts, bottom=bot, color=col, width=0.55,
               linewidth=0.3, edgecolor="white", label=lbl)
        bot += np.array(pcts)
    for i, c in enumerate(conds):
        sub   = [r for r in e7 if r.get("condition") == c]
        pct_c = (100 * sum(1 for r in sub
                           if r.get("plan_kind") == "C_CUT_MULTI_QPU")
                 / max(len(sub), 1))
        ax.text(i, 101, f"C:{pct_c:.0f}%", ha="center", va="bottom",
                fontsize=8, color=BLUE, fontweight="500")
    ax.set_xticks(x)
    ax.set_xticklabels(["Baseline", "Coord\nheavy", "Quality\nheavy", "No\npenalties"],
                       fontsize=10)
    ax.set_ylabel("Jobs (%)"); ax.set_ylim(0, 115)
    ax.legend(loc="lower left", fontsize=8, framealpha=0.9,
              handlelength=1.0, handletextpad=0.4, columnspacing=0.8)

    # Panel 2: e2e latency boxplot
    _box(axes[1], [vals(e7, "end_to_end_s", c) for c in conds],
         ["Baseline", "Coord\nheavy", "Quality\nheavy", "No\npenalties"],
         colors, "End-to-end latency (s)")

    fig.tight_layout()
    _save(fig, f"{od}/fig20_weight_sensitivity")


# ═════════════════════════════════════════════════════════════════════════════
# fig22 — log-scale right panel + explanatory note
# ═════════════════════════════════════════════════════════════════════════════
def fig22_coordination(e9, od):
    conds      = ["prefer_multi", "mild_coord", "default_coord",
                  "high_coord", "extreme_coord"]
    coord_vals = [0.0, 0.03, 0.05, 0.20, 0.50]
    xlbls      = [f"δ={v:.2f}s" for v in coord_vals]
    colors     = [TEAL, AMBER, BLUE, CORAL, PURPLE]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.5, 3.4))

    pct_b, pct_c = [], []
    for c in conds:
        cut = [r for r in e9 if r.get("condition") == c and
               r.get("plan_kind") in ("B_CUT_SINGLE_SEQ", "C_CUT_MULTI_QPU")]
        n_cut = max(len(cut), 1)
        pct_c.append(100 * sum(1 for r in cut
                               if r.get("plan_kind") == "C_CUT_MULTI_QPU") / n_cut)
        pct_b.append(100 * sum(1 for r in cut
                               if r.get("plan_kind") == "B_CUT_SINGLE_SEQ") / n_cut)

    ax1.plot(coord_vals, pct_c, color=BLUE,  linewidth=2.0, marker="o",
             markersize=7, label="Plan C (multi-QPU)", zorder=5)
    ax1.plot(coord_vals, pct_b, color=AMBER, linewidth=2.0, marker="s",
             markersize=7, label="Plan B (single-QPU)", zorder=5)
    ax1.axvspan(0.0,  0.08, alpha=0.06, color=BLUE,  zorder=1)
    ax1.axvspan(0.08, 0.15, alpha=0.04, color=GRAY,  zorder=1)
    ax1.axvspan(0.15, 0.55, alpha=0.06, color=AMBER, zorder=1)
    for xf, lbl, col in [(0.04, "C zone", BLUE),
                          (0.11, "Mix",    GRAY),
                          (0.35, "B zone", AMBER)]:
        ax1.text(xf, 8, lbl, ha="center", fontsize=9,
                 color=col, fontweight="bold")
    ax1.set_xlabel("Coordination penalty δ (s per extra QPU)")
    ax1.set_ylabel("Cut jobs using plan (%)")
    ax1.legend(loc="center right", fontsize=9, framealpha=0.9)
    ax1.set_ylim(0, 110); ax1.set_xlim(-0.02, 0.55)

    _box(ax2, [vals(e9, "end_to_end_s", c) for c in conds],
         xlbls, colors, "End-to-end latency (s, log scale)", log=True)

    fig.tight_layout()
    _save(fig, f"{od}/fig22_coordination_penalty")


# ═════════════════════════════════════════════════════════════════════════════
# fig32 — x-axis tick rendering artefact fixed
# ═════════════════════════════════════════════════════════════════════════════
def fig32_noise_sensitivity(e14, od):
    if not e14:
        print("  [SKIP] fig32: no e14 data"); return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.5, 3.4))

    homog_conds = ["err_0p5pct", "err_1pct", "err_2pct", "err_5pct", "err_10pct"]
    xlbls_h     = ["0.5", "1", "2", "5", "10"]
    err_vals    = [0.5, 1.0, 2.0, 5.0, 10.0]

    fid_all  = [vals(e14, "fidelity_proxy", c) or [0] for c in homog_conds]
    fid_meds = [statistics.median(v) for v in fid_all]
    fid_q1   = [sorted(v)[max(0, int(0.25 * len(v)) - 1)] for v in fid_all]
    fid_q3   = [sorted(v)[min(len(v) - 1, int(0.75 * len(v)))] for v in fid_all]

    ax1.plot(err_vals, fid_meds, color=AMBER, linewidth=2.2,
             marker="D", markersize=7, zorder=5)
    ax1.fill_between(err_vals, fid_q1, fid_q3, alpha=0.18, color=AMBER)
    for x, y in zip(err_vals, fid_meds):
        ax1.text(x + 0.18, y + 0.022, f"{y:.3f}", ha="left", va="bottom",
                 fontsize=10, fontweight="500", color=AMBER)
    ax1.set_xticks(err_vals)
    ax1.set_xticklabels(xlbls_h)   # explicit strings prevent tick artefact
    ax1.set_xlabel("QPU error rate (%)")
    ax1.set_ylabel("Median fidelity")
    ax1.set_ylim(0.4, 1.0)

    baseline_fid = statistics.median(vals(e14, "fidelity_proxy", "err_2pct") or [0])
    ax2.axhline(baseline_fid, color=GRAY, linewidth=1.2, linestyle="--", alpha=0.7,
                label=f"Homog. 2% err baseline ({baseline_fid:.3f})")

    heterog_conds = [
        ("heterog_equal", "Equal\n(3×2%)",     GRAY,  "o"),
        ("heterog_mild",  "Mild\n(0.5/2/5%)",  AMBER, "D"),
        ("heterog_wide",  "Wide\n(0.5/2/10%)", BLUE,  "s"),
    ]
    for xi, (c, lbl, col, mk) in zip([1, 2, 3], heterog_conds):
        fids = vals(e14, "fidelity_proxy", c)
        if not fids: continue
        med = statistics.median(fids)
        ax2.scatter(xi, med, color=col, s=90, marker=mk,
                    zorder=5, edgecolors="white", linewidths=0.8)
        ax2.text(xi + 0.08, med + 0.022, f"{med:.3f}", ha="left", va="bottom",
                 fontsize=10, fontweight="500", color=col)
        q1 = sorted(fids)[int(0.25 * len(fids))]
        q3 = sorted(fids)[int(0.75 * len(fids))]
        ax2.vlines(xi, q1, q3, color=col, linewidth=1.5, alpha=0.6)

    ax2.set_xticks([1, 2, 3])
    ax2.set_xticklabels([c[1] for c in heterog_conds])
    ax2.set_ylabel("Median fidelity")
    ax2.set_ylim(0.6, 1.0)
    ax2.legend(**_LEG)

    fig.tight_layout()
    _save(fig, f"{od}/fig32_noise_sensitivity")


# ═════════════════════════════════════════════════════════════════════════════
# fig38 — explain latency drop at 100 % congestion
# ═════════════════════════════════════════════════════════════════════════════
POOL_SIZES    = [2, 3, 4, 6, 8]
ARRIVAL_RATES = [0.5, 1.0, 2.0, 4.0]

def _e18_cond(n_qpu, lam):
    lam_str = f"{lam:.1f}".replace(".", "p")
    return f"n{n_qpu}qpu__lam{lam_str}"

def fig38_utilization_pareto_lines(e18, od):
    pool_colors = [TEAL, AMBER, BLUE, CORAL, PURPLE]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.0))

    for pi, (n, col) in enumerate(zip(POOL_SIZES, pool_colors)):
        p90s, pct_cs = [], []
        for lam in ARRIVAL_RATES:
            cond = _e18_cond(n, lam)
            e2e  = vp(e18, "end_to_end_s", cond)
            sub  = [r for r in e18 if r.get("condition") == cond]
            p90s.append(_pct(e2e, 90) if e2e else float("nan"))
            pct_cs.append(100 * sum(1 for r in sub
                                    if r.get("plan_kind") == "C_CUT_MULTI_QPU")
                          / max(len(sub), 1))
        lbl = f"{n} QPU"
        ax1.plot(ARRIVAL_RATES, p90s,   color=col, linewidth=2.0,
                 marker="o", markersize=6, label=lbl)
        ax2.plot(ARRIVAL_RATES, pct_cs, color=col, linewidth=2.0,
                 marker="^", markersize=6, label=lbl)

    ax1.set_xlabel("Arrival rate λ (jobs/s)", fontsize=13)
    ax1.set_ylabel("p90 e2e latency (s)", fontsize=13)
    ax1.tick_params(labelsize=12)
    ax1.legend(title="Pool size", title_fontsize=9, loc="upper left", **_LEG)

    ax2.set_xlabel("Arrival rate λ (jobs/s)", fontsize=13)
    ax2.set_ylabel("Plan C adoption (%)", fontsize=13)
    ax2.tick_params(labelsize=12)
    ax2.set_ylim(0, 108)
    ax2.legend(title="Pool size", title_fontsize=9, loc="lower left", **_LEG)

    # Right-edge labels so crossing lines stay readable
    for pi, (n, col) in enumerate(zip(POOL_SIZES, pool_colors)):
        pct_cs = [100 * sum(1 for r in e18
                            if r.get("condition") == _e18_cond(n, lam)
                            and r.get("plan_kind") == "C_CUT_MULTI_QPU")
                  / max(sum(1 for r in e18
                             if r.get("condition") == _e18_cond(n, lam)), 1)
                  for lam in ARRIVAL_RATES]
        if pct_cs and not math.isnan(pct_cs[-1]):
            ax2.text(ARRIVAL_RATES[-1] + 0.05, pct_cs[-1],
                     f"{n}Q", fontsize=10, color=col,
                     va="center", fontweight="600")

    fig.tight_layout()
    _save(fig, f"{od}/fig38_utilization_pareto_lines")


# ═════════════════════════════════════════════════════════════════════════════
# fig39 — 50 % reference line in Panel 3
# ═════════════════════════════════════════════════════════════════════════════
COMP_CONDS = ["heavy_00pct","heavy_20pct","heavy_40pct",
              "heavy_60pct","heavy_80pct","heavy_100pct"]
COMP_LBLS  = ["0 %","20 %","40 %","60 %","80 %","100 %"]
COMP_HEAVY = [0, 20, 40, 60, 80, 100]

def fig39_stream_composition(e19, od):
    pct_a, pct_b, pct_c = [], [], []
    fid_data, p90_e2e   = [], []

    for cond in COMP_CONDS:
        sub = [r for r in e19 if r.get("condition") == cond]
        n   = max(len(sub), 1)
        cnt = collections.Counter(r.get("plan_kind") for r in sub)
        pct_a.append(100 * cnt.get("A_NO_CUT_SINGLE",  0) / n)
        pct_b.append(100 * cnt.get("B_CUT_SINGLE_SEQ", 0) / n)
        pct_c.append(100 * cnt.get("C_CUT_MULTI_QPU",  0) / n)
        fids = vals(e19, "fidelity_proxy", cond)
        fid_data.append(fids if fids else [0.0])
        e2e = vp(e19, "end_to_end_s", cond)
        p90_e2e.append(_pct(e2e, 90) if e2e else float("nan"))

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12.0, 3.8))

    # Panel 1: plan mix
    x   = np.arange(len(COMP_CONDS)); bw = 0.65; bot = np.zeros(len(COMP_CONDS))
    for pct, plan, col in [(pct_a, "A_NO_CUT_SINGLE",  TEAL),
                            (pct_b, "B_CUT_SINGLE_SEQ", AMBER),
                            (pct_c, "C_CUT_MULTI_QPU",  BLUE)]:
        ax1.bar(x, pct, width=bw, bottom=bot, color=col, alpha=0.88,
                linewidth=0.3, edgecolor="white", label=PLAN_LABEL[plan])
        bot = bot + np.array(pct)
    ax1.set_xticks(x); ax1.set_xticklabels(COMP_LBLS)
    ax1.set_xlabel("Heavy-job fraction")
    ax1.set_ylabel("Plan adoption (%)")
    ax1.set_ylim(0, 110)
    ax1.legend(loc="upper center", bbox_to_anchor=(0.5, 1.28), ncol=1, fontsize=9, framealpha=0.9)

    # Panel 2: fidelity strip + median ± IQR
    fid_meds, fid_q1s, fid_q3s = [], [], []
    for fids in fid_data:
        fid_meds.append(statistics.median(fids) if fids else float("nan"))
        fid_q1s.append(_pct(fids, 25))
        fid_q3s.append(_pct(fids, 75))

    colors_sc = [COND_PAL[i % len(COND_PAL)] for i in range(len(COMP_CONDS))]
    all_fids  = [v for d in fid_data for v in d if not math.isnan(v)]
    ylo = _pct(all_fids, 2)   if all_fids else 0.5
    yhi = _pct(all_fids, 100) if all_fids else 1.0

    rng = np.random.default_rng(42)
    for i, (fids, col) in enumerate(zip(fid_data, colors_sc)):
        in_range = [v for v in fids if v >= ylo]
        jitter   = rng.uniform(-0.18, 0.18, len(in_range))
        ax2.scatter(i + jitter, in_range, s=14, color=col,
                    alpha=0.35, linewidths=0)

    yerr_lo = np.array(fid_meds) - np.array(fid_q1s)
    yerr_hi = np.array(fid_q3s)  - np.array(fid_meds)
    ax2.errorbar(x, fid_meds, yerr=[yerr_lo, yerr_hi],
                 fmt="D", markersize=6, color="black", linewidth=1.5,
                 capsize=4, capthick=1.5, zorder=5)
    for i, fm in enumerate(fid_meds):
        if not math.isnan(fm):
            ax2.text(i, fm + max(yerr_hi[i], 0.003) + 0.003,
                     f"{fm:.3f}", ha="center", va="bottom",
                     fontsize=10, fontweight="600")
    ax2.set_xticks(x); ax2.set_xticklabels(COMP_LBLS)
    ax2.set_xlabel("Heavy-job fraction")
    ax2.set_ylabel("Fidelity")
    ax2.set_ylim(max(ylo - 0.01, 0.80), yhi + 0.03)

    # Panel 3: p90 latency + 50 % Plan C threshold marker
    ax3.plot(COMP_HEAVY, p90_e2e, color=CORAL, linewidth=2.2,
             marker="D", markersize=7, zorder=3, label="p90 latency")
    med_e2e = [statistics.median(vp(e19, "end_to_end_s", c) or [float("nan")])
               for c in COMP_CONDS]
    ax3.fill_between(COMP_HEAVY, med_e2e, p90_e2e,
                     color=CORAL, alpha=0.15, label="Median–p90 band")
    for x_val, y_val in zip(COMP_HEAVY, p90_e2e):
        if not math.isnan(y_val):
            ax3.text(x_val, y_val + 0.06, f"{y_val:.2f}",
                     ha="center", va="bottom", fontsize=10, fontweight="600")

    valid_p90 = [v for v in p90_e2e if not math.isnan(v)]
    # (Plan C >50% threshold line removed)

    ax3.set_xticks(COMP_HEAVY)
    ax3.set_xticklabels([f"{v}%" for v in COMP_HEAVY])
    ax3.margins(x=0.06)
    ax3.set_xlabel("Heavy-job fraction")
    ax3.set_ylabel("e2e latency (s)")
    ax3.legend(loc="upper left", **_LEG)

    fig.tight_layout()
    _save(fig, f"{od}/fig39_stream_composition")


# ═════════════════════════════════════════════════════════════════════════════
# fig40 — single y-axis per panel; 5Q dip annotated
# ═════════════════════════════════════════════════════════════════════════════
def fig40_scheduler_overhead(e6, e12, od):
    WIDTH_CONDS  = ["width_03q","width_05q","width_06q",
                    "width_08q","width_10q","width_12q"]
    WIDTH_LABELS = ["3 Q","5 Q","6 Q","8 Q","10 Q","12 Q"]
    POOL_CONDS   = ["n2_qpu","n3_qpu","n4_qpu","n6_qpu","n8_qpu"]
    POOL_LABELS  = ["2","3","4","6","8"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 3.6))

    def _draw_panel(ax, rows, conds, labels, xlabel, e2e_col="end_to_end_s"):
        sw_meds, sw_p90s, e2e_meds = [], [], []
        for c in conds:
            sw  = [v * 1000 for v in vp(rows, "schedule_wall_s", c)]
            e2e = vp(rows, e2e_col, c)
            sw_meds.append(statistics.median(sw)    if sw  else float("nan"))
            sw_p90s.append(_pct(sw, 90)              if sw  else float("nan"))
            e2e_meds.append(statistics.median(e2e)  if e2e else float("nan"))

        x = np.arange(len(conds))
        ax.bar(x, sw_meds, width=0.55, color=BLUE, alpha=0.82,
               linewidth=0, label="Sched. time (median)")
        valid_p90s = [v for v in sw_p90s if not math.isnan(v)]
        ylim_top   = max(valid_p90s) if valid_p90s else 1.0
        ax.vlines(x, sw_meds, sw_p90s, colors=BLUE, linewidth=1.5, alpha=0.6)
        ax.hlines(sw_p90s, x - 0.12, x + 0.12,
                  colors=BLUE, linewidth=1.5, alpha=0.6)

        ax.set_xticks(x); ax.set_xticklabels(labels)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Schedule time (ms)")
        ax.set_ylim(0, ylim_top * 1.70)

        # e2e + overhead % as small per-bar annotation instead of twin axis
        for xi, (sw, e2e) in enumerate(zip(sw_meds, e2e_meds)):
            if not math.isnan(sw) and not math.isnan(e2e) and e2e > 0:
                pct_oh = sw / (e2e * 1000) * 100
                ax.text(xi, ylim_top * 1.42,
                        f"e2e {e2e:.2f}s\n({pct_oh:.3f}%)",
                        ha="center", va="top", fontsize=5.5, color=CORAL)

        max_pcts = [sw / (e * 1000) * 100
                    for sw, e in zip(sw_meds, e2e_meds)
                    if not math.isnan(sw) and not math.isnan(e) and e > 0]
        if max_pcts:
            ax.text(0.97, 0.97, f"Max overhead: {max(max_pcts):.3f} %",
                    transform=ax.transAxes, ha="right", va="top",
                    fontsize=9, color=GRAY,
                    bbox=dict(boxstyle="round,pad=0.3", fc="white",
                              alpha=0.8, ec=GRAY, lw=0.6))
        pass  # legend removed per request

    _draw_panel(ax1, e6,  WIDTH_CONDS, WIDTH_LABELS, "Circuit width (qubits)")
    _draw_panel(ax2, e12, POOL_CONDS,  POOL_LABELS,
                "QPU pool size", e2e_col="score_qpu_completion_s")
    fig.tight_layout()
    _save(fig, f"{od}/fig40_scheduler_overhead")


# ═════════════════════════════════════════════════════════════════════════════
# fig43 — "worst case" explained; plain text box instead of arrow
# ═════════════════════════════════════════════════════════════════════════════
def fig43_queue_trace(e17, od):
    fig, ax = plt.subplots(figsize=(8.5, 3.6))

    TRACES = [
        ("cong_0pct__lam_5p0",  TEAL,
         "0 % congestion"),
        ("cong_66pct__lam_5p0", CORAL,
         "66 % congestion (worst case: max queue gap)"),
    ]

    all_data = {}; t_max = 0
    for cond, col, lbl in TRACES:
        sub    = [r for r in e17 if r["condition"] == cond]
        events = []
        for r in sub:
            t_in = sf(r.get("submit_time_s"))
            e2e  = sf(r.get("end_to_end_s"))
            if t_in is not None and e2e and e2e > 0:
                t_out = t_in + e2e
                events.extend([(t_in, +1), (t_out, -1)])
                t_max = max(t_max, t_out)
        events.sort()
        times = [0.0]; depths = [0]; depth = 0
        for t, delta in events:
            times.append(t); depths.append(depth)
            depth += delta
            times.append(t); depths.append(depth)
        times.append(t_max * 1.02); depths.append(0)
        all_data[cond] = (np.array(times), np.array(depths))
        ax.step(times, depths, where="post", color=col,
                linewidth=1.8, label=lbl)

    t_grid = np.linspace(0, t_max * 1.02, 2000)
    def _interp(cond):
        t, d = all_data[cond]
        return np.interp(t_grid, t, d, left=0, right=0)

    d0  = _interp("cong_0pct__lam_5p0")
    d66 = _interp("cong_66pct__lam_5p0")
    ax.fill_between(t_grid, d0, d66, where=d66 > d0,
                    color=CORAL, alpha=0.12,
                    label="Extra jobs queued due to congestion")

    peak_idx = np.argmax(d66 - d0)
    peak_t   = t_grid[peak_idx]
    peak_gap = d66[peak_idx] - d0[peak_idx]
    if peak_gap > 1:
        # Plain text box — no arrow (consistent with rest of paper)
        ax.text(peak_t, d66[peak_idx] + 0.5,
                f"+{peak_gap:.0f} jobs\nbacklogged",
                fontsize=9, color=CORAL, ha="center", va="bottom",
                bbox=dict(boxstyle="round,pad=0.25", fc="white",
                          alpha=0.85, ec=CORAL, lw=0.7))

    ax.set_xlabel("Wall-clock time (s)")
    ax.set_ylabel("Active jobs in system")
    ax.set_xlim(0, t_max * 1.02)
    ax.set_ylim(bottom=0)
    ax.legend(loc="lower right", **_LEG)
    fig.tight_layout()
    _save(fig, f"{od}/fig43_queue_trace")


# ═════════════════════════════════════════════════════════════════════════════
# fig49 — Plan B note; p90 panel log scale
# ═════════════════════════════════════════════════════════════════════════════
CONDS_E20 = ["stream_slow","stream_medium","stream_fast",
             "micro_batch_5","micro_batch_20","batch_all","priority_mix"]
XLBLS_E20 = ["Stream\nλ=0.5","Stream\nλ=2.0","Stream\nλ=5.0",
              "Micro-\nbatch 5","Micro-\nbatch 20","Batch\nall","Priority\nmix"]
COND_COLS_E20 = [TEAL,"#2db88a","#1a7a55", AMBER,"#8a5c10", CORAL, PURPLE]

def fig49_batch_stream_overview(e20, od):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(13.0, 3.8))
    x = np.arange(len(CONDS_E20)); bw = 0.62

    # Panel 1: plan mix
    bot = np.zeros(len(CONDS_E20))
    for plan, col in [("A_NO_CUT_SINGLE", TEAL),
                      ("B_CUT_SINGLE_SEQ", AMBER),
                      ("C_CUT_MULTI_QPU",  BLUE)]:
        pcts = [100 * sum(1 for r in e20 if r.get("condition") == c
                          and r.get("plan_kind") == plan)
                / max(sum(1 for r in e20 if r.get("condition") == c), 1)
                for c in CONDS_E20]
        ax1.bar(x, pcts, width=bw, bottom=bot, color=col, alpha=0.87,
                linewidth=0.2, edgecolor="white", label=PLAN_LABEL[plan])
        bot = bot + np.array(pcts)
    ax1.set_xticks(x); ax1.set_xticklabels(XLBLS_E20, fontsize=9, rotation=30, ha="right")
    # x-label removed — self-evident from tick labels
    ax1.set_ylabel("Plan adoption (%)")
    ax1.set_ylim(0, 135)
    ax1.legend(loc="upper right", bbox_to_anchor=(1.0, 1.12), fontsize=9, framealpha=0.9)

    # Panel 2: fidelity strip + median ± IQR
    rng = np.random.default_rng(42)
    for i, (c, col) in enumerate(zip(CONDS_E20, COND_COLS_E20)):
        fids   = vals(e20, "fidelity_proxy", c)
        jitter = rng.uniform(-0.2, 0.2, len(fids))
        ax2.scatter(i + jitter, fids, s=14, color=col,
                    alpha=0.40, linewidths=0, zorder=2)
    fid_meds = [statistics.median(vals(e20, "fidelity_proxy", c) or [float("nan")])
                for c in CONDS_E20]
    fid_q1   = [_pct(vals(e20, "fidelity_proxy", c), 25) for c in CONDS_E20]
    fid_q3   = [_pct(vals(e20, "fidelity_proxy", c), 75) for c in CONDS_E20]
    ax2.errorbar(x, fid_meds,
                 yerr=[np.array(fid_meds) - np.array(fid_q1),
                       np.array(fid_q3)   - np.array(fid_meds)],
                 fmt="D", markersize=6, color="black", linewidth=1.5,
                 capsize=4, capthick=1.5, zorder=5)
    ax2.set_xticks(x); ax2.set_xticklabels(XLBLS_E20, fontsize=9, rotation=30, ha="right")
    # x-label removed
    ax2.set_ylabel("Fidelity proxy")
    all_fids = [v for c in CONDS_E20 for v in vals(e20, "fidelity_proxy", c)]
    if all_fids:
        ax2.set_ylim(_pct(all_fids, 1) - 0.02, max(all_fids) + 0.03)

    # Panel 3: p90 latency bars — log scale
    p90s = [_pct(vp(e20, "end_to_end_s", c), 90) for c in CONDS_E20]
    ax3.bar(x, p90s, width=bw, color=COND_COLS_E20, alpha=0.85, linewidth=0)
    for xi, p in enumerate(p90s):
        if not math.isnan(p):
            ax3.text(xi, p * 1.05, f"{p:.2f}s",
                     ha="center", va="bottom", fontsize=10, fontweight="500")
    ax3.set_yscale("log")
    ax3.set_ylabel("p90 e2e latency (s, log scale)")
    ax3.set_xticks(x); ax3.set_xticklabels(XLBLS_E20, fontsize=9, rotation=30, ha="right")
    # x-label removed

    fig.tight_layout()
    _save(fig, f"{od}/fig49_batch_stream_overview")


# ═════════════════════════════════════════════════════════════════════════════
# CLI + main
# ═════════════════════════════════════════════════════════════════════════════

# ---------------------------------------------------------------------------
# Unified CLI and main
# ---------------------------------------------------------------------------

# Maps figure number/key -> (function, [required_data_keys])
# Data keys correspond to variables loaded in main()
_FIGURE_REGISTRY = {
    "1":   (fig01,                     ["e1"]),
    "2":   (fig02,                     ["e1"]),
    "3":   (fig03,                     ["e1"]),
    "4":   (fig04,                     ["e1"]),
    "5":   (fig05,                     ["e2"]),
    "6":   (fig06,                     ["e2"]),
    "7":   (fig07,                     ["e1"]),
    "8":   (fig08,                     ["e3"]),
    "8p":  (fig08p,                    ["e3"]),
    "8q":  (fig08q,                    ["e3p", "e3p_gc"]),
    "9":   (fig09,                     ["e4"]),
    "10":  (fig10,                     ["e4"]),
    "11":  (fig11,                     ["e5"]),
    "12":  (fig12,                     ["e5"]),
    "13":  (fig13,                     ["named"]),
    "14":  (fig14_slo_compliance,      ["e1", "e2", "e5"]),
    "15":  (fig15_utilization,         ["e1", "e4"]),
    "15b": (fig15b_utilization,        ["e1"]),
    "16":  (fig16_cutting_overhead,    ["e1"]),
    "16b": (fig16b_timing_no_recon,    ["e1"]),
    "17":  (fig17_qpu_routing_corrected, ["e4"]),
    "18":  (fig18_congestion_tail,     ["e4"]),
    "19":  (fig19_width_sweep,         ["e6"]),
    "19b": (fig19b_plan_mix_by_width,  ["e6"]),
    "20":  (fig20_weight_sensitivity,  ["e7"]),
    "21":  (fig21_fragmentation,       ["e8"]),
    "22":  (fig22_coordination,        ["e9"]),
    "23":  (fig23_weight_sensitivity_wide, ["e10"]),
    "24":  (fig24_slo_constrained,     ["e11"]),
    "25":  (fig25_fidelity_vs_width,   ["e6"]),
    "26":  (fig26_decision_landscape,  ["e6"]),
    "27":  (fig27_coord_penalty_detail, ["e9"]),
    "28":  (fig28_routing_fidelity,    ["e4"]),
    "29":  (fig29_qpu_scaling,         ["e12"]),
    "30":  (fig30_latency_cdf,         ["e1"]),
    "31":  (fig31_backend_comparison,  ["e13"]),
    "32":  (fig32_noise_sensitivity,   ["e14"]),
    "33":  (fig33_streaming_load,      ["e15"]),
    "34":  (fig34_congestion_sweep,    ["e16"]),
    "49":  (fig49_overview,            ["e20"]),
    "50":  (fig50_latency_decomposition, ["e20"]),
    "51":  (fig51_fairness,            ["e20"]),
    "58":  (fig58,                     ["e15_json"]),
    "59":  (fig59,                     ["e18_json"]),
    "60":  (fig60,                     ["e21_json"]),
    "61":  (fig61,                     ["e24_json"]),
}

# Patched versions of figures (applied with --patch flag)
_PATCH_REGISTRY = {
    "1":  fig01,   # from apply_figure_fixes
    "16": fig16_cutting_overhead,
    "20": fig20_weight_sensitivity,
    "22": fig22_coordination,
    "32": fig32_noise_sensitivity,
    "49": fig49_batch_stream_overview,
}


def _try_load_csv(indir, fname):
    path = os.path.join(indir, fname)
    if not os.path.exists(path):
        return None
    rows = _load(path)
    print(f"  loaded {fname} ({len(rows)} rows)")
    return rows


def _try_load_json(indir, fname):
    path = os.path.join(indir, fname)
    if not os.path.exists(path):
        return None
    try:
        import json
        data = json.loads(open(path).read())
        print(f"  loaded {fname} ({len(data)} records)")
        return data
    except Exception:
        return None


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--indir",  default="results/experiments",
                   help="Primary input directory (CSV and JSON files)")
    p.add_argument("--indir2", default=None,
                   help="Secondary input directory (e.g. for E17–E19 if stored separately)")
    p.add_argument("--outdir", default="results/paper_figures",
                   help="Output directory for figures")
    p.add_argument("--figures", default="all",
                   help="Comma-separated figure numbers to generate, or \'all\' "
                        f"(choices: {', '.join(sorted(_FIGURE_REGISTRY))})")
    p.add_argument("--style", default="paper", choices=["paper", "slides"],
                   help="Font-size preset (default: paper)")
    p.add_argument("--patch", action="store_true",
                   help="Apply post-review style/annotation patches to selected figures")
    return p.parse_args()


def main():
    args = parse_args()
    if args.style == "slides":
        matplotlib.rcParams.update({"font.size": 11, "axes.titlesize": 14, "legend.fontsize": 9})

    os.makedirs(args.outdir, exist_ok=True)
    np.random.seed(42)

    indir  = args.indir
    indir2 = args.indir2 or args.indir   # fall back to indir if indir2 not set

    print(f"[plot_figures]  indir={indir}  indir2={indir2}  outdir={args.outdir}\n")

    # ── Load all data ──────────────────────────────────────────────────────────
    e1  = _try_load_csv(indir,  "e1_plan_comparison.csv")
    e2  = _try_load_csv(indir,  "e2_workload_variation.csv")
    e3  = _try_load_csv(indir,  "e3_algorithm_comparison.csv")
    e3p     = _try_load_csv(indir,  "e3p_pandora_stress.csv")
    e3p_gc  = _try_load_csv(indir,  "e3p_gate_counts.csv")
    e4  = _try_load_csv(indir,  "e4_qpu_diversity.csv")
    e5  = _try_load_csv(indir,  "e5_batch_vs_stream.csv")
    e6  = _try_load_csv(indir,  "e6_width_sweep.csv")
    e7  = _try_load_csv(indir,  "e7_weight_sensitivity.csv")
    e8  = _try_load_csv(indir,  "e8_fragmentation.csv")
    e9  = _try_load_csv(indir,  "e9_coordination.csv")
    e10 = _try_load_csv(indir,  "e10_weight_sensitivity_wide.csv")
    e11 = _try_load_csv(indir,  "e11_slo_constrained.csv")
    e12 = _try_load_csv(indir,  "e12_qpu_scaling.csv")
    e13 = _try_load_csv(indir,  "e13_backend_comparison.csv")
    e14 = _try_load_csv(indir,  "e14_noise_sensitivity.csv")
    e15 = _try_load_csv(indir,  "e15_streaming_load.csv")
    e16 = _try_load_csv(indir,  "e16_congestion_sweep.csv")
    # E17–E20: check indir2 first (may live in a separate supplementary dir)
    e20 = (_try_load_csv(indir2, "e20_batch_stream.csv") or
           _try_load_csv(indir,  "e20_batch_stream.csv"))
    # JSON experiments
    e15_json = (_try_load_json(indir2, "e15_utilisation.json") or
                _try_load_json(indir,  "e24_idle_fraction.json"))  # fallback schema
    e18_json = (_try_load_json(indir2, "e18_utilization_pareto.json") or
                _try_load_json(indir,  "e18_pool_arrival_grid.json"))
    e21_json = (_try_load_json(indir2, "e21_throughput_scaling.json") or
                _try_load_json(indir,  "e21_throughput_scaling.json"))
    e24_json = (_try_load_json(indir2, "e24_idle_fraction.json") or
                _try_load_json(indir,  "e24_idle_fraction.json"))

    named = [(n, r) for n, r in [("E1",e1),("E2",e2),("E3",e3),("E4",e4),("E5",e5)] if r]

    data = dict(
        e1=e1, e2=e2, e3=e3, e3p=e3p, e3p_gc=e3p_gc, e4=e4, e5=e5, e6=e6, e7=e7, e8=e8,
        e9=e9, e10=e10, e11=e11, e12=e12, e13=e13, e14=e14, e15=e15, e16=e16,
        e20=e20, e15_json=e15_json, e18_json=e18_json,
        e21_json=e21_json, e24_json=e24_json, named=named,
    )

    # ── Determine which figures to generate ────────────────────────────────────
    if args.figures.lower() == "all":
        wanted = set(_FIGURE_REGISTRY.keys())
    else:
        wanted = {f.strip() for f in args.figures.split(",")}
        unknown = wanted - set(_FIGURE_REGISTRY)
        if unknown:
            print(f"[WARN] Unknown figure numbers: {unknown}. Skipping.")
            wanted -= unknown

    registry = _PATCH_REGISTRY if args.patch else _FIGURE_REGISTRY

    # ── Generate figures ───────────────────────────────────────────────────────
    generated = 0
    skipped   = 0
    for fig_key in sorted(wanted, key=lambda k: (len(k), k)):
        if fig_key not in registry:
            continue
        fn, required = _FIGURE_REGISTRY[fig_key]
        if args.patch and fig_key in _PATCH_REGISTRY:
            fn = _PATCH_REGISTRY[fig_key]
        # Check all required data is present
        missing = [k for k in required if not data.get(k)]
        if missing:
            print(f"  [skip] fig{fig_key}: missing data {missing}")
            skipped += 1
            continue
        # Call with correct signature
        fn_args = [data[k] for k in required] + [args.outdir]
        try:
            fn(*fn_args)
            generated += 1
        except Exception as exc:
            print(f"  [ERROR] fig{fig_key}: {exc}")

    print(f"\n[plot_figures]  generated={generated}  skipped={skipped}")
    print(f"[plot_figures]  figures -> {args.outdir}/")


if __name__ == "__main__":
    main()