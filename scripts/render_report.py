"""Render a self-contained, high-resolution PNG from benchmark JSON."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
COLORS = {"oval":"#2878B5", "circuit":"#D9534F", "procedural":"#399764", "other":"#777777"}
def fmt(value, digits=3):
    if value is None: return "N/A"
    if isinstance(value, (float, np.floating)): return f"{value:.{digits}f}"
    return str(value)
def render_report(source: Path, destination: Path) -> None:
    report = json.loads(source.read_text()); config = report["config"]; summary = report["overall_summary"]
    episodes = report["episodes"]; tracks = report["track_summaries"]
    if not episodes: raise ValueError("The report contains no episode records.")
    run_id = config.get("run_id", source.stem); model_name = config.get("model", {}).get("name", "Unknown model")
    completed, count = summary.get("completed_episodes", 0), summary.get("episodes", len(episodes)); rate = summary.get("completion_rate")
    fig = plt.figure(figsize=(16, 13), layout="constrained")
    grid = fig.add_gridspec(4, 2, height_ratios=[0.8, 1.25, 1.25, 0.65], hspace=0.24, wspace=0.16)
    header = fig.add_subplot(grid[0, :]); header.axis("off")
    header.text(.01,.94,"NEURORACELAB / EXPERIMENT REPORT",fontsize=11,fontweight="bold",transform=header.transAxes)
    header.text(.01,.70,model_name,fontsize=25,fontweight="bold",transform=header.transAxes)
    header.text(.01,.48,f"Run: {run_id} | Purpose: {config.get('purpose','Not specified')}",fontsize=9,transform=header.transAxes)
    metrics = [("COMPLETION",f"{completed}/{count}"),("COMPLETION RATE",f"{rate:.1%}" if rate is not None else "N/A"),
               ("MEAN PROGRESS",fmt(summary.get("mean_progress"))),("MEAN RETURN",fmt(summary.get("mean_return"))),
               ("INFERENCE LATENCY",f"{fmt(summary.get('mean_inference_latency_ms'),4)} ms")]
    for i,(label,value) in enumerate(metrics):
        x=.01+i*.198; header.text(x,.22,label,fontsize=8,color="#555",transform=header.transAxes); header.text(x,.02,value,fontsize=16,fontweight="bold",transform=header.transAxes)
    ax=fig.add_subplot(grid[1,0]); names=list(tracks); rates=[tracks[n].get("completion_rate",0) for n in names]
    bars=ax.bar(names,rates,color=[COLORS.get(n,COLORS["other"]) for n in names]); ax.set_title("Completion rate by track"); ax.set_ylabel("Completed / all episodes"); ax.set_ylim(0,1.12); ax.grid(axis="y",alpha=.25)
    for b,n in zip(bars,names): ax.text(b.get_x()+b.get_width()/2,b.get_height()+.035,f"{tracks[n].get('completed_episodes',0)}/{tracks[n].get('episodes',0)}",ha="center",fontsize=10)
    labels=[f"{r['track_family']}\n{r['seed']}" for r in episodes]; colors=[COLORS.get(r["track_family"],COLORS["other"]) for r in episodes]
    ax=fig.add_subplot(grid[1,1]); progress=[r.get("progress",0) for r in episodes]; bars=ax.bar(range(len(episodes)),progress,color=colors); ax.axhline(1,ls="--",lw=1,color="#333")
    ax.set_xticks(range(len(episodes)),labels,fontsize=8); ax.set_ylim(0,max(1.12,max(progress,default=1)+.08)); ax.set_title("Episode progress"); ax.set_ylabel("Laps completed (fraction)"); ax.grid(axis="y",alpha=.25)
    for b,r in zip(bars,episodes):
        if r.get("completed"): ax.text(b.get_x()+b.get_width()/2,b.get_height()+.015,"✓",ha="center",fontsize=10)
    ax=fig.add_subplot(grid[2,0]); latency=[r.get("inference_latency_mean_ms") for r in episodes]; valid=[(i,v) for i,v in enumerate(latency) if v is not None]
    if valid:
        ix,vals=zip(*valid); ax.bar(ix,vals,color=[colors[i] for i in ix])
    ax.set_xticks(range(len(episodes)),labels,fontsize=8); ax.set_title("Mean inference latency per episode"); ax.set_ylabel("Milliseconds per call"); ax.grid(axis="y",alpha=.25)
    ax=fig.add_subplot(grid[2,1]); reasons=summary.get("termination_reasons",{})
    if reasons:
        rnames=list(reasons); counts=[reasons[n] for n in rnames]; bars=ax.barh(rnames,counts,color="#777")
        ax.set_title("Termination reasons"); ax.set_xlabel("Episode count"); ax.grid(axis="x",alpha=.25)
        for b,v in zip(bars,counts): ax.text(v+.03,b.get_y()+b.get_height()/2,str(v),va="center",fontsize=9)
    else: ax.text(.5,.5,"No termination data",ha="center",va="center"); ax.set_axis_off()
    footer=fig.add_subplot(grid[3,:]); footer.axis("off"); cols=["Track","Completion","Mean progress","Mean return","Mean |lateral error| (m)"]; rows=[]
    for name,s in tracks.items():
        recs=[r for r in episodes if r["track_family"]==name]; errs=[r["mean_abs_lateral_error_m"] for r in recs if r.get("mean_abs_lateral_error_m") is not None]
        rows.append([name,f"{s.get('completed_episodes',0)}/{s.get('episodes',0)}",fmt(s.get("mean_progress")),fmt(s.get("mean_return")),fmt(float(np.mean(errs))) if errs else "N/A"])
    table=footer.table(cellText=rows,colLabels=cols,loc="upper center",cellLoc="center",bbox=[0,.20,1,.75]); table.auto_set_font_size(False); table.set_fontsize(8)
    footer.text(0,.05,f"Seeds: {config.get('seeds','N/A')} | Max steps: {config.get('max_steps','N/A')} | Width: {config.get('track_width_m','N/A')} m | Training curves: not applicable to baseline",fontsize=8,transform=footer.transAxes)
    destination.parent.mkdir(parents=True,exist_ok=True); fig.savefig(destination,dpi=180,bbox_inches="tight",facecolor="white"); plt.close(fig)
def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("report",type=Path); p.add_argument("--output",type=Path); a=p.parse_args()
    src=a.report.expanduser().resolve()
    if not src.is_file(): p.error(f"Report file not found: {src}")
    dst=a.output.expanduser().resolve() if a.output else src.with_suffix(".png"); render_report(src,dst); print(f"PNG report: {dst}")
if __name__=="__main__": main()
