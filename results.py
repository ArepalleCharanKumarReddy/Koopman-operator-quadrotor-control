"""Generate Table I, Table II and Figures 2-6 from saved experiment results."""
from __future__ import annotations

import csv
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = Path("results")
DPI = 220


def load(path):
    if not path.exists():
        raise FileNotFoundError(path.resolve())
    with np.load(path, allow_pickle=False) as d:
        return {k: d[k] for k in d.files}


def settling_time(t, y, target=0.0, fraction=0.02):
    y = np.asarray(y)
    e = np.abs(y-target)
    band = fraction * max(abs(float(y[0]-target)), 1e-12)
    for k in range(len(t)):
        if np.all(e[k:] <= band):
            return float(t[k])
    return np.nan


def overshoot(y, target=0.0):
    y = np.asarray(y)
    d0 = float(y[0]-target)
    if abs(d0) < 1e-12:
        return np.nan
    opposite = (y-target)*d0 < 0
    if not np.any(opposite):
        return 0.0
    return 100.0 * float(np.max(np.abs(y[opposite]-target))) / abs(d0)


def table_i():
    rows=[]
    for N in (3,4,5,6,7,8):
        d=load(Path("models")/f"koopman_N{N}.npz")
        rows.append((N,float(d["prediction_error_percent"])))
    ref=dict(rows)[8]
    with (RESULTS_DIR/"table_I.csv").open("w",newline="") as f:
        w=csv.writer(f); w.writerow(["N","raw_validation_error_percent","relative_error_to_N8"])
        for N,e in rows: w.writerow([N,e,e/max(ref,1e-12)])
    print("Table I")
    for N,e in rows: print(f"N={N}: raw={e:.6f}% relative={e/max(ref,1e-12):.4f}")


def fig2(rep):
    t=rep["time"]
    fig,ax=plt.subplots(3,1,figsize=(8,7),sharex=True)
    labels=[r"$\phi$ (deg)",r"$\theta$ (deg)",r"$\psi$ (deg)"]
    for i,a in enumerate(ax):
        a.plot(t,np.rad2deg(rep["rpy_true"][:,i]),label="Real value")
        a.plot(t,np.rad2deg(rep["rpy_pred"][:,i]),"--",label="Koopman predict value")
        a.set_ylabel(labels[i]); a.grid(True,alpha=.25)
        if i==0:a.legend(loc="best")
    ax[-1].set_xlabel("time (sec)"); fig.tight_layout(); fig.savefig(RESULTS_DIR/"fig2.png",dpi=DPI); plt.close(fig)


def fig3(rep):
    t=rep["time"]
    fig,ax=plt.subplots(3,1,figsize=(8,7),sharex=True)
    labels=[r"$\omega_x$ (rad/s)",r"$\omega_y$ (rad/s)",r"$\omega_z$ (rad/s)"]
    for i,a in enumerate(ax):
        a.plot(t,rep["omega_true"][:,i],label="Real value")
        a.plot(t,rep["omega_pred"][:,i],"--",label="Koopman predict value")
        a.set_ylabel(labels[i]); a.grid(True,alpha=.25)
        if i==0:a.legend(loc="best")
    ax[-1].set_xlabel("time (sec)"); fig.tight_layout(); fig.savefig(RESULTS_DIR/"fig3.png",dpi=DPI); plt.close(fig)


def fig4_5_6(cl):
    t=cl["time"]
    labs=[r"$\phi$ (deg)",r"$\theta$ (deg)",r"$\psi$ (deg)"]
    fig,ax=plt.subplots(3,1,figsize=(8,7),sharex=True)
    for i,a in enumerate(ax):
        a.plot(t,np.rad2deg(cl["reference"][:,i]),"--",label="Reference")
        a.plot(t,np.rad2deg(cl["pid_rpy"][:,i]),label="PID")
        a.plot(t,np.rad2deg(cl["koopman_lqr_rpy"][:,i]),"--",label="Koopman-LQR")
        a.set_ylabel(labs[i]);a.grid(True,alpha=.25)
        if i==0:a.legend(loc="best")
    ax[-1].set_xlabel("time (sec)");fig.tight_layout();fig.savefig(RESULTS_DIR/"fig4.png",dpi=DPI);plt.close(fig)

    labs=[r"$\omega_x$ (rad/s)",r"$\omega_y$ (rad/s)",r"$\omega_z$ (rad/s)"]
    fig,ax=plt.subplots(3,1,figsize=(8,7),sharex=True)
    for i,a in enumerate(ax):
        a.plot(t,cl["pid_omega"][:,i],label="PID")
        a.plot(t,cl["koopman_lqr_omega"][:,i],"--",label="Koopman-LQR")
        a.set_ylabel(labs[i]);a.grid(True,alpha=.25)
        if i==0:a.legend(loc="best")
    ax[-1].set_xlabel("time (sec)");fig.tight_layout();fig.savefig(RESULTS_DIR/"fig5.png",dpi=DPI);plt.close(fig)

    labs=[r"$\tau_x$ (Nm)",r"$\tau_y$ (Nm)",r"$\tau_z$ (Nm)"]
    fig,ax=plt.subplots(3,1,figsize=(8,7),sharex=True)
    tt=t[:-1]
    for i,a in enumerate(ax):
        a.plot(tt,cl["pid_tau"][:,i],label="PID")
        a.plot(tt,cl["koopman_lqr_tau"][:,i],"--",label="Koopman-LQR")
        a.set_ylabel(labs[i]);a.grid(True,alpha=.25)
        if i==0:a.legend(loc="best")
    ax[-1].set_xlabel("time (sec)");fig.tight_layout();fig.savefig(RESULTS_DIR/"fig6.png",dpi=DPI);plt.close(fig)


def table_ii(cl):
    rows=[]
    for ch,i in [("phi",0),("theta",1),("psi",2)]:
        for name,key in [("PID","pid_rpy"),("Koopman-LQR","koopman_lqr_rpy")]:
            y=cl[key][:,i]; st=settling_time(cl["time"],y); ov="-" if ch=="psi" else f"{overshoot(y):.2f}%"
            rows.append([ch,name,f"{st:.2f}s" if np.isfinite(st) else "-",ov])
    with (RESULTS_DIR/"table_II.csv").open("w",newline="") as f:
        w=csv.writer(f);w.writerow(["Channel","Algorithm","Settling time","Overshoot"]);w.writerows(rows)
    print("\nTable II")
    for r in rows: print(", ".join(r))


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    table_i()
    rep=load(RESULTS_DIR/"validation_representative_N5.npz")
    fig2(rep);fig3(rep)
    cl=load(RESULTS_DIR/"closed_loop.npz")
    fig4_5_6(cl);table_ii(cl)
    print("\nGenerated Figure 2-6 and Tables I-II in results/.")


if __name__ == "__main__":
    main()
