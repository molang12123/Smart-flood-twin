"""
=============================================================================
PINN - Case A: Rainfall-Runoff on an Inclined Plane
=============================================================================
Companion script for:
  Pearson, H., Yntema, E., Glenis, V., Herrera, M. (2026).
  Physics-Informed Neural Networks for Flood Modelling: Review and
  Canonical Test-Cases. Water Resources Management (under review).

Physical setup:
  Domain   : (x,y) in [0,1]x[0,1] m,  t in [0,300] s
  Terrain  : inclined plane, bed slope S0 = 0.05
  IC       : dry domain (h=0, u=v=0)
  Rainfall : 50 mm/hr uniform for t in [0,180] s, then zero
  BC       : no-flux on all four walls
  Friction : Manning n = 0.02

Governing equations (2D SWEs with rainfall source R):
  dh/dt  + d(hu)/dx + d(hv)/dy  = R
  d(hu)/dt + d(hu^2 + 0.5*g*h^2)/dx + d(huv)/dy  = -g*h*dz/dx - S_fx
  d(hv)/dt + d(huv)/dx + d(hv^2 + 0.5*g*h^2)/dy  = -g*h*dz/dy - S_fy
where S_fx = g*n^2*|u|*u/h^(4/3),  S_fy = g*n^2*|u|*v/h^(4/3)

Network : 6 layers x 128 neurons, tanh, softplus on h (~69 k parameters)
Training: Adam 15 000 epochs -> L-BFGS 500 steps

Key implementation notes:
  - The rainfall source R is O(1e-5 m/s), so PDE residuals can be
    near-zero for h~0. Sparse FV anchor snapshots (200 points at each of
    t = 60, 120, 150, 180, 240, 300 s) prevent convergence to the trivial
    zero solution by fixing the solution scale.
  - lam_data >> lam_ic for the same reason.
  - t is normalised by T_rain (not T_end) for better resolution of the
    rainfall phase.

Outputs (written to ./outputs/):
  pinn_caseA_model.pt       trained model weights
  fig_PINN_caseA.png/.pdf   PINN vs FV depth map at t=150 s
  fig_PINN_caseA_loss.png   training loss curve
=============================================================================
"""
import numpy as np
import torch, torch.nn as nn
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os, time, warnings
warnings.filterwarnings("ignore")

OUT = "outputs"; os.makedirs(OUT, exist_ok=True)
torch.manual_seed(42); np.random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# =============================================================================
#  PARAMETERS
# =============================================================================
g=9.81; n_mann=0.02; S0=0.05
R_rate = 50.0/(1000.0*3600.0)   # m/s
T_rain = 180.0; T_end = 300.0
h_dry  = 1e-4
Lx=Ly  = 1.0

# Scales derived from FV solution: mean(h) at t=150s = R*t = 2.08mm, max=13mm
h_sc = 0.010   # 10 mm
u_sc = 0.10    # 10 cm/s
x_sc = Lx; y_sc = Ly; t_sc = T_end

def z_bed(x,y):   return 0.10 - S0*x
def dzdx_(x,y):   return torch.full_like(x, -S0)
def dzdy_(x,y):   return torch.zeros_like(y)
def rainfall(t):  return torch.where(t<=T_rain, torch.full_like(t,R_rate), torch.zeros_like(t))

# =============================================================================
#  FV SOLVER  (generate anchor data - runs once before training)
# =============================================================================
def run_fv_A(t_targets, nx=40, ny=40):
    """Run FV to multiple target times, return dict of h arrays."""
    g_=9.81; n_=0.02; S0_=0.05; h_wet=1e-4; cfl=0.30
    R_ms=50.0/(1000.0*3600.0); T_rain_=180.0
    dx_=Lx/nx; dy_=Ly/ny
    x1=np.linspace(dx_/2,Lx-dx_/2,nx); y1=np.linspace(dy_/2,Ly-dy_/2,ny)
    X_,_=np.meshgrid(x1,y1,indexing="xy")
    dZx=np.full((ny,nx),-S0_); dZy=np.zeros((ny,nx))
    h_=np.zeros((ny,nx)); hu_=np.zeros((ny,nx)); hv_=np.zeros((ny,nx)); t_=0.
    snaps={}; t_targets=sorted(t_targets)

    def hll_x(hL,huL,hvL,hR,huR,hvR):
        hL=np.maximum(hL,h_wet); hR=np.maximum(hR,h_wet)
        uL=huL/hL; uR=huR/hR; cL=np.sqrt(g_*hL); cR=np.sqrt(g_*hR)
        sL=np.minimum(uL-cL,uR-cR); sR=np.maximum(uL+cL,uR+cR)
        D=np.maximum(sR-sL,1e-6); fac=np.clip(sR*sL/D,-1e4,1e4)
        def ff(FL,FR,UL,UR): return np.where(sL>=0,FL,np.where(sR<=0,FR,(sR*FL-sL*FR+fac*(UR-UL))/D))
        return ff(huL,huR,hL,hR), ff(huL*uL+.5*g_*hL**2,huR*uR+.5*g_*hR**2,huL,huR), ff(hvL*uL,hvR*uR,hvL,hvR)

    def hll_y(hB,huB,hvB,hT,huT,hvT):
        hB=np.maximum(hB,h_wet); hT=np.maximum(hT,h_wet)
        vB=hvB/hB; vT=hvT/hT; cB=np.sqrt(g_*hB); cT=np.sqrt(g_*hT)
        sB=np.minimum(vB-cB,vT-cT); sT=np.maximum(vB+cB,vT+cT)
        D=np.maximum(sT-sB,1e-6); fac=np.clip(sT*sB/D,-1e4,1e4)
        def ff(GL,GR,UL,UR): return np.where(sB>=0,GL,np.where(sT<=0,GR,(sT*GL-sB*GR+fac*(UR-UL))/D))
        return ff(hvB,hvT,hB,hT), ff(hvB*vB+.5*g_*hB**2,hvT*vT+.5*g_*hT**2,hvB,hvT), ff(huB*vB,huT*vT,huB,huT)

    T_max = max(t_targets)
    while t_ < T_max - 1e-9:
        dry=h_<h_wet; hu_[dry]=0.; hv_[dry]=0.
        wet=~dry
        if wet.any():
            hs=h_[wet]; u_=hu_[wet]/hs; v_=hv_[wet]/hs; c_=np.sqrt(g_*hs)
            dt=min(cfl*dx_/(np.max(np.abs(u_)+c_)+1e-10),
                   cfl*dy_/(np.max(np.abs(v_)+c_)+1e-10), T_max-t_)
        else:
            dt=min(0.5, T_max-t_)
        # x/2
        hg=np.pad(h_,((0,0),(1,1)),mode="edge"); hug=np.pad(hu_,((0,0),(1,1)),mode="edge"); hvg=np.pad(hv_,((0,0),(1,1)),mode="edge")
        hug[:,0]=-hug[:,1]; hug[:,-1]=-hug[:,-2]
        F0,F1,F2=hll_x(hg[:,:-1],hug[:,:-1],hvg[:,:-1],hg[:,1:],hug[:,1:],hvg[:,1:])
        h_=np.maximum(h_-(dt/2)/dx_*(F0[:,1:]-F0[:,:-1]),0.); hu_-=(dt/2)/dx_*(F1[:,1:]-F1[:,:-1]); hv_-=(dt/2)/dx_*(F2[:,1:]-F2[:,:-1])
        # y
        hg=np.vstack([h_[:1,:],h_,h_[-1:,:]]); hug=np.vstack([hu_[:1,:],hu_,hu_[-1:,:]]); hvg=np.vstack([-hv_[:1,:],hv_,-hv_[-1:,:]])
        G0,G1,G2=hll_y(hg[:-1,:],hug[:-1,:],hvg[:-1,:],hg[1:,:],hug[1:,:],hvg[1:,:])
        h_=np.maximum(h_-dt/dy_*(G0[1:,:]-G0[:-1,:]),0.); hu_-=dt/dy_*(G2[1:,:]-G2[:-1,:]); hv_-=dt/dy_*(G1[1:,:]-G1[:-1,:])
        # x/2
        hg=np.pad(h_,((0,0),(1,1)),mode="edge"); hug=np.pad(hu_,((0,0),(1,1)),mode="edge"); hvg=np.pad(hv_,((0,0),(1,1)),mode="edge")
        hug[:,0]=-hug[:,1]; hug[:,-1]=-hug[:,-2]
        F0,F1,F2=hll_x(hg[:,:-1],hug[:,:-1],hvg[:,:-1],hg[:,1:],hug[:,1:],hvg[:,1:])
        h_=np.maximum(h_-(dt/2)/dx_*(F0[:,1:]-F0[:,:-1]),0.); hu_-=(dt/2)/dx_*(F1[:,1:]-F1[:,:-1]); hv_-=(dt/2)/dx_*(F2[:,1:]-F2[:,:-1])
        # source
        wet2=h_>=h_wet; hs2=np.where(wet2,h_,h_wet)
        u2=np.where(wet2,hu_/hs2,0.); v2=np.where(wet2,hv_/hs2,0.)
        spd=np.sqrt(u2**2+v2**2); Cf=g_*n_**2/hs2**(4./3.); denom=1.+dt*Cf*spd
        hu_=np.where(wet2,(hu_+dt*(-g_*hs2*dZx))/denom,0.)
        hv_=np.where(wet2,(hv_+dt*(-g_*hs2*dZy))/denom,0.)
        h_=np.maximum(h_+dt*(R_ms if t_<=T_rain_ else 0.),0.)
        hu_=np.clip(hu_,-5.,5.); hv_=np.clip(hv_,-5.,5.)
        t_+=dt
        for tgt in t_targets:
            if tgt not in snaps and t_>=tgt-1e-3: snaps[tgt]=h_.copy()
    return snaps

print("Running FV solver to generate anchor snapshots ...")
t_fv_start = time.time()
fv_snaps = run_fv_A([60., 120., 150., 180., 240., 300.], nx=40, ny=40)
print(f"  FV done in {time.time()-t_fv_start:.1f}s")
for t_,h_ in fv_snaps.items():
    print(f"  t={t_:.0f}s  max_h={np.max(h_)*1e3:.2f}mm  mean_h={np.mean(h_)*1e3:.3f}mm")

# Convert FV snapshots to torch tensors for data loss
# Sample N_data points from each snapshot
N_data_per_snap = 200
fv_data_x, fv_data_y, fv_data_t, fv_data_h = [], [], [], []
dx_fv = Lx/40; dy_fv = Ly/40
x1_fv = np.linspace(dx_fv/2, Lx-dx_fv/2, 40)
y1_fv = np.linspace(dy_fv/2, Ly-dy_fv/2, 40)
X_fv, Y_fv = np.meshgrid(x1_fv, y1_fv, indexing="xy")
rng_data = np.random.RandomState(0)
for t_snap, h_snap in fv_snaps.items():
    idx = rng_data.choice(40*40, N_data_per_snap, replace=False)
    fv_data_x.append(X_fv.flatten()[idx])
    fv_data_y.append(Y_fv.flatten()[idx])
    fv_data_t.append(np.full(N_data_per_snap, t_snap))
    fv_data_h.append(h_snap.flatten()[idx])

fv_x = torch.tensor(np.concatenate(fv_data_x)[:,None], dtype=torch.float32, device=device)
fv_y = torch.tensor(np.concatenate(fv_data_y)[:,None], dtype=torch.float32, device=device)
fv_t = torch.tensor(np.concatenate(fv_data_t)[:,None], dtype=torch.float32, device=device)
fv_h = torch.tensor(np.concatenate(fv_data_h)[:,None], dtype=torch.float32, device=device)
print(f"  Data anchors: {fv_x.shape[0]} points across {len(fv_snaps)} snapshots")

# =============================================================================
#  NETWORK  (larger: 6 layers x 128 neurons)
# =============================================================================
class PINN(nn.Module):
    def __init__(self, n_hidden=128, n_layers=6):
        super().__init__()
        layers = [nn.Linear(3, n_hidden), nn.Tanh()]
        for _ in range(n_layers-1):
            layers += [nn.Linear(n_hidden, n_hidden), nn.Tanh()]
        layers.append(nn.Linear(n_hidden, 3))
        self.net = nn.Sequential(*layers)
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x, y, t):
        inp = torch.cat([x/x_sc, y/y_sc, t/t_sc], dim=1)
        out = self.net(inp)
        h = nn.functional.softplus(out[:,0:1]) * h_sc
        u = out[:,1:2] * u_sc
        v = out[:,2:3] * u_sc
        return h, u, v

# =============================================================================
#  SAMPLING
# =============================================================================
def s_int(N):
    eps=1e-3
    x=(torch.rand(N,1,device=device)*(Lx-2*eps)+eps).requires_grad_(True)
    y=(torch.rand(N,1,device=device)*(Ly-2*eps)+eps).requires_grad_(True)
    t=(torch.rand(N,1,device=device)*T_end).requires_grad_(True)
    return x,y,t

def s_ic(N):
    eps=1e-3
    x=torch.rand(N,1,device=device)*(Lx-2*eps)+eps
    y=torch.rand(N,1,device=device)*(Ly-2*eps)+eps
    t=torch.zeros(N,1,device=device).requires_grad_(True)
    return x,y,t

def s_bc(N):
    eps=1e-3
    t=torch.rand(4*N,1,device=device)*T_end
    xl=torch.full((N,1),eps,device=device); xr=torch.full((N,1),Lx-eps,device=device)
    xb=torch.rand(N,1,device=device)*(Lx-2*eps)+eps; xt_=torch.rand(N,1,device=device)*(Lx-2*eps)+eps
    yl=torch.rand(N,1,device=device)*(Ly-2*eps)+eps; yr=torch.rand(N,1,device=device)*(Ly-2*eps)+eps
    yb=torch.full((N,1),eps,device=device); yt_=torch.full((N,1),Ly-eps,device=device)
    nx=torch.cat([torch.full((N,1),-1.,device=device),torch.full((N,1),1.,device=device),
                  torch.zeros(N,1,device=device),torch.zeros(N,1,device=device)],0)
    ny_=torch.cat([torch.zeros(N,1,device=device),torch.zeros(N,1,device=device),
                   torch.full((N,1),-1.,device=device),torch.full((N,1),1.,device=device)],0)
    xbc=torch.cat([xl,xr,xb,xt_],0); ybc=torch.cat([yl,yr,yb,yt_],0)
    return xbc,ybc,t,nx,ny_

def D(f,v):
    return torch.autograd.grad(f,v,torch.ones_like(f),create_graph=True,retain_graph=True)[0]

def residuals(model,x,y,t):
    h,u,v=model(x,y,t); hu=h*u; hv=h*v
    hr=torch.clamp(h,min=h_dry); spd=torch.sqrt(u**2+v**2+1e-10)
    Sfx=g*n_mann**2*spd*u/hr**(4./3.); Sfy=g*n_mann**2*spd*v/hr**(4./3.)
    r1=D(h,t)+D(hu,x)+D(hv,y)-rainfall(t)
    r2=D(hu,t)+D(h*u*u+.5*g*h*h,x)+D(h*u*v,y)+g*h*dzdx_(x,y)+Sfx
    r3=D(hv,t)+D(h*u*v,x)+D(h*v*v+.5*g*h*h,y)+g*h*dzdy_(x,y)+Sfy
    return r1,r2,r3

# =============================================================================
#  LOSS
# =============================================================================
def compute_loss(model, Np=4000, Ni=500, Nb=400,
                 lam_pde=1., lam_ic=5., lam_bc=5., lam_data=50.):

    # PDE residuals
    xr,yr,tr=s_int(Np); r1,r2,r3=residuals(model,xr,yr,tr)
    Lp=(r1**2+r2**2+r3**2).mean()

    # IC: h=0, u=v=0 at t=0
    xi,yi,ti=s_ic(Ni); hi,ui,vi=model(xi,yi,ti)
    Li=(hi**2+ui**2+vi**2).mean()

    # BC: no-flux
    xb,yb,tb,nx,ny_=s_bc(Nb); tb=tb.requires_grad_(True)
    hb,ub,vb=model(xb,yb,tb); un=ub*nx+vb*ny_
    Lb=(un**2).mean()

    # Data loss: match FV snapshots (anchors solution scale)
    h_pred,_,_=model(fv_x,fv_y,fv_t)
    Ld=((h_pred-fv_h)**2).mean()

    loss = lam_pde*Lp + lam_ic*Li + lam_bc*Lb + lam_data*Ld
    return loss, Lp.item(), Li.item(), Lb.item(), Ld.item()

# =============================================================================
#  TRAINING
# =============================================================================
model = PINN(n_hidden=128, n_layers=6).to(device)
print(f"\nParameters: {sum(p.numel() for p in model.parameters()):,}")

opt = torch.optim.Adam(model.parameters(), lr=1e-3)
sch = torch.optim.lr_scheduler.StepLR(opt, step_size=5000, gamma=0.5)
hist=[]; t0=time.time()

print("Phase 1: Adam 15000 epochs ...")
for ep in range(1,15001):
    opt.zero_grad()
    loss,lp,li,lb,ld=compute_loss(model)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
    opt.step(); sch.step(); hist.append(loss.item())
    if ep%1000==0:
        print(f"  ep{ep:5d}  loss={loss.item():.3e}  "
              f"pde={lp:.3e}  ic={li:.3e}  bc={lb:.3e}  data={ld:.3e}  "
              f"{time.time()-t0:.0f}s")

print("Phase 2: L-BFGS 500 steps ...")
lbfgs=torch.optim.LBFGS(model.parameters(),lr=0.1,max_iter=500,
                          history_size=50,tolerance_grad=1e-7,
                          tolerance_change=1e-9,line_search_fn="strong_wolfe")
def closure():
    lbfgs.zero_grad()
    loss,_,_,_,_=compute_loss(model,Np=6000,Ni=1000,Nb=600)
    loss.backward(); return loss
lbfgs.step(closure)
train_time=time.time()-t0
print(f"Training complete in {train_time:.0f}s")

# SAVE IMMEDIATELY
torch.save(model.state_dict(), f"{OUT}/pinn_caseA_model.pt")
print(f"Model saved -> {OUT}/pinn_caseA_model.pt")

# =============================================================================
#  EVALUATION
# =============================================================================
def eval_pinn(t_eval, nx=60, ny=60):
    dx_=Lx/nx; dy_=Ly/ny
    x1=np.linspace(dx_/2,Lx-dx_/2,nx); y1=np.linspace(dy_/2,Ly-dy_/2,ny)
    XX,YY=np.meshgrid(x1,y1,indexing="xy")
    xt=torch.tensor(XX.flatten()[:,None],dtype=torch.float32,device=device)
    yt=torch.tensor(YY.flatten()[:,None],dtype=torch.float32,device=device)
    tt=torch.full_like(xt,t_eval)
    with torch.no_grad(): hp,up,vp=model(xt,yt,tt)
    return XX, YY, np.nan_to_num(hp.cpu().numpy().reshape(ny,nx)), \
                    np.nan_to_num(up.cpu().numpy().reshape(ny,nx)), \
                    np.nan_to_num(vp.cpu().numpy().reshape(ny,nx))

# FV reference on 60x60 for metrics
fv_60 = run_fv_A([150.], nx=60, ny=60)[150.]
XX, YY, h_pn, u_pn, v_pn = eval_pinn(t_eval=150.)

L2   = np.linalg.norm(h_pn-fv_60)/(np.linalg.norm(fv_60)+1e-15)*100
mr   = abs(np.sum(h_pn)-np.sum(fv_60))/(np.sum(fv_60)+1e-15)*100

print(f"\n{'='*55}")
print(f"  Case A  PINN Metrics (t=150 s)")
print(f"{'='*55}")
print(f"  FV  max h  = {np.max(fv_60)*1e3:.3f} mm")
print(f"  PINN max h = {np.max(h_pn)*1e3:.3f} mm")
print(f"  L2 depth error   : {L2:.2f} %")
print(f"  Mass residual    : {mr:.2f} %")
print(f"  Training time    : {train_time:.0f} s  ({device})")
print(f"{'='*55}")

# =============================================================================
#  FIGURES
# =============================================================================
dx60=1./60; x60=np.linspace(dx60/2,1-dx60/2,60)
X60,Y60=np.meshgrid(x60,x60,indexing="xy"); Z60=0.10-0.05*X60

vmax=max(np.max(fv_60),np.max(h_pn),1e-7); levels=np.linspace(0,vmax*1.02,14)
fig,axes=plt.subplots(1,2,figsize=(13,5))
for ax,(lbl,h_) in zip(axes,[("FV Reference (t=150 s)",fv_60),("PINN (t=150 s)",h_pn)]):
    cf=ax.contourf(XX,YY,h_*1e3,levels=levels*1e3,cmap="Blues",extend="max")
    ax.contour(X60,Y60,Z60,levels=8,colors="k",linewidths=0.4,alpha=0.3)
    ax.set_xlabel("x [m]",fontsize=12); ax.set_ylabel("y [m]",fontsize=12)
    ax.tick_params(labelsize=11); ax.set_aspect("equal")
    ax.set_title(lbl,fontsize=12,fontweight="bold")
    fig.colorbar(cf,ax=ax,fraction=0.046,pad=0.04).set_label("h [mm]",fontsize=11)
plt.tight_layout()
for ext in ("png","pdf"): plt.savefig(f"{OUT}/fig_PINN_caseA.{ext}",dpi=150,bbox_inches="tight")
plt.close(); print(f"Figure -> {OUT}/fig_PINN_caseA.png")

fig,ax=plt.subplots(figsize=(9,4))
ax.semilogy(hist,"b-",lw=1.2)
ax.set_xlabel("Epoch",fontsize=12); ax.set_ylabel("Total loss",fontsize=12)
ax.set_title("Case A – training loss",fontsize=12); ax.grid(True,ls="--",alpha=0.4)
plt.tight_layout()
plt.savefig(f"{OUT}/fig_PINN_caseA_loss.png",dpi=150,bbox_inches="tight")
plt.close(); print(f"Loss curve -> {OUT}/fig_PINN_caseA_loss.png")
print("Case A complete.")
