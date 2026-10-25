"""Legacy Case A physics/network, no import-time training.
Reference anchors must be explicitly passed to compute_loss.
"""
import torch
from torch import nn

g=9.81; n_mann=.02; S0=.05
R_rate=50/3600000; T_rain=180.; T_end=300.
h_dry=1e-4; Lx=Ly=1.
h_sc=.01; u_sc=.1; x_sc=y_sc=1.; t_sc=T_end
device=torch.device('cpu')

def z_bed(x,y):   return 0.10 - S0*x

def dzdx_(x,y):   return torch.full_like(x, -S0)

def dzdy_(x,y):   return torch.zeros_like(y)

def rainfall(t):  return torch.where((t>=0)&(t<T_rain), torch.full_like(t,R_rate), torch.zeros_like(t))

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

def manning_source(h, u, v, n=n_mann, gravity=g, dry_depth=h_dry):
    """Conservative depth-integrated Manning source, in m²/s².

    The factor h is essential: without it this is an acceleration term.
    """
    hr=torch.clamp(h,min=dry_depth)
    spd=torch.sqrt(u**2+v**2+1e-10)
    return (gravity*n**2*h*spd*u/hr**(4./3.),
            gravity*n**2*h*spd*v/hr**(4./3.))


def residuals(model,x,y,t):
    h,u,v=model(x,y,t); hu=h*u; hv=h*v
    Sfx,Sfy=manning_source(h,u,v)
    r1=D(h,t)+D(hu,x)+D(hv,y)-rainfall(t)
    r2=D(hu,t)+D(h*u*u+.5*g*h*h,x)+D(h*u*v,y)+g*h*dzdx_(x,y)+Sfx
    r3=D(hv,t)+D(h*u*v,x)+D(h*v*v+.5*g*h*h,y)+g*h*dzdy_(x,y)+Sfy
    return r1,r2,r3

def compute_loss(model, Np=4000, Ni=500, Nb=400,
                 lam_pde=1., lam_ic=5., lam_bc=5., lam_data=50., samples=None, anchors=None):

    # PDE residuals
    xr,yr,tr=s_int(Np) if samples is None else samples[0]; r1,r2,r3=residuals(model,xr,yr,tr)
    Lp=(r1**2+r2**2+r3**2).mean()

    # IC: h=0, u=v=0 at t=0
    xi,yi,ti=s_ic(Ni) if samples is None else samples[1]; hi,ui,vi=model(xi,yi,ti)
    Li=(hi**2+ui**2+vi**2).mean()

    # BC: no-flux
    xb,yb,tb,nx,ny_=s_bc(Nb) if samples is None else samples[2]; tb=tb.requires_grad_(True)
    hb,ub,vb=model(xb,yb,tb); un=ub*nx+vb*ny_
    Lb=(un**2).mean()

    # Data loss: match FV snapshots (anchors solution scale)
    if anchors is None:
        if lam_data:
            raise ValueError("Data loss requires explicit reference anchors")
        Ld=Lp.new_zeros(())
    else:
        ax,ay,at,ah=anchors
        h_pred,_,_=model(ax,ay,at)
        Ld=((h_pred-ah)**2).mean()

    loss = lam_pde*Lp + lam_ic*Li + lam_bc*Lb + lam_data*Ld
    return loss, Lp.item(), Li.item(), Lb.item(), Ld.item()

def make_lbfgs_closure(model, optimizer, anchors=None, lam_data=0., Np=4000, Ni=500, Nb=400):
    """Freeze all collocation sets for a complete deterministic LBFGS call."""
    samples=(s_int(Np),s_ic(Ni),s_bc(Nb))
    def closure():
        optimizer.zero_grad()
        for group in samples:
            for tensor in group:
                if tensor.grad is not None: tensor.grad=None
        loss,*_=compute_loss(model, samples=samples, anchors=anchors, lam_data=lam_data)
        if not torch.isfinite(loss): raise FloatingPointError("Nonfinite loss")
        loss.backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise FloatingPointError("Nonfinite gradient")
        return loss
    return closure
