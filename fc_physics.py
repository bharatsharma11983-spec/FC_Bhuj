#!/usr/bin/env python3
"""
fc_physics.py — SFF Physics Engine for Bhuj Case 7
All corner frequency calculations, station readers, IS1893, Bhuj lithology.
Import from fc_app.py.
"""
import numpy as np, re, os
from scipy.fft import rfft, rfftfreq
from scipy.ndimage import gaussian_filter1d
_trapz = getattr(np,'trapezoid',None) or getattr(np,'trapz',None)

# ── Bhuj lithological profile (13 layers) ──────────────────────────────────
BHUJ_LAYERS = [
    ("Quaternary alluvium/Rann",      0.00, 0.01, 1.85, "Recent alluvium, Rann of Kutch clay-silt"),
    ("Holocene-Pleistocene alluvium", 0.01, 0.05, 1.90, "Fluvial deposits, sandy alluvium"),
    ("Late Quaternary sediments",     0.05, 0.20, 1.95, "Consolidated clay-sand, saline deposits"),
    ("Tertiary (Nummulitic limestone)",0.20, 1.00, 2.35, "Eocene limestone, marine sediments"),
    ("Deccan Trap basalt (Cretaceous)",1.00, 1.50, 2.60, "Basalt intrusive sills, dykes"),
    ("Jurassic marine sed. (Jhurio)", 1.50, 3.00, 2.45, "Limestone, shale, Kutch Supergroup"),
    ("Triassic-Jurassic clastic",     3.00, 5.00, 2.50, "Sandstone, shale, Patcham Formation"),
    ("Permian continental red beds",  5.00, 6.00, 2.55, "Continental red beds, Kaladongar Fm"),
    ("Late Proterozoic metased.",     6.00, 8.00, 2.62, "Phyllite, quartzite, Delhi Supergroup"),
    ("Early Proterozoic granite",     8.00,10.00, 2.70, "Granite-gneiss, Aravalli basement"),
    ("Precambrian metamorphic",      10.00,12.00, 2.75, "Granulite, charnockite, lower crust"),
    ("Upper mantle transition",      12.00,14.00, 2.80, "Mafic lower crust, gabbro"),
    ("Lower crust/Moho zone",        14.00,20.00, 2.85, "Moho ~35 km; focal depth zone"),
]

# Bhuj Vs profile (shear-wave velocity by layer) — km/s
BHUJ_VS_LAYERS = [
    ("Quaternary alluvium/Rann",      0.00, 0.01, 0.25),
    ("Holocene-Pleistocene alluvium", 0.01, 0.05, 0.35),
    ("Late Quaternary sediments",     0.05, 0.20, 0.55),
    ("Tertiary (Nummulitic limestone)",0.20, 1.00, 2.20),
    ("Deccan Trap basalt",            1.00, 1.50, 3.10),
    ("Jurassic marine sed.",          1.50, 3.00, 2.80),
    ("Triassic-Jurassic clastic",     3.00, 5.00, 2.90),
    ("Permian continental",           5.00, 6.00, 3.10),
    ("Late Proterozoic metased.",     6.00, 8.00, 3.30),
    ("Early Proterozoic granite",     8.00,10.00, 3.50),
    ("Precambrian metamorphic",      10.00,12.00, 3.65),
    ("Upper mantle transition",      12.00,14.00, 3.75),
    ("Lower crust/Moho zone",        14.00,20.00, 3.80),
]

def weighted_density(depth_km, layers=None):
    """Depth-weighted average density from surface to depth_km."""
    if layers is None: layers = BHUJ_LAYERS
    if depth_km <= 0: return layers[0][3]
    total_wt = total_d = 0.0
    for _, zt, zb, rho, _ in layers:
        if zb <= 0 or zt >= depth_km: continue
        dz = min(zb, depth_km) - max(zt, 0)
        total_wt += rho * dz; total_d += dz
    return total_wt / total_d if total_d > 0 else 2.8

def weighted_vs(depth_km, layers=None):
    """Depth-weighted average Vs from surface to depth_km."""
    if layers is None: layers = BHUJ_VS_LAYERS
    if depth_km <= 0: return layers[0][3]
    total_wt = total_d = 0.0
    for _, zt, zb, vs, *_ in layers:
        if zb <= 0 or zt >= depth_km: continue
        dz = min(zb, depth_km) - max(zt, 0)
        total_wt += vs * dz; total_d += dz
    return total_wt / total_d if total_d > 0 else 3.8

def is1893_spectrum(Z=0.36, version='2016', soil='medium'):
    """IS 1893 design spectrum. Returns (T, Sa_g)."""
    T = np.logspace(-2, np.log10(20), 300)
    if version == '1984':
        SF = {'hard':1.0,'medium':1.2,'soft':1.5}.get(soil, 1.0)
        Sa = np.where(T<=0.10, Z*SF*(1+15*T),
             np.where(T<=0.40, Z*SF*2.5,
             np.where(T<=4.0,  Z*SF*1.0/T, Z*SF*0.25)))
    elif version == '2002':
        t_p = {'hard':0.40,'medium':0.55,'soft':0.67}[soil]
        SF  = {'hard':1.0,'medium':1.36,'soft':1.67}[soil]
        Sa  = np.where(T<=0.10, Z/2*SF*(1+15*T)*2.5,
              np.where(T<=t_p,  Z/2*SF*2.5,
              np.where(T<=4.0,  Z/2*SF*2.5*t_p/T,
                                Z/2*SF*2.5*t_p/4.0)))
    elif version == '2016':
        Sa = np.where(T<=0.10, Z*(1+15*T),
             np.where(T<=0.55, Z*2.5,
             np.where(T<=4.0,  Z*2.5*0.55/T,
                               Z*2.5*0.55/4.0*(4/T)**2)))
    elif version == '2024_wd':
        Z24 = 0.50
        Sa  = np.where(T<=0.10, Z24*(1+20*T),
              np.where(T<=0.60, Z24*3.0,
              np.where(T<=5.0,  Z24*3.0*0.60/T,
                                Z24*3.0*0.60/5.0*(5.0/T)**1.5)))
    else:
        raise ValueError(version)
    return T, Sa

def read_pesmos_dat(fp):
    with open(fp,'r',errors='replace') as f: lines=f.readlines()
    m = re.search(r'(\d+)\s+Acceleration.*at\s+([\d.]+)\s+sec', lines[5])
    if not m: raise ValueError(f"Bad header: {fp}")
    npts,dt = int(m.group(1)),float(m.group(2))
    pk = re.search(r'Peak Acceleration\s*=\s*([-\d.E+]+)', lines[4])
    raw=[]
    for line in lines[6:]:
        for tok in line.split():
            try: raw.append(float(tok))
            except: pass
    acc = np.array(raw[:npts])
    meta = {'npts':npts,'dt':dt,'pga':float(pk.group(1)) if pk else None,'comp':lines[1].strip()}
    return np.arange(npts)*dt, acc, meta

def read_pesmos_vs(fp):
    pers,psa=[],[]
    with open(fp,'r',errors='replace') as f:
        for line in f:
            if line.startswith('#') or not line.strip(): continue
            p=line.split()
            if len(p)>=6:
                try: pers.append(float(p[0])); psa.append(float(p[5]))
                except: pass
    return np.array(pers), np.array(psa)

# ── SFF Engine ────────────────────────────────────────────────────────────────
class SFFEngine:
    """
    Stochastic Finite-Fault engine.
    Reference: Motazedian & Atkinson (2005) BSSA 95:995.
    """
    def __init__(self, p): self.p = p

    def M0(self): return 10**(1.5*self.p['Mw']+16.05)

    # ── Corner frequencies ──────────────────────────────────────────────
    def fc_brune(self):
        """Eq.1: Brune (1970) whole-fault fc = 4.9e6·β·(Δσ/M₀)^(1/3)"""
        return 4.9e6*self.p['beta']*(self.p['stress_drop']/self.M0())**(1/3)

    def fc_static(self, N):
        """Eq.2: FINSIM static fc — grid-dependent (UNPHYSICAL for M₀,sub=M₀/N)"""
        return 4.9e6*self.p['beta']*(self.p['stress_drop']/(self.M0()/N))**(1/3)

    def fc_dynamic(self, N, NR):
        """Eq.3: EXSIM dynamic fc — grid-independent (Motazedian & Atkinson 2005)"""
        return 4.9e6*self.p['beta']*(self.p['stress_drop']/(NR*self.M0()/N))**(1/3)

    def fc_double(self):
        """Eq.4: Double-corner (Atkinson & Silva 2000) — fa,fb,epsilon"""
        fc = self.fc_brune(); return fc*0.20, fc*3.50, 0.55

    def fc_tang(self, N, NR, Vr, dl, dw):
        """Eq.5: Tang (2022) — explicit Vr coupling: k·Vr·√(π/(NR·Δl·Δw))"""
        return 0.5*Vr*np.sqrt(np.pi/(NR*dl*dw))

    # ── Path & site filters ──────────────────────────────────────────────
    def G(self, R):
        """Piecewise geometric spreading (Table 4.2), CGS units (1/cm)."""
        if R<40:    g=R**(-1.0)
        elif R<=80: g=40**(-1.0)*(R/40)**(-0.5)
        else:       g=40**(-1.0)*2**(-0.5)*(R/80)**(-0.55)
        return g/1e5   # km⁻¹ → cm⁻¹

    def P(self, f, R):
        """Anelastic attenuation: exp(−πfR/Q(f)β)"""
        Q=self.p['Q0']*np.maximum(f,1e-6)**self.p['eta']
        return np.exp(-np.pi*f*R/(Q*self.p['beta']))

    def K(self, f):
        """Site kappa filter: exp(−πκ₀f)"""
        return np.exp(-np.pi*self.p['kappa']*f)

    def site_amp(self):
        """Generic Vs30 amplification: (760/Vs30)^0.3"""
        return (760/self.p['Vs30'])**0.3

    def C_rad(self):
        """Radiation constant Cr = (Rθφ·Fs)/(4π·ρ·β³)"""
        bc = self.p['beta']*1e5   # km/s → cm/s
        return (0.55*2.0)/(4*np.pi*self.p['rho']*bc**3)

    def T_path(self):
        """Path duration (Table 4.2): T_path = a·R + b"""
        a = self.p.get('Tpath_a', 0.108)
        b = self.p.get('Tpath_b', 15.5939)
        return a*self.p['R'] + b

    # ── Fourier acceleration spectrum ────────────────────────────────────
    def fourier_spectrum(self, f, model):
        """
        Complete SFF Fourier acceleration spectrum |A(f)| [cm/s].
        Eq.6: |A(f)| = Cr·M₀·H_ij·[f²/(fc²+f²)]·G(R)·P(f,R)·K(f)·AF
        """
        p=self.p; M0=self.M0(); N=p['N']; dl=p['dl']; dw=p['dw']
        Vr=p['Vr_frac']*p['beta']; Cr=self.C_rad(); sf=np.maximum(f,1e-6)

        if model in ('brune','natural'):
            fc=self.fc_brune(); E=Cr*M0*sf**2/(fc**2+sf**2)
        elif model=='static':
            fc=self.fc_static(N); E=Cr*M0*sf**2/(fc**2+sf**2)
        elif model=='dynamic':
            M0s=M0/N; E=np.zeros_like(sf); step=max(1,N//40)
            for k in range(1,N+1,step):
                fck=self.fc_dynamic(N,k); E+=Cr*M0s*step*sf**2/(fck**2+sf**2)
        elif model=='double':
            fa,fb,eps=self.fc_double()
            E=Cr*M0*sf**2*(eps/(1+(sf/fa)**2)+(1-eps)/(1+(sf/fb)**2))
        elif model=='tang':
            M0s=M0/N; E=np.zeros_like(sf); step=max(1,N//40)
            for k in range(1,N+1,step):
                fct=max(self.fc_tang(N,k,Vr,dl,dw),1e-5)
                E+=Cr*M0s*step*sf**2/(fct**2+sf**2)
        else: raise ValueError(model)
        return E*self.G(p['R'])*self.P(sf,p['R'])*self.K(sf)*self.site_amp()

    # ── Stochastic synthesis ─────────────────────────────────────────────
    def simulate(self, model, seed=42, dt=0.005):
        """
        Boore (1983/2003) stochastic synthesis.
        KEY FIX: RMS normalization divides by √duration (Parseval theorem).
        σ_target = √(2∫|A(f)|²df / T_dur) — correct scaling for time series.
        Returns (time_s, acc_m_s2).
        """
        np.random.seed(seed)
        Tsrc = 1.0/self.fc_brune()
        dur  = max(Tsrc + self.T_path() + self.p['R']/self.p['beta'], 80.0)
        npts = int(dur/dt); npts += npts%2
        t    = np.arange(npts)*dt
        freqs = rfftfreq(npts,dt); freqs[0]=freqs[1]

        A_f = self.fourier_spectrum(freqs, model)   # cm/s

        # Saragoni-Hart taper window
        n1=int(0.2*npts); n2=int(0.85*npts)
        win=np.ones(npts)
        win[:n1]=(np.arange(n1)/max(n1,1))**2
        win[n2:]=((npts-np.arange(n2,npts))/max(npts-n2,1))**2

        # White noise → shape with A_f
        noise   = np.random.randn(npts)*win
        phases  = np.angle(rfft(noise))
        X       = A_f*np.exp(1j*phases); X[0]=0
        acc_cm  = np.real(np.fft.irfft(X,n=npts))*win

        # ── KEY FIX: divide by √(duration) per Boore 2003 Eq.3 ──────
        # σ² = (2/T)∫|A(f)|² df  ← one-sided PSD × bandwidth / duration
        rms_tgt = np.sqrt(2*_trapz(A_f**2, freqs) / dur)
        rms_now = np.sqrt(np.mean(acc_cm**2))
        if rms_now > 1e-30: acc_cm *= rms_tgt / rms_now

        return t, acc_cm/100.0   # → m/s²

    # ── FAS from time history ────────────────────────────────────────────
    def fas(self, time, acc, sigma=2):
        dt=time[1]-time[0]; sp=np.abs(rfft(acc))*dt
        fr=rfftfreq(len(acc),dt)
        if sigma>0: sp=gaussian_filter1d(sp,sigma=sigma)
        return fr[1:], sp[1:]

    # ── Response spectrum (Newmark-β) ────────────────────────────────────
    def response_spectrum(self, time, acc, periods=None, zeta=0.05):
        """
        5%-damped PSA via Newmark average acceleration (Chopra 2012 §5.2).
        Eq.7: k_eff = ω²+4ζω/Δt+4/Δt²; ΔP̂ = Δp+(4/Δt+4ζω)v+2a
        Returns (periods_s, Sa_g).
        """
        if periods is None: periods=np.logspace(-2,1,100)
        dt=float(time[1]-time[0]); Sa=np.zeros(len(periods)); g=9.81
        for i,T in enumerate(periods):
            T=max(float(T),2*dt); om=2*np.pi/T
            k_eff=om**2+4*zeta*om/dt+4/dt**2
            u=v=0; a_s=-acc[0]; u_max=0
            for j in range(1,len(acc)):
                dp=-(acc[j]-acc[j-1])
                dp_hat=dp+(4/dt+4*zeta*om)*v+2*a_s
                du=dp_hat/k_eff; dv=(2/dt)*du-2*v
                da=(4/dt**2)*du-(4/dt)*v-2*a_s
                u+=du; v+=dv; a_s+=da; u_max=max(u_max,abs(u))
            Sa[i]=om**2*u_max/g
        return periods, Sa

# ── Dynamic Layer Generation (based on depth) ────────────────────────
def generate_layers_from_depth(depth_km, num_layers=15):
    """Generate N/num_layers equally spaced layers for given depth.
    If depth=26km with num_layers=15, creates 15 layers from 0 to 26km.
    """
    if depth_km <= 0:
        return []
    
    step = depth_km / num_layers
    layers = []
    
    # Density values (typical for Bhuj crust)
    densities = [1.85, 1.90, 1.95, 2.35, 2.60, 2.45, 2.50, 2.55, 2.62, 2.70, 2.75, 2.80, 2.85, 3.00, 3.20]
    vs_values = [0.25, 0.35, 0.55, 2.20, 3.10, 2.80, 2.90, 3.10, 3.30, 3.50, 3.65, 3.75, 3.80, 3.85, 3.90]
    
    for i in range(num_layers):
        d1 = i * step
        d2 = (i + 1) * step
        rho = densities[i] if i < len(densities) else densities[-1]
        vs = vs_values[i] if i < len(vs_values) else vs_values[-1]
        name = f"Layer_{i+1}"
        layers.append((name, d1, d2, rho))
    
    return layers

def generate_vs_layers_from_depth(depth_km, num_layers=15):
    """Generate Vs layers from depth."""
    if depth_km <= 0:
        return []
    
    step = depth_km / num_layers
    vs_values = [0.25, 0.35, 0.55, 2.20, 3.10, 2.80, 2.90, 3.10, 3.30, 3.50, 3.65, 3.75, 3.80, 3.85, 3.90]
    
    layers = []
    for i in range(num_layers):
        d1 = i * step
        d2 = (i + 1) * step
        vs = vs_values[i] if i < len(vs_values) else vs_values[-1]
        name = f"Layer_{i+1}"
        layers.append((name, d1, d2, vs))
    
    return layers

# ── Slip-Weighted Fc Model ────────────────────────────────────────
class SlipWeightedFC:
    """Slip-weighted corner frequency model for asperity-rich faults.
    
    Equation: fc_ij(t) = k * Vr * (D_ij/D̄)^0.5 * √(π/A_eff(t))
    """
    def __init__(self, p, k=0.5, Vr=None):
        self.p = p
        self.k = k  # scaling factor
        self.Vr = Vr or p.get('Vr_frac', 0.8) * 3.0  #rupture velocity
        
    def fc_slip_weighted(self, D_ij, D_avg, A_eff):
        """Calculate slip-weighted fc.
        
        Args:
            D_ij: Local slip on subfault
            D_avg: Average slip across fault
            A_eff: Effective rupture area
        """
        if D_avg <= 0 or A_eff <= 0:
            return 0.0
            
        slip_ratio = (D_ij / D_avg) ** 0.5  # (D_ij/D̄)^0.5
        
        return self.k * self.Vr * slip_ratio * np.sqrt(np.pi / A_eff)
    
    def generate_slip_field(self, nx=10, ny=7, stress_het=1.0):
        """Generate slip distribution with asperities.
        
        Args:
            nx, ny: Fault grid dimensions
            stress_het: Heterogeneity factor (1.0 = uniform, >1.0 = more asperities)
        """
        np.random.seed(self.p.get('seed', 42))
        
        # Base slip
        D_avg = 1.0
        
        # Generate heterogeneous slip field
        slip = np.random.exponential(D_avg, (ny, nx)) * stress_het
        
        return slip
    
    def simulate_slip_weighted(self, nx=10, ny=7, seed=42):
        """Simulate slip-weighted corner frequency field."""
        np.random.seed(seed)
        
        slip_field = self.generate_slip_field(nx, ny, stress_het=1.5)
        D_avg = np.mean(slip_field)
        
        # Generate fc for each subfault
        fc_field = np.zeros_like(slip_field)
        
        for i in range(ny):
            for j in range(nx):
                D_ij = slip_field[i, j]
                A_eff = nx * ny  # effective area
                fc_field[i, j] = self.fc_slip_weighted(D_ij, D_avg, A_eff)
        
        return fc_field, slip_field

