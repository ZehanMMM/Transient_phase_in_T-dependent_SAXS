"""Three-stage comparison: face axial rotation, tip full 3D rotation.

Adiabatic Brownian approximation: allowed body rotations equilibrate much
faster than 20 s. Face axial-sign populations evolve via a reduced Neel
master equation whose well/saddle free energies integrate the fast twist.
Tip lab-frame dipoles can equilibrate by body rotation even if Neel-blocked.
Intrinsic spins stay at cubic minima. Fixed nominal vdW is shown separately,
never included in the face/tip main curves at any stage.
No lost-energy penalty and no equilibrium-deficit term is added to any curve.
"""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import Boltzmann, mu_0
from scipy.integrate import quad
from scipy.special import expit, i0e, i1e, logsumexp, xlogy

import geometry_model as geometry
from pair_energy_model import brownian_time_s

OUT=Path(__file__).resolve().parents[1]/'outputs'/'axial_face_free_tip'


def log_i0(x):
    return np.log(i0e(x))+np.abs(x)


def binary_entropy_cost(p):
    return float(xlogy(p,2*p)+xlogy(1-p,2*(1-p)))


def face_model(C_J, temperature_K, window_s):
    """Exact axial-spin reduction under fast independent rotations about [100].

    s1x,s2x = +/-1/sqrt(3). With a=sign(s1x*s2x), phi=relative azimuth,
    Edd = -J*a + J*cos(phi), J=2C/3. Fast phi integral gives I0(beta J).
    The 32 states of each a group have equal probability by symmetry.
    Only an x-sign Neel flip changes a, one per particle. Both pathways
    pass through an x=0 saddle with transverse amplitude C*sqrt(2/3).
    """
    kbt=Boltzmann*temperature_K
    J=2*C_J/3
    x=J/kbt
    logz_phi=log_i0(x)
    u_phi=-J*i1e(x)/i0e(x)
    saddle_F=-kbt*log_i0(C_J*np.sqrt(2/3)/kbt)
    well_F=np.array([-J,J])-kbt*logz_phi
    barriers=geometry.PARAMS.zfc_fc_activation_barrier_J+saddle_F-well_F
    if np.min(barriers)<0:
        raise ValueError('Negative barrier: adiabatic transition-state model invalid')
    rates=2/(3*geometry.PARAMS.attempt_time_s)*np.exp(-barriers/kbt)
    rate=float(rates.sum())
    peq=float(expit(2*x))
    if not np.isclose(rates[1]/rate,peq,atol=1e-12):
        raise ArithmeticError('Detailed balance failed')

    def p_at(t):
        return peq+(.5-peq)*np.exp(-rate*t)

    argument=rate*window_s
    decay_mean=-np.expm1(-argument)/argument if argument else 1.0
    pmean=peq+(.5-peq)*decay_mean
    # Direct entropy integral, not Feq + excess. Resolve initial fast modes.
    first=min(window_s,.01/rate)
    nodes=np.unique(np.r_[0.,np.geomspace(first,window_s,16)])
    entropy_mean=sum(quad(lambda t: binary_entropy_cost(p_at(t)),a,b,
                          epsabs=window_s*1e-11,epsrel=1e-9)[0]
                     for a,b in zip(nodes[:-1],nodes[1:]))/window_s
    u=-J*(2*pmean-1)+u_phi
    f=-J*(2*pmean-1)-kbt*logz_phi+kbt*entropy_mean
    feq=-kbt*(logz_phi+np.logaddexp(x,-x)-np.log(2))
    return dict(Udd_J=u,F_orient_J=f,orientation_entropy_cost_J=f-u,
                F_orient_eq_J=feq,p_parallel_initial=.5,p_parallel_mean=pmean,
                p_parallel_end=p_at(window_s),p_parallel_eq=peq,
                face_sign_relaxation_s=1/rate,
                fast_azimuth_Udd_J=u_phi,
                slow_axial_Udd_J=-J*(2*pmean-1))


def tip_model(C_J, temperature_K, order=160):
    """Normalized partition of two freely rotating fixed-magnitude dipoles.

    Integrating the second sphere analytically gives Z = integral[-1,1]
    sinh(lambda*sqrt(1+3u^2))/(lambda*sqrt(1+3u^2)) du/2.
    """
    kbt=Boltzmann*temperature_K
    u,w=np.polynomial.legendre.leggauss(order)
    v=np.sqrt(1+3*u*u)
    x=C_J*v/kbt
    if np.max(np.abs(x))<1e-5:
        log_sinhc=x*x/6-x**4/180
        langevin=x/3-x**3/45
    else:
        log_sinhc=x+np.log(-np.expm1(-2*x))-np.log(2*x)
        langevin=1/np.tanh(x)-1/x
    logs=np.log(w/2)+log_sinhc
    lz=logsumexp(logs)
    p=np.exp(logs-lz)
    energy=-C_J*float(p@(v*langevin))
    free=-kbt*lz
    return dict(Udd_J=energy,F_orient_J=free,
                orientation_entropy_cost_J=free-energy,F_orient_eq_J=free,
                p_parallel_initial=np.nan,p_parallel_mean=np.nan,
                p_parallel_end=np.nan,p_parallel_eq=np.nan,
                face_sign_relaxation_s=np.nan,
                fast_azimuth_Udd_J=np.nan,slow_axial_Udd_J=np.nan)


def cage_cost(kind, temperature_K, tilt_deg=2):
    """Geometric measure only. Alignment entropy within domain is already in F.

    Face: any of 6 <100> normals in a small cone, twist unrestricted.
    Tip/free-body branch: entire SO(3), f=1. No factor 8 or extra twist cost.
    Face cone regularizes exact-axis constraint; magnetic calculation uses
    the centre of this narrow cone (small-cone approximation).
    """
    if not 0<tilt_deg<10:
        raise ValueError('Small face cone must have halfangle in (0,10) degrees')
    fraction=3*(1-np.cos(np.deg2rad(tilt_deg))) if kind=='face' else 1.
    return -2*Boltzmann*temperature_K*np.log(fraction),fraction


def compute(temperature_K,window_s=20,tilt_deg=2):
    rows=[]
    params=geometry.PARAMS
    for kind,direction in [('face',np.array([1.,0,0])),('tip',np.ones(3)/np.sqrt(3))]:
        r=geometry.center_distance_at_gap_m(direction,3e-9,params)
        moment=params.saturation_magnetization_Apm*params.particle_volume_m3
        C=mu_0*moment**2/(4*np.pi*r**3)
        values=face_model(C,temperature_K,window_s) if kind=='face' else tip_model(C,temperature_K)
        vdw=geometry.pair_vdw_energy_J(r*direction,params)
        rot,fraction=cage_cost(kind,temperature_K,tilt_deg)
        row=dict(geometry=kind,temperature_K=temperature_K,window_s=window_s,
                 center_distance_nm=r*1e9,face_tilt_halfangle_deg=tilt_deg,
                 orientation_fraction=fraction,**values,
                 UvdW_J=vdw,rotation_constraint_F_J=rot,
                 stage1_Udd_J=values['Udd_J'],
                 stage2_F_orient_J=values['F_orient_J'],
                 stage3_F_with_rotation_J=values['F_orient_J']+rot,
                 free_Brownian_time_s=float(brownian_time_s(np.array([temperature_K]))[0]),
                 assembly_free_energy_J=np.nan)
        if values['orientation_entropy_cost_J'] < -1e-10*Boltzmann*temperature_K:
            raise ArithmeticError('Negative entropy loss relative to uniform allowed states')
        for key,value in list(row.items()):
            if key.endswith('_J'):
                row[key[:-2]+'_kBT']=value/(Boltzmann*temperature_K)
        rows.append(row)
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--window-s',type=float,default=20)
    parser.add_argument('--face-tilt-deg',type=float,default=2)
    parser.add_argument('--output-dir',type=Path,default=OUT)
    args=parser.parse_args()
    if args.window_s<=0:
        parser.error('Window must be positive')
    out=args.output_dir.resolve()
    out.mkdir(parents=True,exist_ok=True)
    rows=[row for t in np.arange(200.,301.) for row in compute(t,args.window_s,args.face_tilt_deg)]
    with (out/'three_stage_data.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    plt.rcParams.update({'font.family':'Arial','font.size':13,'axes.labelsize':16,
                         'axes.grid':False,'axes.linewidth':1.8,'pdf.fonttype':42})
    stages=[('stage1_Udd','1. Dipolar energy (vdW shown separately)'),
            ('stage2_F_orient','2. Add dipole-orientation entropy'),
            ('stage3_F_with_rotation','3. Add body-orientation constraint')]
    for number,(field,title) in enumerate(stages,1):
        for unit in ('kBT','J'):
            fig,ax=plt.subplots(figsize=(7.3,9.1))
            ax.set_box_aspect(1)
            ax.axvspan(233.15,293.15,color='#EEEEEE',zorder=0,label='Experimental range')
            ax.axvspan(253.15,273.15,color='#F4EAA2',alpha=.8,zorder=1,label='Transient aggregation (experiment)')
            for kind,color in [('face','#B54535'),('tip','#DF9A45')]:
                data=[r for r in rows if r['geometry']==kind]
                ax.plot([r['temperature_K'] for r in data],[r[field+'_'+unit] for r in data],
                        color=color,lw=2.5,label=kind+' (no vdW)')
            data=[r for r in rows if r['geometry']=='face']
            ax.plot([r['temperature_K'] for r in data],[r['UvdW_'+unit] for r in data],
                    color='#287A96',lw=1.8,label='face vdW (separate)')
            ax.axhline(0,color='0.5',lw=.8)
            ax.set(xlabel='Temperature (K)',ylabel=('Energy' if number==1 else 'Partial free energy')+
                   (r' / $k_BT$' if unit=='kBT' else ' (J)'),xlim=(200,300),
                   title=title+f'\n{args.window_s:g} s Neel; fast allowed body rotation')
            ax.title.set_fontsize(14)
            if number in (1,2):
                if unit=='kBT':
                    ax.set_ylim(-10,1)
                else:
                    lower=min(r['stage1_Udd_J'] for r in rows)
                    ax.set_ylim(1.12*lower,-.06*lower)
            if unit=='J':
                ax.ticklabel_format(axis='y',style='sci',scilimits=(0,0))
            h,l=ax.get_legend_handles_labels()
            ax.legend(h[2:]+h[:2],l[2:]+l[:2],loc='upper left',bbox_to_anchor=(0,-.17),frameon=False)
            fig.subplots_adjust(left=.18,right=.96,top=.89,bottom=.31)
            for ext in ('png','pdf'):
                fig.savefig(out/f'stage{number}_{unit}.{ext}',dpi=300,bbox_inches='tight')
            plt.close(fig)
    # Directly show the nonzero entropy increment, without changing stage axes.
    fig,ax=plt.subplots(figsize=(6.7,5.5))
    for kind,color in [('face','#B54535'),('tip','#DF9A45')]:
        data=[r for r in rows if r['geometry']==kind]
        ax.plot([r['temperature_K'] for r in data],[r['orientation_entropy_cost_kBT'] for r in data],
                color=color,lw=2,label=kind)
    ax.set(xlabel='Temperature (K)',ylabel=r'$(F_{orientation}-U_{dd})/k_BT$',
           title='Orientation-entropy contribution (stage 2 minus stage 1)')
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out/'entropy_increment.png',dpi=250)
    plt.close(fig)
    config=dict(window_s=args.window_s,face='axial rotation only',tip='full 3D body rotation',
                Brownian_treatment='adiabatic conditional equilibrium, not explicit trajectory',
                intrinsic_moments='8 cubic minima in body frame',
                face_tilt_halfangle_deg=args.face_tilt_deg,
                face_tilt_treatment='entropy regularization only; magnetic small-cone limit',
                rate_model='new adiabatic angular free energies at wells and saddles',
                reference='two uncoupled particles with free body orientations',
                vdW='inherited nominal-posture 4^3 voxel energy; not angle averaged',
                plotted_main_curves='stage1 Udd; stage2 F_orient; stage3 F_orient + rotation constraint; all exclude vdW',
                geometry_warning='free tip rotations do not maintain a literal tip-contact geometry',
                assembly_free_energy='not computed: ligand PMF, positional entropy, many-body constraints absent')
    (out/'model_config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    for row in rows:
        if row['temperature_K'] in (200,250,300):
            print({k:row[k] for k in ['geometry','temperature_K','Udd_kBT',
                  'orientation_entropy_cost_kBT','rotation_constraint_F_kBT',
                  'stage1_Udd_kBT','stage2_F_orient_kBT','stage3_F_with_rotation_kBT']})


if __name__=='__main__':
    main()
