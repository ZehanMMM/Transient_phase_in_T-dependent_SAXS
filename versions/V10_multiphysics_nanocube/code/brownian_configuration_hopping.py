"""20 s magnetic energy with whole-cube configuration hopping.

Endpoint cube registries differ by +/-90 degree rotations about <100> axes.
Each preserves an endpoint face/tip geometry of an ideal cubic-symmetric
particle, but can change which material face/vertex points to the partner.
Intrinsic moments remain at body-frame easy minima during Brownian jumps.
The 24 body registries project onto 8 lab easy directions (3-fold redundant),
so the energy/rate process closes on the inherited 64 pair directions.

Brownian prefactors reproduce the isolated first-rank orientational
correlation time tauB, NOT a mean first-escape time. Magnetic path barriers
are evaluated along rigid rotations. Additional ligand/contact barriers
are zero only as an explicitly optimistic accessibility baseline.
No rotational entropy, orientation entropy, or energy-deficit penalty is
added to Udd. vdW is a separate nominal-geometry reference curve.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.constants import Boltzmann
from scipy.special import logsumexp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import geometry_model as geometry
from pair_energy_model import brownian_time_s,hexane_viscosity_Pa_s
from pair_free_reference import ReversibleEvolution

OUT=Path(__file__).resolve().parents[1]/'outputs'/'brownian_configuration_hopping'


def path_maximum(C, moment, other, axis, link):
    """Exact max of dipolar energy along a +pi/2 rotation about signed axis."""
    field=other-3*np.dot(other,link)*link
    parallel=np.dot(moment,axis)*axis
    A=C*np.dot(moment-parallel,field)
    B=C*np.dot(np.cross(axis,moment),field)
    D=C*np.dot(parallel,field)
    candidates=[D+A,D+B]
    stationary=np.arctan2(B,A)
    for theta in (stationary,stationary+np.pi,stationary-np.pi):
        if 0<=theta<=np.pi/2:
            candidates.append(D+A*np.cos(theta)+B*np.sin(theta))
    return max(candidates)


def brownian_generator(temperature_K,energies,C,states,link,tau_B,
                       contact_barrier_J=0.,mobility=1.):
    if tau_B<=0 or mobility<0 or contact_barrier_J<0:
        raise ValueError('Positive tauB and nonnegative mobility/contact barrier required')
    Q=np.zeros((64,64))
    lookup={tuple(np.sign(s).astype(int)):i for i,s in enumerate(states)}
    # Six signed quarter-turn moves per particle. At zero coupling their
    # action on a vector is sum(R-I)m=-4m, so k0=1/(4*tauB).
    attempt=mobility/(4*tau_B)
    largest_barrier=0.
    for i,m1 in enumerate(states):
        for j,m2 in enumerate(states):
            source=8*i+j
            for particle in (0,1):
                m,other=(m1,m2) if particle==0 else (m2,m1)
                for axis0 in np.eye(3):
                    for sign in (-1,1):
                        axis=sign*axis0
                        destination_m=np.cross(axis,m)+axis*np.dot(axis,m)
                        dest=lookup[tuple(np.sign(destination_m).astype(int))]
                        destination=8*dest+j if particle==0 else 8*i+dest
                        saddle=path_maximum(C,m,other,axis,link)
                        barrier=max(saddle-energies[source],0.)+contact_barrier_J
                        largest_barrier=max(largest_barrier,barrier)
                        Q[source,destination]+=attempt*np.exp(-barrier/(Boltzmann*temperature_K))
            Q[source,source]=-Q[source].sum()
    return Q,largest_barrier


def compute(temperature_K,kind,window_s=20.,mobility=1.,contact_barrier_kBT=0.):
    params=geometry.PARAMS
    link=np.array([1.,0,0]) if kind=='face' else np.ones(3)/np.sqrt(3)
    r=geometry.center_distance_at_gap_m(link,3e-9,params)
    energy,C,states,link=geometry.easy_axis_pair_energies_J(r*link,params)
    kbt=Boltzmann*temperature_K
    pi=np.exp(-energy/kbt-logsumexp(-energy/kbt))
    initial=np.full(64,1/64)
    QN=geometry.neel_pair_generator(temperature_K,params.zfc_fc_activation_barrier_J,
                                   energy,C,states,link,params)
    tau_B=float(brownian_time_s(np.array([temperature_K]))[0])
    QB,maxbar=brownian_generator(temperature_K,energy,C,states,link,tau_B,
                                contact_barrier_kBT*kbt,mobility)
    evo=ReversibleEvolution(QN+QB,pi,initial)
    mean=evo.mean(window_s);end=evo.at(window_s)
    old=ReversibleEvolution(QN,pi,initial)
    # Relaxation time of the slowest nonstationary mode of the joint graph.
    tau_joint=-1/evo.eigenvalues[-2]
    row=dict(geometry=kind,temperature_K=temperature_K,window_s=window_s,
             center_distance_nm=r*1e9,rotation_mobility=mobility,
             extra_contact_barrier_kBT=contact_barrier_kBT,
             tau_B_free_s=tau_B,tau_joint_s=tau_joint,
             hexane_viscosity_Pa_s=float(hexane_viscosity_Pa_s(np.array([temperature_K]))[0]),
             Udd_time_mean_J=float(mean@energy),Udd_end_J=float(end@energy),
             Udd_eq_J=float(pi@energy),Udd_initial_J=float(initial@energy),
             Neel_only_Udd_time_mean_J=float(old.mean(window_s)@energy),
             vdW_J=geometry.pair_vdw_energy_J(r*link,params),
             Brownian_max_transition_barrier_J=maxbar,
             detailed_balance_relative_error=evo.balance_error,kBT_J=kbt)
    for field,value in list(row.items()):
        if field.endswith('_J'):
            row[field[:-2]+'_kBT']=value/kbt
    return row


def draw(rows,out,window):
    plt.rcParams.update({'font.family':'Arial','font.size':14,'axes.labelsize':17,
                         'axes.linewidth':1.8,'axes.grid':False,'pdf.fonttype':42})
    for unit in ('kBT','J'):
        fig,ax=plt.subplots(figsize=(7.3,10.5));ax.set_box_aspect(1)
        for kind,color in [('face','#B54535'),('tip','#DE9945')]:
            data=[r for r in rows if r['geometry']==kind];ts=[r['temperature_K'] for r in data]
            ax.plot(ts,[r['Udd_time_mean_'+unit] for r in data],color=color,lw=2.8,
                    label=kind+f', {window:g} s')
            ax.plot(ts,[r['Udd_eq_'+unit] for r in data],color=color,lw=1.6,ls='--',
                    label=kind+', equilibrium')
        data=[r for r in rows if r['geometry']=='face'];ts=[r['temperature_K'] for r in data]
        ax.plot(ts,[r['vdW_'+unit] for r in data],color='#287A96',lw=2,label='face vdW (separate)')
        ax.plot(ts,[1 if unit=='kBT' else r['kBT_J'] for r in data],color='black',lw=1.5,label=r'$k_BT$ (reference)')
        ax.axvspan(233.15,293.15,color='#EEEEEE',zorder=0,label='Experimental range')
        ax.axvspan(253.15,273.15,color='#F4EAA2',alpha=.8,zorder=1,label='Transient aggregation (experiment)')
        ax.axhline(0,color='0.6',lw=.8)
        ax.set(xlabel='Temperature (K)',ylabel=r'Energy / $k_BT$' if unit=='kBT' else 'Energy (J)',
               xlim=(200,300),title=f'Magnetic energy: Neel + body-configuration hopping\n{window:g} s; no additional contact barrier')
        ax.title.set_fontsize(13)
        if unit=='kBT':ax.set_ylim(-10,2)
        else:ax.ticklabel_format(axis='y',style='sci',scilimits=(0,0))
        ax.legend(loc='upper left',bbox_to_anchor=(0,-.17),frameon=False)
        fig.subplots_adjust(left=.18,right=.97,top=.89,bottom=.40)
        for ext in ('png','pdf'):fig.savefig(out/f'magnetic_energy_{unit}.{ext}',dpi=300,bbox_inches='tight')
        plt.close(fig)
    # Keep old-vs-new comparisons and fast relaxation auditable, off main plot.
    fig,ax=plt.subplots(figsize=(7.3,5.8))
    for kind,color in [('face','#B54535'),('tip','#DE9945')]:
        data=[r for r in rows if r['geometry']==kind];ts=[r['temperature_K'] for r in data]
        ax.plot(ts,[r['Udd_time_mean_kBT'] for r in data],color=color,lw=2.5,label=kind+', Neel + body hopping')
        ax.plot(ts,[r['Neel_only_Udd_time_mean_kBT'] for r in data],color=color,lw=1.7,ls=':',label=kind+', old Neel only')
    ax.set(xlabel='Temperature (K)',ylabel=r'$U_{dd}/k_BT$',xlim=(200,300),title='Effect of configuration hopping')
    ax.legend(fontsize=10);fig.tight_layout();fig.savefig(out/'comparison_with_Neel_only.png',dpi=300);plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--window-s',type=float,default=20.)
    parser.add_argument('--output-dir',type=Path,default=OUT)
    args=parser.parse_args()
    if args.window_s<=0:parser.error('window must be positive')
    args.output_dir.mkdir(parents=True,exist_ok=True)
    rows=[compute(t,g,args.window_s) for g in ('face','tip') for t in np.arange(200.,301.)]
    with (args.output_dir/'energy_data.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    draw(rows,args.output_dir,args.window_s)
    config=dict(window_s=args.window_s,body_moves='six +/-90-degree <100> rigid rotations per particle',
                Brownian_per_path_attempt='1/(4*tau_B), calibrated to first-rank isolated decay',
                Brownian_barrier='exact maximum dipolar energy along each rigid rotation minus initial energy',
                contact_barrier='zero baseline, not experimentally measured',
                initial='uniform 64 lab directions, independently reset at every T',
                Hamiltonian='same old 64 endpoint Udd values; body-frame Eani at zero minima',
                geometry='same endpoint face/tip centre distances; no face-to-tip conversion or translation simulated',
                path_warning='hard-core/ligand path accessibility NOT verified; face recontact may require separating particles',
                state_reduction='24 cubic body registries map onto 8 lab moments; other 3-fold degeneracy is lumped',
                plotted='Udd only; no entropy, deficit, rotational penalty or vdW added',
                drag='inherited sphere-equivalent hexane rotational drag; no many-body contact friction')
    (args.output_dir/'model_config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    for r in rows:
        if r['temperature_K'] in (200,250,300):
            print({key:r[key] for key in ('geometry','temperature_K','Udd_time_mean_kBT','Udd_eq_kBT','Neel_only_Udd_time_mean_kBT','tau_B_free_s','tau_joint_s')})


if __name__=='__main__':main()
