"""Inherited 64-state Neel probabilities along a stepped cooling protocol.

No particle rotation, orientation entropy or rotational entropy. Only Udd is
plotted for each pair branch. Nominal face vdW and kBT are separate curves.
Ramp propagation uses midpoint constant-temperature exponential substeps.
Each hold is propagated and averaged exactly by spectral evolution.
"""
from dataclasses import dataclass
from pathlib import Path
import csv
import json

import numpy as np
from scipy.constants import Boltzmann
from scipy.interpolate import PchipInterpolator
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import geometry_model as geometry
from pair_free_reference import ReversibleEvolution,equilibrium

OUT=Path(__file__).resolve().parents[1]/'outputs'/'neel_cooling_history'


@dataclass(frozen=True)
class Protocol:
    start_K: float=300.
    end_K: float=200.
    step_K: float=10.
    ramp_s: float=120.
    exposure_s: float=20.
    ramp_substep_K: float=.25


def calculate(kind,protocol=Protocol(),rates_scale=1.):
    """Return SAXS frame averages, full endpoint history and state audit."""
    if min(protocol.step_K,protocol.exposure_s,protocol.ramp_substep_K)<=0 or protocol.ramp_s<0:
        raise ValueError('Invalid protocol durations/grid')
    if protocol.start_K<protocol.end_K or rates_scale<0:
        raise ValueError('Cooling and nonnegative rate scale required')
    params=geometry.PARAMS
    n=np.array([1.,0,0]) if kind=='face' else np.ones(3)/np.sqrt(3)
    r=geometry.center_distance_at_gap_m(n,3e-9,params)
    energies,C,states,n=geometry.easy_axis_pair_energies_J(r*n,params)
    vdw=geometry.pair_vdw_energy_J(r*n,params)
    p=np.full(64,1/64)
    elapsed=0.
    frames=[];trajectory=[];probabilities=[]
    setpoints=np.arange(protocol.start_K,protocol.end_K-1e-8,-protocol.step_K)
    if abs(setpoints[-1]-protocol.end_K)>1e-8:
        raise ValueError('Temperature interval must be divisible by step_K')

    def evolution(T,prior):
        peq,_=equilibrium(energies,T)
        Q=geometry.neel_pair_generator(T,params.zfc_fc_activation_barrier_J,energies,C,states,n,params)
        return ReversibleEvolution(rates_scale*Q,peq,prior),peq

    def snapshot(T,stage,prob):
        return dict(geometry=kind,exposure_s=protocol.exposure_s,time_s=elapsed,
                    temperature_K=T,stage=stage,Udd_J=float(prob@energies),
                    Udd_kBT=float(prob@energies)/(Boltzmann*T))

    trajectory.append(snapshot(protocol.start_K,'initial',p))
    for index,T in enumerate(setpoints):
        previous_hold_end=p.copy()
        ramp_start=elapsed
        if index>0 and protocol.ramp_s>0:
            old_T=setpoints[index-1]
            count=int(np.ceil((old_T-T)/protocol.ramp_substep_K))
            boundaries=np.linspace(old_T,T,count+1)
            dt=protocol.ramp_s/count
            for left,right in zip(boundaries[:-1],boundaries[1:]):
                evo,_=evolution((left+right)/2,p)
                p=evo.at(dt)
                elapsed+=dt
                trajectory.append(snapshot(right,'ramp',p))
        incoming=p.copy()
        hold_start=elapsed
        evo,peq=evolution(T,incoming)
        mean=evo.mean(protocol.exposure_s)
        p=evo.at(protocol.exposure_s)
        # Independent-start comparator uses the SAME hold length, no ramp.
        reset_evo,_=evolution(T,np.full(64,1/64))
        reset_mean=reset_evo.mean(protocol.exposure_s)
        # Hold trajectory resolves the fast initial transient without
        # replacing the exposure integral by endpoint/midpoint energy.
        for dt in np.geomspace(1e-4,protocol.exposure_s,25):
            elapsed=hold_start+dt
            trajectory.append(snapshot(T,'hold',evo.at(dt)))
        elapsed=hold_start+protocol.exposure_s
        row=dict(geometry=kind,temperature_K=T,exposure_s=protocol.exposure_s,
                 ramp_duration_s=protocol.ramp_s,frame_index=index,
                 ramp_start_s=ramp_start,hold_start_s=hold_start,hold_end_s=elapsed,
                 center_distance_nm=r*1e9,Udd_hold_start_J=float(incoming@energies),
                 Udd_hold_mean_J=float(mean@energies),Udd_hold_end_J=float(p@energies),
                 Udd_eq_J=float(peq@energies),Udd_reset_same_window_J=float(reset_mean@energies),
                 vdW_J=vdw,kBT_J=Boltzmann*T,
                 probability_normalization_error=abs(p.sum()-1),
                 probability_change_during_ramp_L1=float(abs(incoming-previous_hold_end).sum()))
        for key,value in list(row.items()):
            if key.endswith('_J'):row[key[:-2]+'_kBT']=value/(Boltzmann*T)
        frames.append(row)
        for state in range(64):
            probabilities.append(dict(geometry=kind,temperature_K=T,exposure_s=protocol.exposure_s,
                state_index=state,p_previous_hold_end=previous_hold_end[state],
                p_hold_start=incoming[state],p_hold_mean=mean[state],p_hold_end=p[state],p_eq=peq[state]))
    return frames,trajectory,probabilities


def write_csv(path,rows):
    with path.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def draw(rows,out,window):
    plt.rcParams.update({'font.family':'Arial','font.size':14,'axes.labelsize':17,
                         'axes.linewidth':1.8,'axes.grid':False,'pdf.fonttype':42})
    dense=np.linspace(200,300,501)
    for unit in ('kBT','J'):
        fig,ax=plt.subplots(figsize=(7.3,9.4));ax.set_box_aspect(1)
        for kind,color in [('face','#B54535'),('tip','#DE9945')]:
            data=sorted([r for r in rows if r['geometry']==kind and r['exposure_s']==window],
                        key=lambda r:r['temperature_K'])
            ts=[r['temperature_K'] for r in data]
            vals=[r['Udd_hold_mean_'+unit] for r in data]
            ax.plot(dense,PchipInterpolator(ts,vals)(dense),color=color,lw=2.5,label=kind+', inherited history')
        face=next(r for r in rows if r['geometry']=='face')
        ax.plot(dense,np.full_like(dense,face['vdW_J'])/(Boltzmann*dense) if unit=='kBT' else
                np.full_like(dense,face['vdW_J']),color='#287A96',lw=2,label='face vdW (separate)')
        ax.plot(dense,np.ones_like(dense) if unit=='kBT' else Boltzmann*dense,
                color='black',lw=1.5,label=r'$k_BT$ (reference)')
        ax.axvspan(233.15,293.15,color='#EEEEEE',zorder=0,label='Experimental range')
        ax.axvspan(253.15,273.15,color='#F4EAA2',alpha=.8,zorder=1,label='Transient aggregation (experiment)')
        ax.axhline(0,color='0.6',lw=.8)
        ax.set(xlabel='Temperature (K)',ylabel=r'Energy / $k_BT$' if unit=='kBT' else 'Energy (J)',
               xlim=(200,300),title=f'Cooling history: Neel dynamics only\n120 s per 10 K ramp; {window:g} s hold average')
        ax.title.set_fontsize(14)
        if unit=='kBT':ax.set_ylim(-10,2)
        else:ax.ticklabel_format(axis='y',style='sci',scilimits=(0,0))
        ax.legend(loc='upper left',bbox_to_anchor=(0,-.17),frameon=False)
        fig.subplots_adjust(left=.18,right=.97,top=.89,bottom=.34)
        for ext in ('png','pdf'):
            fig.savefig(out/f'magnetic_energy_history_{window:g}s_{unit}.{ext}',dpi=300,bbox_inches='tight')
        plt.close(fig)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    frames=[];trajectory=[];probabilities=[]
    for window in (20.,150.):
        for kind in ('face','tip'):
            f,t,p=calculate(kind,Protocol(exposure_s=window))
            frames+=f;trajectory+=t;probabilities+=p
            print(kind,window,[(r['temperature_K'],round(r['Udd_hold_mean_kBT'],6)) for r in f if r['temperature_K'] in (200,250,300)],flush=True)
    write_csv(OUT/'frame_energies.csv',frames)
    write_csv(OUT/'time_trajectory.csv',trajectory)
    write_csv(OUT/'state_probabilities.csv',probabilities)
    for window in (20.,150.):draw(frames,OUT,window)
    config=dict(initial='uniform 64-state prior at 300 K, only once',start_hold='included at 300 K',
        end_K=200,step_K=10,ramp_s=120,hold_s=[20,150],ramp_substep_K=.25,
        plotted='exact hold-average Udd, independent face vdW and kBT references',
        dynamics='Neel only, probability inherited through every ramp and hold',
        body_rotation=False,entropy_terms=False,penalty_terms=False,
        smoothing='PCHIP display of 11 actual frame averages, not extra temperature measurements',
        geometry='fixed hypothetical face and tip pairs throughout, no structure conversion',
        assumptions='constant Ms; discrete easy-axis minima; no intrawell fluctuations or assembly population')
    (OUT/'model_config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')


if __name__=='__main__':main()
