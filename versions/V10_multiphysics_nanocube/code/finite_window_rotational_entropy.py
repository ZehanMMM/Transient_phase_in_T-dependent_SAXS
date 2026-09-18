"""20 s Neel baseline plus a factorized, equilibrated local rotation entropy.

This is NOT coupled Brownian/Neel dynamics: body orientations do not change
the inherited generator. Energies are unchanged. Only free energies acquire
the SO(3) cage-measure cost. The original 4^3 voxel vdW is retained explicitly
to isolate this change from the existing 20 s baseline.
"""
from pathlib import Path
import argparse
import csv
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from pair_free_reference import compute_point
from rotational_entropy_model import orientation_fraction

OUT = Path(__file__).resolve().parents[1]/'outputs'/'finite_window_rotational_entropy'

# All main fields exclude vdW. Keep the original fixed-geometry generator
# and the original face +/-5 degree / tip axial-twist entropy convention.
SEPARATED_STAGES = (
    ('Udd_time_mean', 'Udd_eq', '1. Magnetic interaction energy'),
    ('delta_Fmag_time_mean', 'delta_Fmag_eq', '2. Magnetic free energy'),
    ('Fmag_plus_rotation_time_mean', 'Fmag_plus_rotation_equilibrium',
     '3. Magnetic free energy + rotation entropy'),
)


def plot_separated_stages(rows, out, window_s):
    """Replot the legacy model, not the adiabatic free-body model."""
    selected=[r for r in rows if r['face_twist_halfwidth_deg']==5 or
              r.get('rotation_domain')=='axial-face-free-tip']
    write_csv(out/'separated_three_stage_data.csv',selected)
    plt.rcParams.update({'font.family':'Arial','font.size':14,'axes.labelsize':17,
                         'axes.linewidth':1.8,'axes.grid':False,'pdf.fonttype':42})
    for number,(field,eq_field,title) in enumerate(SEPARATED_STAGES,1):
        for unit in ('kBT','J'):
            fig,ax=plt.subplots(figsize=(7.3,10.5))
            ax.set_box_aspect(1)
            for kind,color in [('face','#B54535'),('tip','#DE9945')]:
                data=[r for r in selected if r['geometry']==kind]
                ts=[r['temperature_K'] for r in data]
                ax.plot(ts,[r[field+'_'+unit] for r in data],color=color,lw=2.5,
                        label=kind+f', {window_s:g} s')
                ax.plot(ts,[r[eq_field+'_'+unit] for r in data],color=color,lw=1.8,
                        ls='--',label=kind+', equilibrium')
            data=[r for r in selected if r['geometry']=='face']
            ts=[r['temperature_K'] for r in data]
            ax.plot(ts,[r['vdW_'+unit] for r in data],color='#287A96',lw=2,
                    label='face vdW (separate)')
            ax.plot(ts,[1 if unit=='kBT' else r['kBT_J'] for r in data],
                    color='black',lw=1.5,label=r'$k_BT$ (reference)')
            ax.axvspan(233.15,293.15,color='#EEEEEE',zorder=0,label='Experimental range')
            ax.axvspan(253.15,273.15,color='#F4EAA2',alpha=.8,zorder=1,
                       label='Transient aggregation (experiment)')
            ax.axhline(0,color='0.6',lw=.8)
            subtitle=(f'{window_s:g} s window; fixed-geometry Neel model' if number<3 else
                      f'{window_s:g} s window; legacy face twist +/- 5 deg')
            if number==3 and selected[0].get('rotation_domain')=='axial-face-free-tip':
                subtitle=f'{window_s:g} s; face axial rotation, tip free rotation'
            ax.set(xlabel='Temperature (K)',ylabel=('Energy' if number==1 else 'Free energy')+
                   (r' / $k_BT$' if unit=='kBT' else ' (J)'),xlim=(200,300),
                   title=title+'\n'+subtitle)
            ax.title.set_fontsize(14)
            if number<3:
                if unit=='kBT':
                    ax.set_ylim(-10,2)
                else:
                    low=min(r['Udd_eq_J'] for r in selected)
                    ax.set_ylim(1.1*low,-.2*low)
            if unit=='J':
                ax.ticklabel_format(axis='y',style='sci',scilimits=(0,0))
            ax.legend(loc='upper left',bbox_to_anchor=(0,-.17),frameon=False)
            fig.subplots_adjust(left=.18,right=.97,top=.89,bottom=.40)
            for ext in ('png','pdf'):
                fig.savefig(out/f'separated_stage{number}_{unit}.{ext}',dpi=300,bbox_inches='tight')
            plt.close(fig)


def updated_rotation_fraction(kind,tilt):
    """Allowed SO(3) measure: full tip, six face-normal cones x free twist.

    Exact axis confinement has zero continuous measure. The inherited tilt
    tolerance regularizes that limit. No rotation affects magnetic dynamics.
    """
    if not 0<tilt<=5:
        raise ValueError('Face tilt tolerance must be in (0,5] degrees')
    if kind=='tip':return 1.
    if kind=='face':return 3*(1-np.cos(np.deg2rad(tilt)))
    raise ValueError(kind)


def add_entropy(base, kind, delta=5., tilt=2., rotation_domain='legacy'):
    if rotation_domain=='legacy':
        fraction=orientation_fraction(kind,tilt,delta)
    elif rotation_domain=='axial-face-free-tip':
        fraction=updated_rotation_fraction(kind,tilt)
    else:
        raise ValueError(rotation_domain)
    cost = -2*np.log(fraction)
    row = dict(geometry=kind, face_twist_halfwidth_deg=(delta if rotation_domain=='legacy' else 180.),
               tilt_halfangle_deg=tilt,rotation_domain=rotation_domain,
               body_orientation_fraction=fraction, **base)
    row['rotation_entropy_change_over_kB'] = -cost
    row['rotation_free_energy_cost_kBT'] = cost
    row['Fmag_plus_rotation_time_mean_kBT'] = base['delta_Fmag_time_mean_kBT']+cost
    row['partial_contact_F_with_rotation_kBT'] = (
        base['delta_Fmag_time_mean_kBT']+cost+base['vdW_kBT'])
    row['Fmag_plus_rotation_equilibrium_kBT'] = base['delta_Fmag_eq_kBT']+cost
    row['partial_contact_F_with_rotation_equilibrium_kBT'] = (
        base['delta_Fmag_eq_kBT']+cost+base['vdW_kBT'])
    for field in ('rotation_free_energy_cost','Fmag_plus_rotation_time_mean',
                  'partial_contact_F_with_rotation',
                  'Fmag_plus_rotation_equilibrium',
                  'partial_contact_F_with_rotation_equilibrium'):
        row[field+'_J'] = row[field+'_kBT']*base['kBT_J']
    return row


def write_csv(path, rows):
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def roots(x,y):
    result=[]
    for i in range(len(x)-1):
        if y[i] == 0:
            result.append(float(x[i]))
        elif y[i]*y[i+1]<0:
            result.append(float(x[i]-y[i]*(x[i+1]-x[i])/(y[i+1]-y[i])))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--window-s',type=float,default=20)
    parser.add_argument('--output-dir',type=Path,default=OUT)
    parser.add_argument('--separated-only',action='store_true',
                        help='Create three no-vdW main-curve plots, preserving legacy figures')
    parser.add_argument('--rotation-domain',choices=('legacy','axial-face-free-tip'),default='legacy')
    parser.add_argument('--face-tilt-deg',type=float,default=2.)
    args=parser.parse_args()
    if args.window_s<=0:
        parser.error('window must be positive')
    out=args.output_dir.resolve()
    if args.rotation_domain!='legacy' and out==OUT.resolve():
        out=out/'axial_face_free_tip_entropy_only'
    out.mkdir(parents=True,exist_ok=True)
    ts=np.arange(200.,301.)
    bases={g:[compute_point(t,d,args.window_s)[0] for t in ts]
           for g,d in [('face',[1,0,0]),('tip',[1,1,1])]}
    if args.rotation_domain!='legacy':
        rows=[add_entropy(b,g,tilt=args.face_tilt_deg,rotation_domain=args.rotation_domain)
              for g in bases for b in bases[g]]
        plot_separated_stages(rows,out,args.window_s)
        write_csv(out/'energies_and_free_energies.csv',rows)
        comparison=[]
        for r in rows:
            old_fraction=orientation_fraction(r['geometry'],args.face_tilt_deg,5.)
            old_cost=-2*np.log(old_fraction)
            comparison.append(dict(geometry=r['geometry'],temperature_K=r['temperature_K'],
                old_rotation_cost_kBT=old_cost,new_rotation_cost_kBT=r['rotation_free_energy_cost_kBT'],
                free_energy_change_kBT=r['rotation_free_energy_cost_kBT']-old_cost,
                Udd_kBT=r['Udd_time_mean_kBT'],
                new_Fmag_plus_rotation_kBT=r['Fmag_plus_rotation_time_mean_kBT']))
        write_csv(out/'rotation_change.csv',comparison)
        config=dict(window_s=args.window_s,rotation_domain=args.rotation_domain,
                    face_tilt_deg=args.face_tilt_deg,face_twist='full 2pi',tip_orientation='full SO(3)',
                    initial='uniform 64 states reset independently at each T; NO cooling history',
                    dynamics='unchanged fixed-body Neel model, not free-body dynamics',
                    entropy='factorized allowed-domain measure, per-pair -2kBT ln(f)',
                    vdW='separate reference, excluded from all main curves',
                    warning='entropy-only sensitivity; not a self-consistent joint rotational/magnetic model')
        (out/'model_config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
        for r in comparison:
            if r['temperature_K']==250:print(r)
        print(f'Entropy-only updated plots: {out}')
        return
    rows=[add_entropy(b,g,delta) for delta in (1.,2.,5.,10.)
          for g in bases for b in bases[g]]
    plot_separated_stages(rows,out,args.window_s)
    if args.separated_only:
        print(f'Separated legacy-model plots: {out}')
        return
    write_csv(out/'energies_and_free_energies.csv',rows)
    differences=[]
    for delta in (1.,2.,5.,10.):
        for f,t in zip(bases['face'],bases['tip']):
            rf,rt=add_entropy(f,'face',delta),add_entropy(t,'tip',delta)
            dc=rt['rotation_free_energy_cost_kBT']-rf['rotation_free_energy_cost_kBT']
            diff=dict(temperature_K=f['temperature_K'],face_twist_halfwidth_deg=delta,
                delta_Fmag_before_rotation_kBT=t['delta_Fmag_time_mean_kBT']-f['delta_Fmag_time_mean_kBT'],
                delta_Frotation_kBT=dc,
                delta_Fmag_after_rotation_kBT=rt['Fmag_plus_rotation_time_mean_kBT']-rf['Fmag_plus_rotation_time_mean_kBT'],
                delta_Fcontact_after_rotation_kBT=rt['partial_contact_F_with_rotation_kBT']-rf['partial_contact_F_with_rotation_kBT'])
            for key,val in list(diff.items()):
                if key.endswith('_kBT'):
                    diff[key[:-4]+'_J']=val*f['kBT_J']
            differences.append(diff)
    write_csv(out/'tip_minus_face.csv',differences)
    summary={}
    for delta in (1.,2.,5.,10.):
        d=[r for r in differences if r['face_twist_halfwidth_deg']==delta]
        summary[str(delta)]={key:roots(ts,[r[key] for r in d]) for key in
                            ('delta_Fmag_after_rotation_kBT','delta_Fcontact_after_rotation_kBT')}
    (out/'crossings.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    plt.rcParams.update({'font.family':'Arial','font.size':14,'axes.labelsize':17,
                         'axes.linewidth':1.8,'axes.grid':False,'pdf.fonttype':42})
    for unit in ('kBT','J'):
        for name in ('free_energy_components','contact_free_energy','energy_unchanged'):
            fig,ax=plt.subplots(figsize=(7.3,10.5) if name=='contact_free_energy' else (7.3,8.8))
            ax.set_box_aspect(1)
            if name=='contact_free_energy':
                experimental=ax.axvspan(233.15,293.15,color='#EEEEEE',zorder=0,
                                       label='Experimental range')
                transient=ax.axvspan(253.15,273.15,color='#F4EAA2',alpha=.8,zorder=1,
                                    label='Transient aggregation (experiment)')
            for g,color in [('face','#B54535'),('tip','#DE9945')]:
                data=[r for r in rows if r['geometry']==g and r['face_twist_halfwidth_deg']==5]
                field={'free_energy_components':'Fmag_plus_rotation_time_mean',
                       'contact_free_energy':'partial_contact_F_with_rotation',
                       'energy_unchanged':'Udd_time_mean'}[name]
                ax.plot(ts,[r[field+'_'+unit] for r in data],color=color,lw=2.5,
                        label=g+f', {args.window_s:g} s' if name=='contact_free_energy' else g)
                if name=='contact_free_energy':
                    ax.plot(ts,[r['partial_contact_F_with_rotation_equilibrium_'+unit] for r in data],
                            color=color,lw=1.8,ls='--',label=g+', equilibrium')
            ax.plot(ts,[b['vdW_'+unit] for b in bases['face']],color='#287A96',lw=2,label='face vdW (component)')
            ax.plot(ts,[1 if unit=='kBT' else b['kBT_J'] for b in bases['face']],color='black',lw=1.5,label=r'$k_BT$ (reference)')
            title={'free_energy_components':'Magnetic free energy + rotation entropy',
                   'contact_free_energy':'Partial contact free energy, including vdW',
                   'energy_unchanged':'Mean magnetic energy (unchanged)'}[name]
            ax.set(xlabel='Temperature (K)',ylabel=('Free energy' if name!='energy_unchanged' else 'Energy')+
                   (r' / $k_BT$' if unit=='kBT' else ' (J)'),
                   title=title+f'\n{args.window_s:g} s window, face twist +/- 5 deg',xlim=(200,300))
            if unit=='J':
                ax.ticklabel_format(axis='y',style='sci',scilimits=(0,0))
            ax.axhline(0,color='0.6',lw=.8)
            handles,labels=ax.get_legend_handles_labels()
            if name=='contact_free_energy':
                handles,labels=handles[2:]+handles[:2],labels[2:]+labels[:2]
            ax.legend(handles,labels,loc='upper left',bbox_to_anchor=(0,-.17),frameon=False)
            fig.subplots_adjust(left=.18,right=.97,top=.89,bottom=.40 if name=='contact_free_energy' else .28)
            for ext in ('png','pdf'):
                fig.savefig(out/f'{name}_{unit}.{ext}',dpi=300,bbox_inches='tight')
            plt.close(fig)
    fig,ax=plt.subplots(figsize=(7.3,8.8))
    ax.set_box_aspect(1)
    for delta,color in zip((1.,2.,5.,10.),('#C44C39','#DC9840','#257C94','#555555')):
        d=[r for r in differences if r['face_twist_halfwidth_deg']==delta]
        ax.plot(ts,[r['delta_Fcontact_after_rotation_kBT'] for r in d],color=color,lw=2.2,label=f'Face twist +/- {delta:g} deg')
    ax.axhline(0,color='black',lw=1)
    ax.set(xlabel='Temperature (K)',ylabel=r'$(F_{tip}-F_{face})/k_BT$',xlim=(200,300),
           title=f'{args.window_s:g} s magnetic dynamics + rotation entropy\nIncluding inherited vdW; positive favors face')
    ax.legend(loc='upper left',bbox_to_anchor=(0,-.17),frameon=False)
    fig.subplots_adjust(left=.18,right=.97,top=.89,bottom=.28)
    for ext in ('png','pdf'):
        fig.savefig(out/f'contact_difference_sensitivity.{ext}',dpi=300,bbox_inches='tight')
    plt.close(fig)
    print(json.dumps(summary,indent=2))
    for d in differences:
        if d['face_twist_halfwidth_deg']==5 and d['temperature_K'] in (200,250,300):
            print(d)


if __name__=='__main__':
    main()
