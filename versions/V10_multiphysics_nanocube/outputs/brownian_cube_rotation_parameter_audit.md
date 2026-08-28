# Parameter provenance for the V10.18 pair-energy model

| Parameter | Value used | Classification | Source or required action |
|---|---:|---|---|
| Saturation magnetization, `Ms` | 2.850e+05 A m^-1 | Present-sample value at 300 K | present-particle magnetometry at 300 K: 55 emu/g = 55 A m2/kg; converted with rho(Fe3O4)=5.18e3 kg/m3. This assumes 55 emu/g is normalized to inorganic Fe3O4 mass; if it includes ligand mass, apply the TGA inorganic-mass correction before conversion. The constant-Ms treatment over 200-300 K remains a model approximation. |
| Neel attempt time, `tau0` | 0.98 ns | Literature value at 300 K | Moreno et al., Phys. Rev. B 112, 024429 (2025), DOI 10.1103/vmwp-q427; reported value is 0.98 +/- 0.13 ns for cubic-anisotropy magnetite. |
| Hamaker constant | 2.00e-20 J | Literature-based representative value | Faure et al., Langmuir 27, 8659-8664 (2011), DOI 10.1021/la201387d reports 9-29 zJ across hexane/toluene. Use a solvent-specific value when the medium is fixed. |
| Particle edge | 16 nm | Experimental sample input | Must come from TEM/SAXS for the present batch; it is not a universal material constant. |
| Blocking temperature | 250 K | Experimental input | Taken from the present ZFC/FC result. |
| ZFC/FC observation time | 100 s | Protocol assumption | Replace by a timescale derived from the actual magnetometry sweep/settling protocol if available. |
| Diameter coefficient of variation | 0.0% | Temporary monodisperse baseline | Replace by the present sample's TEM size-distribution fit when available. |
| Solvent viscosity | temperature-dependent n-hexane | Literature reference correlation | Michailidou et al., J. Phys. Chem. Ref. Data 42, 033104 (2013), DOI 10.1063/1.4818980; the Table 4 saturation values are fitted in reciprocal temperature, with extrapolation below 250 K. |
| Ligand length per surface | 1.5 nm | User-specified sample input | Confirm by ligand identity/chain conformation or scattering/TGA characterization. |
| Face and tip surface gaps | 3, 3 nm | Geometric model inputs | Face gap was user specified; tip gap equals two ligand lengths. |
| Cube rounding radius | 1.5 nm | Unsupported morphology input | Replace by a TEM-derived corner radius; it materially affects the tip center distance. |
| Effective SAXS window | 20 s | Experimental protocol convention | Midpoint representation of the 150 s integration, as requested. |
| Angular quadrature | 40 x 80 | Numerical setting | Requires convergence checking, not a literature citation. |
| Barrier quadrature | 11 points | Numerical setting | Gauss-Hermite integration; relevant only when a nonzero measured diameter CV is supplied. |

The factor 1/3 in each adjacent-well rate is not an empirical parameter. A <111> minimum has three nearest <110>-saddle exits, and the cited tau0 describes the total mean residence/escape time. Therefore each equivalent path receives one third of the total isolated escape rate; the other four <111> wells require multiple elementary hops.
