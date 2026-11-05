"""3D quasi-static extraction (the methods of FastHenry and FastCap).

geometry: nodes and rectangular conductor segments in 3D, split into filaments (graded towards
the faces, so skin and proximity effect come out of the solve); kernels: partial inductances of
filaments; system: the mesh-current network, its port impedance matrix at any frequency and a
passive reduced-order model (PRIMA) valid over the whole band; fasthenry: input files for
FastHenry and its results, for cross-checks.

All lengths in metres, all values SI.
"""
