# Scientific Scope

Primary question:
Does the definition of a subject-specific DWI-derived structural network alter held-out tau-PET prediction accuracy or the form and stability of the symbolically discovered local reaction law?

Structural connectivity is derived from DWI. Connectivity matrices `W` encode subject-specific structural edges. Graph Laplacians `L` are built from those matrices. Tau transport is the network-mediated term `-rho_m L_s^(m) y_s`. The local net reaction is represented by `r_theta,m(y_{s,i})`. The neural reaction function is the learned compact approximation. The symbolic reaction equation is fitted later from input `y` to target `r_theta(y)`.

Real-data analysis requires authorized data, approved preprocessing decisions, ROI-order verification, QC, and frozen subject-level splits. Synthetic validation is only a software smoke-test path and does not support biological claims.
