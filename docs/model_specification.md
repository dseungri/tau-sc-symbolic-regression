# Model Specification

Primary equation:

```text
dy_{s,i}/dt
    = -rho_m (L_s^(m) y_s)_i
      + r_{theta,m}(y_{s,i})
```

For an unnormalized Laplacian, the transport term expands to:

```text
-rho_m (L_s^(m) y_s)_i
    = rho_m sum_j W_s,ij^(m) (y_s,j - y_s,i)
```

The exact Laplacian normalization remains pending. Dr. Wu or the analysis lead must approve the convention before real modeling.

The MLP maps:

```text
y -> r_theta(y)
```

Later symbolic regression fits:

```text
input: y
target: r_theta(y)
```

Do not reverse these variables. Do not apply symbolic regression to the transport term in the primary analysis.
