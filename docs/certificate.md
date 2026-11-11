# A posteriori certificate with Rademacher probes

## Setting
After recovery we hold A_hat. With m' fresh probes phi_1..phi_m' (independent Rademacher vectors over
the sensed targets), m' extra mixed passes give Z = Phi'(A - A_hat), Phi' = [phi_i^T] / sqrt(m').
For one column, e = a_s - a_hat_s and z_i = phi_i . e / sqrt(m').

## Point estimate
E[(phi . e)^2] = sum_j e_j^2 = ||e||^2, so ||z||^2 = (1/m') sum_i (phi_i . e)^2 is unbiased for
||e||^2. The same holds for the Frobenius norm of the whole error matrix.

## High-probability bound
Halko, Martinsson and Tropp (2011, Section 4.3, Lemma 4.1) bound the spectral norm with Gaussian probes
(constant 10 sqrt(2/pi), failure probability 10^-r with r probes). That argument uses Gaussian
anti-concentration and does not transfer to Rademacher probes: for e = (1, 1, 0, ...) the Rademacher
sum phi . e is exactly 0 with probability 1/2. We derive a bound that holds for Rademacher probes.

Let X = (phi . e)^2 >= 0, mu = E X = ||e||^2.
1. Fourth moment (Khintchine, exact for Rademacher):
   E X^2 = E (phi . e)^4 = 3 ||e||^4 - 2 sum_j e_j^4 <= 3 mu^2.
2. Paley-Zygmund: for theta in (0, 1), P(X > theta mu) >= (1 - theta)^2 mu^2 / E X^2 >= (1 - theta)^2 / 3.
3. Independent probes: P(max_i X_i <= theta mu) <= (1 - (1 - theta)^2 / 3)^m' =: delta(theta).
4. So with probability at least 1 - delta(theta):
   max_i (phi_i . e)^2 > theta ||e||^2, that is ||e|| < sqrt(m' / theta) max_i |z_i|.
For a target failure probability delta we take the largest theta with delta(theta) <= delta
(`cct.certify.posterior.theta_for`), giving the constant c = sqrt(m' / theta).

| m' | delta | theta | c |
|---|---|---|---|
| 16 | 0.05 | 0.284 | 7.5 |
| 16 | 0.01 | 0.134 | 10.9 |
| 32 | 0.01 | 0.366 | 9.4 |
(Values from `bound_constant`; c is stated relative to the normalised z.)

**Frobenius version.** With X = ||phi^T E||^2 = sum_s (phi . e_s)^2, E X = ||E||_F^2 and, by
Cauchy-Schwarz on the cross terms, E X^2 <= 3 ||E||_F^4, so the same constant bounds ||E||_F by
c max_i ||z_i||, with z_i the i-th row of Z.

**Per-column bounds are marginal.** Each column's bound fails with probability at most delta; a
simultaneous statement over N columns needs a union bound (delta / N per column, which raises c; the
dependence of c on delta is mild: c grows like sqrt(m' / theta) with (1 - (1 - theta)^2/3)^m' = delta).

Test T12 checks coverage by Monte Carlo (1,000 trials) on random, two-spike (worst case), one-hot and
power-law error vectors.

## From column bounds to influence intervals
See `cct/certify/neumann.py`: column bounds c_s bound every entry of their column; row sums and
normalised entries get interval bounds; total influence is monotone in the nonnegative normalised
entries, so propagating the lower and upper matrices through the finite Neumann series (the graph is a
DAG) gives a per-node interval that contains the true influence whenever the column bounds hold
(test T11).
