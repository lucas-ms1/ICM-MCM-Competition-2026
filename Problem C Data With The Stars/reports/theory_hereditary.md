# Stochastic hereditary differential equation (theory)

## Continuous-time idealization

The mean-field voter model with fading memory can be stated as a **stochastic hereditary (functional) differential equation**. Let \(Y_t^{(g)}\) denote the latent preference vector for community \(g\) at time \(t\), and \(p_t^{(g)} = \mathrm{softmax}(Y_t^{(g)})\) the fan-share distribution. A continuous-time idealization is:

\[
dY_t^{(g)} = b\bigl(t,\, Y_t^{(g)},\, \int_0^t K(t-s)\, G(Y_s)\, ds\bigr)\, dt + \Sigma\, dB_t,
\qquad
p_t^{(g)} = \mathrm{softmax}(Y_t^{(g)}).
\]

- **\(Y\)**: latent preference (utility) vector; \(\mathrm{softmax}(Y)\) gives the share distribution.
- **\(K(t-s)\)**: memory kernel weighting the past; e.g. exponential \(K(\tau)\propto e^{-\alpha\tau}\) or power-law \(K(\tau)\propto (\tau+1)^{-d}\).
- **\(G(Y_s)\)**: past signal (e.g. judge score or popularity) driven by the latent state.
- **\(b\)**: drift depending on current state and the kernel-weighted history (hereditary term).
- **\(\Sigma\, dB_t\)**: diffusion (optional; in the code, optional logit-normal shock plays a similar role).

So the drift depends not only on the current state but on a **convolution** of the past with the kernel \(K\)—hence “hereditary” or “Volterra” type.

## Discrete-time (weekly) implementation

The **weekly discrete-time update** in the code is an **Euler-type discretization in time** and **rectangular (or trapezoidal) quadrature** for the hereditary integral \(\int_0^t K(t-s)\, G(Y_s)\, ds\):

1. **Time**: One step per week; \(Y_{t+1}\) is computed from \(Y_t\) and the current drift.
2. **Hereditary integral**: The integral is approximated by \(\sum_{\ell=0}^{L} K(\ell)\, S_{t-\ell}\) (or the same with \(p\) for social buzz), i.e. a discrete convolution with kernel weights \(K(0), K(1), \ldots, K(L)\). This corresponds to rectangular quadrature on the lag variable with step size 1 (one week).
3. **Markovian state (optional)**: The exponential fading state \(m_{i,t} = (1-\lambda)m_{i,t-1} + \lambda S_{i,t}\) is the discrete-time analogue of an exponential kernel with a one-dimensional state; it is consistent with the same continuous-time interpretation with \(K(\tau)\propto e^{-\alpha\tau}\) and state augmentation.

So the weekly mean-field step (with kernel-weighted history and optional Markovian \(m_t\)) is the **Euler + rectangular quadrature** discretization of the stochastic hereditary DE above.

## Discrete kernels in code

- **Exponential**: \(K(\ell) \propto \mathrm{decay}^{\ell}\); the constant is absorbed in the coefficient \(\eta_S\) or \(\eta_m\).
- **Power-law**: \(K(\ell) \propto (\ell+1)^{-d}\) for \(\ell=0,\ldots,L\), normalized to sum to 1; gives slow decay (“long memory”).
- **Rectangular**: \(K(\ell) = 1/L\) for \(\ell \le L\), else 0; finite window.

These are the discrete kernels used in `make_kernel` and `kernel_weighted_history`; they define the weights in \(M_{i,t} = \sum_{\ell=0}^{L} K(\ell)\, S_{i,t-\ell}\) and the analogous term for social buzz.
