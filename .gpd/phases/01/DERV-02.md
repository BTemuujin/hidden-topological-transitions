# DERV-02: Free Energy Functional for Exact Steepest Descent

## Objective
Formulate the variational free energy functional $G(\{S_0\})$ to be maximized to find the thermodynamic limit of the partition function.

## Theoretical Framework
The exact steepest descent method approximates the partition function by finding the configuration that maximizes the functional $G$.

### 1. The Functional $G$
The functional is defined as:
$$G = -\beta H + \text{Entropy Term}$$

### 2. Thermal Magnitude and Langevin Function
Each spin site $i$ has a local field $v_i$. The thermal average of the spin magnitude is given by the Langevin function:
$$m_i = L(v_i) = \coth(v_i) - \frac{1}{v_i}$$
The spin vector is $\mathbf{S}_i = m_i \hat{n}_i$, where $\hat{n}_i$ is a unit vector defined by $(\theta_i, \phi_i)$.

### 3. Entropy Term
The entropy for classical unit vectors in 3D, incorporating the $\ln(4\pi)$ offset to match the exact partition function, is:
$$\text{Entropy} = \frac{1}{N} \sum_i \left[ \ln(4\pi) + \ln\left(\frac{\sinh v_i}{v_i}\right) - v_i L(v_i) \right]$$

## Computational Implementation
- **Optimization**: $G$ is maximized with respect to $\{\theta_i, \phi_i, v_i\}$.
- **Numerical Stability**: $\ln(\sinh v / v)$ is computed using a stable approximation for $v > 20$ to prevent overflow.
- **Constraints**: $v_i > 0$ is enforced via a `softplus` activation in JAX.
