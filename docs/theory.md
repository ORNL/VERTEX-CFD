---
parent: VERTEX-CFD v1.0 User Guide
title: Theory
nav_order: 2
usemathjax: true
---

# Theory

---

## Governing equations

VERTEX-CFD implements the entropically damped artificial compressibility (EDAC) Navier-Stokes equations, a temperature equation, and a magneto-hydrodynamics (MHD) equation (inductionless equation). Coupling between the different equations is ensured by the buoyancy force and the Lorentz force.

$$
\begin{aligned}
\left\{
\begin{matrix}
    \nabla \cdot \mathbf{u} = 0 \\
    \partial_t \mathbf{u} + (\mathbf{u} \cdot \nabla) \mathbf{u} = -\frac{1}{\rho}\nabla P + \nu \Delta \mathbf{u} + \frac{1}{\rho}\mathbf{J} \times \mathbf{B^0} - \mathbf{g} \beta (T - T_0) \\
    \rho C_p \left( \partial_t T + \mathbf{u} \cdot \nabla T \right) = \nabla \cdot (k \nabla T ) + q^{'''} \\
    \mathbf{J} = \sigma ( -\nabla \varphi + \mathbf{u} \times \mathbf{B^0} ) \\
    \nabla \cdot (\sigma \nabla \varphi) = \nabla \cdot [ \sigma \mathbf{u} \times \mathbf{B^0} ]
\end{matrix}
\right.
\end{aligned}
$$

The equations are recast in a conservative form and solved for the pressure $$P$$, the velocity $$\mathbf{u}$$, the temperature $$T$$, and the electric potential $$\varphi$$.

$$
\begin{aligned}\label{eq:pdes}
\left\{
\begin{matrix}
    \nabla \cdot \mathbf{u} = 0 \\
    \partial_t \rho \mathbf{u} + \rho (\mathbf{u} \cdot \nabla) \mathbf{u} = -\nabla P + \rho \nu \Delta \mathbf{u} + f^L - \rho \mathbf{g} \beta (T - T_0) \\
    f^L = \mathbf{J} \times \mathbf{B^0} = \sigma \left( -\nabla \varphi \times \mathbf{B^0} + (\mathbf{B} \cdot \mathbf{u}) \cdot \mathbf{B^0} - ||\mathbf{B^0}||^2 \mathbf{u} \right) \\
    \rho C_p \left( \partial_t T + \mathbf{u} \cdot \nabla T \right) = \nabla \cdot (k \nabla T ) + q^{'''} \\
    \nabla \cdot (\sigma \nabla \varphi) = \nabla \cdot [ \sigma \mathbf{u} \times \mathbf{B^0} ]
\end{matrix}
\right.
\end{aligned}
$$


## Discretized equations

VERTEX-CFD employs a finite element discretization method and high-order implicit temporal integrators to integrate partial differential equations (PDEs). Numerical stability of the solution is ensured by using an L-stable implicit temporal integrator and the appropriate mesh density.

A continuous Galerkin finite element method from the Trilinos package, Panzer \cite{panzer-website}, is employed to discretize the equations presented as Eq. \ref{eq:pdes}. Considering a finite dimensional subspace $$V^p_h$$, an approximate solution $$U_h$$ of $$U$$ can be expressed as

$$
\begin{equation}
    U_h = \sum_i U_i \phi_i~~,
\end{equation}
$$

where $$\phi_i \in V^p_h$$ is a basis function. A weak form of Eq. \ref{eq:pdes} is obtained by substituting $$U$$ with $$U_h$$, taking an inner product between each term of the partial differential equation and a test function and integrating over the computational volume:

$$
\begin{equation}\label{eq:weak-form}
    \iiint \left[ \phi^T \partial_t U + \phi^T \nabla \cdot F(U) - \phi^T \nabla \cdot G(U, \nabla U) - \phi^T S(U) \right] d\Omega = 0.0~~.
\end{equation}
$$

Equation \ref{eq:weak-form} can be further transformed by integrating per part the conservative fluxes to yield boundary fluxes denoted by the under script $$bc$$:

$$
\begin{align}\label{eq:weak-form-bc}
\iiint \phi^T \partial_t U d\Omega - \iiint \nabla \phi^T \cdot F(U) d\Omega + \\ \iiint \nabla \cdot \phi^T \cdot G(U, \nabla U) d\Omega - \iiint \phi^T S(U) d\Omega \nonumber = \\ -\iint \phi^T F_{bc}(U) \vec{n} d \delta \Omega + \iint \phi^T G_{bc}(U, \nabla U) \vec{n} d \delta \Omega~~.
\end{align}
$$

Boundary fluxes are evaluated at quadrature points, and their contribution is added to the global residuals. The boundary flux $$G$$ is evaluated with the symmetric interior penalty method \cite{penaltyMethod}. Implementation of the boundary conditions is further detailed in Section [Boundary conditions](#boundary-conditions).

The time-derivative terms $$\partial_t U$$ are evaluated with a high-order temporal integrator (SDIRK-22 or SDIRK-54) from the Tempus package \cite{tempus-website}. Given a test function $$\phi_i$$ of order $$p$$, integral terms are evaluated with a $$p+1$$ quadrature rule. VERTEX-CFD does not currently implement any numerical method to stabilize the numerical solution and solely relies on the numerical dissipation from the discretization method and the implicit temporal integrator. This strategy has been sufficient for laminar flows, as demonstrated in the following sections.


## Boundary conditions

Boundary conditions are weakly imposed by computing numerical flux at the boundaries' provided boundary values. The boundary conditions implemented in VERTEX-CFD are listed below:

The boundary conditions implemented in VERTEX-CFD are listed below:

- [Periodic boundary](#periodic-boundary)
- [Dirichlet with time-transient variation](#dirichlet-boundary)
- [Symmetry for isothermal flow](#symmetry-boundary-condition)
- [No-slip for viscous flow](#no-slip-boundary-condition)
- [Rotating wall for isothermal flow](#rotating-wall-boundary-condition)
- [Laminar flow](#laminar-flow)
- [Outflow with back pressure](#outflow-boundary-condition)
- [Cavity Lid](#cavity-Lid)

The vector solution is denoted by $$U_{bc} = (P_{p,{bc}}, \mathbf{u}_{bc}, T_{bc}, \varphi_{bc})$$ at the boundary. It should be noted that when the energy equation and the electric potential equation are not solved, the temperature $$T_{bc}$$ and the electric potential $$\varphi_{bc}$$ are ignored.

### Periodic boundary
Users can set periodic boundaries in the input file by using the Trilinos specific syntax described in [Panzer_STK class](https://docs.trilinos.org/dev/packages/panzer/doc/html/Panzer__STK__PeriodicBC__Parser_8cpp_source.html). Meshes on periodic faces must be identical for the logic to work properly.

### Dirichlet boundary
The Dirichlet boundary condition denotes the Dirichlet boundary condition in VERTEX-CFD. The velocity is set equal to the user-specified values or Dirichlet values $$\mathbf{u}_D$$ while the Lagrange pressure and the boundary gradients are set to the interior values. The temperature is also set to a user-specified value $T_{bc}$. Linear ramping in time is also available and can be used to vary each primitive variable independently.

$$
\begin{equation}
\left\{ \
\begin{matrix}
    u_{i,bc}(\mathbf{r}, t) = u_{i,D}(t) \\
    P_{p,{bc}}(\mathbf{r}, t) = P(\mathbf{r}, t) \\
    \partial_i U_{bc}(\mathbf{r}, t) = \partial_i U(\mathbf{r}, t) \\
    T_{bc}(\mathbf{r}, t) = T_{D}
\end{matrix}
\right.
\end{equation}
$$

### Symmetry boundary condition
The symmetry boundary condition is a no-penetration condition. The normal component of the fluid velocity to the wall is zero, whereas the tangential component is unrestricted. The same observation is also valid for the temeprature gradient. Assuming the outward normal vector to a wall boundary is denoted by $$\mathbf{n}_{bc} = \left(n_{bc,x}, n_{bc,y}, n_{bc,z} \right)$$ in a 3D computational domain, the boundary condition for the primitive variables reads as follows:

$$
\begin{equation}
\left\{
\begin{matrix}
    P_{bc}(\mathbf{r}, t) &=& P(\mathbf{r}, t) \\
    \mathbf{u}_{bc}(\mathbf{r}, t) &=& \mathbf{u}(\mathbf{r}, t) - \left( \mathbf{u}(\mathbf{r}, t) \cdot \mathbf{n}_{bc} \right) \mathbf{n_{bc}} \\
    T_{bc}(\mathbf{r}, t) &=& T(\mathbf{r}, t) \\
    \varphi_{bc}(\mathbf{r}, t) &=& \varphi(\mathbf{r}, t)
\end{matrix}
\right.
\end{equation}
$$

The boundary gradients are a function of the interior values:

$$
\begin{align}
    \partial_i U_{bc}(\mathbf{r}, t) =& \partial_i U(\mathbf{r}, t) \nonumber\\
    \partial_i T_{bc}(\mathbf{r}, t) =& \partial_i T(\mathbf{r}, t) - \left(\partial_i T(\mathbf{r}, t) \cdot \mathbf{n}_{bc} \right) n_{i} \\
    \partial_i \varphi_{bc}(\mathbf{r}, t) =& \partial_i \varphi(\mathbf{r}, t) - \left(\partial_i \varphi(\mathbf{r}, t) \cdot \mathbf{n}_{bc} \right) n_{i} \nonumber
\end{align}
$$

### No-slip boundary condition
The \emph{no-slip} boundary condition is used to model a viscous flow interacting with a moving wall. The velocity of the flow and the wall are the same: this can be put in the following mathematical form:

$$
\begin{equation}
    \mathbf{u_{bc}} = \mathbf{u_{wall}} \ ,
\end{equation}
$$

where $$\mathbf{u_{wall}}$$ is a user input parameter that is defaulted to the zero vector. Because this is a wall, the temperature is set to the wall temperature $$T_{wall}$$. The Lagrange pressure is set to its interior value:

$$
\begin{equation}
\left\{
\begin{matrix}
    P_{bc}(\mathbf{r}, t) &=& P(\mathbf{r}, t) \\
    \mathbf{u}_{bc}(\mathbf{r}, t) &=& \mathbf{u_{wall}} \\
    T_{bc}(\mathbf{r}, t) &=& T_{wall}
\end{matrix}
\right.
\end{equation}
$$

All boundary gradients are set to the interior values: 

$$
\begin{equation}
    \partial_i f_{bc}(\mathbf{r}, t) = \partial_i f(\mathbf{r}, t)
\end{equation}
$$

In VERTEX-CFD, the wall boundary velocity $$\mathbf{u_{wall}}$$ can linearly vary as a function of time. This is particularly useful when modeling flow with non-natural initial conditions (non-stationary flow over a stationary obstacle). The wall boundary velocity is set to initially match the flow velocity and linearly decreases to zero with time. This strategy is commonly adopted to ease the work of the numerical solver while developing steady-state flow.


### Rotating wall boundary condition
The \emph{rotating wall} boundary condition is used to model the interaction of a fluid with a rotating wall. The wall in contact with the fluid rotates with an angular velocity $$\omega_w$$ in the XY plane. The Lagrange pressure is set to the interior value, and the fluid velocity is computed from the angular velocity and the fluid temperature set to the wall temperature $$T_{wall}$$, as follows:

$$
\begin{equation}
\left\{
    \begin{matrix}
    P_{bc}(\mathbf{r}, t) &=& P(\mathbf{r}, t) \\
    \mathbf{u}_{bc}(\mathbf{r}, t) &=& \omega_{wall} \left(-y, x, 0.0 \right) \\
    T_{bc}(\mathbf{r}, t) &=& T_{wall}
    \end{matrix}
\right.
\end{equation}
$$

The Cartesian coordinates are defined as $$x$$ and $$y$$, and the z-component of the velocity is assumed to be zero when used in 3D.

All boundary gradients are set to the interior values such as the following:

$$
\begin{equation}
    \partial_i f_{bc}(\mathbf{r}, t) = \partial_i f(\mathbf{r}, t)
\end{equation}
$$

### Laminar flow
The laminar flow boundary condition sets a parabolic profile for the velocity in the wall normal direction. The input parameters are the average velocity amplitude $u_i^{avg}$, the coordinates of the two points on the circle which can be connected by a line that passes through the circle's center. The resultant inlet velocity profiles can be calculated as follows:

$$
\begin{equation}
    u_i(x,y,z) = C_i u_i^{avg}\cdot\mathbf{n}\left(1.0 - \frac{r^2}{R^2}\right)~~,
\end{equation}
$$

where $$R$$ is the characteristic radius of the circle, $$\mathbf{r}$$ is the distance to the circle center which is calculated from the two points provided by the user, and $$C_i$$ is a constant based on the dimensionality of the problem. In 2D, $$C_i$$ is $3/2$, whereas in 3D, it is $2.0$. This boundary condition can be used with 2D rectangular channels and 3D pipes. The flow direction is calculated based on the surface normals where the boundary condition is applied. A uniform inlet temperature can also be provided if solving for the temperature equation. The boundary conditions are as follows:

$$
\begin{equation}
\left\{
    \begin{matrix}
    P_{bc}(\mathbf{r}, t) &=& P(\mathbf{r}, t) \\
    \mathbf{u}_{bc}(\mathbf{r}, t) &=& \left(u_i\cdot n_x, u_i\cdot n_y, u_i\cdot n_z \right) \\
    T_{bc}(\mathbf{r}, t) &=& T_{inlet}
    \end{matrix}
\right.
\end{equation}
$$

All boundary gradients are set to interior values such as the following:

$$
\begin{equation}
    \partial_i f_{bc}(\mathbf{r}, t) = \partial_i f(\mathbf{r}, t)
\end{equation}
$$

### Outflow boundary condition
The outflow boundary condition is employed when the back pressure at an outlet is known. The Lagrange pressure is computed from the back pressure $$P_b$$:

$$
\begin{equation}
    P_{p,bc} = P_b
\end{equation}
$$

The velocity, the temperature, and all boundary gradients are set to their respective interior values, as follows:

$$
\begin{equation}
\left\{
    \begin{matrix}
    \mathbf{u}_{bc}(\mathbf{r}, t) &=& \mathbf{u}(\mathbf{r}, t) \\
    \partial_i U_{bc}(\mathbf{r}, t) &=& \partial_i U(\mathbf{r}, t) \\
    T_{bc}(\mathbf{r}, t) &=& T(\mathbf{r}, t) \\
    \partial_i T_{bc}(\mathbf{r}, t) &=& \partial_i T(\mathbf{r}, t)
    \end{matrix}
\right.
\end{equation}
$$

### Cavity Lid

A special boundary condition is implemented for the lid-driven cavity case to provide a smooth transition from the lid velocity to the no-slip condition at the wall, as described by Leriche and Gavrilakis \cite{Leriche2000}. The condition assumes that the domain consists of a 2D or 3D cube with a half width of $$h$$, centered on the origin. The Lagrange pressure at the boundary is set to the interior value:

$$
\begin{equation}
    P_{bc} \left(\mathbf{r}, t \right) = P \left(\mathbf{r}, t \right)
\end{equation}
$$

If the energy equation is to be solved, then the boundary temperature is set to a specified constant value:

$$
\begin{equation}
    T_{bc} \left(\mathbf{r}, t \right)= T_b
\end{equation}
$$

The user is required to specify the index $$n$$ aligned with the boundary normal vector and the index $$v$$ of the direction in which the velocity is aligned. The velocity components are then defined as follows:

$$
\begin{equation}
    u_{i} \left(\mathbf{r}, t \right) = \left\{ \begin{matrix}
    0, i \neq v \\
    U_0 \prod_{j=0, j \neq n}^{N} \left( 1 - \left(r_i / h \right)^{18} \right)^2
    \end{matrix}
    \right,
\end{equation}
$$

where $$U_0$$ is the nominal wall velocity, $$N$$ is the number of spatial directions, and $$r_i$$ is the distance of the current location from the origin in the $$i^{th}$$ direction.

## Initial conditions

The initial conditions are specified for the velocity $$\mathbf{u}$$, the Lagrange pressure $$P$$, and the temperature $$T$$ when solving for the energy equation. The electric potential $$\varphi$$ is assumed constant and is set to $$\varphi_0$$ in all initial conditions presented below.

### Uniform initial conditions
The normalized Lagrange pressure, the velocity, and the temperature are initialized to constant values across each block of the computational domain:

$$
\begin{align}
    \left\{
    \begin{matrix}
    P(\mathbf{r}, t) &=& P_{p,0} \\
    \mathbf{u}(\mathbf{r}, t) &=& \mathbf{u}_0 \\
    T(\mathbf{r}, t) &=& T_0
    \end{matrix}
    \right.
\end{align}
$$

### Laminar flow
When using the laminar flow initial condition, the x-component of the velocity is initialized with a parabolic profile. The Lagrange pressure and the temperature are initialized to constant values. The current implementation can be used for two geometries: a 2D rectangular channel and a 3D pipe. The parabolic profile is function of the average velocity and the channel height $$h$$ in 2D or the pipe diameter $$D$$ in 3D. It is assumed that the flow direction is aligned with the x-axis, which yields the following expressions:

$$
\begin{align}
    \left\{
    \begin{matrix}
    P(\mathbf{r}, t) &=& P_{p,0} \\
    \mathbf{u}(\mathbf{r}, t) &=& \left(\frac{3}{2}\cdot u_{avg}\left(1.0 - \frac{y^2}{h}\right), 0.0 \right) \text{ in 2D} \\
    \mathbf{u}(\mathbf{r}, t) &=& \left(2\cdot u_{avg}\left(1.0 - \frac{y^2 + z^2}{D}\right), 0.0, 0.0 \right) \text{ in 3D} \\
    T(\mathbf{r}, t) &=& T_0
    \end{matrix}
    \right.
\end{align}
$$
