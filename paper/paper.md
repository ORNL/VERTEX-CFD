---
title: 'VERTEX-CFD: A multiphysics platform for fusion applications'
tags:
  - Computational Fluid Dynamics
  - CPU and GPU
  - Finite Element Method
authors:
  - name: Marco Delchini
    affiliation: 1
    corresponding: true
  - name: Kellis Kincaid
    affiliation: 1
  - name: Furkan Oz
    affiliation: 1
  - name: Jason DeGraw
    affiliation: 2
  - name: Kalyan Gottiparthi
    affiliation: 3
  - name: Ryan Glasby
    affiliation: 4
  - name: Franklin Stuart
    affiliation: 5
affiliations:
  - name: Nuclear Energy and Fuel Cycle Division, Oak Ridge National Laboratory
    index: 1
  - name: Building and Transportation Division, Oak Ridge National Laboratory
    index: 2
  - name: National Center for Computation Science Division, Oak Ridge National Laboratory
    index: 3
  - name: Computational Science and Engineering Division, Oak Ridge National Laboratory
    index: 4
  - name: Enrichment Science and Engineering Division, Oak Ridge National Laboratory
    index: 5
date: 19 December 2024
bibliography: paper.bib
---

# Summary:

The demand for high-performance computational fluid dynamics and multiphysics software packages has grown in recent years as a response to effort in complex engineering and research applications. While the widespread deployment of high-performance computing (HPC) resources has enabled larger, more complex simulations to be conducted, few commercial or open-source software packages are available which scale performantly on CPU and GPU computing architectures, and represent the multitude of physical processes relevant to these applications. The VERTEX initiative at Oak Ridge National Laboratory is developed to address this technical gap, with a special emphasis on high-fidelity multiphysics modeling of coupled turbulent fluid flow, heat transfer, and magnetohydrodynamics for applications in fusion and fission energy, and other spaces. The VERTEX-CFD module was developed to solve the governing equations of these problems using a high-order continuous Galerkin finite element framework, and fully-implicit monolithic solvers. Special attention is being paid during the development process to verify and to validate the solver, and to ensure performance portability across both CPU and GPU computing platforms. A comprehensive verification and validation (V&V) suite and unit tests were designed to assess the accuracy and convergence behavior of the VERTEX-CFD module for problems taken from the published literature.

# Statement of need

The core work of the cross-cutting VERTEX Laboratory Directed Research and Development (LDRD) initiative aims to create a new multiphysics simulation framework supporting physical phenomena key to Oak Ridge National Laboratory (ORNL) mission-critical challenges. ORNL has clearly demonstrated needs in modeling and simulation of gas dynamics, rarefied flow, plasma-surface interaction, electromagnetics, magneto-hydrodynamics (MHD), and thermal hydraulics for conducting fluids, collisionless and collisional plasma, and structural mechanics.
As part of the VERTEX initiative, the primary mission of the VERTEX-CFD team is to develop modeling and simulation capabilities to accurately model the physics in fusion blanket design. It thus requires a multiphysics solver to implement the incompressible Navier-Stokes (NS) equation to conjugate a heat transfer model and an MHD solver. Solvers, finite element methods, and other relevant tools are provided by the [Trilinos package](https://trilinos.github.io/) [@trilinos-website]. The VERTEX-CFD solver is designed to scale and to be compatible with various CPU and GPU architectures on HPC platforms by leveraging Kokkos [@kokkos] programming language.


# Current capabilities and development workflow

VERTEX-CFD solver is still under active development and currently implements the following capabilities: incompressible Navier-Stokes equations [@Clausen2013], temperature equation, induction-less and full-induction MHD models, RANS turbulence models and WALE (LES) [@nicoud:hal-00910373] turbulence model. Each new physics is implemented in closure models with unit tests. Physical models and coupling between equations were verified and validated against benchmark problems taken from the published literature: isothermal flows [@Taylor-green-vortex, @10.1115/1.3240731; @Clausen2013], heated flows [@Kuehn_Goldstein_1976; @tritton_1959], transient and steady-state cases, turbulent cases [@nicoud:hal-00910373; @nasa-web], and MHD flows [@SMOLENTSEV201565]. VERTEX-CFD solver has demonstrated second-order temporal and spatial accuracy. Scaling of the VERTEX-CFD solver was assessed on CPUs and GPUs architecture. It was found that strong and weak scaling were comparable to other CFD solvers alike NekRS. (ADD FIGURE).

The long term objectives of the VERTEX initiative is to facilitate the addition of new physical models by relying on a plug-and-play architecture, and also guarantee the correctness of the implemented model over time. New physics and equations are easily added to the global tree and allow for quick deployment of new physical model on HPC platforms. Such approach can only be made possible by setting clear requirements and review process for all developers contributing to the project code: any changes and additions to the source code is reviewed and tested before being merged. VERTEX-CFD solver is tested daily on a continuous integration (CI) workflow that is hosted on ORNL network.

# Conclusions

VERTEX-CFD is an open-source CFD solver that relies on a finite element discretization method to solve for the incompressible Navier-Stokes equations coupled to a temperature equation and MHD equation. Reynolds Averaged Navier-Stokes (RANS) turbulence models and large eddy simulation model are also available. The code relies on the  Trilinos package and offers a wide range of temporal integrators, solvers and preconditioners to run on CPU- and GPU-enabled platforms. VERTEX-CFD solver was verified and validated for steady and unsteady incompressible flows with benchmark cases taken from the published literature: natural convection, viscous heating, laminar flow over a circle, and turbulent channels. It was also demonstrated that VERTEX-CFD solver scales on CPUs (Perlmutter) and GPUs (Perlmutter and Summit [@olcf-web]) architectures.

Future development will focus on implementing wall functions for RANS models, and adding conjugate heat transfer capabilities.


# Acknowledgements

This work was funded by the Laboratory Directed Research and Development (LDRD) program at Oak Ridge National Laboratory, and the Scientific Discovery through Advanced Computing (SciDac) program.


# Disclaimer

This manuscript has been authored by UT-Battelle, LLC, under contract DE-AC05-00OR22725 with the US Department of Energy (DOE). The US government retains and the publisher, by accepting the article for publication, acknowledges that the US government retains a nonexclusive, paid-up, irrevocable, worldwide license to publish or reproduce the published form of this manuscript, or allow others to do so, for US government purposes. DOE will provide public access to these results of federally sponsored research in accordance with the [DOE Public Access Plan](http://energy.gov/downloads/doe-public-access-plan).