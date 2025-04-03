`Vertex-CFD` is a free, open source computational fluid dynamics (CFD) and multiphysics code released by Oak Ridge National Laboratory. It is based upon Trilinos, an open source finite element library released by Sandia National Laboratory. `Vertex-CFD` was developed with performance portability as the primary goal, and as such is compatible with a variety of CPU and GPU computing architectures. `Vertex-CFD` currently supports single phase, incompressible flow, with options to include RANS and LES turbulence modeling, heat transfer, and magnetohydrodynamics. The governing equations are discretized using an implicit continuous Galerkin finite element framework and solved in a monolithic fashion using efficient numerical solvers inherited from the Trilinos ecosystem.

## Documentation
[![Documentation Status][docs-badge]][docs-url]

The documentation for `Vertex-CFD` is hosted on [GitHub Pages](https://ornl.github.io/VERTEX-CFD/).

### Repository Features
| Link                                                | Description                              |
|-----------------------------------------------------------|------------------------------------------|
| [src](src)                                             | Source code for VERTEX-CFD        |
| [examples](examples)                              | Example input and mesh files for simple cases |
| [regression_test](regression_test)           | Official regression tests with gold standard results |

## Installation and Dependencies
`Vertex-CFD` is built upon [Trilinos](https://github.com/trilinos/Trilinos), which is freely available to the public.

Detailed instructions for compiling Trilinos and Vertex-CFD are provided in the [installation instructions](https://ornl.github.io/VERTEX-CFD/docs/installation/) section of the GitHub Pages.

## Citing
If you use Vertex-CFD in your research, please consider citing the Zenodo DOI [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14907174.svg)](https://doi.org/10.5281/zenodo.14907174)
 as a software citation, and/or one of our [recent publications](#recent-publications).
 
```bibtex
@software{VERTEX-CFD-v1.0.0,
  author       = {Delchini, Marc Olivier and
                  Furkan Oz and
                  Kincaid, Kellis and
                  Erwin, Jon Taylor and
                  Slattery, Stuart and
                  Curtis, Franklin and
                  Glasby, Ryan and
                  Brandao, Filipe and
                  Gottiparthi, Kalyana and
                  DeGraw, Jason and
                  Ryan Savery},
  title        = {ORNL/VERTEX-CFD: vertex-cfd-v1.0.0-alpha},
  month        = feb,
  year         = 2025,
  publisher    = {Zenodo},
  version      = {v1.0.0},
  doi          = {10.5281/zenodo.14907174},
  url          = {https://doi.org/10.5281/zenodo.14907174},
  swhid        = {swh:1:dir:3ea1469d6974d2e4972bc56707b48e67267354dd
                   ;origin=https://doi.org/10.5281/zenodo.14907173;vi
                   sit=swh:1:snp:f5cf93739bbdcf134c41ed8f6db4c3a694dc
                   972a;anchor=swh:1:rel:4d02f164be05b6b66380bcae1ea5
                   1cf54d291275;path=ORNL-VERTEX-CFD-1ff10fb
                  },
}
```

## License
Vertex-CFD is free software which can be redistributed or modified under the terms of the 3-clause BSD license. See the file `LICENSE` in this directory or the [license text](https://opensource.org/license/BSD-3-Clause) for a complete description of the terms and conditions.

## Contributing
We encourage external users to contribute to Vertex-CFD! Please see the [contributing](https://ornl.github.io/VERTEX-CFD/docs/contribution/) section of the user guide for instructions.

## Contributors
- [Marcho Delchini](https://www.ornl.gov/staff-profile/marc-olivier-delchini)
- [Kellis Kincaid](https://www.ornl.gov/staff-profile/kellis-c-kincaid)
- [Furkan Oz](https://www.ornl.gov/staff-profile/furkan-oz)
- [Kalyan Gottiparthi](https://www.ornl.gov/staff-profile/kalyan-c-gottiparthi)
- [Jason W. DeGraw](https://www.ornl.gov/staff-profile/jason-w-degraw)
- [Doug Stefanski](https://www.ornl.gov/staff-profile/douglas-l-stefanski)
- [Filipe L. Brandao](https://www.ornl.gov/staff-profile/filipe-leite-brandao)
- [Ryan Savery](https://impact.ornl.gov/en/persons/ryan-savery)

## Recent Publications
- Furkan Oz, Kellis Kincaid, Marc-Olivier G. Delchini, Kalyan C. Gottiparthi, Ryan Glasby and Franklin Curtis. "Comparison of Artificial Compressibility Methods for Coupled Laminar Fluid Flow and Heat Transfer," AIAA 2025-1563. AIAA SCITECH 2025 Forum. January 2025. https://doi.org/10.2514/6.2025-1563
- Delchini, M. O., Kincaid, K. C., Stefanski, D., Glasby, R., & Curtis, F. (2024). VERTEX-CFD: A MULTIPHYSICS SOLVER. Transactions of the American Nuclear Society, 130(1), 1191-1194. https://doi.org/10.13182/T130-44950

[docs-badge]: https://img.shields.io/badge/docs-latest-brightgreen.svg
[docs-url]: https://ornl.github.io/VERTEX-CFD/
