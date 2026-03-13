#ifndef VERTEXCFD_CONSTANTSPECIESPROPERTIES_HPP
#define VERTEXCFD_CONSTANTSPECIESPROPERTIES_HPP

#include <Teuchos_ParameterList.hpp>

#include <Kokkos_Core.hpp>

namespace VertexCFD
{
namespace SpeciesProperties
{
//---------------------------------------------------------------------------//
// Constant species properties
//---------------------------------------------------------------------------//

class ConstantSpeciesProperties
{
  public:
    ConstantSpeciesProperties() = default;
    explicit ConstantSpeciesProperties(
        const Teuchos::ParameterList& model_params,
        const Teuchos::ParameterList& reaction_params)
        : _num_species(model_params.get<int>("Number of Species"))
        , _build_bateman(model_params.isType<bool>("Build Reaction Bateman "
                                                   "Source")
                             ? model_params.get<bool>("Build Reaction Bateman "
                                                      "Source")
                             : false)
        , _build_advection(model_params.isType<bool>("Build Advection")
                               ? model_params.get<bool>("Build Advection")
                               : false)
        , _build_diffusion(model_params.isType<bool>("Build Diffusion")
                               ? model_params.get<bool>("Build Diffusion")
                               : false)
        , _build_fission_source(model_params.isType<bool>("Build Fission "
                                                          "Source")
                                    ? model_params.get<bool>("Build Fission "
                                                             "Source")
                                    : false)
        , _build_transmutation_source(
              model_params.isType<bool>("Build Reaction Transmutation Source")
                  ? model_params.get<bool>("Build Reaction Transmutation "
                                           "Source")
                  : false)
        , _build_vapor_removal_source(
              model_params.isType<bool>("Build Vapor Removal Source")
                  ? model_params.get<bool>("Build Vapor Removal Source")
                  : false)
        , _diffusion_coef(_build_diffusion
                              ? model_params.get<double>("Diffusion "
                                                         "Coefficient")
                              : std::numeric_limits<double>::quiet_NaN())
        , _decay("species_decay", _num_species, _num_species)
        , _xs(_build_fission_source
                  ? reaction_params.get<double>("Fission Cross-Section")
                  : std::numeric_limits<double>::quiet_NaN())
        , _gamma("atoms_per_species", _num_species)
        , _vapor_removal_multiplier("vapor_removal_multiplier", _num_species)
        , _sigma("microscopic_cross_section", _num_species, _num_species)
    {
        // Bateman matrix
        if (_build_bateman)
        {
            const auto decay
                = reaction_params.get<Teuchos::Array<double>>("Species Decay");
            for (int i = 0; i < _num_species; ++i)
            {
                for (int j = 0; j < _num_species; ++j)
                {
                    _decay(i, j) = decay[_num_species * i + j];
                }
            }
        }

        // Microscopic Cross-Section
        if (_build_transmutation_source)
        {
            const auto sigma = reaction_params.get<Teuchos::Array<double>>(
                "Microscopic Cross-Section");
            for (int i = 0; i < _num_species; ++i)
            {
                for (int j = 0; j < _num_species; ++j)
                {
                    _sigma(i, j) = sigma[_num_species * i + j];
                }
            }
        }

        // Atoms per species
        if (_build_fission_source)
        {
            auto gamma = reaction_params.get<Teuchos::Array<double>>(
                "Number of atoms per species");
            const auto gamma_view
                = Kokkos::View<double*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>(
                    gamma.data(), gamma.length());
            Kokkos::deep_copy(_gamma, gamma_view);
        }

        // Vapor removal ratio multiplier
        if (_build_vapor_removal_source)
        {
            auto vapor_removal_multiplier
                = reaction_params.get<Teuchos::Array<double>>(
                    "Vapor Removal Ratio Multiplier");
            const auto vapor_removal_multiplier_view
                = Kokkos::View<double*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>(
                    vapor_removal_multiplier.data(),
                    vapor_removal_multiplier.length());
            Kokkos::deep_copy(_vapor_removal_multiplier,
                              vapor_removal_multiplier_view);
        }
    }

    // Bateman matrix
    KOKKOS_INLINE_FUNCTION auto numSpecies() const { return _num_species; }

    // Bateman matrix
    KOKKOS_INLINE_FUNCTION auto batemanMatrix() const { return _decay; }

    // Include reaction term
    KOKKOS_INLINE_FUNCTION bool buildBateman() const { return _build_bateman; }

    // Include advection term
    KOKKOS_INLINE_FUNCTION bool buildAdvection() const
    {
        return _build_advection;
    }

    // Include diffusion term
    KOKKOS_INLINE_FUNCTION bool buildDiffusion() const
    {
        return _build_diffusion;
    }

    // Include fission source term
    KOKKOS_INLINE_FUNCTION bool buildFissionSource() const
    {
        return _build_fission_source;
    }

    // Include transmutation term
    KOKKOS_INLINE_FUNCTION bool buildTransmutationSource() const
    {
        return _build_transmutation_source;
    }

    // Include vapor removal source term
    KOKKOS_INLINE_FUNCTION bool buildVaporRemovalSource() const
    {
        return _build_vapor_removal_source;
    }

    // Constant diffusion coefficient
    KOKKOS_INLINE_FUNCTION double constantDiffusionCoefficient() const
    {
        return _diffusion_coef;
    }

    // Constant vapor removal ratio multiplier
    KOKKOS_INLINE_FUNCTION auto vaporRemovalRatioMultiplier() const
    {
        return _vapor_removal_multiplier;
    }

    // Fission cross section
    KOKKOS_INLINE_FUNCTION double fissionCrossSection() const { return _xs; }

    // Number of atoms per species
    KOKKOS_INLINE_FUNCTION auto atomsPerSpecies() const { return _gamma; }

    // Microscopic cross section
    KOKKOS_INLINE_FUNCTION auto microscopicCrossSection() const
    {
        return _sigma;
    }

  private:
    int _num_species;
    bool _build_bateman;
    bool _build_advection;
    bool _build_diffusion;
    bool _build_fission_source;
    bool _build_transmutation_source;
    bool _build_vapor_removal_source;
    double _diffusion_coef;
    Kokkos::View<double**, Kokkos::LayoutLeft, Kokkos::HostSpace> _decay;
    double _xs;
    Kokkos::View<double*, Kokkos::LayoutLeft, Kokkos::HostSpace> _gamma;
    Kokkos::View<double*, Kokkos::LayoutLeft, Kokkos::HostSpace>
        _vapor_removal_multiplier;
    Kokkos::View<double**, Kokkos::LayoutLeft, Kokkos::HostSpace> _sigma;
};

} // namespace SpeciesProperties
} // namespace VertexCFD

#endif // VERTEXCFD_CONSTANTSPECIESPROPERTIES_HPP
