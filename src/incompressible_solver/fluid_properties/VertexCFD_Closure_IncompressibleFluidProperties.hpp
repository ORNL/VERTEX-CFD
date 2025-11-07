#ifndef VERTEXCFD_CLOSURE_INCOMPRESSIBLEFLUIDPROPERTIES_HPP
#define VERTEXCFD_CLOSURE_INCOMPRESSIBLEFLUIDPROPERTIES_HPP

#include <Panzer_Dimension.hpp>
#include <Panzer_Evaluator_WithBaseImpl.hpp>
#include <Panzer_IntegrationRule.hpp>

#include <Phalanx_Evaluator_Derived.hpp>
#include <Phalanx_KokkosDeviceTypes.hpp>
#include <Phalanx_MDField.hpp>

#include <Kokkos_Core.hpp>

#include <string>

namespace VertexCFD
{
namespace FluidProperties
{
//---------------------------------------------------------------------------//
// Fluid properties evaluation for fluid solver.
//---------------------------------------------------------------------------//
template<class EvalType, class Traits>
class IncompressibleFluidProperties
    : public panzer::EvaluatorWithBaseImpl<Traits>,
      public PHX::EvaluatorDerived<EvalType, Traits>
{
  public:
    using scalar_type = typename EvalType::ScalarT;

    IncompressibleFluidProperties(const panzer::IntegrationRule& ir,
                                  const Teuchos::ParameterList& fluid_params);

    void evaluateFields(typename Traits::EvalData workset) override;

    KOKKOS_INLINE_FUNCTION
    void operator()(
        const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const;

    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _density;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _kinematic_viscosity;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _thermal_conductivity;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _specific_heat_capacity;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _electrical_conductivity;

  private:
    double _rho;
    double _nu;
    double _k;
    double _cp;
    double _sigma;

    enum FluidPropertyType
    {
        constant
    };

    FluidPropertyType _fluid_prop_type;
};

//---------------------------------------------------------------------------//

} // end namespace FluidProperties
} // end namespace VertexCFD

#endif // end VERTEXCFD_CLOSURE_INCOMPRESSIBLEFLUIDPROPERTIES_HPP
