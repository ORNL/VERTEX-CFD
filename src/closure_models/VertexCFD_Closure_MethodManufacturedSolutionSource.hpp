#ifndef VERTEXCFD_CLOSURE_METHODMANUFACTUREDSOLUTIONSOURCE_HPP
#define VERTEXCFD_CLOSURE_METHODMANUFACTUREDSOLUTIONSOURCE_HPP

#include <Panzer_Dimension.hpp>
#include <Panzer_Evaluator_WithBaseImpl.hpp>

#include <Phalanx_Evaluator_Derived.hpp>
#include <Phalanx_Evaluator_WithBaseImpl.hpp>
#include <Phalanx_FieldManager.hpp>
#include <Phalanx_config.hpp>

#include <Kokkos_Core.hpp>

namespace VertexCFD
{
namespace ClosureModel
{
//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
class MethodManufacturedSolutionSource
    : public panzer::EvaluatorWithBaseImpl<Traits>,
      public PHX::EvaluatorDerived<EvalType, Traits>
{
  public:
    using scalar_type = typename EvalType::ScalarT;
    static constexpr int num_space_dim = NumSpaceDim;
    static constexpr int num_coeff = 2 * (num_space_dim + 1);
    static constexpr int num_conserve = num_space_dim + 2;

    MethodManufacturedSolutionSource(
        const panzer::IntegrationRule& ir,
        const bool build_viscous_flux,
        const Teuchos::ParameterList& closure_params);

    void postRegistrationSetup(typename Traits::SetupData sd,
                               PHX::FieldManager<Traits>& fm) override;

    void evaluateFields(typename Traits::EvalData workset) override;

    KOKKOS_INLINE_FUNCTION
    void operator()(const int cell) const;

    template<typename T>
    KOKKOS_INLINE_FUNCTION T
    set_function(const Kokkos::Array<double, num_coeff>& coeff,
                 const Kokkos::Array<T, num_space_dim>& x) const;

    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _continuity_mms_source;
    Kokkos::Array<PHX::MDField<scalar_type, panzer::Cell, panzer::Point>,
                  num_space_dim>
        _momentum_mms_source;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _energy_mms_source;

  private:
    int _ir_degree;
    int _ir_index;

    PHX::MDField<const double, panzer::Cell, panzer::Point, panzer::Dim> _ip_coords;

    Kokkos::Array<double, num_coeff> _phi_coeff;
    Kokkos::Array<Kokkos::Array<double, num_coeff>, num_space_dim> _vel_coeff;
    Kokkos::Array<double, num_coeff> _T_coeff;

    bool _build_viscous_flux;
    double _rho;
    double _nu;
    double _kappa;
};

//---------------------------------------------------------------------------//

} // end namespace ClosureModel
} // end namespace VertexCFD

#endif // end VERTEXCFD_CLOSURE_METHODMANUFACTUREDSOLUTIONSOURCE_HPP
