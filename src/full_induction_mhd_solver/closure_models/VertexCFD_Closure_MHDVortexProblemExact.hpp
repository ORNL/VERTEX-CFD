#ifndef VERTEXCFD_CLOSURE_MHDVORTEXPROBLEMEXACT_HPP
#define VERTEXCFD_CLOSURE_MHDVORTEXPROBLEMEXACT_HPP

#include <Panzer_Dimension.hpp>
#include <Panzer_Evaluator_WithBaseImpl.hpp>

#include <Phalanx_Evaluator_Derived.hpp>
#include <Phalanx_Evaluator_WithBaseImpl.hpp>
#include <Phalanx_FieldManager.hpp>
#include <Phalanx_config.hpp>

#include <Kokkos_Core.hpp>

#include <string>

namespace VertexCFD
{
namespace ClosureModel
{
//---------------------------------------------------------------------------//
// Exact solution for MHD vortex problem
//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
class MHDVortexProblemExact : public panzer::EvaluatorWithBaseImpl<Traits>,
                              public PHX::EvaluatorDerived<EvalType, Traits>
{
  public:
    using scalar_type = typename EvalType::ScalarT;
    static constexpr int num_space_dim = NumSpaceDim;

    MHDVortexProblemExact(const panzer::IntegrationRule& ir,
                          const Teuchos::ParameterList& vortex_params);

    void postRegistrationSetup(typename Traits::SetupData sd,
                               PHX::FieldManager<Traits>& fm) override;

    void evaluateFields(typename Traits::EvalData workset) override;

    KOKKOS_INLINE_FUNCTION
    void operator()(
        const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const;

  private:
    int _ir_degree;
    int _ir_index;
    int _basis_index;
    Kokkos::Array<double, 2> _vel_0;
    Kokkos::Array<double, 2> _xy_0;
    double _time;
    PHX::MDField<const double, panzer::Cell, panzer::Point, panzer::Dim> _ip_coords;

  public:
    PHX::MDField<double, panzer::Cell, panzer::Point> _lagrange_pressure;
    Kokkos::Array<PHX::MDField<double, panzer::Cell, panzer::Point>, num_space_dim>
        _velocity;
    Kokkos::Array<PHX::MDField<double, panzer::Cell, panzer::Point>, num_space_dim>
        _induced_magnetic_field;
};

//---------------------------------------------------------------------------//

} // end namespace ClosureModel
} // end namespace VertexCFD

#endif // end
       // VERTEXCFD_CLOSURE_MHDVORTEXPROBLEMEXACT_HPP
