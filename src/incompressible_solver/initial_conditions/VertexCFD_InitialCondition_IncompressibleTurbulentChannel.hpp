#ifndef VERTEXCFD_INITIALCONDITION_INCOMPRESSIBLETURBULENTCHANNEL_HPP
#define VERTEXCFD_INITIALCONDITION_INCOMPRESSIBLETURBULENTCHANNEL_HPP

#include <Panzer_Dimension.hpp>
#include <Panzer_Evaluator_WithBaseImpl.hpp>
#include <Panzer_PureBasis.hpp>

#include <Phalanx_Evaluator_Derived.hpp>
#include <Phalanx_Evaluator_WithBaseImpl.hpp>
#include <Phalanx_FieldManager.hpp>
#include <Phalanx_config.hpp>

#include <Teuchos_ParameterList.hpp>

#include <Kokkos_Core.hpp>

namespace VertexCFD
{
namespace InitialCondition
{
//---------------------------------------------------------------------------//
// Initial condition for 3D turbulent channel flow. Assumptions are:
// - Fully developed turbulent flow imposed in x-direction
// - Walls at +/- h in y-direction, zero y velocity
// - Option to add random perturbations in x-direction (default true)
// - Option to add Fourier modes in x- and z-directions (default 3 modes)
// Based on work by Kajzer and Pozorski (2018).
//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
class IncompressibleTurbulentChannel
    : public panzer::EvaluatorWithBaseImpl<Traits>,
      public PHX::EvaluatorDerived<EvalType, Traits>
{
  public:
    using scalar_type = typename EvalType::ScalarT;
    using view_layout = typename PHX::DevLayout<scalar_type>::type;
    static constexpr int num_space_dim = NumSpaceDim;
    static constexpr auto pi = Kokkos::numbers::pi_v<double>;

    IncompressibleTurbulentChannel(const Teuchos::ParameterList& ic_params,
                                   const panzer::PureBasis& basis);

    void postRegistrationSetup(typename Traits::SetupData sd,
                               PHX::FieldManager<Traits>& fm) override;

    void evaluateFields(typename Traits::EvalData workset) override;

    KOKKOS_INLINE_FUNCTION
    void operator()(
        const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const;

  public:
    PHX::MDField<scalar_type, panzer::Cell, panzer::BASIS> _lagrange_pressure;
    Kokkos::Array<PHX::MDField<scalar_type, panzer::Cell, panzer::BASIS>,
                  num_space_dim>
        _velocity;
    PHX::MDField<scalar_type, panzer::Cell, panzer::BASIS> _temperature;

  private:
    std::string _basis_name;
    int _basis_index;
    PHX::MDField<double, panzer::Cell, panzer::BASIS, panzer::Dim> _basis_coords;

    double _nu;
    double _h;
    double _Re_tau;
    double _u_tau;
    double _L_x;
    double _L_z;
    bool _add_rands;
    Kokkos::View<double**, view_layout, PHX::mem_space> _rands;
    int _nb_modes;
    double _U_0;
    bool _solve_temp;
    double _T_init;
};

//---------------------------------------------------------------------------//

} // end namespace InitialCondition
} // end namespace VertexCFD

#endif // end VERTEXCFD_INITIALCONDITION_INCOMPRESSIBLETURBULENTCHANNEL_HPP
