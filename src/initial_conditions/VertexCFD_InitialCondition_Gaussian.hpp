#ifndef VERTEXCFD_INITIALCONDITION_GAUSSIAN_HPP
#define VERTEXCFD_INITIALCONDITION_GAUSSIAN_HPP

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
template<class EvalType, class Traits, int NumSpaceDim>
class Gaussian : public panzer::EvaluatorWithBaseImpl<Traits>,
                 public PHX::EvaluatorDerived<EvalType, Traits>
{
  public:
    using scalar_type = typename EvalType::ScalarT;
    using view_layout = typename PHX::DevLayout<scalar_type>::type;
    static constexpr int num_space_dim = NumSpaceDim;

    Gaussian(const Teuchos::ParameterList& params,
             const panzer::PureBasis& basis);

    void postRegistrationSetup(typename Traits::SetupData sd,
                               PHX::FieldManager<Traits>& fm) override;

    void evaluateFields(typename Traits::EvalData workset) override;

    KOKKOS_INLINE_FUNCTION
    void operator()(
        const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const;

  public:
    PHX::MDField<scalar_type, panzer::Cell, panzer::BASIS> _ic;

  private:
    std::string _basis_name;
    Kokkos::Array<double, num_space_dim> _a;
    Kokkos::Array<double, num_space_dim> _b;
    Kokkos::Array<double, num_space_dim> _c;
    double _d;
    int _basis_index;
    PHX::MDField<double, panzer::Cell, panzer::BASIS, panzer::Dim> _basis_coords;

    enum Scaling
    {
        scaled,
        unscaled
    };

    Scaling _scaling;
};

//---------------------------------------------------------------------------//

} // end namespace InitialCondition
} // end namespace VertexCFD

#endif // end VERTEXCFD_INITIALCONDITION_GAUSSIAN_HPP
