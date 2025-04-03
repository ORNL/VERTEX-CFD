#ifndef VERTEXCFD_CLOSURE_EXTERNALMAGNETICFIELD_HPP
#define VERTEXCFD_CLOSURE_EXTERNALMAGNETICFIELD_HPP

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
// External magnetic field.
//---------------------------------------------------------------------------//
template<class EvalType, class Traits>
class ExternalMagneticField : public panzer::EvaluatorWithBaseImpl<Traits>,
                              public PHX::EvaluatorDerived<EvalType, Traits>
{
  public:
    using scalar_type = typename EvalType::ScalarT;
    static constexpr int field_size = 3;

    ExternalMagneticField(const panzer::IntegrationRule& ir,
                          const Teuchos::ParameterList& user_params);

    void postRegistrationSetup(typename Traits::SetupData sd,
                               PHX::FieldManager<Traits>& fm) override;

    void evaluateFields(typename Traits::EvalData workset) override;

    KOKKOS_INLINE_FUNCTION
    void operator()(
        const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const;

    Kokkos::Array<PHX::MDField<scalar_type, panzer::Cell, panzer::Point>, field_size>
        _ext_magn_field;

  private:
    Kokkos::Array<double, field_size> _ext_magn_vct;
    Kokkos::Array<double, field_size> _d_ext_magn_vct_dt;
    double _toroidal_field_magn;
    int _ir_degree;
    int _ir_index;
    double _time;

    enum ExtMagnType
    {
        constant,
        toroidal
    };

    ExtMagnType _ext_magn_type;
    PHX::MDField<const double, panzer::Cell, panzer::Point, panzer::Dim> _ip_coords;
};

//---------------------------------------------------------------------------//

} // end namespace ClosureModel
} // end namespace VertexCFD

#endif // end VERTEXCFD_CLOSURE_EXTERNALMAGNETICFIELD_HPP
