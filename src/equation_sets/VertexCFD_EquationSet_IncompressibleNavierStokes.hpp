#ifndef VERTEXCFD_EQUATIONSET_INCOMPRESSIBLE_NAVIERSTOKES_HPP
#define VERTEXCFD_EQUATIONSET_INCOMPRESSIBLE_NAVIERSTOKES_HPP

#include <Panzer_CellData.hpp>
#include <Panzer_EquationSet_DefaultImpl.hpp>
#include <Panzer_GlobalData.hpp>
#include <Panzer_Traits.hpp>

#include <Phalanx_FieldManager.hpp>

#include <Teuchos_ParameterList.hpp>
#include <Teuchos_RCP.hpp>

#include <string>
#include <unordered_map>
#include <vector>

namespace VertexCFD
{
namespace EquationSet
{
//---------------------------------------------------------------------------//
// Isothermal Navier Stokes equations for incompressible flow.
// Navier Stokes equations for incompressible flow.
//---------------------------------------------------------------------------//
template<class EvalType>
class IncompressibleNavierStokes
    : public panzer::EquationSet_DefaultImpl<EvalType>
{
  public:
    IncompressibleNavierStokes(const Teuchos::RCP<Teuchos::ParameterList>& params,
                               const int& default_integration_order,
                               const panzer::CellData& cell_data,
                               const Teuchos::RCP<panzer::GlobalData>& gd,
                               const bool build_transient_support);

    void buildAndRegisterEquationSetEvaluators(
        PHX::FieldManager<panzer::Traits>& fm,
        const panzer::FieldLibrary& field_library,
        const Teuchos::ParameterList& user_data) const override;

  private:
    int _num_space_dim;
    std::unordered_map<std::string, std::string> _equ_dof_ns_pair;
    std::unordered_map<std::string, std::string> _equ_dof_ep_pair;
    std::unordered_map<std::string, std::string> _equ_dof_tm_pair;
    std::unordered_map<std::string, std::unordered_map<std::string, bool>>
        _equ_source_term;
    bool _build_viscous_flux;
    bool _build_temp_equ;
    bool _build_ind_less_equ;
    bool _build_constant_source;
    bool _build_buoyancy_source;
    bool _build_viscous_heat;
    std::string _turbulence_model;
    std::string _stabilization_method;
    bool _build_godunov_powell_source;
    bool _build_joule_heating_source;

    enum class ConModel
    {
        AC,
        EDAC,
        EDACTempNC
    };
    ConModel _continuity_model;
};

//---------------------------------------------------------------------------//

} // end namespace EquationSet
} // end namespace VertexCFD

#endif // end VERTEXCFD_EQUATIONSET_INCOMPRESSIBLE_NAVIERSTOKES_HPP
