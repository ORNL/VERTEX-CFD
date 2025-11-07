#ifndef VERTEXCFD_EQUATIONSET_INCOMPRESSIBLE_LSVOF_HPP
#define VERTEXCFD_EQUATIONSET_INCOMPRESSIBLE_LSVOF_HPP

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
// Isothermal incompressible Navier Stokes equations for two phase flow
// with volume of fluid and/or level set interface tracking methods
//---------------------------------------------------------------------------//
template<class EvalType>
class IncompressibleLSVOF : public panzer::EquationSet_DefaultImpl<EvalType>
{
  public:
    IncompressibleLSVOF(const Teuchos::RCP<Teuchos::ParameterList>& params,
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
    std::unordered_map<std::string, std::string> _equ_dof_lsvof_pair;
    bool _build_lsvofmom_equ;
    bool _build_lsvof_buoyancy_source;
    bool _build_lsvof_surface_tension;
    std::string _lsvof_model_name;
};

//---------------------------------------------------------------------------//

} // end namespace EquationSet
} // end namespace VertexCFD

#endif // end VERTEXCFD_EQUATIONSET_INCOMPRESSIBLE_LSVOF_HPP
