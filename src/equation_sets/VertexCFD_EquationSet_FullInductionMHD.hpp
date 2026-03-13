#ifndef VERTEXCFD_EQUATIONSET_FULLINDUCTIONMHD_HPP
#define VERTEXCFD_EQUATIONSET_FULLINDUCTIONMHD_HPP

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
// Induction equations for MHD.
//---------------------------------------------------------------------------//
template<class EvalType>
class FullInductionMHD : public panzer::EquationSet_DefaultImpl<EvalType>
{
  public:
    FullInductionMHD(const Teuchos::RCP<Teuchos::ParameterList>& params,
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
    std::unordered_map<std::string, std::string> _equ_dof_fim_pair;
    std::unordered_map<std::string, std::unordered_map<std::string, bool>>
        _equ_source_term;
    bool _build_convective_flux;
    bool _build_resistive_flux;
};

//---------------------------------------------------------------------------//

} // end namespace EquationSet
} // end namespace VertexCFD

#endif // end VERTEXCFD_EQUATIONSET_FULLINDUCTIONMHD_HPP
