#ifndef VERTEXCFD_EQUATIONSET_CONDUCTION_HPP
#define VERTEXCFD_EQUATIONSET_CONDUCTION_HPP

#include <Panzer_CellData.hpp>
#include <Panzer_EquationSet_DefaultImpl.hpp>
#include <Panzer_FieldLibrary.hpp>
#include <Panzer_GlobalData.hpp>
#include <Panzer_Traits.hpp>

#include <Phalanx_FieldManager.hpp>

#include <Teuchos_ParameterList.hpp>
#include <Teuchos_RCP.hpp>

#include <string>

namespace VertexCFD
{
namespace EquationSet
{
//---------------------------------------------------------------------------//
// Conductive heat equation
//---------------------------------------------------------------------------//
template<class EvalType>
class Conduction : public panzer::EquationSet_DefaultImpl<EvalType>
{
  public:
    Conduction(const Teuchos::RCP<Teuchos::ParameterList>& params,
               const int& default_integration_order,
               const panzer::CellData& cell_data,
               const Teuchos::RCP<panzer::GlobalData>& gd,
               const bool build_transient_support);

    void buildAndRegisterEquationSetEvaluators(
        PHX::FieldManager<panzer::Traits>& fm,
        const panzer::FieldLibrary& field_library,
        const Teuchos::ParameterList& user_data) const override;

    std::string fieldName(const int dof) const;

  private:
    const std::string _dof_name;
    const std::string _equ_name;
    bool _build_source_term;
};

//---------------------------------------------------------------------------//

} // end namespace EquationSet
} // end namespace VertexCFD

#endif // end VERTEXCFD_EQUATIONSET_CONDUCTION_HPP
