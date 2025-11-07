#ifndef VERTEXCFD_CONDUCTIONCLOSUREMODELFACTORY_HPP
#define VERTEXCFD_CONDUCTIONCLOSUREMODELFACTORY_HPP

#include <Panzer_ClosureModel_Factory.hpp>
#include <Panzer_Traits.hpp>

#include <Phalanx_Evaluator.hpp>

#include <Teuchos_ParameterList.hpp>
#include <Teuchos_RCP.hpp>

namespace VertexCFD
{
namespace ClosureModel
{
//---------------------------------------------------------------------------//
template<class EvalType, int NumSpaceDim>
class ConductionFactory : public panzer::ClosureModelFactory<EvalType>
{
  public:
    Teuchos::RCP<std::vector<Teuchos::RCP<PHX::Evaluator<panzer::Traits>>>>
    buildClosureModels(const std::string& model_id,
                       const Teuchos::ParameterList& model_params,
                       const panzer::FieldLayoutLibrary& fl,
                       const Teuchos::RCP<panzer::IntegrationRule>& ir,
                       const Teuchos::ParameterList& default_params,
                       const Teuchos::ParameterList& user_params,
                       const Teuchos::RCP<panzer::GlobalData>& global_data,
                       PHX::FieldManager<panzer::Traits>& fm) const override;
};

//---------------------------------------------------------------------------//

} // end namespace ClosureModel
} // end namespace VertexCFD

#endif // end VERTEXCFD_CONDUCTIONCLOSUREMODELFACTORY_HPP
