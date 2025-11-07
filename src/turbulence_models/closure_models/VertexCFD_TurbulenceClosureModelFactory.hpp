#ifndef VERTEXCFD_TURBULENCECLOSUREMODELFACTORY_HPP
#define VERTEXCFD_TURBULENCECLOSUREMODELFACTORY_HPP

#include <Panzer_GlobalData.hpp>
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
class TurbulenceFactory
{
  public:
    static constexpr int num_space_dim = NumSpaceDim;

    void buildClosureModel(
        const Teuchos::RCP<panzer::IntegrationRule>& ir,
        const Teuchos::RCP<panzer::GlobalData>& global_data,
        const Teuchos::ParameterList& user_params,
        const std::string& turbulence_model_name,
        Teuchos::RCP<std::vector<Teuchos::RCP<PHX::Evaluator<panzer::Traits>>>>
            evaluators);
};

//---------------------------------------------------------------------------//

} // end namespace ClosureModel
} // end namespace VertexCFD

#endif // end VERTEXCFD_TURBULENCECLOSUREMODELFACTORY_HPP
