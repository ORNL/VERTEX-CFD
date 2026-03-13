#ifndef VERTEXCFD_INCOMPRESSIBLELSVOFCLOSUREMODELFACTORY_HPP
#define VERTEXCFD_INCOMPRESSIBLELSVOFCLOSUREMODELFACTORY_HPP

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
class IncompressibleLSVOFFactory
{
  public:
    static constexpr int num_space_dim = NumSpaceDim;

    void buildClosureModel(
        const std::string& closure_type,
        const Teuchos::RCP<panzer::IntegrationRule>& ir,
        const Teuchos::ParameterList& user_params,
        const Teuchos::ParameterList& closure_params,
        const Teuchos::RCP<panzer::GlobalData>& global_data,
        bool& found_model,
        std::string& error_msg,
        Teuchos::RCP<std::vector<Teuchos::RCP<PHX::Evaluator<panzer::Traits>>>>
            evaluators);

    void buildDefaultClosureModels(
        const Teuchos::RCP<panzer::IntegrationRule>& ir,
        const Teuchos::ParameterList& closure_model_list,
        const Teuchos::ParameterList& user_params,
        const Teuchos::RCP<panzer::GlobalData>& global_data,
        Teuchos::RCP<std::vector<Teuchos::RCP<PHX::Evaluator<panzer::Traits>>>>
            evaluators);

    bool isDefaultClosureModel(const std::string& closure_type);

  private:
    enum LSVOFModelType
    {
        None,
        CLS,
        VOF
    };

    enum LSNormalReconstructionType
    {
        NonReconstructed
    };
};

//---------------------------------------------------------------------------//

} // end namespace ClosureModel
} // end namespace VertexCFD

#endif // end VERTEXCFD_INCOMPRESSIBLELSVOFCLOSUREMODELFACTORY_HPP
