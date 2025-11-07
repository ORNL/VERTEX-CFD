#ifndef VERTEXCFD_FULLINDUCTIONINITIALCONDITIONFACTORY_HPP
#define VERTEXCFD_FULLINDUCTIONINITIALCONDITIONFACTORY_HPP

#include <Panzer_Traits.hpp>

#include <Phalanx_Evaluator.hpp>

#include <Panzer_PureBasis.hpp>

#include <Teuchos_ParameterList.hpp>
#include <Teuchos_RCP.hpp>

namespace VertexCFD
{
namespace InitialCondition
{
//---------------------------------------------------------------------------//
template<class EvalType, int NumSpaceDim>
class FullInductionICFactory
{
  public:
    static constexpr int num_space_dim = NumSpaceDim;

    void buildClosureModel(
        const std::string& closure_type,
        const std::vector<Teuchos::RCP<const panzer::PureBasis>>& bases,
        const Teuchos::ParameterList& user_params,
        const Teuchos::ParameterList& ic_params,
        bool& found_model,
        std::string& error_msg,
        Teuchos::RCP<std::vector<Teuchos::RCP<PHX::Evaluator<panzer::Traits>>>>
            evaluators);

    std::string availableInitialConditions()
    {
        return "DivergenceAdvectionTest\n"
               "MHDVortexProblem\n";
    }
};

//---------------------------------------------------------------------------//

} // end namespace InitialCondition
} // end namespace VertexCFD

#endif // end VERTEXCFD_FULLINDUCTIONINITIALCONDITIONFACTORY_HPP
