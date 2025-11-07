#ifndef VERTEXCFD_FULLINDUCTIONINITIALCONDITIONFACTORY_IMPL_HPP
#define VERTEXCFD_FULLINDUCTIONINITIALCONDITIONFACTORY_IMPL_HPP

#include "VertexCFD_InitialCondition_DivergenceAdvectionTest.hpp"
#include "VertexCFD_InitialCondition_MHDVortexProblem.hpp"

namespace VertexCFD
{
namespace InitialCondition
{
//---------------------------------------------------------------------------//
template<class EvalType, int NumSpaceDim>
void FullInductionICFactory<EvalType, NumSpaceDim>::buildClosureModel(
    const std::string& ic_type,
    const std::vector<Teuchos::RCP<const panzer::PureBasis>>& bases,
    const Teuchos::ParameterList& /*user_params*/,
    const Teuchos::ParameterList& ic_params,
    bool& found_model,
    std::string& error_msg,
    Teuchos::RCP<std::vector<Teuchos::RCP<PHX::Evaluator<panzer::Traits>>>>
        evaluators)
{
    if (ic_type == "MHDVortexProblem")
    {
        for (const auto& b : bases)
        {
            auto eval = Teuchos::rcp(
                new MHDVortexProblem<EvalType, panzer::Traits, num_space_dim>(
                    ic_params, *b));
            evaluators->push_back(eval);
            found_model = true;
        }
    }

    if (ic_type == "DivergenceAdvectionTest")
    {
        for (const auto& b : bases)
        {
            auto eval = Teuchos::rcp(
                new DivergenceAdvectionTest<EvalType, panzer::Traits, num_space_dim>(
                    ic_params, *b));
            evaluators->push_back(eval);
            found_model = true;
        }
    }

    error_msg = availableInitialConditions();
}

//---------------------------------------------------------------------------//

} // end namespace InitialCondition
} // end namespace VertexCFD

#endif // end VERTEXCFD_FULLINDUCTIONINITIALCONDITIONFACTORY_IMPL_HPP
