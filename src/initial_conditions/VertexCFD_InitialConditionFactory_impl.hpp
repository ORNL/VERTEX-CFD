#ifndef VERTEXCFD_INITIALCONDITIONFACTORY_IMPL_HPP
#define VERTEXCFD_INITIALCONDITIONFACTORY_IMPL_HPP

#include "VertexCFD_InitialCondition_Circle.hpp"
#include "VertexCFD_InitialCondition_Constant.hpp"
#include "VertexCFD_InitialCondition_Gaussian.hpp"
#include "VertexCFD_InitialCondition_InverseGaussian.hpp"
#include "VertexCFD_InitialCondition_MethodManufacturedSolution.hpp"
#include "VertexCFD_InitialCondition_Sinusoidal.hpp"
#include "VertexCFD_InitialCondition_Step.hpp"

#ifdef VERTEXCFD_ENABLE_FULL_INDUCTION_MHD
#include "full_induction_mhd_solver/initial_conditions/VertexCFD_FullInductionInitialConditionFactory.hpp"
#endif

#include "incompressible_lsvof_solver/initial_conditions/VertexCFD_InitialCondition_IncompressibleLSVOFBubble.hpp"
#include "incompressible_solver/initial_conditions/VertexCFD_InitialCondition_IncompressibleLaminarFlow.hpp"
#include "incompressible_solver/initial_conditions/VertexCFD_InitialCondition_IncompressibleTaylorGreenVortex.hpp"
#include "incompressible_solver/initial_conditions/VertexCFD_InitialCondition_IncompressibleTurbulentChannel.hpp"
#include "incompressible_solver/initial_conditions/VertexCFD_InitialCondition_IncompressibleVortexInBox.hpp"

#include <Panzer_FieldLibrary.hpp>
#include <Panzer_PureBasis.hpp>
#include <Panzer_STK_GatherFields.hpp>
#include <Panzer_String_Utilities.hpp>

#include <Teuchos_RCP.hpp>

#include <vector>

namespace VertexCFD
{
namespace InitialCondition
{
//---------------------------------------------------------------------------//
template<class EvalType, int NumSpaceDim>
Factory<EvalType, NumSpaceDim>::Factory(
    Teuchos::RCP<const panzer_stk::STK_Interface> mesh)
    : _mesh{mesh}
{
}

//---------------------------------------------------------------------------//
template<class EvalType, int NumSpaceDim>
Teuchos::RCP<std::vector<Teuchos::RCP<PHX::Evaluator<panzer::Traits>>>>
Factory<EvalType, NumSpaceDim>::buildClosureModels(
    const std::string& block_id,
    const Teuchos::ParameterList& block_params,
    const panzer::FieldLayoutLibrary& fl,
    const Teuchos::RCP<panzer::IntegrationRule>&,
    const Teuchos::ParameterList&,
    const Teuchos::ParameterList& user_params,
    const Teuchos::RCP<panzer::GlobalData>&,
    PHX::FieldManager<panzer::Traits>&) const
{
    auto evaluators = Teuchos::rcp(
        new std::vector<Teuchos::RCP<PHX::Evaluator<panzer::Traits>>>);

    // Space dimension
    constexpr int num_dim_space = NumSpaceDim;

    if (!block_params.isSublist(block_id))
    {
        throw std::runtime_error("Initial condition block id not in list");
    }

    // Initial conditions apply to the bases.
    std::vector<Teuchos::RCP<const panzer::PureBasis>> bases;
    fl.uniqueBases(bases);

#ifdef VERTEXCFD_ENABLE_FULL_INDUCTION_MHD
    // Full induction solver factory objects
    FullInductionICFactory<EvalType, NumSpaceDim> full_induction_factory;
    std::string full_ind_error_msg = "None";
#endif

    // Loop over initial conditions
    const Teuchos::ParameterList& ic_params = block_params.sublist(block_id);
    for (auto param_itr = ic_params.begin(); param_itr != ic_params.end();
         ++param_itr)
    {
        bool found_model = false;

        auto key = param_itr->first;
        const auto& p
            = Teuchos::getValue<Teuchos::ParameterList>(param_itr->second);

        if (p.isType<std::string>("Type"))
        {
            const std::string type = p.get<std::string>("Type");

            // Read initial conditions from corresponding fields of the input
            // mesh file.
            if (type == "From File")
            {
                std::vector<std::string> field_names;
                panzer::StringTokenizer(
                    field_names, p.get<std::string>("Field Names"), ",", true);

                // Assume we use the same basis for all DoFs.
                auto basis = fl.lookupLayout(field_names.at(0));

                auto plist
                    = Teuchos::ParameterList{}
                          .set("Field Names", Teuchos::rcpFromRef(field_names))
                          .set("Basis", basis);

                auto eval = Teuchos::rcp(
                    new panzer_stk::GatherFields<EvalType, panzer::Traits>(
                        _mesh, plist));
                evaluators->push_back(eval);
                found_model = true;
            }

#ifdef VERTEXCFD_ENABLE_FULL_INDUCTION_MHD
            if (std::string::npos
                != full_induction_factory.availableInitialConditions().find(
                    type))
            {
                full_induction_factory.buildClosureModel(type,
                                                         bases,
                                                         user_params,
                                                         p,
                                                         found_model,
                                                         full_ind_error_msg,
                                                         evaluators);
            }
#endif

            if (type == "Constant")
            {
                for (const auto& b : bases)
                {
                    auto eval = Teuchos::rcp(
                        new Constant<EvalType, panzer::Traits>(p, *b));
                    evaluators->push_back(eval);
                    found_model = true;
                }
            }

            if (type == "Gaussian")
            {
                for (const auto& b : bases)
                {
                    auto eval = Teuchos::rcp(
                        new Gaussian<EvalType, panzer::Traits, num_dim_space>(
                            p, *b));
                    evaluators->push_back(eval);
                    found_model = true;
                }
            }

            if (type == "Step")
            {
                for (const auto& b : bases)
                {
                    auto eval = Teuchos::rcp(
                        new Step<EvalType, panzer::Traits>(p, *b));
                    evaluators->push_back(eval);
                    found_model = true;
                }
            }

            if (type == "Circle")
            {
                for (const auto& b : bases)
                {
                    auto eval = Teuchos::rcp(
                        new Circle<EvalType, panzer::Traits, num_dim_space>(
                            p, *b));
                    evaluators->push_back(eval);
                    found_model = true;
                }
            }

            if (type == "InverseGaussian")
            {
                for (const auto& b : bases)
                {
                    auto eval = Teuchos::rcp(
                        new InverseGaussian<EvalType, panzer::Traits, num_dim_space>(
                            p, *b));
                    evaluators->push_back(eval);
                    found_model = true;
                }
            }

            if (type == "MethodManufacturedSolution")
            {
                for (const auto& b : bases)
                {
                    auto eval = Teuchos::rcp(
                        new MethodManufacturedSolution<EvalType,
                                                       panzer::Traits,
                                                       num_dim_space>(*b));
                    evaluators->push_back(eval);
                    found_model = true;
                }
            }

            if (type == "IncompressibleVortexInBox")
            {
                for (const auto& b : bases)
                {
                    auto eval = Teuchos::rcp(
                        new IncompressibleVortexInBox<EvalType, panzer::Traits>(
                            *b));
                    evaluators->push_back(eval);
                    found_model = true;
                }
            }

            if (type == "IncompressibleTaylorGreenVortex")
            {
                for (const auto& b : bases)
                {
                    auto eval = Teuchos::rcp(
                        new IncompressibleTaylorGreenVortex<EvalType,
                                                            panzer::Traits,
                                                            num_dim_space>(*b));
                    evaluators->push_back(eval);
                    found_model = true;
                }
            }

            if (type == "IncompressibleLaminarFlow")
            {
                for (const auto& b : bases)
                {
                    auto eval = Teuchos::rcp(
                        new IncompressibleLaminarFlow<EvalType,
                                                      panzer::Traits,
                                                      num_dim_space>(p, *b));
                    evaluators->push_back(eval);
                    found_model = true;
                }
            }

            if (type == "IncompressibleTurbulentChannel")
            {
                for (const auto& b : bases)
                {
                    auto eval = Teuchos::rcp(
                        new IncompressibleTurbulentChannel<EvalType,
                                                           panzer::Traits,
                                                           num_dim_space>(p,
                                                                          *b));
                    evaluators->push_back(eval);
                    found_model = true;
                }
            }

            if (type == "Sinusoidal")
            {
                for (const auto& b : bases)
                {
                    auto eval = Teuchos::rcp(
                        new Sinusoidal<EvalType, panzer::Traits>(p, *b));
                    evaluators->push_back(eval);
                    found_model = true;
                }
            }

            if (type == "IncompressibleLSVOFBubble")
            {
                for (const auto& b : bases)
                {
                    auto eval = Teuchos::rcp(
                        new IncompressibleLSVOFBubble<EvalType,
                                                      panzer::Traits,
                                                      num_dim_space>(p, *b));
                    evaluators->push_back(eval);
                    found_model = true;
                }
            }
        }

        if (!found_model)
        {
            std::string msg = "Initial condition " + key
                              + "  failed to build.\n";
            msg += "The initial conditions implemented in VertexCFD are:\n";
            msg += "Circle\n";
            msg += "Constant\n";
            msg += "From File\n";
            msg += "Gaussian\n";
            msg += "IncompressibleLaminarFlow\n";
            msg += "IncompressibleLSVOFBubble\n";
            msg += "IncompressibleTaylorGreenVortex\n";
            msg += "IncompressibleTurbulentFlow\n";
            msg += "IncompressibleVortexInBox\n";
            msg += "InverseGaussian\n";
            msg += "MethodManufacturedSolution\n";
            msg += "Sinusoidal\n";
            msg += "Step\n";
#ifdef VERTEXCFD_ENABLE_FULL_INDUCTION_MHD
            msg += "=================================\n";
            msg += "Full induction MHD closure models:\n";
            msg += full_ind_error_msg;
#endif
            throw std::runtime_error(msg);
        }
    }

    return evaluators;
}

//---------------------------------------------------------------------------//

} // end namespace InitialCondition
} // end namespace VertexCFD

#endif // end VERTEXCFD_INITIALCONDITIONFACTORY_IMPL_HPP
