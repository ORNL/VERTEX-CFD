#ifndef VERTEXCFD_COMPUTE_ERRORNORMS_HPP
#define VERTEXCFD_COMPUTE_ERRORNORMS_HPP

#include "boundary_conditions/VertexCFD_BCStrategy_Factory.hpp"

#include <Panzer_FieldManagerBuilder.hpp>
#include <Panzer_GlobalIndexer.hpp>
#include <Panzer_ResponseEvaluatorFactory_Functional.hpp>
#include <Panzer_ResponseLibrary.hpp>
#include <Panzer_Response_Functional.hpp>

#include <Thyra_VectorBase.hpp>

#include <Teuchos_ParameterList.hpp>
#include <Teuchos_RCP.hpp>

#include <Tempus_SolutionHistory_decl.hpp>

#include <vector>

namespace VertexCFD
{
namespace ComputeErrorNorms
{
//---------------------------------------------------------------------------//
template<class Scalar>
class ErrorNorms
{
  public:
    ErrorNorms(
        const Teuchos::RCP<panzer_stk::STK_Interface>& mesh,
        const Teuchos::RCP<const panzer::LinearObjFactory<panzer::Traits>>& lof,
        const Teuchos::RCP<panzer::ResponseLibrary<panzer::Traits>>&
            response_library,
        const std::vector<Teuchos::RCP<panzer::PhysicsBlock>>& physics_blocks,
        const panzer::ClosureModelFactory_TemplateManager<panzer::Traits>&
            cm_factory,
        const Teuchos::ParameterList& closure_params,
        const Teuchos::ParameterList& user_params,
        const Teuchos::RCP<panzer::EquationSetFactory>& eq_set_factory,
        const double volume,
        const int integration_order);

    void ComputeNorms(
        const Teuchos::RCP<Tempus::SolutionState<Scalar>>& working_state);

    struct DofErrorNorm
    {
        std::string name{};
        double error_norm = 0.0;

        explicit DofErrorNorm(const std::string& dof_name)
            : name(dof_name)
        {
        }
        ~DofErrorNorm() = default;
        DofErrorNorm(const DofErrorNorm&) = default;
        DofErrorNorm& operator=(const DofErrorNorm&) = default;
        DofErrorNorm(DofErrorNorm&&) = default;
        DofErrorNorm& operator=(DofErrorNorm&&) = default;
    };

  private:
    Teuchos::RCP<const panzer::LinearObjFactory<panzer::Traits>> _lof;
    Teuchos::RCP<panzer::ResponseLibrary<panzer::Traits>> _response_library;
    std::vector<Teuchos::RCP<panzer::PhysicsBlock>> _physics_blocks;
    panzer::ClosureModelFactory_TemplateManager<panzer::Traits> _cm_factory;
    Teuchos::ParameterList _closure_params;
    Teuchos::ParameterList _user_params;

    Teuchos::RCP<panzer::EquationSetFactory> _eq_set_factory;

    double _volume;
    std::vector<DofErrorNorm> _L1_error_norms;
    std::vector<DofErrorNorm> _L2_error_norms;
    int _L1_error_norm_order;
    int _L2_error_norm_order;
    int _num_space_dim;

  public:
    const std::vector<DofErrorNorm>& L1_errorNorms() const
    {
        return _L1_error_norms;
    }

    const std::vector<DofErrorNorm>& L2_errorNorms() const
    {
        return _L2_error_norms;
    }
};

//---------------------------------------------------------------------------//

} // end namespace ComputeErrorNorms
} // end namespace VertexCFD

#include "VertexCFD_Compute_ErrorNorms_impl.hpp"

#endif // end VERTEXCFD_COMPUTE_ERRORNORMS_HPP
