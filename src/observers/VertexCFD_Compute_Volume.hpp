#ifndef VERTEXCFD_COMPUTE_VOLUME_HPP
#define VERTEXCFD_COMPUTE_VOLUME_HPP

#include <Panzer_BCStrategy_Factory.hpp>
#include <Panzer_FieldManagerBuilder.hpp>
#include <Panzer_GlobalIndexer.hpp>
#include <Panzer_ResponseEvaluatorFactory_Functional.hpp>
#include <Panzer_ResponseLibrary.hpp>
#include <Panzer_Response_Functional.hpp>

#include <Thyra_VectorBase.hpp>

#include <Teuchos_ParameterList.hpp>
#include <Teuchos_RCP.hpp>

#include <Tempus_SolutionHistory_decl.hpp>

namespace VertexCFD
{
namespace ComputeVolume
{
//---------------------------------------------------------------------------//
template<class Scalar>
class Volume
{
  public:
    Volume(
        const Teuchos::RCP<panzer_stk::STK_Interface>& mesh,
        const Teuchos::RCP<const panzer::LinearObjFactory<panzer::Traits>>& lof,
        const Teuchos::RCP<panzer::ResponseLibrary<panzer::Traits>>&
            response_library,
        const std::vector<Teuchos::RCP<panzer::PhysicsBlock>>& physics_blocks,
        const panzer::ClosureModelFactory_TemplateManager<panzer::Traits>&
            cm_factory,
        const Teuchos::ParameterList& closure_params,
        const Teuchos::ParameterList& user_params,
        const Teuchos::RCP<panzer::WorksetContainer>& workset_container,
        const std::vector<panzer::BC>& bcs,
        const Teuchos::RCP<panzer::BCStrategyFactory>& bc_factory,
        const Teuchos::RCP<panzer::EquationSetFactory>& eq_set_factory,
        const int integration_order);

    void ComputeVol();
    double volume() const;

  private:
    Teuchos::RCP<const panzer::LinearObjFactory<panzer::Traits>> _lof;
    Teuchos::RCP<panzer::ResponseLibrary<panzer::Traits>> _response_library;
    std::vector<Teuchos::RCP<panzer::PhysicsBlock>> _physics_blocks;
    panzer::ClosureModelFactory_TemplateManager<panzer::Traits> _cm_factory;
    Teuchos::ParameterList _closure_params;
    Teuchos::ParameterList _user_params;
    Teuchos::RCP<panzer::WorksetContainer> _workset_container;

    std::vector<panzer::BC> _bcs;
    Teuchos::RCP<panzer::BCStrategyFactory> _bc_factory;
    Teuchos::RCP<panzer::EquationSetFactory> _eq_set_factory;

    double _volume;
};

//---------------------------------------------------------------------------//

} // end namespace ComputeVolume
} // end namespace VertexCFD

#include "VertexCFD_Compute_Volume_impl.hpp"

#endif // end VERTEXCFD_COMPUTE_VOLUME_HPP
