#ifndef VERTEXCFD_BOUNDARYCONDITION_INCOMPRESSIBLEBOUNDARYFLUX_HPP
#define VERTEXCFD_BOUNDARYCONDITION_INCOMPRESSIBLEBOUNDARYFLUX_HPP

#include "boundary_conditions/VertexCFD_BCStrategy_BoundaryFluxBase.hpp"

#include <Panzer_BCStrategy.hpp>
#include <Panzer_Evaluator_WithBaseImpl.hpp>
#include <Panzer_GlobalDataAcceptor_DefaultImpl.hpp>
#include <Panzer_PhysicsBlock.hpp>
#include <Panzer_Traits.hpp>

#include <Phalanx_Evaluator_WithBaseImpl.hpp>
#include <Phalanx_FieldManager.hpp>
#include <Phalanx_MDField.hpp>

#include <Teuchos_RCP.hpp>

#include <unordered_map>

namespace VertexCFD
{
namespace BoundaryCondition
{
//---------------------------------------------------------------------------//
template<class EvalType, int NumSpaceDim>
class IncompressibleBoundaryFlux
    : public BoundaryFluxBase<EvalType, NumSpaceDim>
{
  public:
    IncompressibleBoundaryFlux(
        const panzer::BC& bc,
        const Teuchos::RCP<panzer::GlobalData>& global_data);

    static constexpr int num_space_dim = NumSpaceDim;

    void setup(const panzer::PhysicsBlock& side_pb,
               const Teuchos::ParameterList& user_data) override;

    void buildAndRegisterEvaluators(
        PHX::FieldManager<panzer::Traits>& fm,
        const panzer::PhysicsBlock& side_pb,
        const panzer::ClosureModelFactory_TemplateManager<panzer::Traits>& factory,
        const Teuchos::ParameterList& models,
        const Teuchos::ParameterList& user_data) const override;

    void buildAndRegisterScatterEvaluators(
        PHX::FieldManager<panzer::Traits>& fm,
        const panzer::PhysicsBlock& side_pb,
        const panzer::LinearObjFactory<panzer::Traits>& lof,
        const Teuchos::ParameterList& user_data) const override;

  private:
    std::unordered_map<std::string, std::string> _equ_dof_ns_pair;
    std::unordered_map<std::string, std::string> _equ_dof_ep_pair;
    std::unordered_map<std::string, std::string> _equ_dof_tm_pair;

    enum class ConModel
    {
        AC,
        EDAC,
        EDACTempNC
    };
    ConModel _continuity_model;

    bool _build_viscous_flux;
    bool _build_temp_equ;
    bool _internal_interface;
    bool _build_ind_less_equ;
    bool _turbulence_model_bool;
    std::string _continuity_model_name;
};

//---------------------------------------------------------------------------//

} // end namespace BoundaryCondition
} // end namespace VertexCFD

#endif // end VERTEXCFD_BOUNDARYCONDITION_INCOMPRESSIBLEBOUNDARYFLUX_HPP
