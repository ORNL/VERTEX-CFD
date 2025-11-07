#ifndef VERTEXCFD_BOUNDARYCONDITION_BOUNDARYFLUXBASE_HPP
#define VERTEXCFD_BOUNDARYCONDITION_BOUNDARYFLUXBASE_HPP

#include <Panzer_BCStrategy.hpp>
#include <Panzer_Evaluator_WithBaseImpl.hpp>
#include <Panzer_GlobalDataAcceptor_DefaultImpl.hpp>
#include <Panzer_PhysicsBlock.hpp>
#include <Panzer_Traits.hpp>

#include <Phalanx_Evaluator_WithBaseImpl.hpp>
#include <Phalanx_FieldManager.hpp>
#include <Phalanx_MDField.hpp>

#include <Teuchos_RCP.hpp>

namespace VertexCFD
{
namespace BoundaryCondition
{
//---------------------------------------------------------------------------//
template<class EvalType, int NumSpaceDim>
class BoundaryFluxBase : public panzer::BCStrategy<EvalType>,
                         public panzer::GlobalDataAcceptorDefaultImpl,
                         public panzer::EvaluatorWithBaseImpl<panzer::Traits>
{
  public:
    // Space dimension
    static constexpr int num_space_dim = NumSpaceDim;

    BoundaryFluxBase(const panzer::BC& bc,
                     const Teuchos::RCP<panzer::GlobalData>& global_data);

    virtual void setup(const panzer::PhysicsBlock& side_pb,
                       const Teuchos::ParameterList& user_data)
        = 0;

    virtual void buildAndRegisterEvaluators(
        PHX::FieldManager<panzer::Traits>& fm,
        const panzer::PhysicsBlock& side_pb,
        const panzer::ClosureModelFactory_TemplateManager<panzer::Traits>& factory,
        const Teuchos::ParameterList& models,
        const Teuchos::ParameterList& user_data) const
        = 0;

    virtual void buildAndRegisterScatterEvaluators(
        PHX::FieldManager<panzer::Traits>& fm,
        const panzer::PhysicsBlock& side_pb,
        const panzer::LinearObjFactory<panzer::Traits>& lof,
        const Teuchos::ParameterList& user_data) const
        = 0;

    virtual void buildAndRegisterGatherAndOrientationEvaluators(
        PHX::FieldManager<panzer::Traits>& fm,
        const panzer::PhysicsBlock& side_pb,
        const panzer::LinearObjFactory<panzer::Traits>& lof,
        const Teuchos::ParameterList& user_data) const
        = 0;

    virtual void postRegistrationSetup(typename panzer::Traits::SetupData d,
                                       PHX::FieldManager<panzer::Traits>& vm)
        = 0;

    virtual void evaluateFields(typename panzer::Traits::EvalData d) = 0;

    // Local members
    void initialize(const panzer::PhysicsBlock& side_pb,
                    std::unordered_map<std::string, std::string>& dof_eq_map);

    void getModelID(const Teuchos::ParameterList& bc_params,
                    const panzer::PhysicsBlock& side_pb,
                    std::string& model_id) const;

    auto getBasisIRLayout(const panzer::PhysicsBlock& side_pb,
                          const std::string& dof_name) const;

    void registerDOFsGradient(PHX::FieldManager<panzer::Traits>& fm,
                              const panzer::PhysicsBlock& side_pb,
                              const std::string& dof_name) const;

    void registerSideNormals(PHX::FieldManager<panzer::Traits>& fm,
                             const panzer::PhysicsBlock& side_pb) const;

    void registerConvectionTypeFluxOperator(
        std::pair<const std::string, std::string> dof_eq_pair,
        std::unordered_map<std::string, std::vector<std::string>>& eq_vct_map,
        const std::string& closure_name,
        PHX::FieldManager<panzer::Traits>& fm,
        const panzer::PhysicsBlock& side_pb,
        const double& multiplier) const;

    void registerPenaltyAndViscousGradientOperator(
        std::pair<const std::string, std::string> dof_eq_pair,
        PHX::FieldManager<panzer::Traits>& fm,
        const panzer::PhysicsBlock& side_pb,
        const Teuchos::ParameterList& bc_params) const;

    void registerViscousTypeFluxOperator(
        std::pair<const std::string, std::string> dof_eq_pair,
        std::unordered_map<std::string, std::vector<std::string>>& eq_vct_map,
        const std::string closure_name,
        PHX::FieldManager<panzer::Traits>& fm,
        const panzer::PhysicsBlock& side_pb,
        const double& multiplier) const;

    void registerResidual(
        std::pair<const std::string, std::string> dof_eq_pair,
        std::unordered_map<std::string, std::vector<std::string>>& eq_vct_map,
        PHX::FieldManager<panzer::Traits>& fm,
        const panzer::PhysicsBlock& side_pb) const;

    void registerScatterOperator(
        std::pair<const std::string, std::string> dof_eq_pair,
        PHX::FieldManager<panzer::Traits>& fm,
        const panzer::PhysicsBlock& side_pb,
        const panzer::LinearObjFactory<panzer::Traits>& lof) const;

    auto integrationRule() const { return _ir; }

  protected:
    std::unordered_map<std::string, std::string> bnd_prefix;
    std::unordered_map<std::string, Teuchos::RCP<panzer::PureBasis>>
        _dof_basis_pair;

  private:
    Teuchos::RCP<panzer::IntegrationRule> _ir;
};

//---------------------------------------------------------------------------//

} // end namespace BoundaryCondition
} // end namespace VertexCFD

#endif // end VERTEXCFD_BOUNDARYCONDITION_BOUNDARYFLUXBASE_HPP
