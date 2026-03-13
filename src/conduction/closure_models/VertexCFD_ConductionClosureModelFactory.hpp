#ifndef VERTEXCFD_CONDUCTIONCLOSUREMODELFACTORY_HPP
#define VERTEXCFD_CONDUCTIONCLOSUREMODELFACTORY_HPP

/**
 * @file VertexCFD_ConductionClosureModelFactory.hpp
 * @brief Factory for creating conduction closure model evaluators.
 *
 * Provides the ConductionFactory class template that builds Phalanx
 * evaluators for conduction physics in the VertexCFD code base.
 */

/**
 * @defgroup ConductionClosureModelFactory Conduction Closure Model Factory
 * @brief Doxygen module for the conduction closure model factory.
 * @{
 */

#include <Panzer_ClosureModel_Factory.hpp>
#include <Panzer_Traits.hpp>

#include <Phalanx_Evaluator.hpp>

#include <Teuchos_ParameterList.hpp>
#include <Teuchos_RCP.hpp>

namespace VertexCFD
{
namespace ClosureModel
{
//-----------------------------------------------------------//
/**
 * @class ConductionFactory
 * @brief Factory for creating conduction closure model evaluators.
 *
 * This class builds a set of Phalanx evaluators that implement the
 * conduction physics for a given evaluation type <tt>EvalType</tt> and
 * spatial dimension <tt>NumSpaceDim</tt>.  It conforms to the Panzer
 * <tt>ClosureModelFactory</tt> interface.
 *
 * @tparam EvalType    Evaluation type (e.g., Residual, Jacobian) used by
 *                     Phalanx.
 * @tparam NumSpaceDim Number of spatial dimensions (e.g., 2 or 3).
 */
template<class EvalType, int NumSpaceDim>
class ConductionFactory : public panzer::ClosureModelFactory<EvalType>
{
  public:
    /**
     * @brief Build the closure model evaluators for a specific model.
     *
     * The method creates a collection of evaluators based on the supplied
     * model identifier and parameter lists and returns them as a
     * reference‑counted vector.  The created evaluators are also registered
     * with the provided field manager.
     *
     * @param model_id          Identifier of the closure model to build.
     * @param model_params      Parameter list containing model‑specific
     * options.
     * @param fl                Field layout library providing field layout
     * information.
     * @param ir                Integration rule defining quadrature points and
     * weights.
     * @param default_params    Default parameters for the model (e.g.,
     * material properties).
     * @param user_params       User‑supplied parameters that may affect model
     * construction.
     * @param global_data       Global data object containing simulation‑wide
     * information.
     * @param fm                Field manager to which the created evaluators
     * are registered.
     *
     * @return Reference‑counted pointer to a vector of pointers to the created
     * evaluators.
     */
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

//-----------------------------------------------------------//

} // end namespace ClosureModel
} // end namespace VertexCFD

/** @} */ // end of ConductionClosureModelFactory

#endif // VERTEXCFD_CONDUCTIONCLOSUREMODELFACTORY_HPP