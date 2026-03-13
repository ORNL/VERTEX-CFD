#ifndef VERTEXCFD_PLASMASPECIESCLOSUREMODELFACTORY_HPP
#define VERTEXCFD_PLASMASPECIESCLOSUREMODELFACTORY_HPP

/**
 * @file
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
//---------------------------------------------------------------------------//
/**
 * @class PlasmaSpeciesFactory
 * @brief Factory for creating plasma species closure model evaluators.
 *
 * This class builds a set of Phalanx evaluators that implement the
 * plasma species physics for a given evaluation type <tt>EvalType</tt> and
 * spatial dimension <tt>NumSpaceDim</tt>.  It conforms to the Panzer
 * <tt>ClosureModelFactory</tt> interface.
 *
 * @tparam EvalType    Evaluation type (e.g., Residual, Jacobian) used by
 *                     Phalanx.
 * @tparam NumSpaceDim Number of spatial dimensions (e.g., 2 or 3).
 */
template<class EvalType, int NumSpaceDim>
class PlasmaSpeciesFactory : public panzer::ClosureModelFactory<EvalType>
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

    /**
     * @brief Check if a default closure model is not listed in input file and
     * returns an error message if found.
     *
     * @param closure_type      Closure model name read from input file
     */
    void checkDefaultClosureModel(const std::string& closure_type) const;

    /**
     * @brief Return number of default closure models.
     *
     * @param num_species      Number of species
     */
    int numDefaultClosures(const int num_species) const
    {
        return _num_default_cm_per_species * num_species;
    }

  private:
    // Mesh dimension
    static constexpr int num_space_dim = NumSpaceDim;

    // Number of default closure models
    const int _num_default_cm_per_species = 3;

    /**
     * @brief List all optional closure models available for this physics.
     *
     * @return return a string containing all optional closure models.
     */
    std::string availableClosureModels() const { return "Dummy\n\n"; };
};

//---------------------------------------------------------------------------//

} // end namespace ClosureModel
} // end namespace VertexCFD

#endif // end VERTEXCFD_PLASMASPECIESCLOSUREMODELFACTORY_HPP
