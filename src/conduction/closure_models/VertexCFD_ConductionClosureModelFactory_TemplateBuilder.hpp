#ifndef VERTEXCFD_CONDUCTIONCLOSUREMODELFACTORY_TEMPLATEBUILDER_HPP
#define VERTEXCFD_CONDUCTIONCLOSUREMODELFACTORY_TEMPLATEBUILDER_HPP

/**
 * @file VertexCFD_ConductionClosureModelFactory_TemplateBuilder.hpp
 * @brief Builder for templated conduction closure model factories.
 *
 * Provides a helper class to instantiate a ConductionFactory for a given
 * evaluation type and spatial dimension. The builder returns a
 * Teuchos::RCP to a generic panzer::ClosureModelFactoryBase.
 */

/**
 * @defgroup VertexCFD_ConductionClosureModelFactoryTemplateBuilder Conduction
 * Closure Model Factory Builder
 * @brief Doxygen group for the conduction closure model factory template
 * builder.
 * @{
 */

#include "VertexCFD_ConductionClosureModelFactory.hpp"

#include <Panzer_ClosureModel_Factory_Base.hpp>

#include <Teuchos_RCP.hpp>

namespace VertexCFD
{
namespace ClosureModel
{

/**
 * @brief Templated builder for conduction closure model factories.
 *
 * @tparam NumSpaceDim Number of spatial dimensions (e.g., 2 or 3).
 *
 * The class provides a single member function template that creates a
 * ConductionFactory specialized for the given evaluation type @c EvalT and the
 * compile‑time spatial dimension.
 */
template<int NumSpaceDim>
class ConductionFactoryTemplateBuilder
{
  public:
    /**
     * @brief Build a closure model factory for a specific evaluation type.
     *
     * @tparam EvalT Evaluation type (e.g., Residual, Jacobian) used by the
     *               Panzer evaluation framework.
     * @return Teuchos::RCP to a panzer::ClosureModelFactoryBase containing the
     *         instantiated ConductionFactory.
     *
     * The method constructs a ConductionFactory<EvalT, NumSpaceDim> and
     * returns it as a base‑class pointer using Teuchos::rcp_static_cast.
     */
    template<typename EvalT>
    Teuchos::RCP<panzer::ClosureModelFactoryBase> build() const
    {
        auto conduction_closure_factory
            = Teuchos::rcp(new ConductionFactory<EvalT, NumSpaceDim>{});
        return Teuchos::rcp_static_cast<panzer::ClosureModelFactoryBase>(
            conduction_closure_factory);
    }
};

} // end namespace ClosureModel
} // end namespace VertexCFD

/** @} */ // end of VertexCFD_ConductionClosureModelFactoryTemplateBuilder

#endif // end VERTEXCFD_CONDUCTIONCLOSUREMODELFACTORY_TEMPLATEBUILDER_HPP