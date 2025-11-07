#include "utils/VertexCFD_Utils_ExplicitTemplateInstantiation.hpp"

#include "VertexCFD_SolidElectricPotentialClosureModelFactory.hpp"
#include "VertexCFD_SolidElectricPotentialClosureModelFactory_impl.hpp"

VERTEXCFD_INSTANTIATE_TEMPLATE_CLASS_EVAL_NUMSPACEDIM(
    VertexCFD::ClosureModel::SolidElectricPotentialFactory)
