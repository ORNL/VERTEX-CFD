#include "utils/VertexCFD_Utils_ExplicitTemplateInstantiation.hpp"

#include "VertexCFD_SolidFullInductionClosureModelFactory.hpp"
#include "VertexCFD_SolidFullInductionClosureModelFactory_impl.hpp"

VERTEXCFD_INSTANTIATE_TEMPLATE_CLASS_EVAL_NUMSPACEDIM(
    VertexCFD::ClosureModel::SolidFullInductionFactory)
