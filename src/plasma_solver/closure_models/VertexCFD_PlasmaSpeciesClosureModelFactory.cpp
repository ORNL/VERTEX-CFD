#include "utils/VertexCFD_Utils_ExplicitTemplateInstantiation.hpp"

#include "VertexCFD_PlasmaSpeciesClosureModelFactory.hpp"
#include "VertexCFD_PlasmaSpeciesClosureModelFactory_impl.hpp"

VERTEXCFD_INSTANTIATE_TEMPLATE_CLASS_EVAL_NUMSPACEDIM(
    VertexCFD::ClosureModel::PlasmaSpeciesFactory)
