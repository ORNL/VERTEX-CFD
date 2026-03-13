#include "utils/VertexCFD_Utils_ExplicitTemplateInstantiation.hpp"

#include "VertexCFD_Closure_PlasmaSpeciesConvectiveFlux.hpp"
#include "VertexCFD_Closure_PlasmaSpeciesConvectiveFlux_impl.hpp"

VERTEXCFD_INSTANTIATE_TEMPLATE_CLASS_EVAL_TRAITS_NUMSPACEDIM(
    VertexCFD::ClosureModel::PlasmaSpeciesConvectiveFlux)
