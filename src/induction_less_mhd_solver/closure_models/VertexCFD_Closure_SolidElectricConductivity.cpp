#include "utils/VertexCFD_Utils_ExplicitTemplateInstantiation.hpp"

#include "VertexCFD_Closure_SolidElectricConductivity.hpp"
#include "VertexCFD_Closure_SolidElectricConductivity_impl.hpp"

VERTEXCFD_INSTANTIATE_TEMPLATE_CLASS_EVAL_TRAITS(
    VertexCFD::ClosureModel::SolidElectricConductivity)
