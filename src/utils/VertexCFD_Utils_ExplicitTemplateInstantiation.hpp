#ifndef VERTEXCFD_UTILS_EXPLICITTEMPLATEINSTANTIATION_HPP
#define VERTEXCFD_UTILS_EXPLICITTEMPLATEINSTANTIATION_HPP

#include <Panzer_ExplicitTemplateInstantiation.hpp>
#include <Panzer_Traits.hpp>

// Explicit instantation of standard <EvalType> panzer classes.
#define VERTEXCFD_INSTANTIATE_TEMPLATE_CLASS_EVAL(name) \
    PANZER_INSTANTIATE_TEMPLATE_CLASS_ONE_T(name)

// Explicit instantation of standard <EvalType,Traits> panzer classes.
#define VERTEXCFD_INSTANTIATE_TEMPLATE_CLASS_EVAL_TRAITS(name) \
    PANZER_INSTANTIATE_TEMPLATE_CLASS_TWO_T(name)

// Explicit instantation of <EvalType,NumSpaceDim> classes.
#define VERTEXCFD_INSTANTIATE_TEMPLATE_CLASS_RESIDUAL_NUMSPACEDIM(name) \
    template class name<panzer::Traits::Residual, 2>;                   \
    template class name<panzer::Traits::Residual, 3>;

#define VERTEXCFD_INSTANTIATE_TEMPLATE_CLASS_TANGENT_NUMSPACEDIM(name) \
    template class name<panzer::Traits::Tangent, 2>;                   \
    template class name<panzer::Traits::Tangent, 3>;

#define VERTEXCFD_INSTANTIATE_TEMPLATE_CLASS_JACOBIAN_NUMSPACEDIM(name) \
    template class name<panzer::Traits::Jacobian, 2>;                   \
    template class name<panzer::Traits::Jacobian, 3>;

#ifdef Panzer_BUILD_HESSIAN_SUPPORT
#define VERTEXCFD_INSTANTIATE_TEMPLATE_CLASS_HESSIAN_NUMSPACEDIM(name) \
    template class name<panzer::Traits::Hessian, 2>;                   \
    template class name<panzer::Traits::Hessian, 3>;
#else
#define VERTEXCFD_INSTANTIATE_TEMPLATE_CLASS_HESSIAN_NUMSPACEDIM(name)
#endif

#define VERTEXCFD_INSTANTIATE_TEMPLATE_CLASS_EVAL_NUMSPACEDIM(name) \
    VERTEXCFD_INSTANTIATE_TEMPLATE_CLASS_RESIDUAL_NUMSPACEDIM(name) \
    VERTEXCFD_INSTANTIATE_TEMPLATE_CLASS_TANGENT_NUMSPACEDIM(name)  \
    VERTEXCFD_INSTANTIATE_TEMPLATE_CLASS_JACOBIAN_NUMSPACEDIM(name) \
    VERTEXCFD_INSTANTIATE_TEMPLATE_CLASS_HESSIAN_NUMSPACEDIM(name)

// Explicit instantation of <EvalType,Traits,NumSpaceDim> classes.
#define VERTEXCFD_INSTANTIATE_TEMPLATE_CLASS_EVAL_TRAITS_NUMSPACEDIM(name) \
    PANZER_INSTANTIATE_TEMPLATE_CLASS_THREE_T(name, 2)                     \
    PANZER_INSTANTIATE_TEMPLATE_CLASS_THREE_T(name, 3)

// Explicit instantation of <EvalType,Traits,NumSpaceDim,ValueType> classes.
#define VERTEXCFD_INSTANTIATE_TEMPLATE_CLASS_EVAL_TRAITS_NUMSPACEDIM_VALUE( \
    name, value)                                                            \
    PANZER_INSTANTIATE_TEMPLATE_CLASS_FOUR_T(name, 2, value)                \
    PANZER_INSTANTIATE_TEMPLATE_CLASS_FOUR_T(name, 3, value)

#endif // end VERTEXCFD_UTILS_EXPLICITTEMPLATEINSTANTIATION_HPP
