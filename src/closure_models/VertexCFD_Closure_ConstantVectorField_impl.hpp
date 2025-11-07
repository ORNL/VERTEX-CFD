#ifndef VERTEXCFD_CLOSURE_CONSTANTVECTORFIELD_IMPL_HPP
#define VERTEXCFD_CLOSURE_CONSTANTVECTORFIELD_IMPL_HPP

#include "utils/VertexCFD_Utils_VectorField.hpp"

#include <Panzer_HierarchicParallelism.hpp>

#include <string>

namespace VertexCFD
{
namespace ClosureModel
{
//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
ConstantVectorField<EvalType, Traits, NumSpaceDim>::ConstantVectorField(
    const panzer::IntegrationRule& ir,
    const Teuchos::ParameterList& closure_params)
    : _ir_degree(ir.cubature_degree)
    , _vec_field_name(closure_params.get<std::string>("Vector Field Name"))
{
    // Get vector field value
    const auto vector
        = closure_params.get<Teuchos::Array<double>>("Vector Field Value");
    for (int dim = 0; dim < num_space_dim; ++dim)
    {
        _vector[dim] = vector[dim];
    }

    // Evaluated fields
    Utils::addEvaluatedVectorField(
        *this, ir.dl_scalar, _vector_field, _vec_field_name + "_");

    this->setName("Constant Vector Field \"" + _vec_field_name + "_" + "\"");
}

template<class EvalType, class Traits, int NumSpaceDim>
void ConstantVectorField<EvalType, Traits, NumSpaceDim>::evaluateFields(
    typename Traits::EvalData /* workset */)
{
    for (int dim = 0; dim < num_space_dim; ++dim)
        Kokkos::deep_copy(_vector_field[dim].get_view(), _vector[dim]);
}

//---------------------------------------------------------------------------//

} // end namespace ClosureModel
} // end namespace VertexCFD

#endif // end VERTEXCFD_CLOSURE_CONSTANTVECTORFIELD_IMPL_HPP
