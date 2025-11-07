#ifndef VERTEXCFD_INITIALCONDITION_CONSTANT_IMPL_HPP
#define VERTEXCFD_INITIALCONDITION_CONSTANT_IMPL_HPP

#include <string>

namespace VertexCFD
{
namespace InitialCondition
{
//---------------------------------------------------------------------------//
template<class EvalType, class Traits>
Constant<EvalType, Traits>::Constant(const Teuchos::ParameterList& params,
                                     const panzer::PureBasis& basis)
{
    _value = params.get<double>("Value");
    const std::string dof_name = params.get<std::string>("Equation Set Name");
    _ic = PHX::MDField<scalar_type, panzer::Cell, panzer::BASIS>(
        dof_name, basis.functional);
    this->addEvaluatedField(_ic);
    this->addUnsharedField(_ic.fieldTag().clone());
    this->setName("Constant Initial Condition: " + dof_name);
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits>
void Constant<EvalType, Traits>::postRegistrationSetup(
    typename Traits::SetupData, PHX::FieldManager<Traits>& fm)
{
    this->utils.setFieldData(_ic, fm);
    _ic.deep_copy(_value);
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits>
void Constant<EvalType, Traits>::evaluateFields(typename Traits::EvalData)
{
}

//---------------------------------------------------------------------------//

} // end namespace InitialCondition
} // end namespace VertexCFD

#endif // end VERTEXCFD_INITIALCONDITION_CONSTANT_IMPL_HPP
