#ifndef VERTEXCFD_GENERALSCALARPARAMETER_IMPL_HPP
#define VERTEXCFD_GENERALSCALARPARAMETER_IMPL_HPP

namespace VertexCFD
{
namespace Parameter
{
//---------------------------------------------------------------------------//
template<class EvalType>
GeneralScalarParameter<EvalType>::GeneralScalarParameter(
    const std::string& name, scalar_type& ref_to_parameter)
    : _name(name)
    , _ref_to_parameter(ref_to_parameter)
{
}

//---------------------------------------------------------------------------//
template<class EvalType>
const std::string& GeneralScalarParameter<EvalType>::name() const
{
    return _name;
}

//---------------------------------------------------------------------------//
template<class EvalType>
void GeneralScalarParameter<EvalType>::update(
    const panzer::Workset& workset,
    const std::unordered_map<std::string, std::unordered_map<std::string, double>>&
        general_scalar_params)
{
    // Lookup parameter values.
    auto param_values = general_scalar_params.find(_name);
    if (param_values == general_scalar_params.end())
    {
        const std::string msg = "GeneralScalar parameter " + _name
                                + " not found";
        throw std::runtime_error(msg);
    }

    // Check to see if this is an element block we have a value for.
    auto block_name = workset.getElementBlock();
    auto block_value = param_values->second.find(block_name);

    // If the block is in the parameter list assign the value.
    if (block_value != param_values->second.end())
    {
        _ref_to_parameter = block_value->second;
    }

    // Othwerwise just assign the default as this block wasn't given a
    // specific value.
    else
    {
        auto default_value = param_values->second.find("Default Value");
        if (default_value != param_values->second.end())
        {
            _ref_to_parameter = default_value->second;
        }
        else
        {
            const std::string msg = "GeneralScalar parameter " + _name
                                    + " does not have a value for block "
                                    + block_name
                                    + " and is also missing a default value";
            throw std::runtime_error(msg);
        }
    }
}

//---------------------------------------------------------------------------//

} // end namespace Parameter
} // end namespace VertexCFD

#endif // end VERTEXCFD_GENERALSCALARPARAMETER_IMPL_HPP
