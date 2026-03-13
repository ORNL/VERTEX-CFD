#ifndef VERTEXCFD_CLOSURE_EXTERNALMAGNETICFIELD_IMPL_HPP
#define VERTEXCFD_CLOSURE_EXTERNALMAGNETICFIELD_IMPL_HPP

#include "utils/VertexCFD_Utils_VectorField.hpp"

#include <Panzer_HierarchicParallelism.hpp>
#include <Panzer_Workset_Utilities.hpp>
#include <Teuchos_StandardParameterEntryValidators.hpp>

#include <string>

namespace VertexCFD
{
namespace ClosureModel
{
//---------------------------------------------------------------------------//
template<class EvalType, class Traits>
ExternalMagneticField<EvalType, Traits>::ExternalMagneticField(
    const panzer::IntegrationRule& ir,
    const Teuchos::ParameterList& ext_mag_field_param_list)
    : _ir_degree(ir.cubature_degree)
    , _ext_magn_type(ExtMagnType::constant)
{
    // Get external magnetic vector type
    if (ext_mag_field_param_list.isType<std::string>("External Magnetic Field "
                                                     "Type"))
    {
        const auto type_validator = Teuchos::rcp(
            new Teuchos::StringToIntegralParameterEntryValidator<ExtMagnType>(
                Teuchos::tuple<std::string>("constant", "toroidal", "x_tanh"),
                "constant"));
        _ext_magn_type = type_validator->getIntegralValue(
            ext_mag_field_param_list.get<std::string>("External Magnetic "
                                                      "Field Type"));
    }

    // Get external magnetic vector value
    if (_ext_magn_type == ExtMagnType::constant)
    {
        const auto ext_magn_vct
            = ext_mag_field_param_list.get<Teuchos::Array<double>>(
                "External Magnetic Field Value");

        const auto zero_array = Teuchos::Array<double>(field_size, 0.0);
        const auto d_ext_magn_vct_dt
            = ext_mag_field_param_list.isType<Teuchos::Array<double>>(
                  "External Magnetic Field Time Rate of "
                  "Change")
                  ? ext_mag_field_param_list.get<Teuchos::Array<double>>(
                        "External Magnetic Field Time Rate "
                        "of Change")
                  : zero_array;

        for (int dim = 0; dim < field_size; ++dim)
        {
            _ext_magn_vct[dim] = ext_magn_vct[dim];
            _d_ext_magn_vct_dt[dim] = d_ext_magn_vct_dt[dim];
        }
    }
    else if (_ext_magn_type == ExtMagnType::toroidal)
    {
        _toroidal_field_magn
            = ext_mag_field_param_list.get<double>("Toroidal Field Magnitude");
    }
    else if (_ext_magn_type == ExtMagnType::x_tanh)
    {
        _offset = ext_mag_field_param_list.get<double>(
            "External Magnetic Field Distribution Offset");
        _curve_coef = ext_mag_field_param_list.get<double>(
            "External Magnetic Field Distribution Curvature Coefficient");

        const auto ext_magn_vct
            = ext_mag_field_param_list.get<Teuchos::Array<double>>(
                "External Magnetic Field Value");

        for (int dim = 0; dim < field_size; ++dim)
            _ext_magn_vct[dim] = ext_magn_vct[dim];
    }

    // Evaluated fields
    Utils::addEvaluatedVectorField(
        *this, ir.dl_scalar, _ext_magn_field, "external_magnetic_field_");

    this->setName("External Magnetic Field");
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits>
void ExternalMagneticField<EvalType, Traits>::postRegistrationSetup(
    typename Traits::SetupData sd, PHX::FieldManager<Traits>&)
{
    _ir_index = panzer::getIntegrationRuleIndex(_ir_degree, (*sd.worksets_)[0]);
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits>
void ExternalMagneticField<EvalType, Traits>::evaluateFields(
    typename Traits::EvalData workset)
{
    _time = workset.time;

    _ip_coords = workset.int_rules[_ir_index]->ip_coordinates;
    auto policy = panzer::HP::inst().teamPolicy<scalar_type, PHX::Device>(
        workset.num_cells);
    Kokkos::parallel_for(this->getName(), policy, *this);
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits>
KOKKOS_INLINE_FUNCTION void ExternalMagneticField<EvalType, Traits>::operator()(
    const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const
{
    const int cell = team.league_rank();
    const int num_point = _ext_magn_field[0].extent(1);
    using Kokkos::sqrt;
    using Kokkos::tanh;

    Kokkos::parallel_for(
        Kokkos::TeamThreadRange(team, 0, num_point), [&](const int point) {
            if (_ext_magn_type == ExtMagnType::constant)
            {
                for (int dim = 0; dim < field_size; ++dim)
                {
                    _ext_magn_field[dim](cell, point)
                        = _ext_magn_vct[dim] + _d_ext_magn_vct_dt[dim] * _time;
                }
            }
            else if (_ext_magn_type == ExtMagnType::toroidal)
            {
                _ext_magn_field[0](cell, point)
                    = -_ip_coords(cell, point, 1) * _toroidal_field_magn
                      / (_ip_coords(cell, point, 0) * _ip_coords(cell, point, 0)
                         + _ip_coords(cell, point, 1)
                               * _ip_coords(cell, point, 1));

                _ext_magn_field[1](cell, point)
                    = _ip_coords(cell, point, 0) * _toroidal_field_magn
                      / (_ip_coords(cell, point, 0) * _ip_coords(cell, point, 0)
                         + _ip_coords(cell, point, 1)
                               * _ip_coords(cell, point, 1));

                _ext_magn_field[2](cell, point) = 0.0;
            }
            else if (_ext_magn_type == ExtMagnType::x_tanh)
            {
                for (int dim = 0; dim < field_size; ++dim)
                    _ext_magn_field[dim](cell, point)
                        = _ext_magn_vct[dim]
                          * (1
                             - tanh(_offset
                                    - _ip_coords(cell, point, 0) / _curve_coef))
                          / 2;
            }
        });
}

//---------------------------------------------------------------------------//

} // end namespace ClosureModel
} // end namespace VertexCFD

#endif // end VERTEXCFD_CLOSURE_EXTERNALMAGNETICFIELD_IMPL_HPP
