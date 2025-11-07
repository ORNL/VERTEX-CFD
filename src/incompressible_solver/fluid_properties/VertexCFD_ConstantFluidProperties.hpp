#ifndef VERTEXCFD_CONSTANTFLUIDPROPERTIES_HPP
#define VERTEXCFD_CONSTANTFLUIDPROPERTIES_HPP

#include <Teuchos_ParameterList.hpp>

#include <Kokkos_Core.hpp>

namespace VertexCFD
{
namespace FluidProperties
{
//---------------------------------------------------------------------------//
// Constant fluid properties
//---------------------------------------------------------------------------//

class ConstantFluidProperties
{
  public:
    ConstantFluidProperties() = default;
    explicit ConstantFluidProperties(const Teuchos::ParameterList& params)
        : _beta(params.get<double>("Artificial compressibility"))
        , _solve_temp(params.get<bool>("Build Temperature Equation"))
        , _build_ind_less_equ(false)
        , _build_buoyancy(false)
    {
        // Thermal parameters
        if (_solve_temp)
        {
            _gamma = params.isType<double>("Heat Capacity Ratio")
                         ? params.get<double>("Heat Capacity Ratio")
                         : 1.0;

            // Check for buoyancy bool
            if (params.isType<bool>("Build Buoyancy Source"))
            {
                _build_buoyancy = params.get<bool>("Build Buoyancy Source");
            }
        }
        else
        {
            _gamma = std::numeric_limits<double>::quiet_NaN();
        }

        // Buoyancy source term

        if (_build_buoyancy)
        {
            _beta_T = params.get<double>("Expansion coefficient");
            _T_ref = params.get<double>("Reference temperature");
        }
        else
        {
            _beta_T = std::numeric_limits<double>::quiet_NaN();
            _T_ref = std::numeric_limits<double>::quiet_NaN();
        }

        // Inductionless MHD equation
        if (params.isType<bool>("Build Inductionless MHD Equation"))
        {
            _build_ind_less_equ
                = params.get<bool>("Build Inductionless MHD Equation");
        }
    }

    // Constant heat capacity ratio
    KOKKOS_INLINE_FUNCTION double constantHeatCapacityRatio() const
    {
        return _gamma;
    }

    // Solve temperature equation
    KOKKOS_INLINE_FUNCTION bool solveTemperature() const
    {
        return _solve_temp;
    }

    // Include buoyancy effects
    KOKKOS_INLINE_FUNCTION bool buildBuoyancy() const
    {
        return _build_buoyancy;
    }

    // Build inductionless MHD equations
    KOKKOS_INLINE_FUNCTION bool buildInductionlessMHD() const
    {
        return _build_ind_less_equ;
    }

    // Expansion coefficient
    KOKKOS_INLINE_FUNCTION double expansionCoefficient() const
    {
        return _beta_T;
    }

    // Reference temperature
    KOKKOS_INLINE_FUNCTION double referenceTemperature() const
    {
        return _T_ref;
    }

    // Artificial compressibility
    KOKKOS_INLINE_FUNCTION double artificialCompressibility() const
    {
        return _beta;
    }

  private:
    double _gamma;
    double _beta;
    double _beta_T;
    double _T_ref;
    bool _solve_temp;
    bool _build_ind_less_equ;
    bool _build_buoyancy;
};

} // namespace FluidProperties
} // namespace VertexCFD

#endif // VERTEXCFD_CONSTANTFLUIDPROPERTIES_HPP
