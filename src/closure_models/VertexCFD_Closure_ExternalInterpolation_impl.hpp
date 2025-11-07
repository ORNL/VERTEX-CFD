#ifndef VERTEXCFD_CLOSURE_EXTERNALINTERPOLATION_IMPL_HPP
#define VERTEXCFD_CLOSURE_EXTERNALINTERPOLATION_IMPL_HPP

#include <Panzer_HierarchicParallelism.hpp>
#include <Panzer_Workset_Utilities.hpp>

#include <exodusII.h>

#include <ArborX_InterpMovingLeastSquares.hpp>

namespace VertexCFD
{
namespace ClosureModel
{

namespace Impl
{
// Copy unique source coordinates to the device in a format
// compatible with ArborX. We can't use lambdas in constructors with Cuda so we
// have to explicitly define a functor.
template<typename SourceView, typename CoordsView, int NumSpaceDim>
struct ConvertExodusToArborXLayout
{
    KOKKOS_FUNCTION void operator()(int i) const
    {
        for (int dim = 0; dim < NumSpaceDim; ++dim)
            _source_points(i)[dim] = _coords(i, dim);
    }

    SourceView _source_points;
    CoordsView _coords;
};
} // namespace Impl

//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
ExternalInterpolation<EvalType, Traits, NumSpaceDim>::ExternalInterpolation(
    const panzer::IntegrationRule& ir,
    const Teuchos::ParameterList& closure_params)
    : _ir_degree(ir.cubature_degree)
    , _interpolated_values("ExternalInterpolation::interpolated_values", 0)
{
    const std::string output_field_name
        = closure_params.get<std::string>("output field name");
    _scalar_field = PHX::MDField<scalar_type, panzer::Cell, panzer::Point>(
        output_field_name, ir.dl_scalar);
    const std::string input_field_name
        = closure_params.get<std::string>("input field name");
    _file_name = closure_params.get<std::string>("file name");

    this->addEvaluatedField(_scalar_field);

    _name = std::string("External Interpolation ")
            + std::to_string(NumSpaceDim) + "D" + " for " + output_field_name;
    this->setName(_name);

    // We read in the input file to extract the source coordinates, deduplicate
    // them, and then allocate memory for the source nodal values. Reading in
    // the nodal values is left to evaluateFields since that depends on the
    // time step we are interested in.

    float version;
    int CPU_word_size = sizeof(double);
    int IO_word_size = 0; /* use what is stored in file */
    const int exoid = ex_open(
        _file_name.c_str(), EX_READ, &CPU_word_size, &IO_word_size, &version);

    const std::size_t num_nodes = ex_inquire_int(exoid, EX_INQ_NODES);

    const Kokkos::View<double* [NumSpaceDim], Kokkos::LayoutLeft, Kokkos::HostSpace>
        coords_host(Kokkos::view_alloc(Kokkos::WithoutInitializing,
                                       "ExternalInterpolation::coords"),
                    num_nodes);

    int error = ex_get_coord(
        exoid,
        Kokkos::subview(coords_host, Kokkos::ALL, 0).data(),
        Kokkos::subview(coords_host, Kokkos::ALL, 1).data(),
        NumSpaceDim == 2 ? nullptr
                         : Kokkos::subview(coords_host, Kokkos::ALL, 2).data());
    if (error != 0)
    {
        const std::string message = _name + ": Error reading coordinates!";
        Kokkos::abort(message.c_str());
    }

    int num_nodal_vars;
    error = ex_get_variable_param(exoid, EX_NODAL, &num_nodal_vars);
    if (error != 0)
    {
        const std::string message = _name + ": Error getting variable params!";
        Kokkos::abort(message.c_str());
    }

    const int name_len = ex_inquire_int(exoid, EX_INQ_MAX_READ_NAME_LENGTH);

    std::vector<char> var_names_buffer(num_nodal_vars * (name_len + 1));
    std::vector<char*> var_names(num_nodal_vars);
    for (int i = 0; i < num_nodal_vars; ++i)
        var_names[i] = var_names_buffer.data() + i * (name_len + 1);

    error = ex_get_variable_names(
        exoid, EX_NODAL, num_nodal_vars, var_names.data());
    if (error != 0)
    {
        const std::string message = _name + ": Error getting variable names!";
        Kokkos::abort(message.c_str());
    }

    const auto it
        = std::find(var_names.begin(), var_names.end(), input_field_name);
    if (it == var_names.end())
    {
        std::stringstream ss;
        ss << _name + ": Error finding field with name <" << input_field_name
           << ">. Available fields are: ";
        for (auto& file_field_name : var_names)
            ss << "- <" << file_field_name << ">\n";
        Kokkos::abort(ss.str().c_str());
    }
    _var_index = std::distance(var_names.begin(), it) + 1;

    _nodal_values_host = Kokkos::View<double*, Kokkos::HostSpace>(
        Kokkos::view_alloc(Kokkos::WithoutInitializing,
                           "ExternalInterpolation::nodal_values"),
        num_nodes);

    int num_time_steps = ex_inquire_int(exoid, EX_INQ_TIME);
    _time_values.resize(num_time_steps);
    error = ex_get_all_times(exoid, _time_values.data());
    if (error != 0)
    {
        const std::string message = _name
                                    + ": Error reading extracting time steps!";
        Kokkos::abort(message.c_str());
    }

    error = ex_close(exoid);
    if (error != 0)
    {
        const std::string message = _name + ": Error closing exodus file!";
        Kokkos::abort(message.c_str());
    }

    // Deduplicate source coordinates and store the mapping
    _reduced_to_global.resize(num_nodes);
    std::iota(
        _reduced_to_global.begin(), _reduced_to_global.end(), std::size_t(0));
    // lexicographic comparison
    std::sort(_reduced_to_global.begin(),
              _reduced_to_global.end(),
              [&](std::size_t i, std::size_t j) {
                  for (int dim = 0; dim < NumSpaceDim; ++dim)
                  {
                      if (coords_host(i, dim) < coords_host(j, dim))
                          return true;
                      if (coords_host(i, dim) > coords_host(j, dim))
                          return false;
                  }
                  return false;
              });
    _reduced_to_global.erase(
        std::unique(_reduced_to_global.begin(),
                    _reduced_to_global.end(),
                    [&](std::size_t i, std::size_t j) {
                        for (int dim = 0; dim < NumSpaceDim; ++dim)
                            if (coords_host(i, dim) != coords_host(j, dim))
                                return false;
                        return true;
                    }),
        _reduced_to_global.end());
    const std::size_t num_unique_nodes = _reduced_to_global.size();

    const Kokkos::View<double* [NumSpaceDim], Kokkos::LayoutLeft, Kokkos::HostSpace>
        coords_host_reduced(Kokkos::view_alloc(Kokkos::WithoutInitializing,
                                               "ExternalInterpolation::"
                                               "coords"),
                            num_unique_nodes);

    _nodal_values_host_reduced = Kokkos::View<double*, Kokkos::HostSpace>(
        Kokkos::view_alloc(Kokkos::WithoutInitializing,
                           "ExternalInterpolation::nodal_values"),
        num_unique_nodes);

    for (size_t i = 0; i < num_unique_nodes; ++i)
    {
        for (int dim = 0; dim < NumSpaceDim; ++dim)
            coords_host_reduced(i, dim)
                = coords_host(_reduced_to_global[i], dim);
    }

    // Copy unique source coordinates and values to the device in a format
    // compatible with ArborX. We can't use lambdas in constructors with Cuda
    // so we have to explicitly define a functor.
    _source_points = Kokkos::View<
        ArborX::ExperimentalHyperGeometry::Point<NumSpaceDim, double>*,
        MemorySpace>(
        Kokkos::view_alloc(Kokkos::WithoutInitializing, "source_points"),
        num_unique_nodes);
    const auto coords = Kokkos::create_mirror_view_and_copy(
        MemorySpace{}, coords_host_reduced);
    _nodal_values = Kokkos::create_mirror_view(MemorySpace{},
                                               _nodal_values_host_reduced);

    ExecutionSpace exec_space;
    Kokkos::parallel_for(
        Kokkos::RangePolicy<ExecutionSpace>(exec_space, 0, num_unique_nodes),
        Impl::ConvertExodusToArborXLayout<decltype(_source_points),
                                          decltype(coords),
                                          NumSpaceDim>{_source_points, coords});
    exec_space.fence();

    read_node_values(0);
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
void ExternalInterpolation<EvalType, Traits, NumSpaceDim>::read_node_values(
    size_t time_step)
{
    float version;
    int CPU_word_size = sizeof(double);
    int IO_word_size = 0; /* use what is stored in file */
    const int exoid = ex_open(
        _file_name.c_str(), EX_READ, &CPU_word_size, &IO_word_size, &version);

    int error = ex_get_var(exoid,
                           time_step + 1,
                           EX_NODAL,
                           _var_index,
                           /*obj_id*/ 1,
                           _nodal_values_host.size(),
                           _nodal_values_host.data());

    if (error != 0)
    {
        const std::string message = _name + ": Error reading nodal values!";
        Kokkos::abort(message.c_str());
    }
    error = ex_close(exoid);
    if (error != 0)
    {
        const std::string message = _name + ": Error closing exodus file!";
        Kokkos::abort(message.c_str());
    }

    for (size_t i = 0; i < _reduced_to_global.size(); ++i)
        _nodal_values_host_reduced(i)
            = _nodal_values_host(_reduced_to_global[i]);

    ExecutionSpace exec;
    Kokkos::deep_copy(exec, _nodal_values, _nodal_values_host_reduced);
    exec.fence();
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
void ExternalInterpolation<EvalType, Traits, NumSpaceDim>::postRegistrationSetup(
    typename Traits::SetupData sd, PHX::FieldManager<Traits>&)
{
    _ir_index = panzer::getIntegrationRuleIndex(_ir_degree, (*sd.worksets_)[0]);
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
void ExternalInterpolation<EvalType, Traits, NumSpaceDim>::evaluateFields(
    typename Traits::EvalData workset)
{
    // Find the time step index such that the corresponding time step is
    // greater than workset.time. If no such index exists, take the last one.
    size_t time_step_index = std::min<int>(
        (std::lower_bound(_time_values.begin(), _time_values.end(), workset.time)
         - _time_values.begin()),
        _time_values.size() - 1);

    // Choose the time step index that is closest to the solver time. This can
    // be the one selected previously or the one before that if the index isn't
    // zero.
    if ((time_step_index != 0)
        && (workset.time - _time_values[time_step_index - 1])
               < (_time_values[time_step_index] - workset.time))
        --time_step_index;

    if (time_step_index != _last_time_step_index)
        read_node_values(time_step_index);

    _last_time_step_index = time_step_index;

    const auto ip_coords = workset.int_rules[_ir_index]->ip_coordinates;
    const int n_points = ip_coords.extent(1);
    const int n_cells = workset.num_cells;

    ExecutionSpace exec_space;

    const Kokkos::View<
        ArborX::ExperimentalHyperGeometry::Point<NumSpaceDim, double>*,
        MemorySpace>
        target_points(Kokkos::view_alloc(Kokkos::WithoutInitializing,
                                         "ExternalInterpolation::targets"),
                      n_cells * n_points);
    Kokkos::parallel_for(
        Kokkos::MDRangePolicy<ExecutionSpace, Kokkos::Rank<2>>(
            exec_space, {0, 0}, {n_cells, n_points}),
        KOKKOS_LAMBDA(int cell, int point) {
            for (int dim = 0; dim < NumSpaceDim; ++dim)
                target_points(cell * n_points + point)[dim]
                    = ip_coords(cell, point, dim);
        });

    const ArborX::Interpolation::MovingLeastSquares<MemorySpace> mls(
        exec_space, _source_points, target_points);

    Kokkos::realloc(Kokkos::view_alloc(Kokkos::WithoutInitializing),
                    _interpolated_values,
                    target_points.size());

    mls.interpolate(exec_space, _nodal_values, _interpolated_values);
    exec_space.fence();

    const auto policy
        = panzer::HP::inst().teamPolicy<scalar_type, PHX::Device>(n_cells);
    Kokkos::parallel_for(this->getName(), policy, *this);
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
KOKKOS_INLINE_FUNCTION void
ExternalInterpolation<EvalType, Traits, NumSpaceDim>::operator()(
    const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const
{
    const int cell = team.league_rank();
    const int num_point = _scalar_field.extent(1);

    Kokkos::parallel_for(
        Kokkos::TeamThreadRange(team, 0, num_point), [&](const int point) {
            _scalar_field(cell, point)
                = _interpolated_values(cell * num_point + point);
        });
}

//---------------------------------------------------------------------------//
} // namespace ClosureModel
} // end namespace VertexCFD

#endif // end VERTEXCFD_CLOSURE_EXTERNALINTERPOLATION_IMPL_HPP
