#ifndef VERTEXCFD_EVALUATORTESTHARNESS_HPP
#define VERTEXCFD_EVALUATORTESTHARNESS_HPP

#include <Panzer_BasisIRLayout.hpp>
#include <Panzer_BasisValues2.hpp>
#include <Panzer_CellData.hpp>
#include <Panzer_CommonArrayFactories.hpp>
#include <Panzer_Dimension.hpp>
#include <Panzer_IntegrationRule.hpp>
#include <Panzer_IntegrationValues2.hpp>
#include <Panzer_Traits.hpp>
#include <Panzer_Workset.hpp>

#include <Phalanx_Evaluator.hpp>
#include <Phalanx_FieldManager.hpp>
#include <Phalanx_KokkosDeviceTypes.hpp>

#include <Shards_BasicTopologies.hpp>
#include <Shards_CellTopology.hpp>
#include <Shards_CellTopologyData.h>
#include <Shards_CellTopologyTraits.hpp>

#include <Teuchos_RCP.hpp>

#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>

#include <Trilinos_version.h>

#include <exception>
#include <sstream>
#include <string>
#include <vector>

namespace VertexCFD
{
namespace Test
{
//---------------------------------------------------------------------------//
// Evaluator cell test fixture.
struct EvaluatorTestFixture
{
    Teuchos::RCP<PHX::FieldManager<panzer::Traits>> fm;
    Teuchos::RCP<shards::CellTopology> cell_topo;
    Teuchos::RCP<panzer::Workset> workset;
    Teuchos::RCP<panzer::CellData> cell_data;
    Teuchos::RCP<panzer::IntegrationRule> ir;
    Teuchos::RCP<panzer::IntegrationValues2<double>> int_values;
    Teuchos::RCP<panzer::BasisIRLayout> basis_ir_layout;
    Teuchos::RCP<panzer::BasisValues2<double>> basis_values;

    using host_coords_view = Kokkos::View<double***,
                                          typename PHX::DevLayout<double>::type,
                                          PHX::Device>::HostMirror;

    // Create an evaluator test fixture object. If a side id is not specified
    // then integration rules will be setup over the volume of the cell
    // instead of the side.
    EvaluatorTestFixture(const CellTopologyData* cell_topo_data,
                         const host_coords_view host_coords,
                         const int integration_order,
                         const int basis_order,
                         const int side_id = -1)
    {
        initialize(cell_topo_data,
                   host_coords,
                   integration_order,
                   basis_order,
                   side_id);
    }

    // Use a default cell topology.
    EvaluatorTestFixture(const int num_space_dim,
                         const int integration_order,
                         const int basis_order,
                         const int side_id = -1)
    {
        // Single cell.
        const int num_cell = 1;

        const CellTopologyData* cell_topo_data = nullptr;
        host_coords_view host_coords;

        // Build a line, quad, or hex as the test cell.
        if (1 == num_space_dim)
        {
            cell_topo_data = shards::getCellTopologyData<shards::Line<2>>();

            host_coords = host_coords_view("coords", num_cell, 2, 1);

            host_coords(0, 0, 0) = 0.0;
            host_coords(0, 1, 0) = 1.0;
        }
        else if (2 == num_space_dim)
        {
            cell_topo_data
                = shards::getCellTopologyData<shards::Quadrilateral<4>>();

            host_coords = host_coords_view("coords", num_cell, 4, 2);

            host_coords(0, 0, 0) = 0.0;
            host_coords(0, 0, 1) = 0.0;

            host_coords(0, 1, 0) = 1.0;
            host_coords(0, 1, 1) = 0.0;

            host_coords(0, 2, 0) = 1.0;
            host_coords(0, 2, 1) = 1.0;

            host_coords(0, 3, 0) = 0.0;
            host_coords(0, 3, 1) = 1.0;
        }
        else if (3 == num_space_dim)
        {
            cell_topo_data
                = shards::getCellTopologyData<shards::Hexahedron<8>>();

            host_coords = host_coords_view("coords", num_cell, 8, 3);

            host_coords(0, 0, 0) = 0.0;
            host_coords(0, 0, 1) = 0.0;
            host_coords(0, 0, 2) = 0.0;

            host_coords(0, 1, 0) = 1.0;
            host_coords(0, 1, 1) = 0.0;
            host_coords(0, 1, 2) = 0.0;

            host_coords(0, 2, 0) = 1.0;
            host_coords(0, 2, 1) = 1.0;
            host_coords(0, 2, 2) = 0.0;

            host_coords(0, 3, 0) = 0.0;
            host_coords(0, 3, 1) = 1.0;
            host_coords(0, 3, 2) = 0.0;

            host_coords(0, 4, 0) = 0.0;
            host_coords(0, 4, 1) = 0.0;
            host_coords(0, 4, 2) = 1.0;

            host_coords(0, 5, 0) = 1.0;
            host_coords(0, 5, 1) = 0.0;
            host_coords(0, 5, 2) = 1.0;

            host_coords(0, 6, 0) = 1.0;
            host_coords(0, 6, 1) = 1.0;
            host_coords(0, 6, 2) = 1.0;

            host_coords(0, 7, 0) = 0.0;
            host_coords(0, 7, 1) = 1.0;
            host_coords(0, 7, 2) = 1.0;
        }
        else
        {
            std::ostringstream msg;
            msg << "Invalid spatial dimensions (" << num_space_dim
                << "): must be 1, 2, or 3.";
            throw std::logic_error(msg.str());
        }

        initialize(cell_topo_data,
                   host_coords,
                   integration_order,
                   basis_order,
                   side_id);
    }

    // Add an evaluator to the manager.
    template<class EvalType>
    void
    registerEvaluator(const Teuchos::RCP<PHX::Evaluator<panzer::Traits>>& eval)
    {
        fm->registerEvaluator<EvalType>(eval);
    }

    // Register fields that will be checked in the test.
    template<class EvalType, class Field>
    void registerTestField(const Field& field)
    {
        fm->requireField<EvalType>(field.fieldTag());
    }

    // Set time.
    void setTime(const double& time) { workset->time = time; }

    // Set time step size.
    void setStepSize(const double& step_size)
    {
        workset->step_size = step_size;
    }

    // Evaluate.
    template<class EvalType>
    void evaluate()
    {
        panzer::Traits::SD setup_data;
        auto worksets = Teuchos::rcp(new std::vector<panzer::Workset>);
        worksets->push_back(*workset);
        setup_data.worksets_ = worksets;
        std::vector<PHX::index_size_type> derivative_dimensions;
        derivative_dimensions.push_back(4);
        fm->setKokkosExtendedDataTypeDimensions<panzer::Traits::Jacobian>(
            derivative_dimensions);
        fm->postRegistrationSetup(setup_data);
        const panzer::Traits::PED ped;
        fm->preEvaluate<EvalType>(ped);
        fm->evaluateFields<EvalType>(*workset);
        fm->postEvaluate<EvalType>(0);
    }

    // Get a test field on the host to test after evaluation.
    template<class EvalType, class Field>
    auto getTestFieldData(const Field& field) const
    {
        auto field_view = field.get_static_view();
        auto field_mirror = Kokkos::create_mirror(field_view);
        Kokkos::deep_copy(field_mirror, field_view);
        return field_mirror;
    }

    // Get the number of quadrature points.
    int numPoint() const { return ir->num_points; }

    // Get the number of basis points.
    int cardinality() const { return basis_ir_layout->cardinality(); }

  private:
    void initialize(const CellTopologyData* cell_topo_data,
                    const host_coords_view host_coords,
                    const int integration_order,
                    const int basis_order,
                    const int side_id)
    {
        cell_topo = Teuchos::rcp(new shards::CellTopology(cell_topo_data));

        const int num_space_dim = cell_topo->getDimension();

        if (num_space_dim != host_coords.extent_int(2))
        {
            std::ostringstream msg;
            msg << "Unexpected spatial dimensions in provided coordinates: "
                << cell_topo->getName() << " expects " << num_space_dim
                << " dimensions, but " << host_coords.extent(2)
                << " were provided.";
            throw std::logic_error(msg.str());
        }

        const int nodes_per_cell = cell_topo->getNodeCount();

        if (nodes_per_cell != host_coords.extent_int(1))
        {
            std::ostringstream msg;
            msg << "Unexpected node count in provided coordinates: "
                << cell_topo->getName() << " expects " << nodes_per_cell
                << " nodes, but " << host_coords.extent(1)
                << " were provided.";
            throw std::logic_error(msg.str());
        }

        const int num_cell = host_coords.extent(0);

        // Create field manager.
        fm = Teuchos::rcp(new PHX::FieldManager<panzer::Traits>);

        // Create the workset.
        workset = Teuchos::rcp(new panzer::Workset);
        workset->num_cells = num_cell;

        // Fill cell local IDs
        {
            const Kokkos::View<int*, PHX::Device> cell_local_ids_k(
                "cell_local_ids_k", num_cell);
            auto cell_local_ids_k_host
                = Kokkos::create_mirror_view(cell_local_ids_k);

            workset->cell_local_ids.resize(num_cell);

            for (int i = 0; i < num_cell; ++i)
            {
                cell_local_ids_k_host(i) = i;
                workset->cell_local_ids[i] = i;
            }

            Kokkos::deep_copy(cell_local_ids_k, cell_local_ids_k_host);
            workset->cell_local_ids_k = cell_local_ids_k;
        }

        workset->block_id = "block_0";
        workset->ir_degrees = Teuchos::rcp(new std::vector<int>);
        workset->basis_names = Teuchos::rcp(new std::vector<std::string>);
        workset->time = 0.0;
        workset->step_size = 0.0;

        const panzer::MDFieldArrayFactory array_factory("", true);
        auto& node_coords = workset->cell_node_coordinates;
        node_coords
            = array_factory
                  .buildStaticArray<double, panzer::Cell, panzer::NODE, panzer::Dim>(
                      "coords", num_cell, nodes_per_cell, num_space_dim);

        Kokkos::deep_copy(node_coords.get_static_view(), host_coords);

        // Setup a cell.
        cell_data
            = Teuchos::rcp(new panzer::CellData(num_cell, side_id, cell_topo));

        // Create integration rule and populate integration values.
        ir = Teuchos::rcp(
            new panzer::IntegrationRule(integration_order, *cell_data));
        int_values
            = Teuchos::rcp(new panzer::IntegrationValues2<double>("", true));
        int_values->setupArrays(ir);
        int_values->evaluateValues(node_coords);
        workset->ir_degrees->push_back(ir->cubature_degree);
        workset->int_rules.push_back(int_values);

        // Create basis and populate basis values.
        basis_ir_layout = panzer::basisIRLayout("HGrad", basis_order, *ir);
        basis_values
            = Teuchos::rcp(new panzer::BasisValues2<double>("", true, true));
        basis_values->setupArrays(basis_ir_layout);
        basis_values->evaluateValues(int_values->cub_points,
                                     int_values->jac,
                                     int_values->jac_det,
                                     int_values->jac_inv,
                                     int_values->weighted_measure,
                                     int_values->node_coordinates);
        workset->bases.push_back(basis_values);
        workset->basis_names->push_back(basis_ir_layout->getBasis()->name());
    }
};

//---------------------------------------------------------------------------//
// For a residual evaluation, the field value is a double and returned as-is.
double accessValue(double v)
{
    return v;
}

//---------------------------------------------------------------------------//
// For a Jacobian evaluation, the field value is Sacado FAD type, so return its
// value.
template<class Value>
double accessValue(Value&& v)
{
    return accessValue(v.val());
}

//---------------------------------------------------------------------------//
// Access a field value at the given indices.
template<class Field, class... Indices>
double fieldValue(Field f, const Indices... i)
{
    return accessValue(f(i...));
}

//---------------------------------------------------------------------------//

} // end namespace Test
} // end namespace VertexCFD

#endif // end VERTEXCFD_EVALUATORTESTHARNESS_HPP
