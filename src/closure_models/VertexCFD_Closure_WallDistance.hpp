#ifndef VERTEXCFD_CLOSURE_WALLDISTANCE_HPP
#define VERTEXCFD_CLOSURE_WALLDISTANCE_HPP

#include "mesh/VertexCFD_Mesh_GeometryData.hpp"

#include <Panzer_Dimension.hpp>
#include <Panzer_Evaluator_WithBaseImpl.hpp>

#include <Phalanx_Evaluator_Derived.hpp>
#include <Phalanx_Evaluator_WithBaseImpl.hpp>
#include <Phalanx_FieldManager.hpp>
#include <Phalanx_config.hpp>

#include <Kokkos_Core.hpp>

namespace VertexCFD
{
namespace ClosureModel
{
//---------------------------------------------------------------------------//
// Wall Distance evaluator.
//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
class WallDistance : public panzer::EvaluatorWithBaseImpl<Traits>,
                     public PHX::EvaluatorDerived<EvalType, Traits>
{
  public:
    struct EvaluateTag
    {
    };
    struct RegistrationTag
    {
    };

    using scalar_type = typename EvalType::ScalarT;
    static constexpr int num_space_dim = NumSpaceDim;

    WallDistance(const panzer::IntegrationRule& ir,
                 Teuchos::RCP<Mesh::Topology::SidesetGeometry> sideset_geometry);

    void postRegistrationSetup(typename Traits::SetupData sd,
                               PHX::FieldManager<Traits>& fm) override;

    void evaluateFields(typename Traits::EvalData workset) override;

    KOKKOS_INLINE_FUNCTION
    void operator()(EvaluateTag, const int cell) const;

    KOKKOS_INLINE_FUNCTION
    void operator()(RegistrationTag, const int cell) const;

  public:
    // Panzer field for storing wall distance
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _distance;

  private:
    // Integration rule cubature degree needed for getting integration point
    // coordinates
    double _ir_degree;

    // Integration rule index needed for getting integration point coordinates
    double _ir_index;

    // number of worksets
    int _number_worksets;

    // current workset, needed for accessing _distance_vector in
    // PostRegistration Setup and EvaluateFields
    std::size_t _current_workset = 0;

    // Field of the coordinates of the integration points
    PHX::MDField<const double, panzer::Cell, panzer::Point, panzer::Dim> _ip_coords;

    unsigned _key;

    // View for storing the vector of global side data
    Kokkos::View<double***, PHX::mem_space> _sides;

    // View for storing the distance field across all worksets
    Kokkos::View<double***, PHX::mem_space> _distance_vector;

    // View for storing the surface normal of global sides
    Kokkos::View<double**, PHX::mem_space> _normals;

    // Vector which stores the workset_id for each workset
    std::vector<std::size_t> _workset_id;
};

//---------------------------------------------------------------------------//

} // end namespace ClosureModel
} // end namespace VertexCFD

#endif // end VERTEXCFD_CLOSURE_WALLDISTANCE_HPP
