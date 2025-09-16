#ifndef VERTEXCFD_RESPONSEMANAGER_HPP
#define VERTEXCFD_RESPONSEMANAGER_HPP

#include "drivers/VertexCFD_PhysicsManager.hpp"

#include <Panzer_WorksetDescriptor.hpp>

#include <Thyra_VectorBase.hpp>

#include <Teuchos_Array.hpp>
#include <Teuchos_RCP.hpp>

#include <string>
#include <unordered_map>
#include <vector>

namespace VertexCFD
{
namespace Response
{
class ResponseManager
{
  public:
    explicit ResponseManager(Teuchos::RCP<PhysicsManager> physics_manager);

    void addFunctionalResponse(
        const std::string& name,
        const std::string& field_name,
        const std::vector<panzer::WorksetDescriptor>& workset_descriptors);
    void addFunctionalResponse(const std::string& name,
                               const std::string& field_name);
    void addMinValueResponse(const std::string& name,
                             const std::string& field_name);
    void addMinValueResponse(
        const std::string& name,
        const std::string& field_name,
        const std::vector<panzer::WorksetDescriptor>& workset_descriptors);
    void addMaxValueResponse(
        const std::string& name,
        const std::string& field_name,
        const std::vector<panzer::WorksetDescriptor>& workset_descriptors);
    void addMaxValueResponse(const std::string& name,
                             const std::string& field_name);
    void addProbeResponse(
        const std::string& name,
        const std::string& field_name,
        const Teuchos::Array<double>& point,
        const std::vector<panzer::WorksetDescriptor>& workset_descriptors);
    void addProbeResponse(const std::string& name,
                          const std::string& field_name,
                          const Teuchos::Array<double>& point);
    void activateResponse(const int index = 0);
    void activateResponse(const std::string& name);
    void deactivateAll();
    void
    evaluateResponses(const Teuchos::RCP<Thyra::VectorBase<double>>& x,
                      const Teuchos::RCP<Thyra::VectorBase<double>>& x_dot,
                      const double t,
                      const double dt);

    int numResponses() const;
    int globalIndex(const int index = 0) const;
    int globalIndex(const std::string& name) const;
    const std::string& name(const int index = 0) const;
    double value(const int index = 0) const;
    double value(const std::string& name) const;

  private:
    int _num_responses;
    Teuchos::RCP<PhysicsManager> _physics_manager;
    std::vector<panzer::WorksetDescriptor> _default_workset_descriptors;
    std::vector<int> _index_map;
    std::unordered_map<std::string, int> _name_map;
    std::vector<Teuchos::RCP<Thyra::VectorBase<double>>> _resp_vectors;
    std::vector<bool> _is_active;

    void addExtremeValueResponse(
        const bool use_max,
        const std::string& name,
        const std::string& field_name,
        const std::vector<panzer::WorksetDescriptor>& workset_descriptors);

    template<class Builder>
    void addResponseFromBuilder(
        const std::string& name,
        const std::vector<panzer::WorksetDescriptor>& workset_descriptors,
        const Builder& builder);
};

} // namespace Response
} // namespace VertexCFD

#endif // VERTEXCFD_RESPONSEMANAGER_HPP
