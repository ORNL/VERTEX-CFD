#ifndef VERTEXCFD_UTILS_PHYSICSMANAGER_HPP
#define VERTEXCFD_UTILS_PHYSICSMANAGER_HPP

#include <Teuchos_ParameterEntry.hpp>
#include <Teuchos_ParameterList.hpp>
#include <Teuchos_RCP.hpp>

#include <Panzer_PhysicsBlock.hpp>

#include <vector>

namespace VertexCFD
{
namespace Utils
{
//---------------------------------------------------------------------------//
void checkIntegrationOrder(const int _integration_order,
                           const Teuchos::ParameterList& physics_parameters)
{
    for (auto ite = physics_parameters.begin(); ite != physics_parameters.end();
         ++ite)
    {
        const Teuchos::ParameterEntry& entry = physics_parameters.entry(ite);

        const Teuchos::ParameterList& physics_block
            = Teuchos::getValue<Teuchos::ParameterList>(entry);

        for (auto ph = physics_block.begin(); ph != physics_block.end(); ++ph)
        {
            const Teuchos::ParameterEntry& eq_set_block
                = physics_block.entry(ph);

            const Teuchos::ParameterList& eq_set
                = Teuchos::getValue<Teuchos::ParameterList>(eq_set_block);

            if (eq_set.get<int>("Integration Order") != _integration_order)
                throw std::runtime_error(
                    "All of the physics blocks must have the same "
                    "integration "
                    "order.");
        }
    }
}

//---------------------------------------------------------------------------//

} // namespace Utils
} // end namespace VertexCFD

#endif // end VERTEXCFD_UTILS_PHYSICSMANAGER_HPP
