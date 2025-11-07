#include <VertexCFD_Utils_Version.hpp>

#include <iostream>

#include <gtest/gtest.h>

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//
namespace Test
{
TEST(vertexcfd_version, version_test)
{
    const auto version_id = VertexCFD::Utils::version();
    EXPECT_TRUE(!version_id.empty());
    std::cout << "VertexCFD version " << version_id << std::endl;

    const auto commit_hash = VertexCFD::Utils::git_commit_hash();
    EXPECT_TRUE(!commit_hash.empty());
    std::cout << "VertexCFD commit hash " << commit_hash << std::endl;
}

} // end namespace Test
