#include "rad_solver/species_properties/VertexCFD_ConstantSpeciesProperties.hpp"

#include <Teuchos_ParameterList.hpp>

#include <gtest/gtest.h>

using namespace VertexCFD::SpeciesProperties;

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//
namespace Test
{
class RADConstantSpeciesPropertiesTest : public ::testing::Test
{
  protected:
    std::unique_ptr<ConstantSpeciesProperties> csp;

    virtual void SetUp() override;
    virtual void SetUpBateman2Species();
    virtual void SetUpBateman10Species();
    virtual void SetUpAdvection2Species();
    virtual void SetUpDiffusion2Species();
};

// Set up all variables but density
void RADConstantSpeciesPropertiesTest::SetUp()
{
    Teuchos::ParameterList csp_model_params;
    Teuchos::ParameterList csp_reaction_params;
    const int num_species = 2;
    csp_model_params.set("Number of Species", num_species);
    csp_model_params.set("Build Reaction", false);
    csp = std::make_unique<ConstantSpeciesProperties>(csp_model_params,
                                                      csp_reaction_params);
}

// Initialize Bateman term with 2 species
void RADConstantSpeciesPropertiesTest::SetUpBateman2Species()
{
    const int num_species = 2;

    Teuchos::Array<double> species_decay(num_species * num_species);
    for (int i = 0; i < num_species * num_species; ++i)
        species_decay[i] = i;

    Teuchos::ParameterList csp_model_params;
    csp_model_params.set("Number of Species", num_species);
    csp_model_params.set("Build Reaction", true);

    Teuchos::ParameterList csp_reaction_params;
    csp_reaction_params.set("Species Decay", species_decay);

    csp = std::make_unique<ConstantSpeciesProperties>(csp_model_params,
                                                      csp_reaction_params);
}

// Initialize Bateman term with 10 species
void RADConstantSpeciesPropertiesTest::SetUpBateman10Species()
{
    const int num_species = 10;

    Teuchos::Array<double> species_decay(num_species * num_species);
    for (int i = 0; i < num_species * num_species; ++i)
        species_decay[i] = i;

    Teuchos::ParameterList csp_model_params;
    csp_model_params.set("Number of Species", num_species);
    csp_model_params.set("Build Reaction", true);

    Teuchos::ParameterList csp_reaction_params;
    csp_reaction_params.set("Species Decay", species_decay);

    csp = std::make_unique<ConstantSpeciesProperties>(csp_model_params,
                                                      csp_reaction_params);
}

// Check advection boolean with 2 species
void RADConstantSpeciesPropertiesTest::SetUpAdvection2Species()
{
    const int num_species = 2;

    Teuchos::ParameterList csp_model_params;
    csp_model_params.set("Number of Species", num_species);
    csp_model_params.set("Build Advection", true);

    Teuchos::ParameterList csp_reaction_params;

    csp = std::make_unique<ConstantSpeciesProperties>(csp_model_params,
                                                      csp_reaction_params);
}

// Check diffusion boolean with 2 species
void RADConstantSpeciesPropertiesTest::SetUpDiffusion2Species()
{
    const int num_species = 2;

    Teuchos::ParameterList csp_model_params;
    csp_model_params.set("Number of Species", num_species);
    csp_model_params.set("Build Diffusion", true);
    csp_model_params.set("Diffusion Coefficient", 0.5);

    Teuchos::ParameterList csp_reaction_params;

    csp = std::make_unique<ConstantSpeciesProperties>(csp_model_params,
                                                      csp_reaction_params);
}

// Without Reaction term
TEST_F(RADConstantSpeciesPropertiesTest, non_reaction)
{
    RADConstantSpeciesPropertiesTest::SetUp();
    const bool reaction_expect = false;
    const bool reaction = csp->buildReaction();
    EXPECT_EQ(reaction_expect, reaction);
}

// Reaction term - 2 species
TEST_F(RADConstantSpeciesPropertiesTest, reaction_2spe)
{
    const int num_species = 2;

    RADConstantSpeciesPropertiesTest::SetUpBateman2Species();
    const auto species_decay = csp->batemanMatrix();

    Teuchos::Array<double> species_decay_exp(num_species * num_species);
    for (int i = 0; i < num_species; ++i)
    {
        for (int j = 0; j < num_species; ++j)
        {
            species_decay_exp[num_species * i + j] = num_species * i + j;
            EXPECT_EQ(species_decay_exp[num_species * i + j],
                      species_decay(i, j));
        }
    }
}

// Reaction term - 10 species
TEST_F(RADConstantSpeciesPropertiesTest, reaction_10spe)
{
    const int num_species = 10;

    RADConstantSpeciesPropertiesTest::SetUpBateman10Species();
    const auto species_decay = csp->batemanMatrix();

    Teuchos::Array<double> species_decay_exp(num_species * num_species);
    for (int i = 0; i < num_species; ++i)
    {
        for (int j = 0; j < num_species; ++j)
        {
            species_decay_exp[num_species * i + j] = num_species * i + j;
            EXPECT_EQ(species_decay_exp[num_species * i + j],
                      species_decay(i, j));
        }
    }
}

// Advection only
TEST_F(RADConstantSpeciesPropertiesTest, advection_2spe)
{
    RADConstantSpeciesPropertiesTest::SetUpAdvection2Species();
    const bool advection_expect = true;
    const int num_species_expect = 2;
    const bool advection = csp->buildAdvection();
    const int num_species = csp->numSpecies();
    EXPECT_EQ(advection_expect, advection);
    EXPECT_EQ(num_species_expect, num_species);
}

// Diffusion only
TEST_F(RADConstantSpeciesPropertiesTest, diffusion_2spe)
{
    RADConstantSpeciesPropertiesTest::SetUpDiffusion2Species();
    const bool diffusion_expect = true;
    const int num_species_expect = 2;
    const double diff_coef_expect = 0.5;
    const bool diffusion = csp->buildDiffusion();
    const double diffusion_coef = csp->constantDiffusionCoefficient();
    const int num_species = csp->numSpecies();
    EXPECT_EQ(diffusion_expect, diffusion);
    EXPECT_EQ(diff_coef_expect, diffusion_coef);
    EXPECT_EQ(num_species_expect, num_species);
}

//---------------------------------------------------------------------------//

} // end namespace Test
