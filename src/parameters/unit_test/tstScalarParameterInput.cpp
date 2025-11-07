#include "parameters/VertexCFD_ParameterDatabase.hpp"
#include "parameters/VertexCFD_ScalarParameterInput.hpp"

#include <Teuchos_ParameterEntryXMLConverterDB.hpp>
#include <Teuchos_ParameterList.hpp>
#include <Teuchos_XMLParameterListCoreHelpers.hpp>

#include <gtest/gtest.h>

#include <iostream>
#include <string>

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//
namespace VertexCFD
{
namespace Test
{
//---------------------------------------------------------------------------//
TEST(ScalarParameterInput, xml_write_read)
{
    // Inject custom type for parameter input.
    TEUCHOS_ADD_TYPE_CONVERTER(Parameter::ScalarParameterInput);

    // Make a parameter list with a scalar parameter input.
    Teuchos::ParameterList write_list;
    const Parameter::ScalarParameterInput write_input = {"Parameter Name"};
    write_list.set("Object Parameter", write_input);

    // Write a parameter list to XML. Write it to cout too so we can see it.
    Teuchos::writeParameterListToXmlFile(write_list,
                                         "scalar_parameter_input_test.xml");
    Teuchos::writeParameterListToXmlOStream(write_list, std::cout);

    // Read the parameter list back in. Also write it to cout so we can see
    // it.
    auto read_list
        = Teuchos::getParametersFromXmlFile("scalar_parameter_input_test.xml");
    std::cout << *read_list;

    // Check result.
    auto read_input
        = read_list->get<Parameter::ScalarParameterInput>("Object Parameter");
    EXPECT_EQ("Parameter Name", read_input.parameter_name);

    // Check equality operator.
    const Parameter::ScalarParameterInput eq_check = {"Parameter Name"};
    EXPECT_EQ(eq_check, write_input);
}

//---------------------------------------------------------------------------//

} // namespace Test
} // namespace VertexCFD
