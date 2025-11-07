#include "VertexCFD_ParameterUnitTestConfig.hpp"

#include "parameters/VertexCFD_ParameterDatabase.hpp"
#include "parameters/VertexCFD_ScalarParameterEvaluator.hpp"
#include "parameters/VertexCFD_ScalarParameterInput.hpp"
#include "parameters/VertexCFD_ScalarParameterManager.hpp"
#include "parameters/VertexCFD_ScalarParameterObserver.hpp"

#include "drivers/VertexCFD_InitialConditionManager.hpp"
#include "drivers/VertexCFD_MeshManager.hpp"
#include "drivers/VertexCFD_PhysicsManager.hpp"

#include "utils/VertexCFD_EvaluatorBase.hpp"

#include <VertexCFD_EvaluatorTestHarness.hpp>

#include <Teuchos_ParameterEntryXMLConverterDB.hpp>
#include <Teuchos_ParameterList.hpp>

#include <gtest/gtest.h>

#include <string>

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//
namespace VertexCFD
{
namespace Test
{
//---------------------------------------------------------------------------//
template<class EvalType, class Traits>
class EvaluatorWithParameter
    : public EvaluatorBase<EvalType, Traits>,
      public Parameter::ScalarParameterObserver<EvalType>
{
  public:
    using scalar_type = typename EvalType::ScalarT;

    EvaluatorWithParameter(const panzer::IntegrationRule& ir,
                           const Teuchos::ParameterList& params)
        : _f1("f1", ir.dl_scalar)
        , _f2("f2", ir.dl_scalar)
        , _f3("f3", ir.dl_scalar)
        , _f4("f4", ir.dl_scalar)
        , _f5("f5", ir.dl_scalar)
    {
        this->addEvaluatedField(_f1);
        this->addEvaluatedField(_f2);
        this->addEvaluatedField(_f3);
        this->addEvaluatedField(_f4);
        this->addEvaluatedField(_f5);
        this->registerParameter("p1", 2.0, params, _p1);
        this->registerParameter("p2", 3.0, params, _p2);
        this->registerParameter("p3", 4.0, params, _p3);
        this->registerParameter("p4", 5.0, params, _p4);
        this->registerParameter("p5", 6.0, params, _p5);
        this->setName("EvaluatorWithParameter");
    }

    scalar_type _p1;
    scalar_type _p2;
    scalar_type _p3;
    scalar_type _p4;
    scalar_type _p5;
    scalar_type _cp1;
    scalar_type _cp2;
    scalar_type _cp3;
    scalar_type _cp4;
    scalar_type _cp5;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _f1;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _f2;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _f3;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _f4;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _f5;

  protected:
    void updateStateWithNewParameters() override
    {
        // Do a copy here to make sure this function gets called.
        _cp1 = _p1;
        _cp2 = _p2;
        _cp3 = _p3;
        _cp4 = _p4;
        _cp5 = _p5;
    }

    void evaluateFieldsImpl(typename Traits::EvalData) override
    {
        _f1.deep_copy(_cp1);
        _f2.deep_copy(_cp2);
        _f3.deep_copy(_cp3);
        _f4.deep_copy(_cp4);
        _f5.deep_copy(_cp5);
    }
};

//---------------------------------------------------------------------------//
template<class EvalType>
void testScalarParameterEvaluator()
{
    // Setup base data structures to make a parameter library.
    auto comm = Teuchos::rcp_dynamic_cast<const Teuchos::MpiComm<int>>(
        Teuchos::DefaultComm<int>::getComm());
    const std::string location = VERTEXCFD_PARAMETER_TEST_DATA_DIR;
    const std::string file = "scalar_parameter_evaluator_test.xml";
    auto parameter_db = Teuchos::rcp(
        new Parameter::ParameterDatabase(comm, location + file));
    parameter_db->physicsParameters()
        ->sublist("FluidPhysicsBlock", true)
        .sublist("Data", true)
        .set("Basis Order", 1);
    auto mesh_manager = Teuchos::rcp(new MeshManager(*parameter_db, comm));
    auto physics_manager = Teuchos::rcp(new PhysicsManager(
        std::integral_constant<int, 2>{}, parameter_db, mesh_manager));
    physics_manager->setupModel();

    // Setup test fixture.
    const int num_space_dim = 2;
    const int integration_order = 1;
    const int basis_order = 1;
    EvaluatorTestFixture test_fixture(
        num_space_dim, integration_order, basis_order);

    // Inject custom type for parameter input.
    TEUCHOS_ADD_TYPE_CONVERTER(Parameter::ScalarParameterInput);

    // Setup parameterized evaluator. The parameter list below in XML would
    // be:
    //
    // <ParameterList>
    //     <Parameter name="p2"  type=”ScalarParameter" value=”Parameter2"/>
    //     <Parameter name="p3"  type="double" value="9.2"/>
    //     <Parameter name="p4"  type="GeneralScalarParameter"
    //     value="GeneralParam1"/> <Parameter name="p5"
    //     type="GeneralScalarParameter" value="GeneralParam2"/>
    // </ParameterList>
    //
    // Note that p1 is not included to trigger the default evaluation.
    Teuchos::ParameterList plist;
    const Parameter::ScalarParameterInput input_2 = {"Parameter2"};
    plist.set("p2", input_2);
    plist.set("p3", 9.2);
    const Parameter::GeneralScalarParameterInput input_4 = {"GeneralParam1"};
    plist.set("p4", input_4);
    const Parameter::GeneralScalarParameterInput input_5 = {"GeneralParam2"};
    plist.set("p5", input_5);
    auto param_eval
        = Teuchos::rcp(new EvaluatorWithParameter<EvalType, panzer::Traits>(
            *test_fixture.ir, plist));
    test_fixture.registerEvaluator<EvalType>(param_eval);

    // Create a parameter observer manager and register the evaluator with it.
    auto param_manager = Teuchos::rcp(
        new Parameter::ScalarParameterManager<EvalType>(*parameter_db));
    param_manager->addObserver(param_eval);

    // Setup scalar parameter evaluator. This will trigger the parameter
    // update first in the graph.
    auto global_data = physics_manager->globalData();
    auto sp_eval = Teuchos::rcp(
        new Parameter::ScalarParameterEvaluator<EvalType, panzer::Traits>(
            param_manager, global_data));
    test_fixture.registerEvaluator<EvalType>(sp_eval);

    // Add required test fields.
    test_fixture.registerTestField<EvalType>(param_eval->_f1);
    test_fixture.registerTestField<EvalType>(param_eval->_f2);
    test_fixture.registerTestField<EvalType>(param_eval->_f3);
    test_fixture.registerTestField<EvalType>(param_eval->_f4);
    test_fixture.registerTestField<EvalType>(param_eval->_f5);

    // Set the test fixture block.
    test_fixture.workset->block_id = "eblock-0_0";

    // Evaluate test fields.
    test_fixture.evaluate<EvalType>();

    // Check the test fields during evaluate.
    //
    // Field 1 should have the default in the evaluator because it is not
    // defined in the evaluator list.
    //
    // Field 2 should have the nominal value from the global scalar parameter
    // list because it is set as a scalar parameter for the object.
    //
    // Field 3 should have the value from the evaluator parameter list because
    // it is set as a local parameter in the evaluator list.
    //
    // Field 4 should have the value of the first general parameter in the
    // given element block.
    //
    // Field 5 should have the default value of the second general
    // parameter in the given element block because no block-specific values
    // are given for the second parameter in the input.
    auto f1_result = test_fixture.getTestFieldData<EvalType>(param_eval->_f1);
    auto f2_result = test_fixture.getTestFieldData<EvalType>(param_eval->_f2);
    auto f3_result = test_fixture.getTestFieldData<EvalType>(param_eval->_f3);
    auto f4_result = test_fixture.getTestFieldData<EvalType>(param_eval->_f4);
    auto f5_result = test_fixture.getTestFieldData<EvalType>(param_eval->_f5);
    EXPECT_DOUBLE_EQ(2.0, fieldValue(f1_result, 0, 0));
    EXPECT_DOUBLE_EQ(5.5, fieldValue(f2_result, 0, 0));
    EXPECT_DOUBLE_EQ(9.2, fieldValue(f3_result, 0, 0));
    EXPECT_DOUBLE_EQ(2.34, fieldValue(f4_result, 0, 0));
    EXPECT_DOUBLE_EQ(3.45, fieldValue(f5_result, 0, 0));
}

//---------------------------------------------------------------------------//
TEST(ScalarParameter, residual)
{
    testScalarParameterEvaluator<panzer::Traits::Residual>();
}

//---------------------------------------------------------------------------//
TEST(ScalarParameter, jacobian)
{
    testScalarParameterEvaluator<panzer::Traits::Jacobian>();
}

//---------------------------------------------------------------------------//

} // namespace Test
} // namespace VertexCFD
