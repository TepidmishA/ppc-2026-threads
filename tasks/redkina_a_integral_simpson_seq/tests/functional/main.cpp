// redkina_a_integral_simpson_seq/tests/functional/main.cpp
#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <functional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "redkina_a_integral_simpson_seq/common/include/common.hpp"
#include "redkina_a_integral_simpson_seq/omp/include/ops_omp.hpp"
#include "redkina_a_integral_simpson_seq/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace redkina_a_integral_simpson_seq {

class RedkinaAIntegralSimpsonFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return "id_" + std::to_string(std::get<0>(test_param));
  }

 protected:
  void SetUp() override {
    auto params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    input_data_ = std::get<1>(params);
    expected_ = std::get<2>(params);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    const double eps = 1e-6;
    return std::fabs(output_data - expected_) < eps;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
  double expected_ = 0.0;
};

namespace {

InputData MakeInput(std::function<double(const std::vector<double> &)> func, std::vector<double> a,
                    std::vector<double> b, std::vector<int> n) {
  return InputData{.func = std::move(func), .a = std::move(a), .b = std::move(b), .n = std::move(n)};
}

// Минимальный набор тестов (3 штуки) для покрытия основных случаев:
// 1D, 2D, 3D; полиномы; разные пределы.
const std::array<TestType, 3> kTestCases = {
    // 1D: квадратичная функция x^2 на [0,1] (точное значение 1/3)
    std::make_tuple(3,
                    MakeInput([](const std::vector<double> &x) { return x[0] * x[0]; }, std::vector<double>{0.0},
                              std::vector<double>{1.0}, std::vector<int>{2}),
                    1.0 / 3.0),

    // 2D: произведение x*y на [0,1]^2 (точное значение 0.25)
    std::make_tuple(9,
                    MakeInput([](const std::vector<double> &x) { return x[0] * x[1]; }, std::vector<double>{0.0, 0.0},
                              std::vector<double>{1.0, 1.0}, std::vector<int>{2, 2}),
                    0.25),

    // 3D: сумма квадратов x^2 + y^2 + z^2 на [-1,1]^3 (точное значение 8.0)
    std::make_tuple(
        19,
        MakeInput([](const std::vector<double> &x) { return (x[0] * x[0]) + (x[1] * x[1]) + (x[2] * x[2]); },
                  std::vector<double>{-1.0, -1.0, -1.0}, std::vector<double>{1.0, 1.0, 1.0}, std::vector<int>{2, 2, 2}),
        8.0)};

const auto kTestTasksListSeq =
    ppc::util::AddFuncTask<RedkinaAIntegralSimpsonSEQ, InType>(kTestCases, PPC_SETTINGS_redkina_a_integral_simpson_seq);

const auto kTestTasksListOmp =
    ppc::util::AddFuncTask<RedkinaAIntegralSimpsonOMP, InType>(kTestCases, PPC_SETTINGS_redkina_a_integral_simpson_seq);

const auto kGtestValuesSeq = ppc::util::ExpandToValues(kTestTasksListSeq);
const auto kGtestValuesOmp = ppc::util::ExpandToValues(kTestTasksListOmp);

const auto kTestName = RedkinaAIntegralSimpsonFuncTests::PrintFuncTestName<RedkinaAIntegralSimpsonFuncTests>;

INSTANTIATE_TEST_SUITE_P(IntegralSimpsonTestsSeq, RedkinaAIntegralSimpsonFuncTests, kGtestValuesSeq, kTestName);
INSTANTIATE_TEST_SUITE_P(IntegralSimpsonTestsOmp, RedkinaAIntegralSimpsonFuncTests, kGtestValuesOmp, kTestName);

TEST_P(RedkinaAIntegralSimpsonFuncTests, CheckIntegral) {
  ExecuteTest(GetParam());
}

}  // namespace

}  // namespace redkina_a_integral_simpson_seq
