// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <random>
#include <vector>

#include "mpi/korotin_e_multidimentional_integrals_monte_carlo/include/ops_mpi.hpp"

namespace korotin_e_multidimentional_integrals_monte_carlo_mpi {

double test_func(double *x) { return x[0] * x[0] + x[1] * x[1] + x[2] * x[2]; }

double ref_integration(const std::vector<double> &left_border, const std::vector<double> &right_border) {
  double res = 0.0;
  for (size_t i = 0; i < left_border.size(); i++) {
    double tmp = right_border[i] * right_border[i] * right_border[i];
    tmp -= left_border[i] * left_border[i] * left_border[i];
    tmp /= 3;
    for (size_t j = 0; j < left_border.size(); j++) {
      if (j == i) continue;
      tmp *= right_border[j] - left_border[j];
    }
    res += tmp;
  }
  return res;
}

}  // namespace korotin_e_multidimentional_integrals_monte_carlo_mpi

TEST(korotin_e_multidimentional_integrals_monte_carlo, test_monte_carlo) {
  boost::mpi::communicator world;
  std::vector<double> left_border(3);
  std::vector<double> right_border(3);
  std::vector<double> res(1, 0);
  std::vector<size_t> N(1, 500);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> distrib(-5, 5);
    for (int i = 0; i < 3; i++) {
      left_border[i] = distrib(gen);
      right_border[i] = distrib(gen);
    }
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(left_border.data()));
    taskDataPar->inputs_count.emplace_back(left_border.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(right_border.data()));
    taskDataPar->inputs_count.emplace_back(right_border.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(N.data()));
    taskDataPar->inputs_count.emplace_back(N.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
    taskDataPar->outputs_count.emplace_back(res.size());
  }

  korotin_e_multidimentional_integrals_monte_carlo_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  testMpiTaskParallel.set_func(korotin_e_multidimentional_integrals_monte_carlo_mpi::test_func);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  double err = testMpiTaskParallel.possible_error();

  if (world.rank() == 0) {
    double ref = korotin_e_multidimentional_integrals_monte_carlo_mpi::ref_integration(left_border, right_border);
    bool ans = (std::abs(res[0] - ref) < err);

    ASSERT_EQ(ans, true);
  }
}
