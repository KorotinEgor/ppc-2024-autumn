// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <random>
#include <vector>

#include "mpi/korotin_e_multidimentional_integrals_monte_carlo/include/ops_mpi.hpp"

namespace korotin_e_multidimentional_integrals_monte_carlo_mpi {

double test_func(double *x) { return x[0] * x[0] + x[1] * x[1] + x[2] * x[2]; }

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
    std::vector<double> ref(1, 0);
    std::vector<std::pair<double,double>> borders(3);

    for (int i = 0; i < 3; i++) {
      borders[i].first = left_border[i];
      borders[i].second = right_border[i];
    }

    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(borders.data()));
    taskDataSeq->inputs_count.emplace_back(borders.size());
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(N.data()));
    taskDataSeq->inputs_count.emplace_back(N.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(ref.data()));
    taskDataSeq->outputs_count.emplace_back(ref.size());

    korotin_e_multidimentional_integrals_monte_carlo_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    testMpiTaskSequential.set_func(korotin_e_multidimentional_integrals_monte_carlo_mpi::test_func);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();
    double seq_err = testMpiTaskSequential.possible_error();

    bool ans = (std::abs(res[0] - ref[0]) < err + seq_err);

    ASSERT_EQ(ans, true);
  }
}
