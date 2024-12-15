// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <vector>

#include "seq/korotin_e_multidimentional_integrals_monte_carlo/include/ops_seq.hpp"

namespace korotin_e_multidimentional_integrals_monte_carlo_seq {

double test_func(double *x) { return x[0] * x[0] + x[1] * x[1] + x[2] * x[2]; }

double ref_integration(const std::vector<std::pair<double, double>> &borders) {
  double res = 0.0;
  for (size_t i = 0; i < borders.size(); i++) {
    double tmp = borders[i].second * borders[i].second * borders[i].second;
    tmp -= borders[i].first * borders[i].first * borders[i].first;
    tmp /= 3;
    for (size_t j = 0; j < borders.size(); j++) {
      if (j == i) continue;
      tmp *= borders[j].second - borders[j].first;
    }
    res += tmp;
  }
  return res;
}

}  // namespace korotin_e_multidimentional_integrals_monte_carlo_seq

TEST(korotin_e_multidimentional_integrals_monte_carlo_seq, test_monte_carlo) {
  std::vector<std::pair<double, double>> borders(3);
  std::vector<double> res(1, 0);
  std::vector<size_t> N(1, 500);
  double ref;

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<double> distrib(0.0, 3.0);
  for (int i = 0; i < 3; i++) {
    borders[i].first = distrib(gen);
    borders[i].second = distrib(gen);
  }

  ref = korotin_e_multidimentional_integrals_monte_carlo_seq::ref_integration(borders);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(borders.data()));
  taskDataSeq->inputs_count.emplace_back(borders.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(N.data()));
  taskDataSeq->inputs_count.emplace_back(N.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
  taskDataSeq->outputs_count.emplace_back(res.size());

  // Create Task
  korotin_e_multidimentional_integrals_monte_carlo_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  testTaskSequential.set_func(korotin_e_multidimentional_integrals_monte_carlo_seq::test_func);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  double err = testTaskSequential.possible_error();
  bool ans = (std::abs(res[0] - ref) < err);
  ASSERT_EQ(ans, true);
}
