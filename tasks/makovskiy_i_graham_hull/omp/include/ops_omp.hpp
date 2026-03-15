#pragma once

#include "makovskiy_i_graham_hull/common/include/common.hpp"

namespace makovskiy_i_graham_hull {

class ConvexHullGrahamOMP : public ppc::task::Task<InType, OutType> {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kOMP;
  }
  explicit ConvexHullGrahamOMP(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace makovskiy_i_graham_hull
