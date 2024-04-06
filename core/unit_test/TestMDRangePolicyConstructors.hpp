//@HEADER
// ************************************************************************
//
//                        Kokkos v. 4.0
//       Copyright (2022) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
// Part of Kokkos, under the Apache License v2.0 with LLVM Exceptions.
// See https://kokkos.org/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//@HEADER

#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>

#include <regex>

namespace {

template <class IndexType>
void construct_mdrange_policy_variable_type() {
  (void)Kokkos::MDRangePolicy<TEST_EXECSPACE, Kokkos::Rank<2>>{
      Kokkos::Array<IndexType, 2>{}, Kokkos::Array<IndexType, 2>{}};

  (void)Kokkos::MDRangePolicy<TEST_EXECSPACE, Kokkos::Rank<2>>{
      {{IndexType(0), IndexType(0)}}, {{IndexType(2), IndexType(2)}}};

  (void)Kokkos::MDRangePolicy<TEST_EXECSPACE, Kokkos::Rank<2>>{
      {IndexType(0), IndexType(0)}, {IndexType(2), IndexType(2)}};
}

TEST(TEST_CATEGORY, md_range_policy_construction_from_arrays) {
  {
    // Check that construction from Kokkos::Array of the specified index type
    // works.
    using IndexType = unsigned long long;
    Kokkos::MDRangePolicy<TEST_EXECSPACE, Kokkos::Rank<2>,
                          Kokkos::IndexType<IndexType>>
        p1(Kokkos::Array<IndexType, 2>{{0, 1}},
           Kokkos::Array<IndexType, 2>{{2, 3}});
    Kokkos::MDRangePolicy<TEST_EXECSPACE, Kokkos::Rank<2>,
                          Kokkos::IndexType<IndexType>>
        p2(Kokkos::Array<IndexType, 2>{{0, 1}},
           Kokkos::Array<IndexType, 2>{{2, 3}});
    Kokkos::MDRangePolicy<TEST_EXECSPACE, Kokkos::Rank<2>,
                          Kokkos::IndexType<IndexType>>
        p3(Kokkos::Array<IndexType, 2>{{0, 1}},
           Kokkos::Array<IndexType, 2>{{2, 3}},
           Kokkos::Array<IndexType, 1>{{4}});
  }
  {
    // Check that construction from double-braced initializer list
    // works.
    using index_type = unsigned long long;
    Kokkos::MDRangePolicy<TEST_EXECSPACE, Kokkos::Rank<2>> p1({{0, 1}},
                                                              {{2, 3}});
    Kokkos::MDRangePolicy<TEST_EXECSPACE, Kokkos::Rank<2>,
                          Kokkos::IndexType<index_type>>
        p2({{0, 1}}, {{2, 3}});
  }
  {
    // Check that construction from Kokkos::Array of long compiles for backwards
    // compability.  This was broken in
    // https://github.com/kokkos/kokkos/pull/3527/commits/88ea8eec6567c84739d77bdd25fdbc647fae28bb#r512323639
    Kokkos::MDRangePolicy<TEST_EXECSPACE, Kokkos::Rank<2>> p1(
        Kokkos::Array<long, 2>{{0, 1}}, Kokkos::Array<long, 2>{{2, 3}});
    Kokkos::MDRangePolicy<TEST_EXECSPACE, Kokkos::Rank<2>> p2(
        Kokkos::Array<long, 2>{{0, 1}}, Kokkos::Array<long, 2>{{2, 3}});
    Kokkos::MDRangePolicy<TEST_EXECSPACE, Kokkos::Rank<2>> p3(
        Kokkos::Array<long, 2>{{0, 1}}, Kokkos::Array<long, 2>{{2, 3}},
        Kokkos::Array<long, 1>{{4}});
  }

  // Check that construction from various index types works.
  construct_mdrange_policy_variable_type<char>();
  construct_mdrange_policy_variable_type<int>();
  construct_mdrange_policy_variable_type<unsigned long>();
  construct_mdrange_policy_variable_type<std::int64_t>();
}

#ifndef KOKKOS_ENABLE_OPENMPTARGET  // FIXME_OPENMPTARGET
TEST(TEST_CATEGORY_DEATH, policy_bounds_unsafe_narrowing_conversions) {
  using Policy = Kokkos::MDRangePolicy<TEST_EXECSPACE, Kokkos::Rank<2>,
                                       Kokkos::IndexType<unsigned>>;

  ::testing::FLAGS_gtest_death_test_style = "threadsafe";
  ASSERT_DEATH(
      {
        (void)Policy({-1, 0}, {2, 3});
      },
      "unsafe narrowing conversion");
}

TEST(TEST_CATEGORY_DEATH, policy_invalid_bounds) {
  using Policy = Kokkos::MDRangePolicy<TEST_EXECSPACE, Kokkos::Rank<2>>;

  ::testing::FLAGS_gtest_death_test_style = "threadsafe";

  auto [dim0, dim1] = (Policy::inner_direction == Kokkos::Iterate::Right)
                          ? std::make_pair(1, 0)
                          : std::make_pair(0, 1);
  std::string msg1 =
      "Kokkos::MDRangePolicy bounds error: The lower bound (100) is greater "
      "than its upper bound (90) in dimension " +
      std::to_string(dim0) + ".\n";

  std::string msg2 =
      "Kokkos::MDRangePolicy bounds error: The lower bound (100) is greater "
      "than its upper bound (90) in dimension " +
      std::to_string(dim1) + ".\n";

#if !defined(KOKKOS_ENABLE_DEPRECATED_CODE_4)
  // escape the parentheses in the regex to match the error message
  msg1 = std::regex_replace(msg1, std::regex("\\(|\\)"), "\\$&");
  (void)msg2;
  ASSERT_DEATH({ (void)Policy({100, 100}, {90, 90}); }, msg1);
#else
  if (!Kokkos::show_warnings()) {
    GTEST_SKIP() << "Kokkos warning messages are disabled";
  }

  ::testing::internal::CaptureStderr();
  (void)Policy({100, 100}, {90, 90});
#ifdef KOKKOS_ENABLE_DEPRECATION_WARNINGS
  ASSERT_EQ(::testing::internal::GetCapturedStderr(), msg1 + msg2);
#else
  ASSERT_TRUE(::testing::internal::GetCapturedStderr().empty());
  (void)msg1;
  (void)msg2;
#endif

#endif
}
#endif

TEST(TEST_CATEGORY, policy_get_tile_size) {
  constexpr int rank = 3;
  using Policy    = Kokkos::MDRangePolicy<TEST_EXECSPACE, Kokkos::Rank<rank>>;
  using tile_type = typename Policy::tile_type;

  std::size_t last_rank =
      (Policy::inner_direction == Kokkos::Iterate::Right) ? rank - 1 : 0;

  auto default_size_properties =
      Kokkos::Impl::get_tile_size_properties(TEST_EXECSPACE());

  {
    int dim_length = 100;
    Policy policy({0, 0, 0}, {dim_length, dim_length, dim_length},
                  tile_type{{2, 4, 16}});

    auto rec_tile_sizes         = policy.tile_size_recommended();
    auto rec_largest_tile_sizes = policy.largest_tile_size_recommended();

    EXPECT_EQ(default_size_properties.max_total_tile_size,
              policy.max_total_tile_size());

    int prod_rec_tile_size = 1;
    for (std::size_t i = 0; i < rank; ++i) {
      EXPECT_GT(rec_tile_sizes[i], 0)
          << " invalid default tile size for rank " << i;
      EXPECT_LE(rec_tile_sizes[i], rec_largest_tile_sizes[i])
          << " invalid default tile size for rank " << i;

      if (default_size_properties.default_largest_tile_size == 0) {
        EXPECT_EQ(dim_length, rec_largest_tile_sizes[i])
            << " incorrect recommended largest tile size returned for rank "
            << i;

        auto expected_rec_tile_size =
            (i == last_rank) ? dim_length
                             : default_size_properties.default_tile_size;
        EXPECT_EQ(expected_rec_tile_size, rec_tile_sizes[i])
            << " incorrect recommended tile size returned for rank " << i;
      } else {
        EXPECT_EQ(default_size_properties.default_largest_tile_size,
                  rec_largest_tile_sizes[i])
            << " incorrect recommended largest tile size returned for rank "
            << i;

        auto expected_rec_tile_size =
            (i == last_rank) ? default_size_properties.default_largest_tile_size
                             : default_size_properties.default_tile_size;
        EXPECT_EQ(expected_rec_tile_size, rec_tile_sizes[i])
            << " incorrect recommended tile size returned for rank " << i;
      }

      prod_rec_tile_size *= rec_tile_sizes[i];
    }
    EXPECT_LT(prod_rec_tile_size, policy.max_total_tile_size());
  }
}

}  // namespace
