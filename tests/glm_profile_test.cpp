#include "strata/core/glm_profile.hpp"
#include <cassert>
#include <cmath>
#include <sstream>

int main() {
    std::istringstream sparse("# comment\n0 2:6 0:2\n1 0:nan\n2 0:1 0:2\n3 4:1\n4 1:3junk\n5 2:-1\n6 0:0\n7 1:inf\n");
    auto a = strata::glmprofile::read_shares(sparse, 8, 4, true);
    assert(a[0].size() == 4 && a[0][0] == 0.75 && a[0][1] == 0.25 && a[0][3] == 0);
    for (int i = 1; i < 8; ++i) assert(a[i].empty());
    std::istringstream dense("0 1 2 3 4\n1 1 2\n2 1 2 3 4 5\n3 1 2 -3 4\n");
    auto b = strata::glmprofile::read_shares(dense, 4, 4, false);
    assert(b[0].size() == 4 && std::abs(b[0][0] - .4) < 1e-12);
    for (int i = 1; i < 4; ++i) assert(b[i].empty());
}
