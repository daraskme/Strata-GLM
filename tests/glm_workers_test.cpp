#include "strata/core/glm_workers.hpp"
#include <array>
#include <cassert>
#include <cstdio>

int main() {
    for (int threads : {1, 2, 8, 16}) {
        for (int spin : {0, 300, 20000}) {
            strata::core::glmfast::Workers pool(threads - 1, spin);
            for (int round = 0; round < 500; ++round) {
                const int n = std::array<int, 8>{0, 1, 17, 31, 32, 33, 127, 256}[round % 8];
                std::array<std::atomic<int>, 256> visits{};
                std::array<int, 256> result{};
                pool.run(n, [&](int i) {
                    ++visits[i];
                    if (i % 11 == 0) std::this_thread::yield();
                    result[i] = i * 17 + round;
                });
                for (int i = 0; i < n; ++i) {
                    assert(visits[i] == 1);
                    assert(result[i] == i * 17 + round);
                }
                if (round % 100 == 0) std::this_thread::sleep_for(std::chrono::milliseconds(2));
            }
        }
    }
    std::puts("Worker publication, exactly-once jobs, wake/sleep: passed");
}
