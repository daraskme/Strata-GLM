#pragma once
#include <algorithm>
#include <cmath>
#include <istream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace strata::glmprofile {
// Counts affect placement only, never routing. Reject malformed rows wholesale;
// omitted sparse experts are zero, and the caller handles uncovered layers.
inline std::vector<std::vector<double>> read_shares(std::istream& in, int layers, int experts, bool sparse) {
    std::vector<std::vector<double>> result(layers);
    std::string line;
    while (std::getline(in, line)) {
        std::istringstream row(line);
        int layer = -1;
        if (!(row >> layer) || layer < 0 || layer >= layers) continue;
        std::vector<double> counts(experts, 0.0);
        std::vector<bool> seen(experts, false);
        std::string token;
        int index = 0;
        bool valid = true;
        double total = 0;
        while (row >> token) {
            try {
                int expert = index++;
                std::string value = token;
                size_t consumed = 0;
                if (sparse) {
                    const auto colon = token.find(':');
                    if (colon == std::string::npos) { valid = false; break; }
                    expert = std::stoi(token.substr(0, colon), &consumed);
                    if (consumed != colon) { valid = false; break; }
                    value = token.substr(colon + 1);
                }
                const double count = std::stod(value, &consumed);
                if (consumed != value.size() || !std::isfinite(count) || count < 0 ||
                    expert < 0 || expert >= experts || seen[expert]) { valid = false; break; }
                seen[expert] = true;
                counts[expert] = count;
                total += count;
            } catch (...) { valid = false; break; }
        }
        if (!valid || !std::isfinite(total) || total <= 0 || (!sparse && index != experts)) continue;
        std::sort(counts.rbegin(), counts.rend());
        for (double& count : counts) count /= total;
        result[layer] = std::move(counts);
    }
    return result;
}
}
