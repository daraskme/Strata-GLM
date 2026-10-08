// Numeric regression for mixed UD expert formats. Reference: ggml dequantized
// weights, CPU double accumulation; no second invocation of the GPU dot kernel.
#include "strata/kernels/glm_fast.hpp"
#include "ggml.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

namespace gf = strata::kernels::glmf;
namespace {
void require(bool ok, const char* why) {
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", why); std::exit(1); }
}
void ck(cudaError_t rc) { require(rc == cudaSuccess, cudaGetErrorString(rc)); }
struct Device {
    void* ptr = nullptr;
    explicit Device(size_t bytes) { ck(cudaMalloc(&ptr, bytes)); }
    ~Device() { cudaFree(ptr); }
    Device(const Device&) = delete;
    Device& operator=(const Device&) = delete;
    template<class T> T* as() { return static_cast<T*>(ptr); }
    void put(const void* src, size_t n) { ck(cudaMemcpy(ptr, src, n, cudaMemcpyHostToDevice)); }
    void get(void* dst, size_t n) { ck(cudaMemcpy(dst, ptr, n, cudaMemcpyDeviceToHost)); }
};
struct Q8 { ggml_fp16_t d, s; int8_t q[32]; };
static_assert(sizeof(Q8) == 36);
std::vector<float> unpack(const std::vector<Q8>& q) {
    std::vector<float> x(q.size() * 32);
    for (size_t b = 0; b < q.size(); ++b)
        for (int j = 0; j < 32; ++j) x[b*32+j] = ggml_fp16_to_fp32(q[b].d) * q[b].q[j];
    return x;
}
double rel(const std::vector<float>& actual, const std::vector<double>& expected) {
    double error = 0, norm = 0;
    for (size_t i = 0; i < actual.size(); ++i) {
        require(std::isfinite(actual[i]), "non-finite output");
        error += std::pow(actual[i] - expected[i], 2);
        norm += expected[i] * expected[i];
    }
    return std::sqrt(error / std::max(norm, 1e-30));
}
void run(int gu, int dn, int k) {
    constexpr int E = 512, F = 256;
    constexpr float limit = 10.f;
    const size_t gr = ggml_row_size((ggml_type) gu, E);
    const size_t dr = ggml_row_size((ggml_type) dn, F);
    const size_t down = 2 * F * gr, blob = down + E * dr;
    require(gf::moe_supported(gu) && gf::moe_supported(dn), "supported expert formats");
    require(gf::row_bytes(gu, E) == gr && gf::row_bytes(dn, F) == dr, "row sizes match ggml");
    std::mt19937 rng(4731 + gu * 19 + dn + k);
    std::normal_distribution<float> nd(0.f, 0.25f);
    std::vector<uint8_t> weights(blob * k);
    std::vector<float> raw(E * F), importance(E, 1.f);
    for (int e = 0; e < k; ++e) {
        for (int mat = 0; mat < 3; ++mat) {
            for (auto& v : raw) v = nd(rng);
            const int t = mat == 2 ? dn : gu;
            const int rows = mat == 2 ? E : F, cols = mat == 2 ? F : E;
            const size_t off = e * blob + (mat == 2 ? down : mat * F * gr);
            const size_t written = ggml_quantize_chunk((ggml_type)t, raw.data(), weights.data()+off,
                                                        0, rows, cols, importance.data());
            require(written == (mat == 2 ? E * dr : F * gr), "quantized matrix length");
        }
    }
    Device w(weights.size()), x(E*4), xq(E/32*sizeof(Q8)), hq(k*F/32*sizeof(Q8)), out(E*4);
    Device ptrs(8*sizeof(unsigned long long)), routes(8*4), flag(4);
    w.put(weights.data(), weights.size());
    std::vector<float> xf(E), route(8, 0);
    for (int i = 0; i < E; ++i) xf[i] = 2.f * std::sin(i * .17f) + .2f * std::cos(i * .71f);
    std::vector<unsigned long long> pp(8, 0);
    for (int e = 0; e < k; ++e) {
        pp[e] = (unsigned long long)(w.as<uint8_t>() + e * blob);
        route[e] = float(e+1) / (k*(k+1)/2);
    }
    const int zero = 0;
    ptrs.put(pp.data(), 8*sizeof(unsigned long long)); routes.put(route.data(), 8*4); flag.put(&zero, 4);
    x.put(xf.data(), E*4);
    gf::MoeDev d; d.plan_ptr = ptrs.as<unsigned long long>(); d.plan_w = routes.as<float>(); d.cpu_flag = flag.as<int>();
    gf::quantize_q8_1(x.as<float>(), xq.ptr, E, nullptr);
    gf::moe_gate_up(gu, d, k, E, F, limit, xq.ptr, hq.ptr, nullptr, 0, nullptr, 0, nullptr, nullptr);
    gf::moe_down(dn, d, k, E, F, down, hq.ptr, nullptr, out.as<float>(), nullptr);
    ck(cudaDeviceSynchronize());
    require(gf::launch_errors() == 0, "no kernel launch failures");
    std::vector<Q8> qx(E/32), qh(k*F/32);
    xq.get(qx.data(), qx.size()*sizeof(Q8)); hq.get(qh.data(), qh.size()*sizeof(Q8));
    const auto xx = unpack(qx), hh = unpack(qh);
    std::vector<double> href(k*F), yref(E, 0);
    std::vector<float> row(E), actual(E);
    const auto* gt = ggml_get_type_traits((ggml_type)gu);
    const auto* dt = ggml_get_type_traits((ggml_type)dn);
    for (int e = 0; e < k; ++e) {
        const uint8_t* b = weights.data() + e*blob;
        for (int r = 0; r < F; ++r) {
            double gate = 0, up = 0;
            gt->to_float(b + r*gr, row.data(), E);
            for (int i = 0; i < E; ++i) gate += double(row[i]) * xx[i];
            gt->to_float(b + (F+r)*gr, row.data(), E);
            for (int i = 0; i < E; ++i) up += double(row[i]) * xx[i];
            gate = std::min(gate, double(limit)); up = std::clamp(up, -double(limit), double(limit));
            href[e*F+r] = gate/(1+std::exp(-gate))*up;
        }
        // Use the actual quantized activation: this isolates down-kernel error
        // from the intentionally lossy Q8 activation between the two matrices.
        for (int r = 0; r < E; ++r) {
            dt->to_float(b + down + r*dr, row.data(), F);
            double sum = 0;
            for (int j = 0; j < F; ++j) sum += double(row[j]) * hh[e*F+j];
            yref[r] += route[e] * sum;
        }
    }
    out.get(actual.data(), E*4);
    const double gh = rel(hh, href), dy = rel(actual, yref);
    std::printf("types=%d/%d experts=%d gate_Q8_rel_L2=%.8f down_rel_L2=%.8f\n", gu, dn, k, gh, dy);
    require(gh < .015, "gate/up CPU reference within Q8 activation tolerance");
    require(dy < .0002, "down CPU reference accuracy");
}
}
int main() {
    for (int t : {12, 13, 14}) for (int k : {1, 8}) run(t, t, k);
    // Real Unsloth UD-IQ4_XS pattern: IQ3_S gate/up with Q6_K down.
    run(21, 14, 8);
    run(21, 23, 8);
    // OrcaRouter Q4_K_M mixes Q4_K gate/up with Q6_K down.
    run(12, 14, 8);
    require(!gf::moe_supported(999), "unknown format rejected");
    const int before = gf::launch_errors();
    gf::MoeDev empty;
    gf::moe_gate_up(999, empty, 1, 512, 256, 10, nullptr, nullptr, nullptr, 0, nullptr, 0, nullptr, nullptr);
    gf::moe_down(999, empty, 1, 512, 256, 0, nullptr, nullptr, nullptr, nullptr);
    require(gf::launch_errors() == before+2, "unsupported dispatch is a sticky failure");
    ggml_quantize_free();
    std::puts("PASS mixed expert quantization parity");
}
