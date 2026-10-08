#pragma once
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace strata::core::glmfast {
inline void cpu_relax() {
#if defined(__x86_64__) && !defined(_WIN32)
    asm volatile("pause" ::: "memory");
#endif
}

// a fixed pool of worker threads: run(n, fn) calls fn(0..n-1) across the workers AND the caller and returns
// when all finished.  Jobs are claimed lock-free; a worker spins ~300 us for the next batch before it sleeps,
// because one miss request runs its batches back to back (reads, gate/up, down) and a futex wake per batch
// was most of the CPU path's latency.
class Workers {
public:
    // spin_us: how long an idle worker spins for the next batch before it sleeps
    explicit Workers(int n, int spin_us = 300) : spin_us_(spin_us) {
        for (int i = 0; i < n; ++i) th_.emplace_back([this] { loop(); });
    }
    ~Workers() {
        quit_.store(true);
        {
            std::lock_guard<std::mutex> lk(mu_);
        }
        cv_.notify_all();
        for (auto& t : th_) t.join();
    }
    void run(int n, const std::function<void(int)>& fn) {
        if (n <= 0) return;
        const uint64_t g = (gen_.load(std::memory_order_relaxed) + 1) & 0xffffffffu;
        // Invalidate old claimers BEFORE changing their bound/callback. A late
        // worker could otherwise see the new larger total with the old ticket
        // generation and claim a job from the next batch using stale state.
        // New workers start only after gen_'s release publication below.
        ticket_.store(g << 32, std::memory_order_release);
        fn_ = &fn;
        // An old worker that observes the new bound must also observe ticket
        // invalidation, even on weakly ordered hosts.
        total_.store(n, std::memory_order_release);
        done_.store(0, std::memory_order_relaxed);
        // the ticket packs (generation, next index): a straggler of an older batch can never claim (or burn)
        // an index of this one - its CAS fails on the generation
        gen_.store(g, std::memory_order_release);
        if (sleeping_.load(std::memory_order_acquire) > 0) {
            std::lock_guard<std::mutex> lk(mu_);
            cv_.notify_all();
        }
        work(g);
        while (done_.load(std::memory_order_acquire) < n) cpu_relax();
    }
    int size() const { return (int) th_.size() + 1; }

private:
    void work(uint64_t g) {
        int completed = 0;
        for (;;) {
            uint64_t t = ticket_.load(std::memory_order_acquire);
            if ((t >> 32) != g) break;
            const int n = total_.load(std::memory_order_acquire);
            if ((int) (t & 0xffffffffu) >= n) break;
            if (!ticket_.compare_exchange_weak(t, t + 1, std::memory_order_acq_rel)) continue;
            (*fn_)((int) (t & 0xffffffffu));
            ++completed;
        }
        // One publication per worker/batch, rather than per tiny row job. run()
        // cannot advance the generation until every claimed job is published.
        if (completed) done_.fetch_add(completed, std::memory_order_acq_rel);
    }
    void loop() {
        uint64_t seen = gen_.load();
        for (;;) {
            const auto t0 = std::chrono::steady_clock::now();
            int spins = 0;
            while (gen_.load(std::memory_order_acquire) == seen) {
                if (quit_.load(std::memory_order_relaxed)) return;
                cpu_relax();
                if ((++spins & 255) == 0 && std::chrono::steady_clock::now() - t0 > std::chrono::microseconds(spin_us_)) {
                    std::unique_lock<std::mutex> lk(mu_);
                    sleeping_.fetch_add(1);
                    cv_.wait(lk, [&] { return quit_.load() || gen_.load() != seen; });
                    sleeping_.fetch_sub(1);
                    break;
                }
            }
            if (quit_.load()) return;
            seen = gen_.load(std::memory_order_acquire);
            work(seen);
        }
    }
    int spin_us_;
    std::vector<std::thread> th_;
    std::mutex mu_;
    std::condition_variable cv_;
    const std::function<void(int)>* volatile fn_ = nullptr;
    std::atomic<int> total_{0}, sleeping_{0};
    // Claim/completion writes must not invalidate the cache line on which idle
    // workers spin waiting for the next generation (especially across P/E cores).
    alignas(64) std::atomic<int> done_{0};
    alignas(64) std::atomic<uint64_t> ticket_{0};
    alignas(64) std::atomic<uint64_t> gen_{0};
    std::atomic<bool> quit_{false};
};

}
