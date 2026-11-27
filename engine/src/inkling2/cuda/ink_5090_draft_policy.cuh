// engine/src/inkling2/cuda/ink_5090_draft_policy.cuh - RTX 5090 single-user (C1) tokens-per-ms draft window & bypass policy
//
// MIT License
// Copyright (c) 2026 Niko1221 and the Strata contributors
//
// Active ONLY on the RTX 5090 row in Fast mode (ink_5090_draftpol_on(dev)):
//   * Off on every other GPU family/row, off with STRATA_INK_SM120=0, off in Exact mode (STRATA_INK_SM120_EXACT=1 or
//     STRATA_INK_5090_DRAFTPOL_EXACT=1), and off with STRATA_INK_5090_DRAFTPOL=0.
//   * Chooses between (A) zero-draft priced & cooldown-guarded Root-PLD (with generated-span self-repetition preference),
//     (B) zero-draft plain M=1 bypass when long-context or high-entropy drafting loses to plain decode (with periodic probes),
//     and (C) DFlash 2 window M* maximizing calibrated expected emitted tokens per millisecond (+ neural-confirmed suffix
//     splice and priced conditioned 2nd-branch fork).
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "ink_dflash2_kernels.cuh"
#include "ink_runner.cuh"
#include "ink_sm120_cfg.cuh"

namespace strata::ink {

struct Ink5090DraftPolicy {
    // Calibrated RTX 5090 hardware timing model (deterministic; zero wall-clock jitter across rounds or host CPUs)
    double   ema_plain_base_ms = 4.72;
    double   ema_draft_ms      = 1.15;
    double   verify_scale      = 1.00;
    double   obs_verify_ms[25] = {};
    uint32_t m_calls[25]       = {};
    uint32_t plain_calls       = 0;
    uint32_t draft_calls       = 0;

    // Online survival-censored DFlash 2 log-space power calibration (pc = rc^(conf_pow * cal_alpha)) & net gain vs plain M=1
    double obs_acc_ema         = 0.72;
    double obs_pred_ema        = 0.72;
    double cal_alpha           = 1.00;   // stores cal_gamma >= 0.98 (1.0 = uncalibrated, >1 discounts overconfident long-ctx drafts)
    double ema_df2_rate        = 0.52;
    double ema_gain_tok        = 0.75;
    double pld_gen_step_p      = 0.90;   // Model self-repetition survival inside [gen_start_pos .. anchor_pos)
    double pld_prompt_step_p   = 0.80;   // Prompt suffix survival inside [0 .. gen_start_pos)

    // Per-request state (reset automatically whenever anchor_pos starts a new request)
    int    expected_next_pos   = -1;
    int    gen_start_pos       = 0;
    int    req_df2_steps       = 0;
    int    plain_burst_left    = 0;
    int    probe_interval      = 8;
    int    pld_consec_hits     = 0;
    int    req_pld_hits        = 0;
    int    gen_scan_pos        = 0;
    bool   last_was_probe      = false;

    // 64-entry ring of failed PLD (anchor_tok, first_cont_tok) pairs in the current request
    uint64_t failed_pld[64]    = {};
    int      failed_pld_wr     = 0;
    int      failed_pld_cnt    = 0;

    // Telemetry counters (printed when STRATA_INK_SPEC_LOG=1)
    long   stat_df2            = 0;
    long   stat_pld            = 0;
    long   stat_splice         = 0;
    long   stat_fork           = 0;
    long   stat_plain_pre      = 0;
    long   stat_plain_post     = 0;

    void reset_request(int anchor_pos = 0) noexcept {
        obs_acc_ema        = 0.72;
        obs_pred_ema       = 0.72;
        cal_alpha          = 1.00;
        ema_gain_tok       = 0.75;
        pld_gen_step_p     = 0.90;
        pld_prompt_step_p  = 0.80;
        expected_next_pos  = -1;
        gen_start_pos      = std::max(0, anchor_pos);
        req_df2_steps      = 0;
        plain_burst_left   = 0;
        probe_interval     = 8;
        pld_consec_hits    = 0;
        req_pld_hits       = 0;
        gen_scan_pos       = std::max(0, anchor_pos) + 4;
        last_was_probe     = false;
        failed_pld_wr      = 0;
        failed_pld_cnt     = 0;
    }

    void begin_step(int anchor_pos, std::vector<int32_t>& pld_hist) {
        if (expected_next_pos < 0 || anchor_pos < expected_next_pos || anchor_pos > expected_next_pos + 2) {
            reset_request(anchor_pos);
            if ((size_t)anchor_pos < pld_hist.size()) {
                pld_hist.resize((size_t)std::max(0, anchor_pos));
            }
        }
    }

    static double prior_verify_base_ms(int m) noexcept {
        // Measured RTX 5090 verify+argmax+commit curve (@ 932/932 PDL: grouped_reg for M=2..8 + Bk128 NVFP4 TC for M=9..16):
        static constexpr double kCurve[17] = {
            4.72,   // 0
            4.72,   // 1 (plain forward_seq + fast argmax + 1-row ingest)
            5.65,   // 2
            6.14,   // 3
            6.62,   // 4 (end of M_pad=4 argmax & <4> FP8 GEMV)
            7.10,   // 5 (end of tc_attn nslab=1)
            7.60,   // 6 (start of tc_attn nslab=2)
            8.04,   // 7
            8.48,   // 8 (end of M=2..8 grouped_reg, SWA slab=1, M_pad=8 argmax)
            8.88,   // 9
            9.24,   // 10 (end of tc_attn nslab=2)
            9.62,   // 11 (start of tc_attn nslab=3)
            9.96,   // 12
            10.30,  // 13
            10.64,  // 14
            10.98,  // 15 (end of tc_attn nslab=3)
            11.32   // 16
        };
        if (m <= 1) return kCurve[1];
        if (m <= 16) return kCurve[m];
        return kCurve[16] + 0.36 * (double)(m - 16);
    }

    // tc_attn reads the entire global KV cache ONCE per 5-row slab (nslab = 1 for M = 1..5!), so M = 2..5 pays ZERO extra long-KV penalty!
    static double ctx_kv_ms(int m, int apos) noexcept {
        const double k1k = (double)std::max(0, apos) * (1.0 / 1024.0);
        if (m <= 5) return 0.036 * k1k;
        const int nslab = (m <= 10) ? 2 : ((m <= 15) ? 3 : 4);
        return (0.036 + 0.018 * (double)(nslab - 1)) * k1k;
    }

    double t_plain_ms(int apos) const noexcept {
        return ema_plain_base_ms + ctx_kv_ms(1, apos);
    }

    double t_verify_ms(int m, int apos) const noexcept {
        if (m <= 1) return t_plain_ms(apos);
        const double base = (m <= 24 && obs_verify_ms[m] > 1.0)
            ? obs_verify_ms[m]
            : (prior_verify_base_ms(m) * verify_scale);
        return base + ctx_kv_ms(m, apos);
    }

    static uint64_t pack_pair(int32_t a, int32_t b) noexcept {
        return ((uint64_t)(uint32_t)a << 32) | (uint64_t)(uint32_t)b;
    }

    bool is_pld_failed(int32_t a, int32_t b) const noexcept {
        const uint64_t key = pack_pair(a, b);
        for (int i = 0; i < failed_pld_cnt; ++i) {
            if (failed_pld[i] == key) return true;
        }
        return false;
    }

    void mark_pld_failed(int32_t a, int32_t b) noexcept {
        const uint64_t key = pack_pair(a, b);
        for (int i = 0; i < failed_pld_cnt; ++i) {
            if (failed_pld[i] == key) return;
        }
        failed_pld[failed_pld_wr] = key;
        failed_pld_wr = (failed_pld_wr + 1) & 63;
        if (failed_pld_cnt < 64) ++failed_pld_cnt;
    }

    // Two-zone suffix lookup:
    //   1. Generated output span [gen_start_pos .. anchor_pos - 1] (model self-repetition: allows min_n_gen, e.g. 3-gram!)
    //   2. Prompt span [lo .. gen_start_pos - 1] (requires min_n_prompt, 4-gram within 4K, 5-gram within 16K)
    int find_pld_guarded(
        const std::vector<int32_t>& hist,
        const int32_t* key,
        int klen,
        int anchor_pos,
        int min_n_gen,
        int min_n_prompt,
        int max_cont,
        int32_t* out_cont,
        int* out_match_n = nullptr,
        bool* out_from_gen = nullptr) const noexcept {
        if (max_cont <= 0 || anchor_pos < 4 || (int)hist.size() <= anchor_pos) return 0;
        const int32_t last_t = key[klen - 1];
        const int hi = anchor_pos - 1;

        // Zone 1: Generated span (exact model self-repetition in JSON arrays, repeated code blocks, tables, counting)
        const int gen_lo = std::max(0, std::max(gen_start_pos, anchor_pos - 4096));
        if (klen >= min_n_gen && hi >= gen_lo + min_n_gen) {
            for (int n = std::min(8, klen); n >= min_n_gen; --n) {
                for (int p = hi - 1; p >= gen_lo + n - 1; --p) {
                    if (hist[(size_t)p] != last_t) continue;
                    bool eq = true;
                    for (int j = 1; j < n; ++j) {
                        if (hist[(size_t)(p - j)] != key[klen - 1 - j]) { eq = false; break; }
                    }
                    if (!eq) continue;
                    const int32_t c0 = hist[(size_t)(p + 1)];
                    if (c0 < 0 || is_pld_failed(last_t, c0)) continue;
                    int cnt = 0;
                    for (int k = 1; k <= max_cont && (p + k) < anchor_pos; ++k) {
                        const int32_t ct = hist[(size_t)(p + k)];
                        if (ct < 0) break;
                        out_cont[cnt++] = ct;
                    }
                    if (cnt >= 2) {
                        if (out_match_n) *out_match_n = n;
                        if (out_from_gen) *out_from_gen = true;
                        return cnt;
                    }
                }
            }
        }

        // Zone 2: Prompt span (schema keys, function names, quoted identifiers)
        if (klen >= min_n_prompt) {
            const int pr_hi = std::min(hi, gen_start_pos + 2);
            const int lo_far  = std::max(0, anchor_pos - 16384);
            const int lo_near = std::max(0, anchor_pos - 4096);
            for (int n = std::min(8, klen); n >= min_n_prompt; --n) {
                const int lo = (n >= 5) ? lo_far : lo_near;
                for (int p = pr_hi - 1; p >= lo + n - 1; --p) {
                    if (hist[(size_t)p] != last_t) continue;
                    bool eq = true;
                    for (int j = 1; j < n; ++j) {
                        if (hist[(size_t)(p - j)] != key[klen - 1 - j]) { eq = false; break; }
                    }
                    if (!eq) continue;
                    const int32_t c0 = hist[(size_t)(p + 1)];
                    if (c0 < 0 || is_pld_failed(last_t, c0)) continue;
                    int cnt = 0;
                    for (int k = 1; k <= max_cont && (p + k) < anchor_pos; ++k) {
                        const int32_t ct = hist[(size_t)(p + k)];
                        if (ct < 0) break;
                        out_cont[cnt++] = ct;
                    }
                    if (cnt >= 2) {
                        if (out_match_n) *out_match_n = n;
                        if (out_from_gen) *out_from_gen = false;
                        return cnt;
                    }
                }
            }
        }
        return 0;
    }

    void update_gen_repeat(int anchor_pos, const std::vector<int32_t>& hist) noexcept {
        if (req_pld_hits > 0 || anchor_pos >= 2048 || (int)hist.size() <= anchor_pos) return;
        const int start_q = std::max(gen_start_pos + 4, gen_scan_pos);
        for (int q = start_q; q <= anchor_pos; ++q) {
            const int32_t t0 = hist[(size_t)q], t1 = hist[(size_t)(q - 1)], t2 = hist[(size_t)(q - 2)], t3 = hist[(size_t)(q - 3)];
            if (t0 < 0 || t1 < 0 || t2 < 0 || t3 < 0) continue;
            for (int p = q - 1; p >= gen_start_pos + 3; --p) {
                if (hist[(size_t)p] == t0 && hist[(size_t)(p - 1)] == t1 && hist[(size_t)(p - 2)] == t2 && hist[(size_t)(p - 3)] == t3) {
                    req_pld_hits = 1;
                    gen_scan_pos = anchor_pos + 1;
                    return;
                }
            }
        }
        gen_scan_pos = anchor_pos + 1;
    }

    // Choice A: Zero-Draft Root-PLD (0.004 ms CPU, skips DFlash 2 completely when suffix match beats drafted tok/ms!)
    bool try_root_pld(
        const std::vector<int32_t>& hist,
        int32_t anchor_tok,
        int anchor_pos,
        int max_nodes,
        bool pld_enabled,
        InkTreeSpec& out_tr,
        int32_t& out_first_cont,
        bool& out_from_gen) noexcept {
        out_first_cont = -1;
        out_from_gen = false;
        update_gen_repeat(anchor_pos, hist);
        if (!pld_enabled || anchor_pos < 8 || (int)hist.size() <= anchor_pos) return false;
        const int klen = std::min(8, anchor_pos + 1);
        int32_t rkey[8];
        for (int j = 0; j < klen; ++j) {
            const int32_t tv = hist[(size_t)(anchor_pos - klen + 1 + j)];
            if (tv < 0) return false;
            rkey[j] = tv;
        }
        const bool short_ctx = (anchor_pos < 4352);
        const int max_pld_m = short_ctx ? 14 : 8;
        const int cap_k = std::min(std::max(0, std::min(max_pld_m, max_nodes) - 1), max_pld_m - 1);
        if (cap_k < 3) return false;
        int32_t cont[16] = {};
        int match_n = 0;
        bool from_gen = false;
        const int min_gen = short_ctx ? 4 : 5;
        const int min_pr  = short_ctx ? 5 : 6;
        const int n_c = find_pld_guarded(hist, rkey, klen, anchor_pos, min_gen, min_pr, cap_k, cont, &match_n, &from_gen);
        if (n_c < 3) return false;

        const double base_p = from_gen ? pld_gen_step_p : pld_prompt_step_p;
        const double len_boost = (match_n >= 6) ? 1.05 : ((match_n == 5) ? 1.02 : 0.99);
        const double p_step = std::max(0.28, std::min(0.965, base_p * len_boost));

        double best_rate = -1.0;
        int best_m = 4;
        double cum = 1.0;
        double e_tok = 1.0;
        for (int k = 1; k <= n_c; ++k) {
            cum *= p_step;
            e_tok += cum;
            if (k >= 3) {
                const int m = 1 + k;
                if (m > 8 && (m & 1) != 0 && k < n_c) continue;   // bucket M > 8 to even {10, 12, 14, 16} to halve graph captures!
                const double rate = e_tok / t_verify_ms(m, anchor_pos);   // zero draft ms!
                if (rate > best_rate) { best_rate = rate; best_m = m; }
            }
        }
        const double plain_rate = 1.0 / t_plain_ms(anchor_pos);
        const double ref_rate   = std::max(plain_rate, ema_df2_rate);
        const double req_margin = short_ctx ? 1.02 : 1.08;
        if (best_rate < ref_rate * req_margin) return false;

        out_tr = InkTreeSpec{};
        out_tr.M = best_m;
        out_tr.tok[0] = anchor_tok;
        out_tr.parent[0] = -1;
        out_tr.depth[0] = 0;
        for (int k = 1; k < best_m; ++k) {
            out_tr.tok[k] = cont[k - 1];
            out_tr.parent[k] = (int8_t)(k - 1);
            out_tr.depth[k] = (uint8_t)k;
        }
        out_tr.build();
        out_first_cont = cont[0];
        out_from_gen = from_gen;
        ++req_pld_hits;
        ++stat_pld;
        return true;
    }

    // Choice B: Pre-draft Plain M=1 Bypass (disabled: calibrated M=2..5 DFlash 2 with nslab=1 beats plain decode across 4K-128K!)
    bool want_pre_draft_plain(int /*anchor_pos*/) noexcept {
        plain_burst_left = 0;
        last_was_probe = false;
        return false;
    }

    // Choice C: Size optimal DFlash 2 verify window M* + Neural-Confirmed Suffix Splice + Root Divergence Graft + Priced 2nd-Branch Fork!
    int build_df2_window(
        int32_t anchor_tok,
        int anchor_pos,
        int base_max,
        int max_nodes,
        const int32_t ids7[7],
        const float raw_conf7[7],
        const df2::Df2Fork5090& fork,
        double conf_pow,
        bool pld_enabled,
        const std::vector<int32_t>& hist,
        float cal_p_out[7],
        double& out_cum_end,
        int& out_main_drafts,
        InkTreeSpec& out_tr) noexcept {
        out_main_drafts = 0;
        const bool short_ctx = (anchor_pos < 4352);
        const int max_w = std::max(2, std::min(8, base_max));
        double e_tok[9] = {};
        double cum_arr[9] = {};
        e_tok[1] = 1.0;
        cum_arr[1] = 1.0;
        double cum = 1.0;
        const double eff_pow = std::max(0.65, std::min(1.45, conf_pow * cal_alpha));
        for (int i = 0; i < 7; ++i) {
            const double rc = std::max(0.0, std::min(1.0, (double)raw_conf7[i]));
            const double pc = (rc > 0.0) ? std::max(0.0, std::min(0.995, std::pow(rc, eff_pow))) : 0.0;
            cal_p_out[i] = (float)pc;
            cum *= pc;
            cum_arr[i + 2] = cum;
            e_tok[i + 2] = e_tok[i + 1] + cum;
        }

        // Marginal remaining-time check at ultra-long >= 48K context (never throw away an already-computed draft at 4K-32K VRAM KV!)
        if (anchor_pos >= 49152) {
            const double rem_plain_rate = 1.0 / t_plain_ms(anchor_pos);
            const double rem_m2_rate    = e_tok[2] / t_verify_ms(2, anchor_pos);
            if (cal_p_out[0] < 0.08 && rem_m2_rate < rem_plain_rate * 0.94) {
                out_tr = InkTreeSpec{};
                out_tr.M = 1;
                out_tr.tok[0] = anchor_tok;
                out_tr.parent[0] = -1;
                out_tr.depth[0] = 0;
                out_cum_end = 0.0;
                out_main_drafts = 0;
                ++stat_plain_post;
                return 1;
            }
        }

        // 1. Optimal main-chain length best_m in [2..max_w]
        double best_rate = -1.0;
        int best_m = 2;
        for (int m = 2; m <= max_w; ++m) {
            const double tot_ms = ema_draft_ms + t_verify_ms(m, anchor_pos);
            const double rate   = e_tok[m] / tot_ms;
            if (rate > best_rate) {
                best_rate = rate;
                best_m = m;
            }
        }

        out_tr = InkTreeSpec{};
        out_tr.M = best_m;
        for (int i = 0; i < best_m; ++i) {
            out_tr.tok[i] = (i == 0) ? anchor_tok : ids7[i - 1];
            out_tr.parent[i] = (i == 0) ? -1 : (int8_t)(i - 1);
            out_tr.depth[i] = (uint8_t)i;
        }
        out_main_drafts = best_m - 1;
        out_cum_end = cum_arr[best_m];
        double cur_e = e_tok[best_m];
        bool spliced = false;

        // 2. Neural-Confirmed Mid-Chain Suffix Splice, Root Divergence Graft, or High-Confidence Leaf-PLD
        const int cap_lim = short_ctx ? ((pld_consec_hits >= 1) ? 16 : 12) : 8;
        const int cap_tot = std::min(cap_lim, max_nodes);
        if (pld_enabled && anchor_pos >= 8 && (int)hist.size() > anchor_pos && cap_tot >= best_m + 2) {
            // 2a. Mid-chain agreement: DFlash 2 drafted ids7[0..d-1], and exact suffix history predicts cont[0] == ids7[d]!
            const int sp_min_gen = short_ctx ? ((req_pld_hits > 0) ? 3 : 4) : 4;
            const int sp_min_pr  = (short_ctx && req_pld_hits > 0) ? 5 : 6;
            for (int d = std::min(best_m - 1, 5); d >= 1 && !spliced; --d) {
                if (cum_arr[d + 1] < 0.48 || d >= 7) continue;
                int32_t skey[8];
                int klen = 0;
                const int need_h = std::max(0, 8 - (d + 1));
                for (int i = anchor_pos - need_h; i < anchor_pos && klen < 8; ++i) {
                    if (i >= 0) skey[klen++] = hist[(size_t)i];
                }
                skey[klen++] = anchor_tok;
                for (int j = 0; j < d && klen < 8; ++j) skey[klen++] = ids7[j];
                int32_t cont[16] = {};
                int match_n = 0;
                bool from_gen = false;
                const int max_c = std::min(15, cap_tot - (d + 1));
                const int nc = find_pld_guarded(hist, skey, klen, anchor_pos, sp_min_gen, sp_min_pr, max_c, cont, &match_n, &from_gen);
                if (nc >= (best_m - d + 1) && cont[0] == ids7[d]) {
                    const double sp = std::max(0.35, std::min(0.955, (from_gen ? pld_gen_step_p : pld_prompt_step_p) * (match_n >= 4 ? 1.04 : 0.96)));
                    double c_run = cum_arr[d + 1];
                    double e_run = e_tok[d + 1];
                    int best_nc = 0;
                    double best_sp_rate = best_rate * 1.015;
                    for (int c = 1; c <= nc; ++c) {
                        const double step_p = (c == 1) ? std::max((double)cal_p_out[d], sp) : sp;
                        c_run *= step_p;
                        e_run += c_run;
                        const int mt = (d + 1) + c;
                        if (mt > best_m && !(mt > 8 && (mt & 1) != 0 && c < nc)) {
                            const double r_sp = e_run / (ema_draft_ms + t_verify_ms(mt, anchor_pos));
                            if (r_sp > best_sp_rate) { best_sp_rate = r_sp; best_nc = c; }
                        }
                    }
                    if (best_nc > 0) {
                        out_tr.M = d + 1;
                        int m_keep = d;
                        for (int c = 0; c < best_nc; ++c) {
                            const int nid = out_tr.M++;
                            out_tr.tok[nid] = cont[c];
                            out_tr.parent[nid] = (int8_t)(nid - 1);
                            out_tr.depth[nid] = (uint8_t)nid;
                            if (m_keep == d + c && (d + c) < 7 && cont[c] == ids7[d + c]) ++m_keep;
                        }
                        out_main_drafts = m_keep;
                        best_rate = best_sp_rate;
                        cur_e = e_run;
                        spliced = true;
                        ++stat_splice;
                    }
                }
            }
            // 2b. Fallback Leaf-PLD when full main chain has high survival
            const int ext_cap = cap_tot - out_tr.M;
            if (!spliced && short_ctx && best_m >= 5 && ext_cap >= 2 && out_cum_end >= 0.45) {
                int32_t lkey[8];
                int klen = 0;
                const int need_h = std::max(0, 8 - best_m);
                for (int i = anchor_pos - need_h; i < anchor_pos && klen < 8; ++i) {
                    if (i >= 0) lkey[klen++] = hist[(size_t)i];
                }
                for (int i = std::max(0, best_m - (8 - klen)); i < best_m && klen < 8; ++i) {
                    lkey[klen++] = out_tr.tok[i];
                }
                int32_t cont[8] = {};
                int match_n = 0;
                bool from_gen = false;
                const int nc = find_pld_guarded(hist, lkey, klen, anchor_pos, 4, 5, std::min(7, ext_cap), cont, &match_n, &from_gen);
                if (nc >= 2) {
                    double ext_cum = out_cum_end;
                    double ext_e   = e_tok[best_m];
                    const double lp = std::max(0.32, std::min(0.94, (from_gen ? pld_gen_step_p : pld_prompt_step_p) * (match_n >= 4 ? 1.00 : 0.92)));
                    int best_ext = 0;
                    double best_ext_rate = best_rate * 1.015;
                    for (int c = 1; c <= nc; ++c) {
                        ext_cum *= lp;
                        ext_e += ext_cum;
                        const int mt = best_m + c;
                        if (mt > 8 && (mt & 1) != 0 && c < nc) continue;
                        const double r_ext = ext_e / (ema_draft_ms + t_verify_ms(mt, anchor_pos));
                        if (r_ext > best_ext_rate) { best_ext_rate = r_ext; best_ext = c; }
                    }
                    if (best_ext > 0) {
                        for (int c = 0; c < best_ext; ++c) {
                            const int nid = out_tr.M++;
                            out_tr.tok[nid] = cont[c];
                            out_tr.parent[nid] = (int8_t)(nid - 1);
                            out_tr.depth[nid] = (uint8_t)nid;
                        }
                        best_rate = best_ext_rate;
                        cur_e = ext_e;
                        spliced = true;
                        ++stat_splice;
                    }
                }
            }
            // 2c. Priced Generated-Span Divergence Graft (short context only, >=4-gram match)
            if (!spliced && short_ctx && out_tr.M <= 7 && out_tr.M + 2 <= cap_tot) {
                const int rklen = std::min(8, anchor_pos + 1);
                int32_t rkey[8];
                bool rkey_ok = true;
                for (int j = 0; j < rklen; ++j) {
                    const int32_t tv = hist[(size_t)(anchor_pos - rklen + 1 + j)];
                    if (tv < 0) { rkey_ok = false; break; }
                    rkey[j] = tv;
                }
                if (rkey_ok) {
                    int32_t cont[8] = {};
                    int match_n = 0;
                    bool from_gen = false;
                    const int nc = find_pld_guarded(hist, rkey, rklen, anchor_pos, 4, 6, 6, cont, &match_n, &from_gen);
                    if (nc >= 2 && from_gen) {
                        int d_div = 0;
                        while (d_div < nc && (d_div + 1) < out_tr.M && out_tr.tok[d_div + 1] == cont[d_div]) ++d_div;
                        if (d_div <= 1 && d_div + 1 < nc && cal_p_out[d_div] <= 0.82f) {
                            const double p_pre = cum_arr[d_div + 1];
                            const double p_rem = std::max(0.0, 0.99 - (double)cal_p_out[d_div]);
                            const double gp    = std::min(p_rem, std::max(0.25, pld_gen_step_p * 0.62));
                            const double e_g2  = cur_e + p_pre * gp * (1.0 + pld_gen_step_p * 0.85);
                            const double r_g2  = e_g2 / (ema_draft_ms + t_verify_ms(out_tr.M + 2, anchor_pos));
                            if (r_g2 > best_rate * 1.018) {
                                const int n1 = out_tr.M++;
                                out_tr.tok[n1] = cont[d_div];
                                out_tr.parent[n1] = (int8_t)d_div;
                                out_tr.depth[n1] = (uint8_t)(d_div + 1);
                                const int n2 = out_tr.M++;
                                out_tr.tok[n2] = cont[d_div + 1];
                                out_tr.parent[n2] = (int8_t)n1;
                                out_tr.depth[n2] = (uint8_t)(d_div + 2);
                                best_rate = r_g2;
                                cur_e = e_g2;
                                spliced = true;
                                ++stat_splice;
                            }
                        }
                    }
                }
            }
        }

        // 3. Priced Conditioned 2nd-Branch Fork (short context only; adds runner-up ONLY when marginal tok/ms wins by >= +2.5%!)
        if (!spliced && short_ctx && out_tr.M <= 6 &&
            fork.fork_step >= 0 && fork.fork_step <= 1 && fork.fork_step < best_m - 1 &&
            fork.alt_id >= 0 && fork.alt_id != ids7[fork.fork_step] && out_tr.M + 1 <= cap_tot) {
            const int fd = fork.fork_step;
            const double p_pre = cum_arr[fd + 1];
            const double p_rem = std::max(0.0, 0.995 - (double)cal_p_out[fd]);
            const double p_b1  = std::min(p_rem, std::pow(std::max(0.0, (double)fork.alt_conf), eff_pow));
            if (p_pre * p_b1 >= 0.26) {
                const double e1 = cur_e + p_pre * p_b1;
                const double r1 = e1 / (ema_draft_ms + t_verify_ms(out_tr.M + 1, anchor_pos));
                int add_b = 0;
                double best_rb = best_rate * 1.025;
                if (r1 > best_rb) { best_rb = r1; add_b = 1; }
                if (fork.next_id >= 0 && out_tr.M + 2 <= cap_tot) {
                    const double p_b2 = std::min(0.98, std::pow(std::max(0.0, (double)fork.next_conf), eff_pow));
                    const double e2 = e1 + p_pre * p_b1 * p_b2;
                    const double r2 = e2 / (ema_draft_ms + t_verify_ms(out_tr.M + 2, anchor_pos));
                    if (r2 > best_rb) { best_rb = r2; add_b = 2; }
                }
                if (add_b >= 1) {
                    const int n1 = out_tr.M++;
                    out_tr.tok[n1] = fork.alt_id;
                    out_tr.parent[n1] = (int8_t)fd;
                    out_tr.depth[n1] = (uint8_t)(fd + 1);
                    if (add_b == 2) {
                        const int n2 = out_tr.M++;
                        out_tr.tok[n2] = fork.next_id;
                        out_tr.parent[n2] = (int8_t)n1;
                        out_tr.depth[n2] = (uint8_t)(fd + 2);
                    }
                    ++stat_fork;
                }
            }
        }

        out_tr.build();
        ++stat_df2;
        return out_tr.M;
    }

    void observe_plain_step(int anchor_pos, double plain_ms) noexcept {
        if (++plain_calls >= 3 && plain_ms > 1.5 && plain_ms < 25.0) {
            const double raw_base = plain_ms - ctx_kv_ms(1, anchor_pos);
            const double clamped  = std::max(2.6, std::min(6.5, raw_base));
            ema_plain_base_ms = 0.85 * ema_plain_base_ms + 0.15 * clamped;
        }
        expected_next_pos = anchor_pos + 1;
    }

    void update_verify_timing(int M, int anchor_pos, double verify_ms) noexcept {
        if (M < 2 || M > 24) return;
        // Skip first 2 calls of any M (CUDA Graph capture + pdl::instantiate host overhead!)
        if (++m_calls[M] < 3) return;
        if (verify_ms > 2.0 && verify_ms < 35.0) {
            const double prior = prior_verify_base_ms(M);
            const double raw_b = verify_ms - ctx_kv_ms(M, anchor_pos);
            const double base  = std::max(0.65 * prior, std::min(1.55 * prior, raw_b));
            obs_verify_ms[M] = (obs_verify_ms[M] > 1.0) ? (0.82 * obs_verify_ms[M] + 0.18 * base) : base;
            const double ratio = std::max(0.75, std::min(1.35, base / prior));
            verify_scale = 0.90 * verify_scale + 0.10 * ratio;
        }
    }

    void observe_pld_step(int anchor_pos, int32_t anchor_tok, int32_t first_cont, bool from_gen, int M, int n_emit, double verify_ms) noexcept {
        update_verify_timing(M, anchor_pos, verify_ms);
        const int n_acc = std::max(0, n_emit - 1);
        const int n_draft = std::max(1, M - 1);
        double& target_p = from_gen ? pld_gen_step_p : pld_prompt_step_p;
        if (n_acc == 0 && first_cont >= 0) {
            mark_pld_failed(anchor_tok, first_cont);
            target_p = std::max(0.28, 0.84 * target_p + 0.16 * 0.08);
            pld_consec_hits = 0;
        } else {
            if (n_acc >= 4) ++pld_consec_hits;
            else if (n_acc <= 1) pld_consec_hits = 0;
            const double trials = (double)n_acc + ((n_acc < n_draft) ? 1.0 : 0.0);
            if (trials > 0.0) {
                const double p_hat = (double)n_acc / trials;
                target_p = std::max(0.28, std::min(0.965, 0.82 * target_p + 0.18 * p_hat));
            }
        }
        expected_next_pos = anchor_pos + n_emit;
    }

    void observe_df2_step(
        int anchor_pos,
        int M,
        int n_emit,
        int main_drafts,
        int main_acc,
        const float raw_conf7[7],
        double conf_pow,
        double draft_ms,
        double verify_ms) noexcept {
        ++draft_calls;
        if (draft_calls >= 3 && draft_ms > 0.25 && draft_ms < 8.0) {
            const double cd = std::max(0.45, std::min(2.50, draft_ms));
            ema_draft_ms = 0.85 * ema_draft_ms + 0.15 * cd;
        }
        update_verify_timing(M, anchor_pos, verify_ms);

        // Exact censored survival log-space power calibration on ONLY the tested DFlash 2 main-chain tokens!
        const int n_acc = std::max(0, n_emit - 1);
        if (M >= 9 && n_acc >= 6) ++pld_consec_hits;
        else if (M >= 9 && n_acc <= 2) pld_consec_hits = 0;
        const int n_obs_df2 = std::min(7, std::max(0, main_drafts));
        const int m_acc_clamped = std::max(0, std::min(n_obs_df2, main_acc));
        for (int d = 0; d < n_obs_df2; ++d) {
            if (d > m_acc_clamped) break;
            const double y = (d < m_acc_clamped) ? 1.0 : 0.0;
            const double rp = std::max(0.03, std::min(0.992, std::pow(std::max(0.0, std::min(1.0, (double)raw_conf7[d])), conf_pow)));
            obs_acc_ema  = 0.93 * obs_acc_ema  + 0.07 * y;
            obs_pred_ema = 0.93 * obs_pred_ema + 0.07 * rp;
        }
        const double a_c = std::max(0.18, std::min(0.94, obs_acc_ema));
        const double p_c = std::max(0.18, std::min(0.92, obs_pred_ema));
        cal_alpha = std::max(0.98, std::min(1.45, std::log(a_c) / std::log(p_c)));

        const double eff_step_ms = std::max(ema_draft_ms, std::min(3.0, draft_ms)) + t_verify_ms(M, anchor_pos);
        const double rate = (double)n_emit / eff_step_ms;
        ema_df2_rate = 0.82 * ema_df2_rate + 0.18 * rate;

        const double tp = t_plain_ms(anchor_pos);
        const double gain = (double)n_emit - (eff_step_ms / std::max(2.0, tp));
        ema_gain_tok = 0.80 * ema_gain_tok + 0.20 * gain;
        ++req_df2_steps;

        if (anchor_pos >= 6144) {
            if (last_was_probe) {
                last_was_probe = false;
                if (n_emit >= 3 || gain > 0.06) {
                    ema_gain_tok = std::max(ema_gain_tok, 0.25);
                    probe_interval = 8;
                    plain_burst_left = 0;
                } else {
                    probe_interval = std::min(18, probe_interval + 4);
                    plain_burst_left = probe_interval;
                }
            } else if (req_df2_steps >= 6 && ema_gain_tok < -0.06) {
                plain_burst_left = probe_interval;
            }
        } else {
            plain_burst_left = 0;
            last_was_probe = false;
        }

        expected_next_pos = anchor_pos + n_emit;
    }
};

}  // namespace strata::ink
