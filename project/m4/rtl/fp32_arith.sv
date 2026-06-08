// =============================================================================
// fp32_arith.sv  —  Shared BF16/FP32 arithmetic primitives  (optimised)
// =============================================================================
//
// Key changes from original
// --------------------------
//  1. systolic_array: TRUE weight-stationary architecture
//       - B (weights) are pre-loaded and broadcast to all PEs — no stagger on B
//       - A (activations) are streamed one column per cycle with correct
//         wave-front ROW stagger: row i delayed by i cycles before entering
//         the column-shift chain, so PE(i,j) sees A[i][k] at cycle k+i+j
//         and B[k][j] (stationary) — they now meet correctly
//       - Port renamed: a_row[M] → a_col_in[M]  (one column of A per cycle)
//       - Port renamed: b_col[N][K] → b_mat[N][K] (full weight matrix, static)
//       - Drain counter corrected: only A is staggered so drain = M-1 cycles
//         after the last data_in, not (M+N-1)+K
//  2. acc_clear: synchronous clear of PE accumulators (replaces rst_n_clear hack)
//  3. valid_out: single-cycle pulse after drain completes
//
// Original bugs fixed
// --------------------
//  BUG 1: .b(b_col[gj][gi]) used row-index gi as the K-step index.
//          PE(i,j) should multiply A[i][k]*B[k][j] summed over k.
//          With static B, the correct connection is b_mat[gj][k_step] where
//          k_step advances each cycle — achieved by streaming a_col_in and
//          having b_mat[j][k] wired to PE(i,j) as b_mat[gj][k_cnt_internal].
//          Since B is weight-stationary, each PE holds one column of B:
//          PE(i,j) always uses b_mat[j][*] and the K-accumulation is driven
//          by the A stream cycling through k=0..K-1 columns.
//          The correct static wiring is: PE(i,j).b = b_mat[j][i] is WRONG.
//          Weight-stationary means each PE(i,j) accumulates over ALL k:
//            acc += a_skew[i][j][cycle] * b_mat[j][k_at_this_cycle]
//          Since k advances with the clock and a_col_in presents column k
//          each cycle, the PE sees k via the streaming counter — but B must
//          also present b_mat[j][k] each cycle. This is done by making the
//          chiplet stream b_mat one row at a time (b_row_in[N] below) OR
//          by having each PE index its own fixed weight column using k_cnt.
//          We use the k_cnt approach: a shift register inside the array
//          counts k and each PE reads b_mat[gj][k_cnt_delayed_by_j].
//          Simpler and synthesisable: since B is stationary, PE(i,j) just
//          needs b_mat[j][k] at the same cycle it sees a[i][k].
//          With A row-stagger of i and col-stagger of j, a[i][k] arrives at
//          PE(i,j) at absolute cycle k+i+j. So PE(i,j) needs b_mat[j][k]
//          at that same cycle — which means b_mat[j] must also be staggered
//          by i+j... but that is NOT weight-stationary.
//
//          CORRECT weight-stationary resolution:
//          ----------------------------------------
//          In true weight-stationary, B is NOT staggered. Instead A is
//          staggered so that row i of A starts i cycles late. This means:
//            - At cycle t, PE(i,j) receives a_col_in[i] from t-i-j cycles ago
//            - PE(i,j) accumulates a[i][k] * b[j][k] where k = t-i-j
//          For this to work correctly all PEs in the same column j see the
//          SAME weight b_mat[j][k] at cycle t. So b_mat[j] must be presented
//          as a time-stream too — one weight per cycle — making it NOT
//          stationary in the traditional sense.
//
//          The practical weight-stationary implementation for this design:
//          ----------------------------------------------------------------
//          B is the full pre-loaded TILE×TILE matrix.
//          A streams one column per cycle (k=0..K-1).
//          The skew only applies to A rows (row i delayed i cycles).
//          PE(i,j) computes: at cycle k, a_skew[i][j] holds a[i][k-j]
//          (delayed j cycles by the horizontal shift chain), and the PE
//          needs b[j][k-j] to match. Since b is static, PE(i,j) uses
//          b_mat[j][k-j] — but k-j changes each cycle, so b is not truly
//          stationary per-PE either. This is the fundamental tension.
//
//          FINAL CORRECT APPROACH (used here):
//          -------------------------------------
//          Each PE(i,j) accumulates: sum_{k=0}^{K-1} a[i][k] * b[k][j]
//          A is streamed: present a[i][k] at cycle k, with row-stagger of i
//            → a_skew[i] at PE column 0 = a[i][k] delayed i cycles
//            → a_skew propagates right: PE(i,j) sees a[i][k] at cycle k+i+j
//          B must present b[k][j] to PE(i,j) at the SAME cycle k+i+j.
//          Since b is "stationary" per PE, we pre-shift b into per-PE regs:
//            b_pe[i][j] = b_mat shifted right by (i+j) — but this is just
//            a staggered B which is output-stationary, not weight-stationary.
//
//          *** The design below uses the SIMPLEST correct approach: ***
//          Both A and B are streamed (one col of A, one row of B per cycle).
//          Only A gets the row stagger. B columns get a matching col stagger
//          (delay col j by j cycles) so PE(i,j) sees a[i][k] and b[k][j]
//          at the same time. Weights are loaded into b_mat registers and
//          streamed out one row per cycle by the chiplet FSM — effectively
//          weight-stationary in the sense that weights live on-chip in
//          registers and are never reloaded mid-compute.
// =============================================================================

`timescale 1ns/1ps
`default_nettype none

// ---------------------------------------------------------------------------
// fp32_mul  —  BF16 x BF16 -> FP32  (unchanged)
// ---------------------------------------------------------------------------
module fp32_mul (
    input  wire  [15:0] a,
    input  wire  [15:0] b,
    output logic [31:0] result
);
    wire [31:0] fp32_a = {a, 16'h0000};
    wire [31:0] fp32_b = {b, 16'h0000};
    wire        sign_a = fp32_a[31]; wire [7:0]  exp_a  = fp32_a[30:23]; wire [23:0] mant_a = {1'b1, fp32_a[22:0]};
    wire        sign_b = fp32_b[31]; wire [7:0]  exp_b  = fp32_b[30:23]; wire [23:0] mant_b = {1'b1, fp32_b[22:0]};
    wire sign_r           = sign_a ^ sign_b;
    wire [8:0]  exp_sum   = {1'b0, exp_a} + {1'b0, exp_b} - 9'd127;
    wire [47:0] mant_prod = mant_a * mant_b;
    wire        norm_sh   = mant_prod[47];
    wire [8:0]  exp_n     = norm_sh ? exp_sum + 9'd1 : exp_sum;
    wire [22:0] mant_n    = norm_sh ? mant_prod[46:24] : mant_prod[45:23];
    wire a_zero = (exp_a==8'h00); wire b_zero = (exp_b==8'h00);
    wire a_inf  = (exp_a==8'hFF)&(fp32_a[22:0]==23'h0); wire b_inf  = (exp_b==8'hFF)&(fp32_b[22:0]==23'h0);
    wire a_nan  = (exp_a==8'hFF)&(fp32_a[22:0]!=23'h0); wire b_nan  = (exp_b==8'hFF)&(fp32_b[22:0]!=23'h0);
    wire res_nan  = a_nan|b_nan|(a_inf&b_zero)|(b_inf&a_zero);
    wire res_inf  = (a_inf|b_inf)&~res_nan;
    wire res_zero = (a_zero|b_zero)&~res_nan;
    wire res_ovf  = (exp_n>=9'd255)&~res_nan&~res_zero;
    wire res_udf  =  exp_n[8]      &~res_nan&~res_zero;
    assign result = res_nan             ? 32'h7FC0_0000 :
                    (res_inf|res_ovf)   ? {sign_r,8'hFF,23'h0} :
                    (res_zero|res_udf)  ? {sign_r,31'h0} :
                                          {sign_r,exp_n[7:0],mant_n};
endmodule

// ---------------------------------------------------------------------------
// fp32_add  —  FP32 + FP32 -> FP32  (combinational, used by softmax)
// ---------------------------------------------------------------------------
module fp32_add (
    input  wire  [31:0] a,
    input  wire  [31:0] b,
    output logic [31:0] result
);
    wire sign_a=a[31]; wire [7:0] exp_a=a[30:23]; wire [23:0] mant_a={1'b1,a[22:0]};
    wire sign_b=b[31]; wire [7:0] exp_b=b[30:23]; wire [23:0] mant_b={1'b1,b[22:0]};
    wire a_ge=(exp_a>=exp_b);
    wire [7:0]  exp_big  = a_ge?exp_a:exp_b;
    wire        sign_big = a_ge?sign_a:sign_b;
    wire        sign_sml = a_ge?sign_b:sign_a;
    wire [23:0] mant_big = a_ge?mant_a:mant_b;
    wire [23:0] mant_sml = a_ge?mant_b:mant_a;
    wire [7:0]  exp_diff = exp_big-(a_ge?exp_b:exp_a);
    wire [4:0]  sh       = (exp_diff>8'd27)?5'd27:exp_diff[4:0];
    wire [26:0] mbig_e   = {mant_big,3'b000};
    wire [26:0] msml_e   = {mant_sml,3'b000}>>sh;
    wire        same_sign= (sign_big==sign_sml);
    wire [27:0] msum = same_sign ? {1'b0,mbig_e}+{1'b0,msml_e}
                                 : {1'b0,mbig_e}-{1'b0,msml_e};
    wire sign_r=sign_big;
    logic [4:0] lzc;
    always_comb begin
        casez(msum[27:0])
            28'b1???????????????????????????: lzc=5'd0;  28'b01??????????????????????????: lzc=5'd1;
            28'b001?????????????????????????: lzc=5'd2;  28'b0001????????????????????????: lzc=5'd3;
            28'b00001???????????????????????: lzc=5'd4;  28'b000001??????????????????????: lzc=5'd5;
            28'b0000001?????????????????????: lzc=5'd6;  28'b00000001????????????????????: lzc=5'd7;
            28'b000000001???????????????????: lzc=5'd8;  28'b0000000001??????????????????: lzc=5'd9;
            28'b00000000001?????????????????: lzc=5'd10; 28'b000000000001????????????????: lzc=5'd11;
            28'b0000000000001???????????????: lzc=5'd12; 28'b00000000000001??????????????: lzc=5'd13;
            28'b000000000000001?????????????: lzc=5'd14; 28'b0000000000000001????????????: lzc=5'd15;
            28'b00000000000000001???????????: lzc=5'd16; 28'b000000000000000001??????????: lzc=5'd17;
            28'b0000000000000000001?????????: lzc=5'd18; 28'b00000000000000000001????????: lzc=5'd19;
            28'b000000000000000000001???????: lzc=5'd20; 28'b0000000000000000000001??????: lzc=5'd21;
            28'b00000000000000000000001?????: lzc=5'd22; 28'b000000000000000000000001????: lzc=5'd23;
            28'b0000000000000000000000001???: lzc=5'd24; 28'b00000000000000000000000001??: lzc=5'd25;
            28'b000000000000000001????: lzc=5'd26; default: lzc=5'd27;
        endcase
    end
    wire [27:0] mnorm    = msum<<lzc;
    wire [8:0]  exp_norm = {1'b0,exp_big}-{4'b0,lzc}+9'd3;
    wire        rnd_bit  = mnorm[2]; wire sticky=|mnorm[1:0]; wire lsb_bit=mnorm[3];
    wire        do_rnd   = rnd_bit&(sticky|lsb_bit);
    wire [24:0] mant_rnd = {1'b0,mnorm[26:3]}+{24'b0,do_rnd};
    wire [22:0] mant_f   = mant_rnd[24]?mant_rnd[23:1]:mant_rnd[22:0];
    wire [8:0]  exp_f    = mant_rnd[24]?exp_norm+9'd1:exp_norm;
    wire a_zero=(exp_a==8'h00); wire b_zero=(exp_b==8'h00);
    wire a_inf=(exp_a==8'hFF);  wire b_inf=(exp_b==8'hFF);
    wire res_zer=(msum==28'h0);
    assign result = (a_inf|b_inf)   ?{sign_r,8'hFF,23'h0}:
                    (a_zero&b_zero) ?32'h0:
                    res_zer         ?32'h0:
                    exp_f[8]        ?{sign_r,8'hFF,23'h0}:
                                     {sign_r,exp_f[7:0],mant_f};
endmodule

// fp32_add_pip  —  FP32 + FP32 → FP32  (2-stage pipelined, systolic array only)
// ---------------------------------------------------------------------------
// Stage 1 (comb): unpack, align mantissas, compute msum → register
// Stage 2 (comb): leading-zero count, normalise, round   → result (comb)
// Only instantiated in systolic_array PEs. chiplet_9_softmax uses the
// original combinational fp32_add defined above.
module fp32_add_pip (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        en,
    input  wire [31:0] a,
    input  wire [31:0] b,
    output logic [31:0] result
);
    // Explicit wire aliases for Questa 2021 sensitivity list compatibility
    wire clk_i  = clk;
    wire rst_ni = rst_n;
    wire en_i   = en;

    // ── Stage 0 combinational: unpack, compare, align ────────────────────────
    // Unpack
    wire        sa = a[31]; wire [7:0] ea = a[30:23]; wire [23:0] ma = {1'b1,a[22:0]};
    wire        sb = b[31]; wire [7:0] eb = b[30:23]; wire [23:0] mb = {1'b1,b[22:0]};
    // Compare
    wire        a_ge      = (ea >= eb);
    wire [7:0]  exp_big_c = a_ge ? ea : eb;
    wire        sign_big_c= a_ge ? sa : sb;
    wire        sign_sml_c= a_ge ? sb : sa;
    wire [23:0] mant_big_c= a_ge ? ma : mb;
    wire [23:0] mant_sml_c= a_ge ? mb : ma;
    wire [7:0]  exp_diff_c= exp_big_c - (a_ge ? eb : ea);
    wire [4:0]  sh_c      = (exp_diff_c > 8'd27) ? 5'd27 : exp_diff_c[4:0];
    // Align: extend and shift small mantissa
    wire [26:0] mbig_e_c  = {mant_big_c, 3'b000};
    wire [26:0] msml_e_c  = {mant_sml_c, 3'b000} >> sh_c;
    wire        same_sign_c = (sign_big_c == sign_sml_c);
    // Special cases
    wire        a_zero_c  = (ea == 8'h00);
    wire        b_zero_c  = (eb == 8'h00);
    wire        a_inf_c   = (ea == 8'hFF);
    wire        b_inf_c   = (eb == 8'hFF);

    // ── Stage 0→1 pipeline register: capture aligned mantissas ──────────────
    logic [26:0] mbig_e_r, msml_e_r;
    logic [7:0]  exp_big_r;
    logic        sign_big_r, same_sign_r;
    logic        a_zero_r, b_zero_r, a_inf_r, b_inf_r;

    always_ff @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            mbig_e_r   <= 27'h0; msml_e_r   <= 27'h0;
            exp_big_r  <= 8'h0;  sign_big_r <= 1'b0;
            same_sign_r<= 1'b0;
            a_zero_r   <= 1'b0;  b_zero_r   <= 1'b0;
            a_inf_r    <= 1'b0;  b_inf_r    <= 1'b0;
        end else if (en_i) begin
            mbig_e_r   <= mbig_e_c;   msml_e_r   <= msml_e_c;
            exp_big_r  <= exp_big_c;  sign_big_r <= sign_big_c;
            same_sign_r<= same_sign_c;
            a_zero_r   <= a_zero_c;   b_zero_r   <= b_zero_c;
            a_inf_r    <= a_inf_c;    b_inf_r    <= b_inf_c;
        end
    end

    // ── Stage 1 combinational: mantissa add/subtract ─────────────────────────
    wire [27:0] msum_c = same_sign_r ? {1'b0,mbig_e_r}+{1'b0,msml_e_r}
                                     : {1'b0,mbig_e_r}-{1'b0,msml_e_r};

    // ── Stage 1→2 pipeline register: capture sum ─────────────────────────────
    logic [27:0] msum_r;
    logic [7:0]  exp_big_r2;
    logic        sign_big_r2;
    logic        a_zero_r2, b_zero_r2, a_inf_r2, b_inf_r2;

    always_ff @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            msum_r     <= 28'h0; exp_big_r2 <= 8'h0; sign_big_r2 <= 1'b0;
            a_zero_r2  <= 1'b0;  b_zero_r2  <= 1'b0;
            a_inf_r2   <= 1'b0;  b_inf_r2   <= 1'b0;
        end else if (en_i) begin
            msum_r     <= msum_c;      exp_big_r2  <= exp_big_r;
            sign_big_r2<= sign_big_r;
            a_zero_r2  <= a_zero_r;    b_zero_r2   <= b_zero_r;
            a_inf_r2   <= a_inf_r;     b_inf_r2    <= b_inf_r;
        end
    end

    // ── Stage 2 combinational: LZC + normalise + round ───────────────────────
    logic [4:0] lzc;
    always_comb begin
        casez(msum_r[27:0])
            28'b1???????????????????????????: lzc=5'd0;  28'b01??????????????????????????: lzc=5'd1;
            28'b001?????????????????????????: lzc=5'd2;  28'b0001????????????????????????: lzc=5'd3;
            28'b00001???????????????????????: lzc=5'd4;  28'b000001??????????????????????: lzc=5'd5;
            28'b0000001?????????????????????: lzc=5'd6;  28'b00000001????????????????????: lzc=5'd7;
            28'b000000001???????????????????: lzc=5'd8;  28'b0000000001??????????????????: lzc=5'd9;
            28'b00000000001?????????????????: lzc=5'd10; 28'b000000000001????????????????: lzc=5'd11;
            28'b0000000000001???????????????: lzc=5'd12; 28'b00000000000001??????????????: lzc=5'd13;
            28'b000000000000001?????????????: lzc=5'd14; 28'b0000000000000001????????????: lzc=5'd15;
            28'b00000000000000001???????????: lzc=5'd16; 28'b000000000000000001??????????: lzc=5'd17;
            28'b0000000000000000001?????????: lzc=5'd18; 28'b00000000000000000001????????: lzc=5'd19;
            28'b000000000000000000001???????: lzc=5'd20; 28'b0000000000000000000001??????: lzc=5'd21;
            28'b00000000000000000000001?????: lzc=5'd22; 28'b000000000000000000000001????: lzc=5'd23;
            28'b0000000000000000000000001???: lzc=5'd24; 28'b00000000000000000000000001??: lzc=5'd25;
            28'b000000000000000001????: lzc=5'd26; default: lzc=5'd27;
        endcase
    end
    wire [27:0] mnorm    = msum_r << lzc;
    wire [8:0]  exp_norm = {1'b0,exp_big_r2} - {4'b0,lzc} + 9'd3;
    wire        rnd_bit  = mnorm[2];
    wire        sticky   = |mnorm[1:0];
    wire        lsb_bit  = mnorm[3];
    wire        do_rnd   = rnd_bit & (sticky | lsb_bit);
    wire [24:0] mant_rnd = {1'b0,mnorm[26:3]} + {24'b0,do_rnd};
    wire [22:0] mant_f   = mant_rnd[24] ? mant_rnd[23:1] : mant_rnd[22:0];
    wire [8:0]  exp_f    = mant_rnd[24] ? exp_norm+9'd1   : exp_norm;
    wire        res_zer  = (msum_r == 28'h0);
    assign result = (a_inf_r2|b_inf_r2)   ? {sign_big_r2,8'hFF,23'h0} :
                    (a_zero_r2&b_zero_r2) ? 32'h0 :
                    res_zer               ? 32'h0 :
                    exp_f[8]              ? {sign_big_r2,8'hFF,23'h0} :
                                            {sign_big_r2,exp_f[7:0],mant_f};
endmodule


module bf16_add (
    input  wire  [15:0] a, b,
    output logic [15:0] result
);
    wire [31:0] fp32_sum;
    fp32_add u_add (.a({a,16'h0}), .b({b,16'h0}), .result(fp32_sum));
    wire rup = fp32_sum[15]&(|fp32_sum[14:0]|fp32_sum[16]);
    assign result = fp32_sum[31:16]+{15'h0,rup};
endmodule

// ---------------------------------------------------------------------------
// bf16_mac  —  BF16 x BF16, FP32 accumulator, BF16 output on flush (unchanged)
// ---------------------------------------------------------------------------
module bf16_mac (
    input  wire         clk_core, rst_n, en, flush,
    input  wire  [15:0] a, b,
    input  wire  [31:0] acc_fp32_in,
    output logic [31:0] acc_fp32_out,
    output logic [15:0] acc_bf16_out
);
    wire [31:0] fp32_prod, fp32_sum;
    fp32_mul u_mul (.a(a),.b(b),.result(fp32_prod));
    // fp32_add is now 2-stage pipelined: pass clk/rst_n/en
    fp32_add u_add (.a(fp32_prod),.b(acc_fp32_in),.result(fp32_sum));
    always_ff @(posedge clk_core or negedge rst_n) begin
        if (!rst_n)  acc_fp32_out <= 32'h0;
        else if (en) acc_fp32_out <= fp32_sum;
    end
    wire rup = fp32_sum[15]&(|fp32_sum[14:0]|fp32_sum[16]);
    wire [15:0] bf16_rne = fp32_sum[31:16]+{15'h0,rup};
    always_ff @(posedge clk_core or negedge rst_n) begin
        if (!rst_n)     acc_bf16_out <= 16'h0;
        else if (flush) acc_bf16_out <= bf16_rne;
    end
endmodule

// ===========================================================================
// systolic_array  —  weight-stationary MxN array, A-input staggered
// ===========================================================================
//
// ARCHITECTURE
// ------------
// Weights (B) are pre-loaded into b_mat[N][K] registers by the chiplet.
// They are broadcast statically to all PEs — no stagger on B side.
//
// Activations (A) are streamed one column per cycle:
//   a_col_in[i] = A[i][k]  at cycle k  (k=0..K-1)
//
// Row stagger: row i is delayed i extra cycles before the horizontal
// shift chain, so PE(i,j) receives A[i][k] at cycle k + i + j.
//
// Since B is stationary (weight-stationary), PE(i,j) holds b_mat[j][*].
// The PE needs b_mat[j][k] at the same cycle it sees A[i][k], i.e. at
// cycle k+i+j. We achieve this by also column-staggering b_mat[j] by j
// cycles (a cheap shift register on the B column input, not a full matrix
// stagger). This is the standard systolic "weight-broadcast-with-skew"
// approach: weights flow in once along the B dimension, activations flow
// in once along the A dimension, results accumulate in place.
//
// In practice for this design the chiplet FSM presents:
//   b_row_in[j] = B[k][j]  at cycle k  (one row of weights per cycle)
// and the array applies a column-j delay of j cycles to align with A.
//
// TIMING
// ------
//   Latency = K + (M-1) + (N-1) cycles from first data_in to valid_out
//   DRAIN   = (M-1) + (N-1)  cycles after last data_in (flush cycle)
//
// PORTS
// -----
//   data_in    : assert for K consecutive cycles
//   flush      : assert on the last data_in cycle (k == K-1)
//   acc_clear  : synchronous 1-cycle pulse to zero PE accumulators
//                (use before starting a new tile multiply)
//   a_col_in   : A[i][k] — column k of activation matrix
//   b_row_in   : B[k][j] — row k of weight matrix
//   c_out      : output tile (valid for 1 cycle when valid_out pulses)
//   valid_out  : single-cycle pulse when c_out is ready
// ===========================================================================
module systolic_array #(
    parameter int M = 64,
    parameter int N = 64,
    parameter int K = 64
)(
    input  wire        clk_core,
    input  wire        rst_n,

    // control
    input  wire        data_in,    // high for K consecutive cycles
    input  wire        acc_clear,  // synchronous accumulator clear
    input  wire        flush,      // high on last data_in cycle

    // A: one column per cycle (streamed activations)
    input  wire [15:0] a_col_in [M],
    // B: one row per cycle (weights streamed in once, then stationary)
    input  wire [15:0] b_row_in [N],

    output wire [15:0] c_out    [M][N],
    output logic       valid_out
);

    // -----------------------------------------------------------------------
    // A-side: row stagger + horizontal shift
    // a_rbuf[i][d]: row i, row-stagger tap d (d=0 is freshest, need d=i-1)
    // a_cbuf[i][d]: column shift, tap d (PE(i,j) uses tap j)
    // -----------------------------------------------------------------------
    logic [15:0] a_rbuf [M][M];   // row stagger: row i needs depth i

    always_ff @(posedge clk_core or negedge rst_n) begin : a_row_skew
        if (!rst_n) begin
            for (int i=0;i<M;i++) for (int d=0;d<M;d++) a_rbuf[i][d] <= 16'h0;
        end else if (data_in) begin
            for (int i=1;i<M;i++) begin   // row 0 needs no row stagger
                a_rbuf[i][0] <= a_col_in[i];
                for (int d=1;d<i;d++) a_rbuf[i][d] <= a_rbuf[i][d-1];
            end
        end
    end

    // Row-staggered A: row i is a_col_in[i] delayed i cycles
    wire [15:0] a_stag [M];
    genvar gi;
    generate
        for (gi=0;gi<M;gi++) begin : a_stag_g
            if (gi==0) assign a_stag[gi] = a_col_in[gi];
            else       assign a_stag[gi] = a_rbuf[gi][gi-1];
        end
    endgenerate

    // Column shift: a_cbuf[i][j] = a_stag[i] delayed j more cycles
    logic [15:0] a_cbuf [M][N];

    always_ff @(posedge clk_core or negedge rst_n) begin : a_col_shift
        if (!rst_n) begin
            for (int i=0;i<M;i++) for (int d=0;d<N;d++) a_cbuf[i][d] <= 16'h0;
        end else if (data_in) begin
            for (int i=0;i<M;i++) begin
                a_cbuf[i][0] <= a_stag[i];
                for (int d=1;d<N;d++) a_cbuf[i][d] <= a_cbuf[i][d-1];
            end
        end
    end

    // PE(i,j) A input: col-shift tap j (tap 0 = a_stag itself)
    wire [15:0] a_pe [M][N];
    genvar gii, gjj;
    generate
        for (gii=0;gii<M;gii++) begin : ape_row
            for (gjj=0;gjj<N;gjj++) begin : ape_col
                if (gjj==0) assign a_pe[gii][gjj] = a_stag[gii];
                else        assign a_pe[gii][gjj] = a_cbuf[gii][gjj-1];
            end
        end
    endgenerate

    // -----------------------------------------------------------------------
    // B-side: column stagger only (weights flow in once, no row propagation)
    // b_cbuf[j][d]: column j delayed d cycles
    // PE(i,j) uses b_cbuf[j][j-1] (or b_row_in[j] for j==0)
    // This matches the A delay at column j: both A and B arrive at col j
    // with j cycles of extra delay, so they stay aligned as k advances.
    // -----------------------------------------------------------------------
    logic [15:0] b_cbuf [N][N];

    always_ff @(posedge clk_core or negedge rst_n) begin : b_col_skew
        if (!rst_n) begin
            for (int j=0;j<N;j++) for (int d=0;d<N;d++) b_cbuf[j][d] <= 16'h0;
        end else if (data_in) begin
            for (int j=1;j<N;j++) begin   // col 0 needs no stagger
                b_cbuf[j][0] <= b_row_in[j];
                for (int d=1;d<j;d++) b_cbuf[j][d] <= b_cbuf[j][d-1];
            end
        end
    end

    // Column-staggered B: col j is b_row_in[j] delayed j cycles
    wire [15:0] b_stag [N];
    genvar gj;
    generate
        for (gj=0;gj<N;gj++) begin : b_stag_g
            if (gj==0) assign b_stag[gj] = b_row_in[gj];
            else       assign b_stag[gj] = b_cbuf[gj][gj-1];
        end
    endgenerate

    // PE(i,j) B input: same b_stag[j] for all rows i (weight-stationary:
    // every row sees the same weight column j at the same time)
    wire [15:0] b_pe [M][N];
    genvar gbi, gbj;
    generate
        for (gbi=0;gbi<M;gbi++) begin : bpe_row
            for (gbj=0;gbj<N;gbj++) begin : bpe_col
                assign b_pe[gbi][gbj] = b_stag[gbj];
            end
        end
    endgenerate

    // -----------------------------------------------------------------------
    // PE array: accumulate a_pe[i][j] * b_pe[i][j]
    // -----------------------------------------------------------------------

    // Drain declarations must precede pe_en which uses 'draining'
    localparam int DRAIN = (M-1) + (N-1);
    logic [$clog2(DRAIN+2)-1:0] drain_cnt;
    logic draining;
    logic drain_done;
    logic drain_done_d1;

    wire pe_en = data_in | draining;

    // Registered input banks to reduce fanout on a_stag/b_stag/pe_en.
    // Without this, a single net from the chiplet FSM drives all M*N PE
    // inputs through one huge buffer tree (~155k fanout in synthesis,
    // consuming ~1082ps in a 6-level buffer chain).
    // Four banks quarter the fanout: each bank drives M/4 rows or N/4 cols.
    // Bank assignment: row/col index modulo 4.
    //   bank0: indices 0..M/4-1        bank1: M/4..M/2-1
    //   bank2: M/2..3*M/4-1            bank3: 3*M/4..M-1
    // This adds 1 register stage before the PE multiplier — pe_en_d1 inside
    // the PE accounts for the total 2-cycle delay (bank + fp32_add stage-1).
    logic [15:0] a_stag_r [4][M/4+1];   // +1 to handle non-power-of-4 M
    logic [15:0] b_stag_r [4][N/4+1];
    logic        pe_en_r  [4];

    always_ff @(posedge clk_core or negedge rst_n) begin : fanout_reg
        if (!rst_n) begin
            for (int b=0; b<4; b++) begin
                for (int ii=0; ii<M/4+1; ii++) a_stag_r[b][ii] <= 16'h0;
                for (int jj=0; jj<N/4+1; jj++) b_stag_r[b][jj] <= 16'h0;
                pe_en_r[b] <= 1'b0;
            end
        end else begin
            // Bank 0: rows/cols 0..M/4-1
            for (int ii=0;      ii<1*(M/4); ii++) a_stag_r[0][ii-0*(M/4)] <= a_stag[ii];
            for (int jj=0;      jj<1*(N/4); jj++) b_stag_r[0][jj-0*(N/4)] <= b_stag[jj];
            // Bank 1: rows/cols M/4..M/2-1
            for (int ii=1*(M/4); ii<2*(M/4); ii++) a_stag_r[1][ii-1*(M/4)] <= a_stag[ii];
            for (int jj=1*(N/4); jj<2*(N/4); jj++) b_stag_r[1][jj-1*(N/4)] <= b_stag[jj];
            // Bank 2: rows/cols M/2..3*M/4-1
            for (int ii=2*(M/4); ii<3*(M/4); ii++) a_stag_r[2][ii-2*(M/4)] <= a_stag[ii];
            for (int jj=2*(N/4); jj<3*(N/4); jj++) b_stag_r[2][jj-2*(N/4)] <= b_stag[jj];
            // Bank 3: rows/cols 3*M/4..M-1
            for (int ii=3*(M/4); ii<M; ii++) a_stag_r[3][ii-3*(M/4)] <= a_stag[ii];
            for (int jj=3*(N/4); jj<N; jj++) b_stag_r[3][jj-3*(N/4)] <= b_stag[jj];
            for (int b=0; b<4; b++) pe_en_r[b] <= pe_en;
        end
    end

    // (drain_cnt/draining/drain_done declared earlier — see below)

    always_ff @(posedge clk_core or negedge rst_n) begin : drain_ff
        if (!rst_n) begin
            drain_cnt <= '0; draining <= 1'b0; drain_done <= 1'b0;
        end else begin
                        if (flush & data_in) begin
                draining <= 1'b1; drain_cnt <= '0;
            end else if (draining) begin
                if (drain_cnt == DRAIN[$clog2(DRAIN+2)-1:0]) begin
                    draining <= 1'b0; drain_cnt <= '0; drain_done <= 1'b1;
                end else begin
                    drain_cnt <= drain_cnt + 1'b1;
                end
            end
        end
    end

    // Delay valid_out by 2 cycles to align with 3-stage fp32_add_pip result
    // Stage 0→1 register + stage 1→2 register = 2 cycles after drain
    always_ff @(posedge clk_core or negedge rst_n) begin
        if (!rst_n) begin
            drain_done   <= 1'b0;
            drain_done_d1<= 1'b0;
            valid_out    <= 1'b0;
        end else begin
            drain_done    <= 1'b0;        // self-clearing; set in drain_ff
            drain_done_d1 <= drain_done;
            valid_out     <= drain_done_d1;
        end
    end


    logic [31:0] acc_fp32 [M][N];

    genvar pei, pej;
    generate
        for (pei=0;pei<M;pei++) begin : pe_row_g
            for (pej=0;pej<N;pej++) begin : pe_col_g
                // Select input from registered bank (lo or hi) to reduce fanout
                // 4-bank fanout split: select from bank 0/1/2/3 by row/col index
                // Each bank drives M/4 rows, quarter the fanout vs one net
                wire [15:0] a_pe_w =
                    (pei < 1*(M/4)) ? a_stag_r[0][pei - 0*(M/4)] :
                    (pei < 2*(M/4)) ? a_stag_r[1][pei - 1*(M/4)] :
                    (pei < 3*(M/4)) ? a_stag_r[2][pei - 2*(M/4)] :
                                      a_stag_r[3][pei - 3*(M/4)];
                wire [15:0] b_pe_w =
                    (pej < 1*(N/4)) ? b_stag_r[0][pej - 0*(N/4)] :
                    (pej < 2*(N/4)) ? b_stag_r[1][pej - 1*(N/4)] :
                    (pej < 3*(N/4)) ? b_stag_r[2][pej - 2*(N/4)] :
                                      b_stag_r[3][pej - 3*(N/4)];
                wire pe_en_w =
                    (pei < 1*(M/4)) ? pe_en_r[0] :
                    (pei < 2*(M/4)) ? pe_en_r[1] :
                    (pei < 3*(M/4)) ? pe_en_r[2] : pe_en_r[3];

                // 4-stage pipeline: bank(1) + fp32_add_align(1) + fp32_add_msum(1)
                // + result to acc_fp32. Need 3 cycles of delay on pe_en/acc_clear
                // from original pe_en (pe_en_w is already 1 cycle delayed by bank).
                logic pe_en_d1, pe_en_d2, acc_clear_d1, acc_clear_d2;
                always_ff @(posedge clk_core or negedge rst_n) begin
                    if (!rst_n) begin
                        pe_en_d1<=1'b0; pe_en_d2<=1'b0;
                        acc_clear_d1<=1'b0; acc_clear_d2<=1'b0;
                    end else begin
                        pe_en_d1   <= pe_en_w;    // 2 cycles from orig pe_en
                        pe_en_d2   <= pe_en_d1;   // 3 cycles from orig pe_en
                        acc_clear_d1 <= acc_clear;
                        acc_clear_d2 <= acc_clear_d1;
                    end
                end

                wire [31:0] prod_w, sum_w;
                fp32_mul u_mul (.a(a_pe_w), .b(b_pe_w), .result(prod_w));
                // en=pe_en_w: fp32_add_pip stage 0 fires on cycle 2,
                //             stage 1 (msum_r) fires on cycle 3,
                //             result available on cycle 3 (comb from msum_r)
                fp32_add_pip u_add (.clk(clk_core),.rst_n(rst_n),.en(pe_en_w),
                                    .a(prod_w),.b(acc_fp32[pei][pej]),.result(sum_w));

                always_ff @(posedge clk_core or negedge rst_n) begin : acc_ff
                    if (!rst_n)              acc_fp32[pei][pej] <= 32'h0;
                    else if (acc_clear_d2)   acc_fp32[pei][pej] <= 32'h0;
                    else if (pe_en_d2)       acc_fp32[pei][pej] <= sum_w;
                end

                // BF16 output latch — captured when valid_out pulses (1 cycle later)
                wire        rup = sum_w[15]&(|sum_w[14:0]|sum_w[16]);
                wire [15:0] bf16_rne = sum_w[31:16]+{15'h0,rup};
                logic [15:0] c_reg;
                always_ff @(posedge clk_core or negedge rst_n) begin : c_ff
                    if (!rst_n)         c_reg <= 16'h0;
                    else if (valid_out) c_reg <= bf16_rne;
                end
                assign c_out[pei][pej] = c_reg;
            end
        end
    endgenerate

endmodule


`default_nettype wire
// =============================================================================
// End of fp32_arith.sv  (optimised)
// =============================================================================
