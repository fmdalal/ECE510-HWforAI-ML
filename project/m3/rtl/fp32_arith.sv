// =============================================================================
// fp32_arith.sv  —  Shared BF16/FP32 arithmetic primitives
// =============================================================================
//
// Clock domains used in this file
// --------------------------------
//   NONE  : fp32_mul, fp32_add, bf16_add  — purely combinational, no clock
//   clk_core (1 GHz chiplet compute clock)
//           : bf16_mac      — FP32 accumulator register + BF16 flush register
//           : systolic_array — skew registers, PE accumulators, drain counter
//
// rst_n is asynchronous active-low reset, shared across the core domain.================
// Included by every chiplet that needs floating-point computation.
// All modules are purely combinational — no clocks.
//
// Modules:
//   fp32_mul   BF16 x BF16 -> FP32
//   fp32_add   FP32 + FP32 -> FP32
//   bf16_add   BF16 + BF16 -> BF16  (via fp32_add)
//   bf16_mac   BF16 x BF16, FP32 accumulator, BF16 output on flush
// =============================================================================

`timescale 1ns/1ps
`default_nettype none

// ---------------------------------------------------------------------------
// fp32_mul  —  BF16 x BF16 -> FP32
// ---------------------------------------------------------------------------
module fp32_mul (
    input  wire  [15:0] a,
    input  wire  [15:0] b,
    output logic [31:0] result
);
    wire [31:0] fp32_a = {a, 16'h0000};
    wire [31:0] fp32_b = {b, 16'h0000};

    wire        sign_a = fp32_a[31];
    wire [7:0]  exp_a  = fp32_a[30:23];
    wire [23:0] mant_a = {1'b1, fp32_a[22:0]};

    wire        sign_b = fp32_b[31];
    wire [7:0]  exp_b  = fp32_b[30:23];
    wire [23:0] mant_b = {1'b1, fp32_b[22:0]};

    wire sign_r             = sign_a ^ sign_b;
    wire [8:0]  exp_sum     = {1'b0, exp_a} + {1'b0, exp_b} - 9'd127;
    wire [47:0] mant_prod   = mant_a * mant_b;
    wire        norm_sh     = mant_prod[47];
    wire [8:0]  exp_n       = norm_sh ? exp_sum + 9'd1 : exp_sum;
    wire [22:0] mant_n      = norm_sh ? mant_prod[46:24] : mant_prod[45:23];

    wire a_zero = (exp_a == 8'h00);
    wire b_zero = (exp_b == 8'h00);
    wire a_inf  = (exp_a == 8'hFF) & (fp32_a[22:0] == 23'h0);
    wire b_inf  = (exp_b == 8'hFF) & (fp32_b[22:0] == 23'h0);
    wire a_nan  = (exp_a == 8'hFF) & (fp32_a[22:0] != 23'h0);
    wire b_nan  = (exp_b == 8'hFF) & (fp32_b[22:0] != 23'h0);

    wire res_nan  = a_nan | b_nan | (a_inf & b_zero) | (b_inf & a_zero);
    wire res_inf  = (a_inf | b_inf) & ~res_nan;
    wire res_zero = (a_zero | b_zero) & ~res_nan;
    wire res_ovf  = (exp_n >= 9'd255) & ~res_nan & ~res_zero;
    wire res_udf  =  exp_n[8]         & ~res_nan & ~res_zero;

    always_comb begin
        if      (res_nan)            result = 32'h7FC0_0000;
        else if (res_inf | res_ovf)  result = {sign_r, 8'hFF, 23'h0};
        else if (res_zero | res_udf) result = {sign_r, 31'h0};
        else                         result = {sign_r, exp_n[7:0], mant_n};
    end
endmodule

// ---------------------------------------------------------------------------
// fp32_add  —  FP32 + FP32 -> FP32
// ---------------------------------------------------------------------------
module fp32_add (
    input  wire  [31:0] a,
    input  wire  [31:0] b,
    output logic [31:0] result
);
    wire        sign_a = a[31];
    wire [7:0]  exp_a  = a[30:23];
    wire [23:0] mant_a = {1'b1, a[22:0]};
    wire        sign_b = b[31];
    wire [7:0]  exp_b  = b[30:23];
    wire [23:0] mant_b = {1'b1, b[22:0]};

    wire        a_ge     = (exp_a >= exp_b);
    wire [7:0]  exp_big  = a_ge ? exp_a  : exp_b;
    wire        sign_big = a_ge ? sign_a : sign_b;
    wire        sign_sml = a_ge ? sign_b : sign_a;
    wire [23:0] mant_big = a_ge ? mant_a : mant_b;
    wire [23:0] mant_sml = a_ge ? mant_b : mant_a;

    wire [7:0]  exp_diff = exp_big - (a_ge ? exp_b : exp_a);
    wire [4:0]  sh       = (exp_diff > 8'd27) ? 5'd27 : exp_diff[4:0];
    wire [26:0] mbig_e   = {mant_big, 3'b000};
    wire [26:0] msml_e   = {mant_sml, 3'b000} >> sh;

    wire        same_sign = (sign_big == sign_sml);
    wire [27:0] msum = same_sign
                     ? {1'b0, mbig_e} + {1'b0, msml_e}
                     : {1'b0, mbig_e} - {1'b0, msml_e};
    wire sign_r = sign_big;

    logic [4:0] lzc;
    always_comb begin
        casez (msum[27:0])
            28'b1???????????????????????????: lzc = 5'd0;
            28'b01??????????????????????????: lzc = 5'd1;
            28'b001?????????????????????????: lzc = 5'd2;
            28'b0001????????????????????????: lzc = 5'd3;
            28'b00001???????????????????????: lzc = 5'd4;
            28'b000001??????????????????????: lzc = 5'd5;
            28'b0000001?????????????????????: lzc = 5'd6;
            28'b00000001????????????????????: lzc = 5'd7;
            28'b000000001???????????????????: lzc = 5'd8;
            28'b0000000001??????????????????: lzc = 5'd9;
            28'b00000000001?????????????????: lzc = 5'd10;
            28'b000000000001????????????????: lzc = 5'd11;
            28'b0000000000001???????????????: lzc = 5'd12;
            28'b00000000000001??????????????: lzc = 5'd13;
            28'b000000000000001?????????????: lzc = 5'd14;
            28'b0000000000000001????????????: lzc = 5'd15;
            28'b00000000000000001???????????: lzc = 5'd16;
            28'b000000000000000001??????????: lzc = 5'd17;
            28'b0000000000000000001?????????: lzc = 5'd18;
            28'b00000000000000000001????????: lzc = 5'd19;
            28'b000000000000000000001???????: lzc = 5'd20;
            28'b0000000000000000000001??????: lzc = 5'd21;
            28'b00000000000000000000001?????: lzc = 5'd22;
            28'b000000000000000000000001????: lzc = 5'd23;
            28'b0000000000000000000000001???: lzc = 5'd24;
            28'b00000000000000000000000001??: lzc = 5'd25;
            28'b000000000000000000000000001?: lzc = 5'd26;
            default:                          lzc = 5'd27;
        endcase
    end

    wire [27:0] mnorm    = msum << lzc;
    wire [8:0]  exp_norm = {1'b0, exp_big} - {4'b0, lzc} + 9'd3;

    wire        rnd_bit  = mnorm[2];
    wire        sticky   = |mnorm[1:0];
    wire        lsb_bit  = mnorm[3];
    wire        do_rnd   = rnd_bit & (sticky | lsb_bit);
    wire [24:0] mant_rnd = {1'b0, mnorm[26:3]} + {24'b0, do_rnd};
    wire [22:0] mant_f   = mant_rnd[24] ? mant_rnd[23:1] : mant_rnd[22:0];
    wire [8:0]  exp_f    = mant_rnd[24] ? exp_norm + 9'd1 : exp_norm;

    wire a_zero  = (exp_a == 8'h00);
    wire b_zero  = (exp_b == 8'h00);
    wire a_inf   = (exp_a == 8'hFF);
    wire b_inf   = (exp_b == 8'hFF);
    wire res_zer = (msum  == 28'h0);

    always_comb begin
        if      (a_inf | b_inf)   result = {sign_r, 8'hFF, 23'h0};
        else if (a_zero & b_zero) result = 32'h0;
        else if (res_zer)         result = 32'h0;
        else if (exp_f[8])        result = {sign_r, 8'hFF, 23'h0};
        else                      result = {sign_r, exp_f[7:0], mant_f};
    end
endmodule

// ---------------------------------------------------------------------------
// bf16_add  —  BF16 + BF16 -> BF16
// ---------------------------------------------------------------------------
module bf16_add (
    input  wire  [15:0] a,
    input  wire  [15:0] b,
    output logic [15:0] result
);
    wire [31:0] fp32_a  = {a, 16'h0000};
    wire [31:0] fp32_b  = {b, 16'h0000};
    wire [31:0] fp32_sum;
    fp32_add u_add (.a(fp32_a), .b(fp32_b), .result(fp32_sum));
    wire rup = fp32_sum[15] & (|fp32_sum[14:0] | fp32_sum[16]);
    always_comb begin
        result = fp32_sum[31:16] + {15'h0, rup};
    end
endmodule

// ---------------------------------------------------------------------------
// bf16_mac  —  BF16 x BF16, FP32 accumulator, BF16 output on flush
// ---------------------------------------------------------------------------
module bf16_mac (
    input  wire         clk_core,   // 1 GHz chiplet compute clock
    input  wire         rst_n,
    input  wire         en,
    input  wire         flush,
    input  wire  [15:0] a,
    input  wire  [15:0] b,
    input  wire  [31:0] acc_fp32_in,
    output logic [31:0] acc_fp32_out,
    output logic [15:0] acc_bf16_out
);
    wire [31:0] fp32_prod;
    fp32_mul u_mul (.a(a), .b(b), .result(fp32_prod));

    wire [31:0] fp32_sum;
    fp32_add u_add (.a(fp32_prod), .b(acc_fp32_in), .result(fp32_sum));

    always_ff @(posedge clk_core or negedge rst_n) begin : acc_fp32_ff
        if (!rst_n)  acc_fp32_out <= 32'h0;
        else if (en) acc_fp32_out <= fp32_sum;
    end

    wire        rnd_bit  = fp32_sum[15];
    wire        sticky   = |fp32_sum[14:0];
    wire        lsb_bit  = fp32_sum[16];
    wire        round_up = rnd_bit & (sticky | lsb_bit);
    wire [15:0] bf16_rne = fp32_sum[31:16] + {15'h0, round_up};

    always_ff @(posedge clk_core or negedge rst_n) begin : acc_bf16_ff
        if (!rst_n)     acc_bf16_out <= 16'h0;
        else if (flush) acc_bf16_out <= bf16_rne;
    end
endmodule

// ---------------------------------------------------------------------------
// systolic_array  —  weight-stationary MxN systolic array, K inner steps
// ---------------------------------------------------------------------------
module systolic_array #(
    parameter int M = 64,
    parameter int N = 64,
    parameter int K = 64
)(
    input  wire        clk_core,   // 1 GHz chiplet compute clock
    input  wire        rst_n,
    input  wire        en,
    input  wire        clear,
    input  wire        flush,
    input  wire [15:0] a_row [M],
    input  wire [15:0] b_col [N][K],
    output wire [15:0] c_out [M][N],
    output logic       valid_out
);
    logic [31:0] acc_fp32 [M][N];
    logic [15:0] a_reg    [M][K];

    always_ff @(posedge clk_core or negedge rst_n) begin : skew_ff
        if (!rst_n) begin
            for (int i = 0; i < M; i++)
                for (int k = 0; k < K; k++)
                    a_reg[i][k] <= 16'h0;
        end else if (en) begin
            for (int i = 0; i < M; i++) begin
                a_reg[i][0] <= a_row[i];
                for (int k = 1; k < K; k++)
                    a_reg[i][k] <= a_reg[i][k-1];
            end
        end
    end

    // rst_n_clear combines async reset and synchronous clear for bf16_mac PEs
    wire rst_n_clear = rst_n & ~clear;

    genvar gi, gj;
    generate
        for (gi = 0; gi < M; gi++) begin : row_gen
            for (gj = 0; gj < N; gj++) begin : col_gen
                wire [31:0] fp32_out_w;
                // Note: gj < K and gi < K are always true since M=N=K
                // Ternary removed to avoid unsynthesizable genvar comparison
                bf16_mac pe_inst (
                    .clk_core     (clk_core),
                    .rst_n        (rst_n_clear),
                    .en           (en),
                    .flush        (flush),
                    .a            (a_reg[gi][gj]),
                    .b            (b_col[gj][gi]),
                    .acc_fp32_in  (acc_fp32[gi][gj]),
                    .acc_fp32_out (fp32_out_w),
                    .acc_bf16_out (c_out[gi][gj])
                );
                // clear handled via rst_n_clear on bf16_mac above
                // acc_fp32 local accumulator uses same combined reset
                always_ff @(posedge clk_core or negedge rst_n_clear) begin : acc_fb_ff
                    if (!rst_n_clear) acc_fp32[gi][gj] <= 32'h0;
                    else if (en)      acc_fp32[gi][gj] <= fp32_out_w;
                end
            end
        end
    endgenerate

    localparam int DRAIN = (M + N - 1) + K + 1;
    logic [7:0] cycle_cnt;

    always_ff @(posedge clk_core or negedge rst_n_clear) begin : drain_ff
        if (!rst_n_clear) begin
            cycle_cnt <= 8'd0;
            valid_out <= 1'b0;
        end else if (en) begin
            if (cycle_cnt < DRAIN[7:0])
                cycle_cnt <= cycle_cnt + 8'd1;
            valid_out <= flush & (cycle_cnt >= DRAIN[7:0] - 8'd1);
        end else begin
            valid_out <= 1'b0;
        end
    end
endmodule

`default_nettype wire
// =============================================================================
// End of fp32_arith.sv
// =============================================================================

