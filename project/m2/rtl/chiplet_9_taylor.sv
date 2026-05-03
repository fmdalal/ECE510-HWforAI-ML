// =============================================================================
// chiplet_9_taylor.sv  —  Chiplet ID 9 (Taylor softmax replacement)
// REVISED: fully pipelined vector architecture, no FSM per row
// =============================================================================
// Pipeline depth = LOG2_N + 3 Horner stages + 2 NR stages + 1 norm stage
// Throughput: 1 row per clock after pipeline fills
// =============================================================================
//
// Clock domains used in this file
// --------------------------------
//   clk_core (1 GHz chiplet compute clock)
//           : ALL registered logic in this module
//           : max reduction tree pipeline registers (LOG2_N levels)
//           : skew delay registers (LOG2_N stages)
//           : Horner chain stage registers (3 stages)
//           : sum reduction tree pipeline registers (LOG2_N levels)
//           : Newton-Raphson reciprocal registers (2 iterations)
//           : output tile assembly register
//           : ucie_rx / ucie_tx FDI instances (one per head)

`timescale 1ns/1ps
`default_nettype none

module chiplet_9_taylor #(
    parameter int NUM_HEADS = 8,
    parameter int TILE      = 64,
    parameter int SEQ_LEN   = 64,
    parameter int LOG2_N    = 6    // log2(SEQ_LEN)
)(
    input  wire        clk_core,   // 1 GHz chiplet compute clock
    input  wire        rst_n,
    input  wire        cfg_start,
    output logic       cfg_done,

    input  wire [511:0] rx_bump_data  [NUM_HEADS],
    input  wire         rx_bump_valid [NUM_HEADS],
    output wire         rx_bump_credit[NUM_HEADS],

    output wire [511:0] tx_bump_data  [NUM_HEADS],
    output wire         tx_bump_valid [NUM_HEADS],
    input  wire         tx_bump_credit[NUM_HEADS]
);
    // BF16 constants
    localparam logic [15:0] ONE     = 16'h3F80;
    localparam logic [15:0] HALF    = 16'h3F00;
    localparam logic [15:0] ONE_6TH = 16'h3E2B;

    // -----------------------------------------------------------------------
    // UCIe RX per head
    // -----------------------------------------------------------------------
    logic        rx_valid [NUM_HEADS];
    logic [15:0] rx_tile  [NUM_HEADS][TILE][TILE];

    genvar gh;
    generate
        for (gh = 0; gh < NUM_HEADS; gh++) begin : rx_gen
            ucie_rx #(.TILE_DIM(TILE)) u_rx (
                .clk_core(clk_core), .rst_n(rst_n),
                .bump_data(rx_bump_data[gh]),
                .bump_valid(rx_bump_valid[gh]),
                .bump_credit(rx_bump_credit[gh]),
                .rx_valid(rx_valid[gh]),
                .rx_src_id(), .rx_tile(rx_tile[gh]),
                .rx_ready(1'b1),
                .rx_crc_err(), .rx_seq_err()
            );
        end
    endgenerate

    // -----------------------------------------------------------------------
    // Process one head at a time: head_idx cycles 0..7
    // For each head, pipe all TILE rows through the vector pipeline
    // -----------------------------------------------------------------------
    logic [2:0]  head_idx;
    logic [5:0]  row_idx;
    logic        pipe_valid;
    logic [15:0] pipe_scores [SEQ_LEN];   // current row being processed

    // Feed rows into the pipeline
    always_ff @(posedge clk_core or negedge rst_n) begin : row_feed_ff
        if (!rst_n) begin
            head_idx   <= '0;
            row_idx    <= '0;
            pipe_valid <= 1'b0;
            cfg_done   <= 1'b0;
        end else begin
            cfg_done   <= 1'b0;
            pipe_valid <= 1'b0;

            if (cfg_start && rx_valid[head_idx]) begin
                for (int j = 0; j < SEQ_LEN; j++)
                    pipe_scores[j] <= rx_tile[head_idx][row_idx][j];
                pipe_valid <= 1'b1;

                if (row_idx == TILE - 1) begin
                    row_idx <= '0;
                    if (head_idx == NUM_HEADS - 1) begin
                        head_idx <= '0;
                        cfg_done <= 1'b1;
                    end else begin
                        head_idx <= head_idx + 3'd1;
                    end
                end else begin
                    row_idx <= row_idx + 6'd1;
                end
            end
        end
    end

    // -----------------------------------------------------------------------
    // Stage 1: max reduction tree (LOG2_N levels of registered comparators)
    // Binary tree: each level halves the number of elements
    // -----------------------------------------------------------------------
    // Level 0: SEQ_LEN/2 comparators
    // Level 1: SEQ_LEN/4 comparators
    // ...
    // Level LOG2_N-1: 1 comparator -> scalar max
    // -----------------------------------------------------------------------
    logic [15:0] max_tree [LOG2_N+1][SEQ_LEN];
    logic        max_valid [LOG2_N+1];

    // Load input row into level 0 of tree
    always_ff @(posedge clk_core or negedge rst_n) begin : max_l0_ff
        if (!rst_n) begin
            max_valid[0] <= 1'b0;
            for (int j = 0; j < SEQ_LEN; j++) max_tree[0][j] <= 16'h0;
        end else begin
            max_valid[0] <= pipe_valid;
            if (pipe_valid)
                for (int j = 0; j < SEQ_LEN; j++)
                    max_tree[0][j] <= pipe_scores[j];
        end
    end

    // Reduction levels
    genvar gl;
    generate
        for (gl = 0; gl < LOG2_N; gl++) begin : max_level
            always_ff @(posedge clk_core or negedge rst_n) begin : max_lev_ff
                if (!rst_n) begin
                    max_valid[gl+1] <= 1'b0;
                    for (int j = 0; j < SEQ_LEN; j++)
                        max_tree[gl+1][j] <= 16'h0;
                end else begin
                    max_valid[gl+1] <= max_valid[gl];
                    for (int j = 0; j < SEQ_LEN >> (gl+1); j++) begin
                        // Signed BF16 compare: pick the larger of pair
                        if ($signed(max_tree[gl][j*2]) >=
                            $signed(max_tree[gl][j*2+1]))
                            max_tree[gl+1][j] <= max_tree[gl][j*2];
                        else
                            max_tree[gl+1][j] <= max_tree[gl][j*2+1];
                    end
                end
            end
        end
    endgenerate

    wire [15:0] row_max   = max_tree[LOG2_N][0];
    wire        max_rdy   = max_valid[LOG2_N];

    // -----------------------------------------------------------------------
    // Align input scores with the max result (pipeline skew compensation)
    // The max takes LOG2_N cycles; delay the original scores to match
    // -----------------------------------------------------------------------
    logic [15:0] scores_dly [LOG2_N][SEQ_LEN];
    logic        valid_dly  [LOG2_N];

    always_ff @(posedge clk_core or negedge rst_n) begin : skew_ff
        if (!rst_n) begin
            for (int l = 0; l < LOG2_N; l++) begin
                valid_dly[l] <= 1'b0;
                for (int j = 0; j < SEQ_LEN; j++)
                    scores_dly[l][j] <= 16'h0;
            end
        end else begin
            valid_dly[0] <= pipe_valid;
            if (pipe_valid)
                for (int j = 0; j < SEQ_LEN; j++)
                    scores_dly[0][j] <= pipe_scores[j];
            for (int l = 1; l < LOG2_N; l++) begin
                valid_dly[l] <= valid_dly[l-1];
                for (int j = 0; j < SEQ_LEN; j++)
                    scores_dly[l][j] <= scores_dly[l-1][j];
            end
        end
    end

    // -----------------------------------------------------------------------
    // Stage 2: subtract max + clip  (element-wise, N parallel bf16_add)
    // -----------------------------------------------------------------------
    wire [15:0] neg_max    = {~row_max[15], row_max[14:0]};
    wire [15:0] shifted_w  [SEQ_LEN];
    wire [15:0] clipped_w  [SEQ_LEN];

    genvar gsi;
    generate
        for (gsi = 0; gsi < SEQ_LEN; gsi++) begin : shift_clip_gen
            bf16_add u_sh (
                .a(scores_dly[LOG2_N-1][gsi]),
                .b(neg_max),
                .result(shifted_w[gsi])
            );
            // Clip: if > 0 -> 0;  if < -8 (BF16 0xC100) -> 0xC100
            wire over  = ~shifted_w[gsi][15] & (shifted_w[gsi] != 16'h0);
            wire under = shifted_w[gsi][15] &
                         ($signed(shifted_w[gsi]) < $signed(16'hC100));
            assign clipped_w[gsi] = over  ? 16'h0000 :
                                    under ? 16'hC100  :
                                            shifted_w[gsi];
        end
    endgenerate

    logic [15:0] clipped_r  [SEQ_LEN];
    logic        clipped_vld;

    always_ff @(posedge clk_core or negedge rst_n) begin : clip_ff
        if (!rst_n) begin
            clipped_vld <= 1'b0;
            for (int j = 0; j < SEQ_LEN; j++) clipped_r[j] <= 16'h0;
        end else begin
            clipped_vld <= max_rdy;
            if (max_rdy)
                for (int j = 0; j < SEQ_LEN; j++)
                    clipped_r[j] <= clipped_w[j];
        end
    end

    // -----------------------------------------------------------------------
    // Stages 3-5: Horner chain — 3 registered MAC stages
    // Each stage: acc[i] = clipped[i] * acc[i] + const
    // All SEQ_LEN elements computed in parallel
    // -----------------------------------------------------------------------
    logic [15:0] h_acc [3][SEQ_LEN];
    logic        h_vld [3];

    // Stage 3 (init): acc = clipped * (1/6) + 0 = 1/6 * x
    genvar ghi;
    generate
        for (ghi = 0; ghi < SEQ_LEN; ghi++) begin : h0_gen
            wire [31:0] h0_fp32_nc;
            bf16_mac h0_mac (
                .clk_core(clk_core), .rst_n(rst_n),
                .en(clipped_vld), .flush(clipped_vld),
                .a(clipped_r[ghi]), .b(ONE_6TH),
                .acc_fp32_in(32'h0),
                .acc_fp32_out(h0_fp32_nc),
                .acc_bf16_out(h_acc[0][ghi])
            );
        end
    endgenerate

    always_ff @(posedge clk_core or negedge rst_n) begin : h0_vld_ff
        if (!rst_n) h_vld[0] <= 1'b0;
        else        h_vld[0] <= clipped_vld;
    end

    // Stage 4: acc = clipped * acc + 0.5
    genvar gh1;
    generate
        for (gh1 = 0; gh1 < SEQ_LEN; gh1++) begin : h1_gen
            wire [31:0] h1_fp32_nc;
            // FP32 mul: clipped * h_acc[0]
            wire [31:0] h1_prod;
            fp32_mul h1_mul (.a(clipped_r[gh1]), .b(h_acc[0][gh1]),
                             .result(h1_prod));
            // FP32 add: prod + 0.5
            wire [31:0] half_fp32 = {HALF, 16'h0000};
            wire [31:0] h1_sum;
            fp32_add h1_add (.a(h1_prod), .b(half_fp32), .result(h1_sum));
            // Register + round to BF16
            wire rup = h1_sum[15] & (|h1_sum[14:0] | h1_sum[16]);
            always_ff @(posedge clk_core or negedge rst_n) begin : h1_reg
                if (!rst_n) h_acc[1][gh1] <= 16'h0;
                else if (h_vld[0])
                    h_acc[1][gh1] <= h1_sum[31:16] + {15'h0, rup};
            end
        end
    endgenerate

    always_ff @(posedge clk_core or negedge rst_n) begin : h1_vld_ff
        if (!rst_n) h_vld[1] <= 1'b0;
        else        h_vld[1] <= h_vld[0];
    end

    // Stage 5: acc = clipped * acc + 1.0
    genvar gh2;
    generate
        for (gh2 = 0; gh2 < SEQ_LEN; gh2++) begin : h2_gen
            wire [31:0] h2_prod;
            fp32_mul h2_mul (.a(clipped_r[gh2]), .b(h_acc[1][gh2]),
                             .result(h2_prod));
            wire [31:0] one_fp32 = {ONE, 16'h0000};
            wire [31:0] h2_sum;
            fp32_add h2_add (.a(h2_prod), .b(one_fp32), .result(h2_sum));
            wire rup = h2_sum[15] & (|h2_sum[14:0] | h2_sum[16]);
            always_ff @(posedge clk_core or negedge rst_n) begin : h2_reg
                if (!rst_n) h_acc[2][gh2] <= 16'h0;
                else if (h_vld[1])
                    h_acc[2][gh2] <= h2_sum[31:16] + {15'h0, rup};
            end
        end
    endgenerate

    always_ff @(posedge clk_core or negedge rst_n) begin : h2_vld_ff
        if (!rst_n) h_vld[2] <= 1'b0;
        else        h_vld[2] <= h_vld[1];
    end

    // -----------------------------------------------------------------------
    // Stage 6: sum reduction tree (LOG2_N adder levels, FP32)
    // -----------------------------------------------------------------------
    logic [31:0] sum_tree [LOG2_N+1][SEQ_LEN];
    logic        sum_vld  [LOG2_N+1];

    always_ff @(posedge clk_core or negedge rst_n) begin : sum_l0_ff
        if (!rst_n) begin
            sum_vld[0] <= 1'b0;
            for (int j = 0; j < SEQ_LEN; j++) sum_tree[0][j] <= 32'h0;
        end else begin
            sum_vld[0] <= h_vld[2];
            if (h_vld[2])
                for (int j = 0; j < SEQ_LEN; j++)
                    sum_tree[0][j] <= {h_acc[2][j], 16'h0};  // BF16->FP32
        end
    end

    genvar gsl;
    generate
        for (gsl = 0; gsl < LOG2_N; gsl++) begin : sum_level
            for (genvar gsp = 0; gsp < SEQ_LEN >> (gsl+1); gsp++) begin : sum_pair
                wire [31:0] pair_sum;
                fp32_add sum_add (
                    .a(sum_tree[gsl][gsp*2]),
                    .b(sum_tree[gsl][gsp*2+1]),
                    .result(pair_sum)
                );
                always_ff @(posedge clk_core or negedge rst_n) begin : sum_reg
                    if (!rst_n) sum_tree[gsl+1][gsp] <= 32'h0;
                    else if (sum_vld[gsl])
                        sum_tree[gsl+1][gsp] <= pair_sum;
                end
            end
            always_ff @(posedge clk_core or negedge rst_n) begin : sum_vld_ff
                if (!rst_n) sum_vld[gsl+1] <= 1'b0;
                else        sum_vld[gsl+1] <= sum_vld[gsl];
            end
        end
    endgenerate

    wire [31:0] total_sum = sum_tree[LOG2_N][0];
    wire        sum_rdy   = sum_vld[LOG2_N];

    // -----------------------------------------------------------------------
    // Stage 7: Newton-Raphson reciprocal (2 iterations, scalar)
    // y_new = y * (2 - S * y)
    // -----------------------------------------------------------------------
    logic [31:0] nr_y;
    logic [1:0]  nr_cnt;
    logic        nr_rdy;

    wire [31:0] sy_prod, two_minus_sy, nr_next;
    wire [15:0] sum_bf16 = total_sum[31:16];
    wire [15:0] y_bf16   = nr_y[31:16];

    fp32_mul nr_m1 (.a(sum_bf16), .b(y_bf16), .result(sy_prod));
    fp32_add nr_a1 (.a(32'h4000_0000),           // 2.0
                    .b({~sy_prod[31], sy_prod[30:0]}),
                    .result(two_minus_sy));
    fp32_mul nr_m2 (.a(y_bf16), .b(two_minus_sy[31:16]), .result(nr_next));

    always_ff @(posedge clk_core or negedge rst_n) begin : nr_ff
        if (!rst_n) begin
            nr_y   <= 32'h3F80_0000;   // init: 1.0
            nr_cnt <= 2'd0;
            nr_rdy <= 1'b0;
        end else begin
            nr_rdy <= 1'b0;
            if (sum_rdy) begin
                nr_y   <= 32'h3F80_0000;
                nr_cnt <= 2'd0;
            end else if (nr_cnt < 2'd2) begin
                nr_y   <= nr_next;
                nr_cnt <= nr_cnt + 2'd1;
                nr_rdy <= (nr_cnt == 2'd1);
            end
        end
    end

    // -----------------------------------------------------------------------
    // Stage 8: normalise — prob[i] = exp[i] * recip  (N parallel fp32_mul)
    // Need to delay exp values to align with reciprocal
    // Pipeline depth for NR = 2 cycles after sum_rdy
    // -----------------------------------------------------------------------
    // Delay exp_approx by (LOG2_N + 2) cycles to align with nr_rdy
    localparam int EXP_DLY = LOG2_N + 2;
    logic [15:0] exp_dly [EXP_DLY][SEQ_LEN];
    logic        exp_vld_dly [EXP_DLY];

    always_ff @(posedge clk_core or negedge rst_n) begin : exp_dly_ff
        if (!rst_n) begin
            for (int l = 0; l < EXP_DLY; l++) begin
                exp_vld_dly[l] <= 1'b0;
                for (int j = 0; j < SEQ_LEN; j++)
                    exp_dly[l][j] <= 16'h0;
            end
        end else begin
            exp_vld_dly[0] <= h_vld[2];
            if (h_vld[2])
                for (int j = 0; j < SEQ_LEN; j++)
                    exp_dly[0][j] <= h_acc[2][j];
            for (int l = 1; l < EXP_DLY; l++) begin
                exp_vld_dly[l] <= exp_vld_dly[l-1];
                for (int j = 0; j < SEQ_LEN; j++)
                    exp_dly[l][j] <= exp_dly[l-1][j];
            end
        end
    end

    wire [15:0] recip_bf16 = nr_y[31:16];
    wire [31:0] norm_fp32  [SEQ_LEN];
    wire [15:0] prob_out   [SEQ_LEN];

    genvar gni;
    generate
        for (gni = 0; gni < SEQ_LEN; gni++) begin : norm_gen
            fp32_mul nm (
                .a(exp_dly[EXP_DLY-1][gni]),
                .b(recip_bf16),
                .result(norm_fp32[gni])
            );
            wire rup = norm_fp32[gni][15] &
                       (|norm_fp32[gni][14:0] | norm_fp32[gni][16]);
            assign prob_out[gni] = norm_fp32[gni][31:16] + {15'h0, rup};
        end
    endgenerate

    // -----------------------------------------------------------------------
    // Pack prob_out into a TILE x TILE tile and transmit
    // -----------------------------------------------------------------------
    logic [15:0] out_tile   [TILE][TILE];
    logic [5:0]  out_row;
    logic        tx_valid_i [NUM_HEADS];
    logic [3:0]  out_head;

    always_ff @(posedge clk_core or negedge rst_n) begin : out_ff
        if (!rst_n) begin
            out_row  <= '0;
            out_head <= '0;
            for (int h = 0; h < NUM_HEADS; h++) tx_valid_i[h] <= 1'b0;
        end else begin
            for (int h = 0; h < NUM_HEADS; h++) tx_valid_i[h] <= 1'b0;

            if (nr_rdy) begin
                for (int j = 0; j < SEQ_LEN; j++)
                    out_tile[out_row][j] <= prob_out[j];

                if (out_row == TILE - 1) begin
                    out_row              <= '0;
                    tx_valid_i[out_head] <= 1'b1;
                    out_head             <= (out_head == NUM_HEADS-1)
                                           ? '0 : out_head + 4'd1;
                end else begin
                    out_row <= out_row + 6'd1;
                end
            end
        end
    end

    generate
        for (gh = 0; gh < NUM_HEADS; gh++) begin : tx_gen
            ucie_tx #(.TILE_DIM(TILE)) u_tx (
                .clk_core(clk_core), .rst_n(rst_n),
                .tx_valid(tx_valid_i[gh]),
                .tx_src_id(4'd9),
                .tx_dst_id(gh[3:0] + 4'd1),
                .tx_tile(out_tile), .tx_ready(),
                .bump_data(tx_bump_data[gh]),
                .bump_valid(tx_bump_valid[gh]),
                .bump_credit(tx_bump_credit[gh])
            );
        end
    endgenerate

endmodule

`default_nettype wire