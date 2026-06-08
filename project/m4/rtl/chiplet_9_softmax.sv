// =============================================================================
// chiplet_9_softmax.sv  —  Chiplet ID 9  [optimised: 8 parallel pipelines]
// =============================================================================
// Fix: parallel softmax pipelines for all 8 heads
// ------------------------------------------------
// Original: single shared pipeline with a head_idx counter stepping through
//           heads 0→7 one at a time, row by row.
//           Cost: 8 × (64 rows + 20 cycle drain) = 672 cycles, then
//                 8 × 284 cycle UCIe TX = 2272 cycles sequential.
//
// Fixed:    8 independent pipeline instances (one per head) instantiated
//           in a generate loop.  Each pipeline has its own:
//             - row_feed FSM (row_idx counter)
//             - max reduction tree
//             - score delay / shift chain
//             - exp array
//             - sum reduction tree
//             - Newton-Raphson reciprocal unit
//             - normalise array
//             - output tile register
//             - ucie_tx
//           All 8 pipelines start simultaneously when rx_valid[h] fires
//           and finish simultaneously.
//           Cost: 1 × (64 + 20) = 84 cycles compute,
//                 1 × 284 cycles UCIe TX (all 8 transmitting in parallel).
//           Speedup: 8× compute, 8× TX.
//
// cfg_done: pulses when ALL 8 heads have transmitted their output tile.
//           Implemented as AND-reduction of per-head done flags.
// =============================================================================

`timescale 1ns/1ps
`default_nettype none

// ---------------------------------------------------------------------------
// softmax_pipe  —  single-head softmax pipeline (parameterised)
// Processes one TILE×TILE score tile row by row.
// Outputs one TILE×TILE probability tile via UCIe TX when done.
// ---------------------------------------------------------------------------
module softmax_pipe #(
    parameter int HEAD_ID  = 0,
    parameter int TILE     = 64,
    parameter int SEQ_LEN  = 64,
    parameter int LOG2_N   = 6
)(
    input  wire        clk_core,
    input  wire        clk_link,      // 2 GHz UCIe PHY clock
    input  wire        rst_n,
    input  wire        cfg_start,
    output logic       pipe_done,     // pulses 1 cycle when tx fires

    // UCIe RX — scores from head chiplet
    input  wire [511:0] rx_bump_data,
    input  wire         rx_bump_valid,
    output wire         rx_bump_credit,

    // UCIe TX — probabilities back to head chiplet
    output wire [511:0] tx_bump_data,
    output wire         tx_bump_valid,
    input  wire         tx_bump_credit
);

    // -------------------------------------------------------------------------
    // UCIe RX
    // -------------------------------------------------------------------------
    logic        rx_valid;
    logic [15:0] rx_tile [TILE][TILE];

    ucie_rx #(.TILE_DIM(TILE)) u_rx (
        .clk_core(clk_core), .clk_link(clk_link), .rst_n(rst_n),
        .bump_data(rx_bump_data), .bump_valid(rx_bump_valid),
        .bump_credit(rx_bump_credit),
        .rx_valid(rx_valid), .rx_src_id(),
        .rx_tile(rx_tile), .rx_ready(1'b1),
        .rx_crc_err(), .rx_seq_err()
    );

    // -------------------------------------------------------------------------
    // Row feed FSM: iterate row_idx = 0..TILE-1 after rx_valid
    // -------------------------------------------------------------------------
    logic [5:0]  row_idx;
    logic        pipe_valid;
    logic [15:0] pipe_scores [SEQ_LEN];

    always_ff @(posedge clk_core or negedge rst_n) begin : row_feed_ff
        if (!rst_n) begin
            row_idx    <= '0;
            pipe_valid <= 1'b0;
        end else begin
            pipe_valid <= 1'b0;
            if (cfg_start && rx_valid) begin
                for (int j = 0; j < SEQ_LEN; j++)
                    pipe_scores[j] <= rx_tile[row_idx][j];
                pipe_valid <= 1'b1;
                row_idx    <= (row_idx == TILE-1) ? '0 : row_idx + 6'd1;
            end
        end
    end

    // -------------------------------------------------------------------------
    // Stage 1: max reduction tree  (LOG2_N registered levels)
    // -------------------------------------------------------------------------
    logic [15:0] max_tree [LOG2_N+1][SEQ_LEN];
    logic        max_valid [LOG2_N+1];

    always_ff @(posedge clk_core or negedge rst_n) begin : max_l0_ff
        if (!rst_n) begin
            max_valid[0] <= 1'b0;
            for (int j=0;j<SEQ_LEN;j++) max_tree[0][j] <= 16'h0;
        end else begin
            max_valid[0] <= pipe_valid;
            if (pipe_valid)
                for (int j=0;j<SEQ_LEN;j++) max_tree[0][j] <= pipe_scores[j];
        end
    end

    genvar gl;
    generate
        for (gl=0;gl<LOG2_N;gl++) begin : max_level
            always_ff @(posedge clk_core or negedge rst_n) begin : max_lev_ff
                if (!rst_n) begin
                    max_valid[gl+1] <= 1'b0;
                    for (int j=0;j<SEQ_LEN;j++) max_tree[gl+1][j] <= 16'h0;
                end else begin
                    max_valid[gl+1] <= max_valid[gl];
                    for (int j=0; j < SEQ_LEN >> (gl+1); j++) begin
                        if ($signed(max_tree[gl][j*2]) >= $signed(max_tree[gl][j*2+1]))
                            max_tree[gl+1][j] <= max_tree[gl][j*2];
                        else
                            max_tree[gl+1][j] <= max_tree[gl][j*2+1];
                    end
                end
            end
        end
    endgenerate

    wire [15:0] row_max = max_tree[LOG2_N][0];
    wire        max_rdy = max_valid[LOG2_N];

    // -------------------------------------------------------------------------
    // Score delay chain (aligns input with max result, LOG2_N cycles)
    // -------------------------------------------------------------------------
    logic [15:0] scores_dly [LOG2_N][SEQ_LEN];
    logic        valid_dly  [LOG2_N];

    always_ff @(posedge clk_core or negedge rst_n) begin : skew_ff
        if (!rst_n) begin
            for (int l=0;l<LOG2_N;l++) begin
                valid_dly[l] <= 1'b0;
                for (int j=0;j<SEQ_LEN;j++) scores_dly[l][j] <= 16'h0;
            end
        end else begin
            valid_dly[0] <= pipe_valid;
            if (pipe_valid)
                for (int j=0;j<SEQ_LEN;j++) scores_dly[0][j] <= pipe_scores[j];
            for (int l=1;l<LOG2_N;l++) begin
                valid_dly[l] <= valid_dly[l-1];
                for (int j=0;j<SEQ_LEN;j++) scores_dly[l][j] <= scores_dly[l-1][j];
            end
        end
    end

    // -------------------------------------------------------------------------
    // Stage 2: subtract max + clip
    // -------------------------------------------------------------------------
    wire [15:0] neg_max   = {~row_max[15], row_max[14:0]};
    wire [15:0] shifted_w [SEQ_LEN];
    wire [15:0] clipped_w [SEQ_LEN];

    genvar gsi;
    generate
        for (gsi=0;gsi<SEQ_LEN;gsi++) begin : shift_clip_gen
            bf16_add u_sh (
                .a(scores_dly[LOG2_N-1][gsi]), .b(neg_max), .result(shifted_w[gsi])
            );
            wire over  = ~shifted_w[gsi][15] & (shifted_w[gsi] != 16'h0);
            wire under =  shifted_w[gsi][15] &
                          ($signed(shifted_w[gsi]) < $signed(16'hC100));
            assign clipped_w[gsi] = over  ? 16'h0000 :
                                    under ? 16'hC100  : shifted_w[gsi];
        end
    endgenerate

    logic [15:0] clipped_r [SEQ_LEN];
    logic        clipped_vld;

    always_ff @(posedge clk_core or negedge rst_n) begin : clip_ff
        if (!rst_n) begin
            clipped_vld <= 1'b0;
            for (int j=0;j<SEQ_LEN;j++) clipped_r[j] <= 16'h0;
        end else begin
            clipped_vld <= max_rdy;
            if (max_rdy) for (int j=0;j<SEQ_LEN;j++) clipped_r[j] <= clipped_w[j];
        end
    end

    // -------------------------------------------------------------------------
    // Stage 3: exponentiation (bf16_exp, 1 registered cycle)
    // -------------------------------------------------------------------------
    logic [15:0] exp_r   [SEQ_LEN];
    logic        h_vld   [3];

    genvar ghi;
    generate
        for (ghi=0;ghi<SEQ_LEN;ghi++) begin : exp_gen
            wire [15:0] exp_comb;
            bf16_exp u_exp (.a(clipped_r[ghi]), .result(exp_comb));
            always_ff @(posedge clk_core or negedge rst_n) begin : exp_reg
                if (!rst_n)          exp_r[ghi] <= 16'h0;
                else if (clipped_vld) exp_r[ghi] <= exp_comb;
            end
        end
    endgenerate

    always_ff @(posedge clk_core or negedge rst_n) begin : exp_vld_ff
        if (!rst_n) begin h_vld[0]<=0; h_vld[1]<=0; h_vld[2]<=0; end
        else begin h_vld[0]<=clipped_vld; h_vld[1]<=h_vld[0]; h_vld[2]<=h_vld[1]; end
    end

    // -------------------------------------------------------------------------
    // Stage 4: sum reduction tree (LOG2_N FP32 adder levels)
    // -------------------------------------------------------------------------
    logic [31:0] sum_tree [LOG2_N+1][SEQ_LEN];
    logic        sum_vld  [LOG2_N+1];

    always_ff @(posedge clk_core or negedge rst_n) begin : sum_l0_ff
        if (!rst_n) begin
            sum_vld[0] <= 1'b0;
            for (int j=0;j<SEQ_LEN;j++) sum_tree[0][j] <= 32'h0;
        end else begin
            sum_vld[0] <= h_vld[2];
            if (h_vld[2])
                for (int j=0;j<SEQ_LEN;j++) sum_tree[0][j] <= {exp_r[j], 16'h0};
        end
    end

    genvar gsl;
    generate
        for (gsl=0;gsl<LOG2_N;gsl++) begin : sum_level
            for (genvar gsp=0; gsp < SEQ_LEN>>(gsl+1); gsp++) begin : sum_pair
                wire [31:0] pair_sum;
                fp32_add sum_add (
                    .a(sum_tree[gsl][gsp*2]), .b(sum_tree[gsl][gsp*2+1]),
                    .result(pair_sum)
                );
                always_ff @(posedge clk_core or negedge rst_n) begin : sum_reg
                    if (!rst_n) sum_tree[gsl+1][gsp] <= 32'h0;
                    else if (sum_vld[gsl]) sum_tree[gsl+1][gsp] <= pair_sum;
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

    // -------------------------------------------------------------------------
    // Stage 5: Newton-Raphson reciprocal (2 iterations)
    // -------------------------------------------------------------------------
    logic [31:0] nr_y;
    logic [1:0]  nr_cnt;
    logic        nr_rdy;

    wire rup_sum = total_sum[15] & (|total_sum[14:0] | total_sum[16]);
	wire [15:0] sum_bf16 = total_sum[31:16] + {15'h0, rup_sum};

    wire [15:0] y_bf16   = nr_y[31:16];
    wire [31:0] sy_prod, two_minus_sy, nr_next;

    fp32_mul nr_m1 (.a(sum_bf16), .b(y_bf16), .result(sy_prod));
    fp32_add nr_a1 (.a(32'h4000_0000), .b({~sy_prod[31],sy_prod[30:0]}),
                    .result(two_minus_sy));
    fp32_mul nr_m2 (.a(y_bf16), .b(two_minus_sy[31:16]), .result(nr_next));

    wire [7:0]  sum_exp_field = total_sum[30:23];
    wire [7:0]  seed_exp      = 8'd253 - sum_exp_field;
    wire [31:0] nr_seed = (sum_exp_field==8'h00) ? 32'h3F80_0000
                                                  : {1'b0, seed_exp, 23'h0};

    always_ff @(posedge clk_core or negedge rst_n) begin : nr_ff
        if (!rst_n) begin nr_y<=32'h3F80_0000; nr_cnt<=2'd0; nr_rdy<=1'b0; end
        else begin
            nr_rdy <= 1'b0;
            if (sum_rdy) begin nr_y<=nr_seed; nr_cnt<=2'd0; end
            else if (nr_cnt < 2'd2) begin
                nr_y   <= nr_next;
                nr_cnt <= nr_cnt + 2'd1;
                nr_rdy <= (nr_cnt == 2'd1);
            end
        end
    end

    // -------------------------------------------------------------------------
    // Stage 6: normalise  (exp * recip, combinational)
    // Exp values delayed (LOG2_N + 2) cycles to align with nr_rdy
    // -------------------------------------------------------------------------
    localparam int EXP_DLY = LOG2_N + 2;
    logic [15:0] exp_dly     [EXP_DLY][SEQ_LEN];
    logic        exp_vld_dly [EXP_DLY];

    always_ff @(posedge clk_core or negedge rst_n) begin : exp_dly_ff
        if (!rst_n) begin
            for (int l=0;l<EXP_DLY;l++) begin
                exp_vld_dly[l] <= 1'b0;
                for (int j=0;j<SEQ_LEN;j++) exp_dly[l][j] <= 16'h0;
            end
        end else begin
            exp_vld_dly[0] <= h_vld[2];
            if (h_vld[2]) for (int j=0;j<SEQ_LEN;j++) exp_dly[0][j] <= exp_r[j];
            for (int l=1;l<EXP_DLY;l++) begin
                exp_vld_dly[l] <= exp_vld_dly[l-1];
                for (int j=0;j<SEQ_LEN;j++) exp_dly[l][j] <= exp_dly[l-1][j];
            end
        end
    end

    wire rup_rec = nr_y[15] & (|nr_y[14:0] | nr_y[16]);
	wire [15:0] recip_bf16 = nr_y[31:16] + {15'h0, rup_rec};
    wire [31:0] norm_fp32  [SEQ_LEN];
    wire [15:0] prob_out   [SEQ_LEN];

    genvar gni;
    generate
        for (gni=0;gni<SEQ_LEN;gni++) begin : norm_gen
            fp32_mul nm (
                .a(exp_dly[EXP_DLY-1][gni]), .b(recip_bf16), .result(norm_fp32[gni])
            );
            wire rup = norm_fp32[gni][15] & (|norm_fp32[gni][14:0] | norm_fp32[gni][16]);
            assign prob_out[gni] = norm_fp32[gni][31:16] + {15'h0, rup};
        end
    endgenerate

    // -------------------------------------------------------------------------
    // Output tile packing + UCIe TX
    // Accumulate all TILE rows, then fire tx_valid once
    // -------------------------------------------------------------------------
    logic [15:0] out_tile [TILE][TILE];
    logic [5:0]  out_row;
    logic        tx_valid_i;

    always_ff @(posedge clk_core or negedge rst_n) begin : out_ff
        if (!rst_n) begin
            out_row    <= '0;
            tx_valid_i <= 1'b0;
            pipe_done  <= 1'b0;
        end else begin
            tx_valid_i <= 1'b0;
            pipe_done  <= 1'b0;
            if (nr_rdy) begin
                for (int j=0;j<SEQ_LEN;j++) out_tile[out_row][j] <= prob_out[j];
                if (out_row == TILE-1) begin
                    out_row    <= '0;
                    tx_valid_i <= 1'b1;
                    pipe_done  <= 1'b1;
                end else begin
                    out_row <= out_row + 6'd1;
                end
            end
        end
    end

    ucie_tx #(.TILE_DIM(TILE)) u_tx (
        .clk_core(clk_core), .clk_link(clk_link), .rst_n(rst_n),
        .tx_valid(tx_valid_i),
        .tx_src_id(4'd9),
        .tx_dst_id(HEAD_ID[3:0] + 4'd1),
        .tx_tile(out_tile), .tx_ready(),
        .bump_data(tx_bump_data),
        .bump_valid(tx_bump_valid),
        .bump_credit(tx_bump_credit)
    );

endmodule


// =============================================================================
// chiplet_9_softmax  —  top-level wrapper: 8 parallel softmax_pipe instances
// =============================================================================
module chiplet_9_softmax #(
    parameter int NUM_HEADS = 8,
    parameter int TILE      = 64,
    parameter int SEQ_LEN   = 64,
    parameter int LOG2_N    = 6
)(
    input  wire        clk_core,
    input  wire        clk_link,      // 2 GHz UCIe PHY clock
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
    // One done flag per head pipeline
    wire pipe_done [NUM_HEADS];

    // Pack into a vector for clean AND-reduction.
    // pipe_done is a 1-cycle pulse; sampling via reduction FF on the same
    // cycle all 8 fire gives a reliable single-cycle cfg_done pulse.
    wire [NUM_HEADS-1:0] pipe_done_vec;
    genvar pd;
    generate
        for (pd = 0; pd < NUM_HEADS; pd++) begin : pd_vec_g
            assign pipe_done_vec[pd] = pipe_done[pd];
        end
    endgenerate

    // cfg_done = registered AND of all 8 pipe_done flags
    always_ff @(posedge clk_core or negedge rst_n) begin : done_ff
        if (!rst_n) cfg_done <= 1'b0;
        else        cfg_done <= &pipe_done_vec;
    end

    // 8 parallel pipelines
    genvar gh;
    generate
        for (gh = 0; gh < NUM_HEADS; gh++) begin : pipe_gen
            softmax_pipe #(
                .HEAD_ID (gh),
                .TILE    (TILE),
                .SEQ_LEN (SEQ_LEN),
                .LOG2_N  (LOG2_N)
            ) u_pipe (
                .clk_core      (clk_core),
                .clk_link      (clk_link),
                .rst_n         (rst_n),
                .cfg_start     (cfg_start),
                .pipe_done     (pipe_done[gh]),
                .rx_bump_data  (rx_bump_data [gh]),
                .rx_bump_valid (rx_bump_valid[gh]),
                .rx_bump_credit(rx_bump_credit[gh]),
                .tx_bump_data  (tx_bump_data [gh]),
                .tx_bump_valid (tx_bump_valid[gh]),
                .tx_bump_credit(tx_bump_credit[gh])
            );
        end
    endgenerate

endmodule

`default_nettype wire
// =============================================================================
// End of chiplet_9_softmax.sv  (optimised: 8 parallel pipelines)
// =============================================================================
