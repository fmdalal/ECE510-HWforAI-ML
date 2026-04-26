// =============================================================================
// conformer_rel_mhsa_accel.sv
//
// Compute-core top-level: Conformer Relative Multi-Head Self-Attention
// accelerator mapped to weight-stationary systolic arrays.
//
// Reference : sooftware/conformer → conformer/attention.py
//             RelativeMultiHeadAttention (Transformer-XL style)
//             INTERSPEECH 2020  /  ACL 2019
//
// Python defaults replicated here
// ─────────────────────────────────────────────────────────────────────────────
//   d_model   = 512    encoder_dim in encoder.py / model.py
//   num_heads = 16     RelativeMultiHeadAttention.__init__ default
//   head_dim  = 32     = 512 / 16
//   sqrt_dim  = √512   self.sqrt_dim = math.sqrt(d_model)  ← NOT √head_dim
//
// Relative attention score (per head h, query row i, key col j)
// ─────────────────────────────────────────────────────────────────────────────
//   content_score[h][i][j] = (Q_h[i] + u_bias[h]) · K_h[j]^T
//   pos_score[h][i][j]     = (Q_h[i] + v_bias[h]) · POS_h[j-i]^T
//   score[h][i][j]         = (content_score + pos_score) / √d_model + mask
//   attn[h]                = softmax(score[h])
//   context[h]             = attn[h] · V_h
//   Y                      = concat(context_h) · W_O
//
// Datapath pipeline (time-multiplexed over one SA instance)
// ─────────────────────────────────────────────────────────────────────────────
//  ST_PROJ_Q    : Q   = X  · W_Q
//  ST_PROJ_K    : K   = X  · W_K
//  ST_PROJ_V    : V   = X  · W_V
//  ST_PROJ_POS  : POS = P  · W_pos
//  ST_BIAS_ADD  : QU  = Q + u_bias (tiled),  QV = Q + v_bias (tiled)
//  ST_CONT_SCORE: content_score_h = QU_h · K_h^T   per head
//  ST_POS_SCORE : pos_score_h     = QV_h · POS_h^T per head
//  ST_REL_SKEW  : cyclic left-shift row i by i positions (Transformer-XL skew)
//  ST_SCALE_ADD : (cont+pos)/√D_MODEL + optional mask
//  ST_SOFTMAX   : row-wise softmax  (iterative fixed-point)
//  ST_CONTEXT   : C_h = attn_h · V_h
//  ST_OUT_PROJ  : Y   = concat(C) · W_O
//  ST_EMIT      : stream Y row-by-row to output port
//
// Fixed-point format
// ─────────────────────────────────────────────────────────────────────────────
//   Q(DATA_WIDTH − FRAC_WIDTH, FRAC_WIDTH)
//   Accumulator: ACCUM_WIDTH ≥ 2 × DATA_WIDTH  (default 16 → 32 OK)
//
// Simulation note
// ─────────────────────────────────────────────────────────────────────────────
//   Default parameters are intentionally reduced (D_MODEL=32, SEQ_LEN=8,
//   NUM_HEADS=2) so the testbench elaborates and runs quickly.
//   Override to (512, 512, 16) for production synthesis.
// =============================================================================

`timescale 1ns / 1ps
`default_nettype none

// ─────────────────────────────────────────────────────────────────────────────
// Processing Element  –  weight-stationary MAC cell
// One PE per grid position; holds its weight for the full tile drain.
// ─────────────────────────────────────────────────────────────────────────────
module sa_pe #(
    parameter int DATA_WIDTH  = 16,
    parameter int FRAC_WIDTH  = 8,
    parameter int ACCUM_WIDTH = 32
)(
    input  wire                    clk,
    input  wire                    rst_n,        // active-low synchronous reset
    // Weight side-load (asserted for one cycle before tile drain begins)
    input  wire                    weight_load,
    input  wire [DATA_WIDTH-1:0]   weight_in,
    // Systolic data / accumulator ports
    input  wire [DATA_WIDTH-1:0]   data_in,
    input  wire [ACCUM_WIDTH-1:0]  acc_in,
    input  wire                    valid_in,
    output logic [DATA_WIDTH-1:0]  data_out,
    output logic [ACCUM_WIDTH-1:0] acc_out,
    output logic                   valid_out
);
    logic [DATA_WIDTH-1:0] weight_reg;

    // ── Weight register  (reset-able, loaded once per tile) ──────────────────
    always_ff @(posedge clk) begin
        if (!rst_n)           weight_reg <= '0;
        else if (weight_load) weight_reg <= weight_in;
    end

    // ── MAC pipeline register  (reset-able) ──────────────────────────────────
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            data_out  <= '0;
            acc_out   <= '0;
            valid_out <= 1'b0;
        end else begin
            data_out  <= data_in;
            valid_out <= valid_in;
            // Fixed-point multiply-accumulate:
            //   product = (a × b) >> FRAC_WIDTH  keeps Q(iw,fw) format
            acc_out <= valid_in
                ? acc_in + ACCUM_WIDTH'(
                    ($signed({{(ACCUM_WIDTH-DATA_WIDTH){data_in[DATA_WIDTH-1]}},   data_in})  *
                     $signed({{(ACCUM_WIDTH-DATA_WIDTH){weight_reg[DATA_WIDTH-1]}}, weight_reg}))
                    >>> FRAC_WIDTH)
                : acc_in;
        end
    end
endmodule : sa_pe

// ─────────────────────────────────────────────────────────────────────────────
// Systolic Array  –  SA_ROWS × SA_COLS weight-stationary grid
// Data flows right across rows; partial sums flow downward through columns.
// ─────────────────────────────────────────────────────────────────────────────
module systolic_array #(
    parameter int DATA_WIDTH  = 16,
    parameter int FRAC_WIDTH  = 8,
    parameter int ACCUM_WIDTH = 32,
    parameter int SA_ROWS     = 4,
    parameter int SA_COLS     = 4
)(
    input  wire                                   clk,
    input  wire                                   rst_n,
    // Weight preload – row-major flat: element[r][c] at [(r*COLS+c+1)*DW-1 -: DW]
    input  wire                                   weight_load,
    input  wire [SA_ROWS*SA_COLS*DATA_WIDTH-1:0]  weight_flat,
    // Input activations – SA_ROWS elements arriving simultaneously
    input  wire [SA_ROWS*DATA_WIDTH-1:0]          row_data_flat,
    input  wire                                   row_valid,
    // Outputs – SA_COLS partial sums at the bottom of each column
    output wire [SA_COLS*ACCUM_WIDTH-1:0]         col_acc_flat,
    output wire                                   col_valid
);
    // Flat inter-PE wire arrays (avoids multi-dim packed-array elaboration bugs)
    localparam int DH = SA_ROWS * (SA_COLS + 1);
    localparam int AV = (SA_ROWS + 1) * SA_COLS;

    wire [DATA_WIDTH-1:0]  dh [0:DH-1];   // tokens flowing right
    wire                   vh [0:DH-1];   // valid flowing right
    wire [ACCUM_WIDTH-1:0] av [0:AV-1];   // accumulators flowing down

    genvar r, c;

    // Left-edge seeds (one row-element per PE row)
    for (r = 0; r < SA_ROWS; r++) begin : g_seed_r
        assign dh[r*(SA_COLS+1)]   = row_data_flat[(r+1)*DATA_WIDTH-1 -: DATA_WIDTH];
        assign vh[r*(SA_COLS+1)]   = row_valid;
    end
    // Top-edge seeds (zero accumulator)
    for (c = 0; c < SA_COLS; c++) begin : g_seed_c
        assign av[c] = '0;
    end

    // PE grid
    for (r = 0; r < SA_ROWS; r++) begin : g_r
        for (c = 0; c < SA_COLS; c++) begin : g_c
            sa_pe #(
                .DATA_WIDTH (DATA_WIDTH),
                .FRAC_WIDTH (FRAC_WIDTH),
                .ACCUM_WIDTH(ACCUM_WIDTH)
            ) u_pe (
                .clk        (clk),
                .rst_n      (rst_n),
                .weight_load(weight_load),
                .weight_in  (weight_flat[(r*SA_COLS+c+1)*DATA_WIDTH-1 -: DATA_WIDTH]),
                .data_in    (dh[r*(SA_COLS+1)+c]),
                .acc_in     (av[r*SA_COLS+c]),
                .valid_in   (vh[r*(SA_COLS+1)+c]),
                .data_out   (dh[r*(SA_COLS+1)+c+1]),
                .acc_out    (av[(r+1)*SA_COLS+c]),
                .valid_out  (vh[r*(SA_COLS+1)+c+1])
            );
        end
    end

    // Bottom-row outputs
    for (c = 0; c < SA_COLS; c++) begin : g_out
        assign col_acc_flat[(c+1)*ACCUM_WIDTH-1 -: ACCUM_WIDTH] = av[SA_ROWS*SA_COLS+c];
    end
    assign col_valid = vh[(SA_ROWS-1)*(SA_COLS+1)+SA_COLS];

endmodule : systolic_array

// ─────────────────────────────────────────────────────────────────────────────
// Softmax unit  –  row-wise iterative fixed-point
//
// Pipeline per row:  FIND_MAX → CALC_EXP → SUM_EXP → NORMALIZE
//
// exp approximation:  exp(x − max) ≈ max(0, 1 + x)
// This is sufficient to demonstrate the HW structure.
// For production replace with a CORDIC core or a 256-entry LUT.
// ─────────────────────────────────────────────────────────────────────────────
module softmax_unit #(
    parameter int DATA_WIDTH  = 16,
    parameter int FRAC_WIDTH  = 8,
    parameter int ACCUM_WIDTH = 32,
    parameter int SEQ_LEN     = 8
)(
    input  wire                             clk,
    input  wire                             rst_n,
    input  wire                             start,
    input  wire [SEQ_LEN*DATA_WIDTH-1:0]    scores_flat,   // one attention row
    output logic [SEQ_LEN*DATA_WIDTH-1:0]   attn_flat,     // softmax output
    output logic                            done
);
    typedef enum logic [2:0] {
        SM_IDLE, SM_MAX, SM_EXP, SM_SUM, SM_NORM, SM_DONE
    } sm_t;
    sm_t sm_st;

    logic [DATA_WIDTH-1:0]         max_val;
    logic [ACCUM_WIDTH-1:0]        exp_sum;
    logic [ACCUM_WIDTH-1:0]        exp_buf [0:SEQ_LEN-1];
    logic [$clog2(SEQ_LEN+1)-1:0]  idx;

    // Current element (indexed by idx)
    wire [DATA_WIDTH-1:0] cur = scores_flat[(int'(idx)+1)*DATA_WIDTH-1 -: DATA_WIDTH];

    // exp(cur − max):  linear approx in Q(iw, FRAC_WIDTH)
    wire signed [ACCUM_WIDTH-1:0] x_sub =
        $signed({{(ACCUM_WIDTH-DATA_WIDTH){cur[DATA_WIDTH-1]}},     cur})     -
        $signed({{(ACCUM_WIDTH-DATA_WIDTH){max_val[DATA_WIDTH-1]}}, max_val});

    wire [ACCUM_WIDTH-1:0] one_fp     = ACCUM_WIDTH'(1) <<< FRAC_WIDTH;
    wire [ACCUM_WIDTH-1:0] exp_approx = x_sub[ACCUM_WIDTH-1] ? '0
                                      : ACCUM_WIDTH'(one_fp + ACCUM_WIDTH'(x_sub));

    // ── Iterative FSM (all registers reset-able) ──────────────────────────────
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            sm_st    <= SM_IDLE;
            done     <= 1'b0;
            max_val  <= '0;
            exp_sum  <= '0;
            idx      <= '0;
            attn_flat<= '0;
        end else begin
            done <= 1'b0;
            case (sm_st)
                SM_IDLE: if (start) begin
                    max_val <= scores_flat[DATA_WIDTH-1:0];
                    idx     <= 1;
                    sm_st   <= SM_MAX;
                end
                SM_MAX: begin
                    if (idx < SEQ_LEN) begin
                        if ($signed(cur) > $signed(max_val)) max_val <= cur;
                        idx <= idx + 1;
                    end else begin idx <= 0; sm_st <= SM_EXP; end
                end
                SM_EXP: begin
                    if (idx < SEQ_LEN) begin
                        exp_buf[idx] <= exp_approx;
                        idx <= idx + 1;
                    end else begin exp_sum <= '0; idx <= 0; sm_st <= SM_SUM; end
                end
                SM_SUM: begin
                    if (idx < SEQ_LEN) begin
                        exp_sum <= exp_sum + exp_buf[idx];
                        idx <= idx + 1;
                    end else begin idx <= 0; sm_st <= SM_NORM; end
                end
                SM_NORM: begin
                    if (idx < SEQ_LEN) begin
                        attn_flat[(int'(idx)+1)*DATA_WIDTH-1 -: DATA_WIDTH] <=
                            (exp_sum != 0)
                            ? DATA_WIDTH'((exp_buf[idx] <<< FRAC_WIDTH) / exp_sum)
                            : DATA_WIDTH'(1 <<< (FRAC_WIDTH - $clog2(SEQ_LEN)));
                        idx <= idx + 1;
                    end else begin done <= 1'b1; sm_st <= SM_DONE; end
                end
                SM_DONE:  sm_st <= SM_IDLE;
                default:  sm_st <= SM_IDLE;
            endcase
        end
    end
endmodule : softmax_unit

// =============================================================================
//  TOP-LEVEL: conformer_rel_mhsa_accel
// =============================================================================
//
// Port groups
// ───────────────────────────────────────────────────────────────────────────
//  clk / rst_n          – clock and active-low synchronous reset
//  start / done / busy  – handshake control
//  x_row_*              – streamed input activations  X  (one row per cycle)
//  p_row_*              – streamed positional embeddings P (one row per cycle)
//  weight_sel/tr/tc/    – tiled weight preload (5 matrices: W_Q,W_K,W_V,W_pos,W_O)
//    tile_flat/load
//  bias_sel/flat/load   – learnable u_bias / v_bias preload
//  attn_mask_flat       – optional additive attention mask [SEQ_LEN × SEQ_LEN]
//  use_mask             – enable the mask addition
//  y_row_*              – streamed output Y (one row per cycle)
//  fsm_state_dbg        – FSM state for waveform debugging
// =============================================================================
module conformer_rel_mhsa_accel #(
    // ── Numeric precision ────────────────────────────────────────────────────
    parameter int DATA_WIDTH  = 16,   // element width (bits)
    parameter int FRAC_WIDTH  = 8,    // Q(DATA_WIDTH−FRAC_WIDTH, FRAC_WIDTH)
    parameter int ACCUM_WIDTH = 32,   // accumulator width  (≥ 2 × DATA_WIDTH)

    // ── Attention dimensions ─────────────────────────────────────────────────
    // Production values  →  D_MODEL=512, NUM_HEADS=16, SEQ_LEN=512
    // Simulation default →  D_MODEL=32,  NUM_HEADS=2,  SEQ_LEN=8
    //   (keeps elaboration time and BRAM usage tractable in cocotb)
    parameter int D_MODEL     = 32,   // embedding dimension
    parameter int NUM_HEADS   = 2,    // attention heads
    parameter int SEQ_LEN     = 8,    // max sequence length

    // ── Systolic array physical tile ─────────────────────────────────────────
    // Constraints:  SA_ROWS | D_MODEL, SA_ROWS | SEQ_LEN,
    //               SA_COLS | D_MODEL, SA_COLS | HEAD_DIM
    parameter int SA_ROWS     = 4,
    parameter int SA_COLS     = 4,

    // ── Derived  (do not override) ───────────────────────────────────────────
    parameter int HEAD_DIM    = D_MODEL / NUM_HEADS,
    parameter int TILE_R      = D_MODEL / SA_ROWS,
    parameter int TILE_C      = D_MODEL / SA_COLS,
    parameter int WTILE_BITS  = SA_ROWS * SA_COLS * DATA_WIDTH,
    parameter int WMAT_BITS   = TILE_R  * TILE_C  * WTILE_BITS,
    parameter int ROW_BITS    = D_MODEL * DATA_WIDTH,
    parameter int MAT_BITS    = SEQ_LEN * ROW_BITS,
    parameter int SCORE_BITS  = NUM_HEADS * SEQ_LEN * SEQ_LEN * DATA_WIDTH,
    parameter int BIAS_BITS   = NUM_HEADS * HEAD_DIM * DATA_WIDTH
)(
    // ── Clock / reset ─────────────────────────────────────────────────────────
    input  wire                                   clk,
    input  wire                                   rst_n,      // active-low synchronous reset

    // ── Handshake ─────────────────────────────────────────────────────────────
    input  wire                                   start,      // single-cycle pulse → begin
    output logic                                  done,       // single-cycle pulse → result valid
    output logic                                  busy,       // high throughout computation

    // ── Input activations X [SEQ_LEN × D_MODEL] ──────────────────────────────
    // Self-attention: query = key = value = X.
    // Present one flat row per cycle with x_valid asserted.
    input  wire [ROW_BITS-1:0]                    x_row_flat,
    input  wire [$clog2(SEQ_LEN)-1:0]             x_row_idx,
    input  wire                                   x_valid,

    // ── Positional embeddings P [SEQ_LEN × D_MODEL] ──────────────────────────
    // Sinusoidal relative encoding; precomputed by host CPU.
    input  wire [ROW_BITS-1:0]                    p_row_flat,
    input  wire [$clog2(SEQ_LEN)-1:0]             p_row_idx,
    input  wire                                   p_valid,

    // ── Weight preloading ─────────────────────────────────────────────────────
    // weight_sel: 0=W_Q  1=W_K  2=W_V  3=W_pos  4=W_O
    // Tile address (weight_tr, weight_tc) selects the SA_ROWS×SA_COLS sub-block.
    // Assert weight_load for one cycle per tile before asserting start.
    input  wire [2:0]                             weight_sel,
    input  wire [$clog2(TILE_R)-1:0]              weight_tr,
    input  wire [$clog2(TILE_C)-1:0]              weight_tc,
    input  wire [WTILE_BITS-1:0]                  weight_tile_flat,
    input  wire                                   weight_load,

    // ── Learnable biases  u_bias / v_bias  [NUM_HEADS × HEAD_DIM] ────────────
    // bias_sel: 0=u_bias  1=v_bias
    input  wire                                   bias_sel,
    input  wire [BIAS_BITS-1:0]                   bias_flat,
    input  wire                                   bias_load,

    // ── Additive attention mask  [SEQ_LEN × SEQ_LEN] ─────────────────────────
    input  wire [SEQ_LEN*SEQ_LEN*DATA_WIDTH-1:0]  attn_mask_flat,
    input  wire                                   use_mask,

    // ── Output Y [SEQ_LEN × D_MODEL] ─────────────────────────────────────────
    output logic [ROW_BITS-1:0]                   y_row_flat,
    output logic [$clog2(SEQ_LEN)-1:0]            y_row_idx,
    output logic                                  y_valid,

    // ── Debug ─────────────────────────────────────────────────────────────────
    output logic [4:0]                            fsm_state_dbg
);

    // =========================================================================
    // FSM state encoding
    // =========================================================================
    typedef enum logic [4:0] {
        ST_IDLE        = 5'd0,
        ST_PROJ_Q      = 5'd1,    // Q   = X  · W_Q
        ST_PROJ_K      = 5'd2,    // K   = X  · W_K
        ST_PROJ_V      = 5'd3,    // V   = X  · W_V
        ST_PROJ_POS    = 5'd4,    // POS = P  · W_pos
        ST_BIAS_ADD    = 5'd5,    // QU  = Q + u_bias,  QV = Q + v_bias
        ST_CONT_SCORE  = 5'd6,    // content_score_h = QU_h · K_h^T
        ST_POS_SCORE   = 5'd7,    // pos_score_h     = QV_h · POS_h^T
        ST_REL_SKEW    = 5'd8,    // Transformer-XL cyclic row-shift skew
        ST_SCALE_ADD   = 5'd9,    // (cont+pos)/√D_MODEL + mask
        ST_SOFTMAX     = 5'd10,   // A_h = softmax(score_h)
        ST_CONTEXT     = 5'd11,   // C_h = A_h · V_h
        ST_OUT_PROJ    = 5'd12,   // Y   = concat(C) · W_O
        ST_EMIT        = 5'd13,   // stream Y row-by-row
        ST_DONE        = 5'd14
    } state_t;

    state_t state;
    assign fsm_state_dbg = state;

    // =========================================================================
    // Scale constant:  1/√D_MODEL  as Q(DATA_WIDTH-FRAC_WIDTH, FRAC_WIDTH)
    // Matches Python:  self.sqrt_dim = math.sqrt(d_model)   (not head_dim)
    // =========================================================================
    localparam real  SCALE_REAL = 1.0 / $sqrt(real'(D_MODEL));
    localparam logic [DATA_WIDTH-1:0] SCALE_FP =
        DATA_WIDTH'($rtoi(SCALE_REAL * real'(1 << FRAC_WIDTH)));

    // =========================================================================
    // On-chip data buffers (flat packed)
    // Row r, column c  →  bits [(r*D_MODEL+c+1)*DATA_WIDTH-1 -: DATA_WIDTH]
    // =========================================================================
    logic [MAT_BITS-1:0]   buf_X;      // input activations
    logic [MAT_BITS-1:0]   buf_P;      // positional embeddings
    logic [MAT_BITS-1:0]   buf_Q;      // Q projection
    logic [MAT_BITS-1:0]   buf_K;      // K projection
    logic [MAT_BITS-1:0]   buf_V;      // V projection
    logic [MAT_BITS-1:0]   buf_POS;    // position projection
    logic [MAT_BITS-1:0]   buf_QU;     // Q + u_bias
    logic [MAT_BITS-1:0]   buf_QV;     // Q + v_bias
    logic [MAT_BITS-1:0]   buf_C;      // context (concat heads)
    logic [MAT_BITS-1:0]   buf_Y;      // output

    // Score / attention buffers  [NUM_HEADS × SEQ_LEN × SEQ_LEN]
    logic [SCORE_BITS-1:0] buf_cont;   // content_score
    logic [SCORE_BITS-1:0] buf_pos;    // pos_score (pre- and post-skew)
    logic [SCORE_BITS-1:0] buf_score;  // combined scaled score
    logic [SCORE_BITS-1:0] buf_attn;   // softmax output

    // Bias registers  [NUM_HEADS × HEAD_DIM]
    logic [BIAS_BITS-1:0]  u_bias;
    logic [BIAS_BITS-1:0]  v_bias;

    // Weight store: 5 matrices (W_Q, W_K, W_V, W_pos, W_O)
    logic [5*WMAT_BITS-1:0] weights;

    // =========================================================================
    // Loop counters  –  all synchronously resettable registers
    // =========================================================================
    logic [$clog2(SEQ_LEN)-1:0]   row_cnt;
    logic [$clog2(TILE_R)-1:0]    tr_cnt;
    logic [$clog2(TILE_C)-1:0]    tc_cnt;
    logic [$clog2(NUM_HEADS)-1:0] head_cnt;
    logic [$clog2(SEQ_LEN)-1:0]   seq_cnt;
    logic                         sm_pending;

    // Integer wires for clean bit-slice arithmetic in always blocks
    wire [31:0] irow  = 32'(row_cnt);
    wire [31:0] itr   = 32'(tr_cnt);
    wire [31:0] itc   = 32'(tc_cnt);
    wire [31:0] ihead = 32'(head_cnt);
    wire [31:0] iseq  = 32'(seq_cnt);

    // =========================================================================
    // Weight and bias loading  (reset-able)
    // =========================================================================
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            weights <= '0;
        end else if (weight_load) begin
            weights[(int'(weight_sel)*WMAT_BITS +
                     (int'(weight_tr)*TILE_C + int'(weight_tc) + 1)*WTILE_BITS - 1)
                    -: WTILE_BITS] <= weight_tile_flat;
        end
    end

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            u_bias <= '0;
            v_bias <= '0;
        end else if (bias_load) begin
            if (!bias_sel) u_bias <= bias_flat;
            else           v_bias <= bias_flat;
        end
    end

    // =========================================================================
    // Input capture  (reset-able)
    // =========================================================================
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            buf_X <= '0;
            buf_P <= '0;
        end else begin
            if (x_valid)
                buf_X[(int'(x_row_idx)*D_MODEL + D_MODEL)*DATA_WIDTH - 1 -: ROW_BITS]
                    <= x_row_flat;
            if (p_valid)
                buf_P[(int'(p_row_idx)*D_MODEL + D_MODEL)*DATA_WIDTH - 1 -: ROW_BITS]
                    <= p_row_flat;
        end
    end

    // =========================================================================
    // Systolic array instance
    // =========================================================================
    logic                           sa_wload;
    logic [WTILE_BITS-1:0]          sa_wflat;
    logic [SA_ROWS*DATA_WIDTH-1:0]  sa_rowflat;
    logic                           sa_rvalid;
    wire  [SA_COLS*ACCUM_WIDTH-1:0] sa_colflat;
    wire                            sa_cvalid;

    systolic_array #(
        .DATA_WIDTH (DATA_WIDTH),
        .FRAC_WIDTH (FRAC_WIDTH),
        .ACCUM_WIDTH(ACCUM_WIDTH),
        .SA_ROWS    (SA_ROWS),
        .SA_COLS    (SA_COLS)
    ) u_sa (
        .clk          (clk),
        .rst_n        (rst_n),
        .weight_load  (sa_wload),
        .weight_flat  (sa_wflat),
        .row_data_flat(sa_rowflat),
        .row_valid    (sa_rvalid),
        .col_acc_flat (sa_colflat),
        .col_valid    (sa_cvalid)
    );

    // =========================================================================
    // Softmax unit instance
    // =========================================================================
    logic                          sm_start;
    logic [SEQ_LEN*DATA_WIDTH-1:0] sm_scores;
    wire  [SEQ_LEN*DATA_WIDTH-1:0] sm_attn;
    wire                           sm_done;

    softmax_unit #(
        .DATA_WIDTH (DATA_WIDTH),
        .FRAC_WIDTH (FRAC_WIDTH),
        .ACCUM_WIDTH(ACCUM_WIDTH),
        .SEQ_LEN    (SEQ_LEN)
    ) u_sm (
        .clk        (clk),
        .rst_n      (rst_n),
        .start      (sm_start),
        .scores_flat(sm_scores),
        .attn_flat  (sm_attn),
        .done       (sm_done)
    );

    // =========================================================================
    // Combinational: SA weight / data steering + softmax feed
    // =========================================================================
    always_comb begin
        sa_wload   = 1'b0;
        sa_wflat   = '0;
        sa_rowflat = '0;
        sa_rvalid  = 1'b0;
        sm_start   = 1'b0;
        sm_scores  = '0;

        case (state)
            // ── Linear projections  Q, K, V, POS, Y ──────────────────────────
            ST_PROJ_Q, ST_PROJ_K, ST_PROJ_V, ST_PROJ_POS, ST_OUT_PROJ: begin
                begin
                    reg [31:0]      ws;
                    reg [MAT_BITS-1:0] src;
                    ws  = (state == ST_PROJ_Q)   ? 32'd0 :
                          (state == ST_PROJ_K)   ? 32'd1 :
                          (state == ST_PROJ_V)   ? 32'd2 :
                          (state == ST_PROJ_POS) ? 32'd3 : 32'd4;
                    src = (state == ST_PROJ_POS) ? buf_P  :
                          (state == ST_OUT_PROJ) ? buf_C  : buf_X;
                    sa_wflat  = weights[(ws*WMAT_BITS +
                                         (itr*TILE_C + itc + 1)*WTILE_BITS - 1)
                                        -: WTILE_BITS];
                    sa_wload  = 1'b1;
                    sa_rvalid = 1'b1;
                    for (int c2 = 0; c2 < SA_ROWS; c2++)
                        sa_rowflat[(c2+1)*DATA_WIDTH-1 -: DATA_WIDTH] =
                            src[(irow*D_MODEL + itr*SA_ROWS + c2 + 1)*DATA_WIDTH - 1
                                -: DATA_WIDTH];
                end
            end

            // ── content_score_h = QU_h · K_h^T ───────────────────────────────
            ST_CONT_SCORE: begin
                for (int kr = 0; kr < SA_ROWS; kr++)
                    for (int kc = 0; kc < SA_COLS; kc++)
                        sa_wflat[(kr*SA_COLS+kc+1)*DATA_WIDTH-1 -: DATA_WIDTH] =
                            buf_K[((itc*SA_COLS+kc)*D_MODEL +
                                    ihead*HEAD_DIM + itr*SA_ROWS + kr + 1)*DATA_WIDTH - 1
                                  -: DATA_WIDTH];
                sa_wload  = 1'b1;
                sa_rvalid = 1'b1;
                for (int qc = 0; qc < SA_ROWS; qc++)
                    sa_rowflat[(qc+1)*DATA_WIDTH-1 -: DATA_WIDTH] =
                        buf_QU[(irow*D_MODEL + ihead*HEAD_DIM + itr*SA_ROWS + qc + 1)*DATA_WIDTH - 1
                               -: DATA_WIDTH];
            end

            // ── pos_score_h = QV_h · POS_h^T ─────────────────────────────────
            ST_POS_SCORE: begin
                for (int pr = 0; pr < SA_ROWS; pr++)
                    for (int pc = 0; pc < SA_COLS; pc++)
                        sa_wflat[(pr*SA_COLS+pc+1)*DATA_WIDTH-1 -: DATA_WIDTH] =
                            buf_POS[((itc*SA_COLS+pc)*D_MODEL +
                                      ihead*HEAD_DIM + itr*SA_ROWS + pr + 1)*DATA_WIDTH - 1
                                    -: DATA_WIDTH];
                sa_wload  = 1'b1;
                sa_rvalid = 1'b1;
                for (int qc = 0; qc < SA_ROWS; qc++)
                    sa_rowflat[(qc+1)*DATA_WIDTH-1 -: DATA_WIDTH] =
                        buf_QV[(irow*D_MODEL + ihead*HEAD_DIM + itr*SA_ROWS + qc + 1)*DATA_WIDTH - 1
                               -: DATA_WIDTH];
            end

            // ── Context C_h = A_h · V_h ───────────────────────────────────────
            ST_CONTEXT: begin
                for (int vr = 0; vr < SA_ROWS; vr++)
                    for (int vc = 0; vc < SA_COLS; vc++)
                        sa_wflat[(vr*SA_COLS+vc+1)*DATA_WIDTH-1 -: DATA_WIDTH] =
                            buf_V[((itr*SA_ROWS+vr)*D_MODEL +
                                    ihead*HEAD_DIM + itc*SA_COLS + vc + 1)*DATA_WIDTH - 1
                                  -: DATA_WIDTH];
                sa_wload  = 1'b1;
                sa_rvalid = 1'b1;
                for (int ac = 0; ac < SA_ROWS; ac++)
                    sa_rowflat[(ac+1)*DATA_WIDTH-1 -: DATA_WIDTH] =
                        buf_attn[(ihead*SEQ_LEN*SEQ_LEN +
                                   irow*SEQ_LEN + itr*SA_ROWS + ac + 1)*DATA_WIDTH - 1
                                 -: DATA_WIDTH];
            end

            // ── Softmax row feed ──────────────────────────────────────────────
            ST_SOFTMAX: begin
                if (sm_pending) begin
                    for (int ki = 0; ki < SEQ_LEN; ki++)
                        sm_scores[(ki+1)*DATA_WIDTH-1 -: DATA_WIDTH] =
                            buf_score[(ihead*SEQ_LEN*SEQ_LEN +
                                       iseq*SEQ_LEN + ki + 1)*DATA_WIDTH - 1
                                      -: DATA_WIDTH];
                    sm_start = 1'b1;
                end
            end

            default: ;
        endcase
    end

    // =========================================================================
    // Main FSM  –  sequential  (all registers synchronously reset to 0)
    // =========================================================================
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            // ── All state registers reset ─────────────────────────────────────
            state      <= ST_IDLE;
            done       <= 1'b0;
            busy       <= 1'b0;
            y_valid    <= 1'b0;
            y_row_flat <= '0;
            y_row_idx  <= '0;
            row_cnt    <= '0;
            tr_cnt     <= '0;
            tc_cnt     <= '0;
            head_cnt   <= '0;
            seq_cnt    <= '0;
            sm_pending <= 1'b0;
            // ── Datapath buffers cleared ──────────────────────────────────────
            buf_Q      <= '0;  buf_K     <= '0;  buf_V     <= '0;
            buf_POS    <= '0;  buf_QU    <= '0;  buf_QV    <= '0;
            buf_C      <= '0;  buf_Y     <= '0;
            buf_cont   <= '0;  buf_pos   <= '0;
            buf_score  <= '0;  buf_attn  <= '0;
        end else begin
            done    <= 1'b0;
            y_valid <= 1'b0;

            case (state)
                // ── Wait for host ─────────────────────────────────────────────
                ST_IDLE: begin
                    busy <= 1'b0;
                    if (start) begin
                        busy       <= 1'b1;
                        row_cnt    <= '0;  tr_cnt  <= '0;  tc_cnt   <= '0;
                        head_cnt   <= '0;  seq_cnt <= '0;  sm_pending <= 1'b0;
                        state      <= ST_PROJ_Q;
                    end
                end

                // ── Q = X · W_Q ───────────────────────────────────────────────
                ST_PROJ_Q: begin
                    if (sa_cvalid) begin
                        for (int c2 = 0; c2 < SA_COLS; c2++)
                            buf_Q[(irow*D_MODEL + itc*SA_COLS + c2 + 1)*DATA_WIDTH - 1
                                  -: DATA_WIDTH]
                                <= DATA_WIDTH'(sa_colflat[(c2+1)*ACCUM_WIDTH-1 -: ACCUM_WIDTH]
                                               >>> FRAC_WIDTH);
                        if (tr_cnt == TILE_R-1) begin tr_cnt <= '0;
                            if (tc_cnt == TILE_C-1) begin tc_cnt <= '0;
                                if (row_cnt == SEQ_LEN-1) begin
                                    row_cnt <= '0; state <= ST_PROJ_K;
                                end else row_cnt <= row_cnt + 1;
                            end else tc_cnt <= tc_cnt + 1;
                        end else tr_cnt <= tr_cnt + 1;
                    end
                end

                // ── K = X · W_K ───────────────────────────────────────────────
                ST_PROJ_K: begin
                    if (sa_cvalid) begin
                        for (int c2 = 0; c2 < SA_COLS; c2++)
                            buf_K[(irow*D_MODEL + itc*SA_COLS + c2 + 1)*DATA_WIDTH - 1
                                  -: DATA_WIDTH]
                                <= DATA_WIDTH'(sa_colflat[(c2+1)*ACCUM_WIDTH-1 -: ACCUM_WIDTH]
                                               >>> FRAC_WIDTH);
                        if (tr_cnt == TILE_R-1) begin tr_cnt <= '0;
                            if (tc_cnt == TILE_C-1) begin tc_cnt <= '0;
                                if (row_cnt == SEQ_LEN-1) begin
                                    row_cnt <= '0; state <= ST_PROJ_V;
                                end else row_cnt <= row_cnt + 1;
                            end else tc_cnt <= tc_cnt + 1;
                        end else tr_cnt <= tr_cnt + 1;
                    end
                end

                // ── V = X · W_V ───────────────────────────────────────────────
                ST_PROJ_V: begin
                    if (sa_cvalid) begin
                        for (int c2 = 0; c2 < SA_COLS; c2++)
                            buf_V[(irow*D_MODEL + itc*SA_COLS + c2 + 1)*DATA_WIDTH - 1
                                  -: DATA_WIDTH]
                                <= DATA_WIDTH'(sa_colflat[(c2+1)*ACCUM_WIDTH-1 -: ACCUM_WIDTH]
                                               >>> FRAC_WIDTH);
                        if (tr_cnt == TILE_R-1) begin tr_cnt <= '0;
                            if (tc_cnt == TILE_C-1) begin tc_cnt <= '0;
                                if (row_cnt == SEQ_LEN-1) begin
                                    row_cnt <= '0; state <= ST_PROJ_POS;
                                end else row_cnt <= row_cnt + 1;
                            end else tc_cnt <= tc_cnt + 1;
                        end else tr_cnt <= tr_cnt + 1;
                    end
                end

                // ── POS = P · W_pos ───────────────────────────────────────────
                ST_PROJ_POS: begin
                    if (sa_cvalid) begin
                        for (int c2 = 0; c2 < SA_COLS; c2++)
                            buf_POS[(irow*D_MODEL + itc*SA_COLS + c2 + 1)*DATA_WIDTH - 1
                                    -: DATA_WIDTH]
                                <= DATA_WIDTH'(sa_colflat[(c2+1)*ACCUM_WIDTH-1 -: ACCUM_WIDTH]
                                               >>> FRAC_WIDTH);
                        if (tr_cnt == TILE_R-1) begin tr_cnt <= '0;
                            if (tc_cnt == TILE_C-1) begin tc_cnt <= '0;
                                if (row_cnt == SEQ_LEN-1) begin
                                    row_cnt <= '0; state <= ST_BIAS_ADD;
                                end else row_cnt <= row_cnt + 1;
                            end else tc_cnt <= tc_cnt + 1;
                        end else tr_cnt <= tr_cnt + 1;
                    end
                end

                // ── QU = Q + u_bias,  QV = Q + v_bias ────────────────────────
                // u/v_bias: [NUM_HEADS × HEAD_DIM], tiled across SEQ_LEN rows.
                ST_BIAS_ADD: begin
                    for (int rr = 0; rr < SEQ_LEN; rr++) begin
                        for (int h2 = 0; h2 < NUM_HEADS; h2++) begin
                            for (int d2 = 0; d2 < HEAD_DIM; d2++) begin
                                reg [DATA_WIDTH-1:0] q_v, ub, vb;
                                q_v = buf_Q [(rr*D_MODEL + h2*HEAD_DIM + d2 + 1)*DATA_WIDTH-1 -: DATA_WIDTH];
                                ub  = u_bias[(h2*HEAD_DIM + d2 + 1)*DATA_WIDTH-1 -: DATA_WIDTH];
                                vb  = v_bias[(h2*HEAD_DIM + d2 + 1)*DATA_WIDTH-1 -: DATA_WIDTH];
                                buf_QU[(rr*D_MODEL + h2*HEAD_DIM + d2 + 1)*DATA_WIDTH-1
                                       -: DATA_WIDTH] <= q_v + ub;
                                buf_QV[(rr*D_MODEL + h2*HEAD_DIM + d2 + 1)*DATA_WIDTH-1
                                       -: DATA_WIDTH] <= q_v + vb;
                            end
                        end
                    end
                    head_cnt <= '0; row_cnt <= '0; tr_cnt <= '0; tc_cnt <= '0;
                    state    <= ST_CONT_SCORE;
                end

                // ── content_score_h = QU_h · K_h^T ───────────────────────────
                ST_CONT_SCORE: begin
                    if (sa_cvalid) begin
                        for (int c2 = 0; c2 < SA_COLS; c2++)
                            buf_cont[(ihead*SEQ_LEN*SEQ_LEN +
                                       irow*SEQ_LEN + itc*SA_COLS + c2 + 1)*DATA_WIDTH - 1
                                     -: DATA_WIDTH]
                                <= DATA_WIDTH'(sa_colflat[(c2+1)*ACCUM_WIDTH-1 -: ACCUM_WIDTH]
                                               >>> FRAC_WIDTH);
                        if (tr_cnt == (HEAD_DIM/SA_ROWS)-1) begin tr_cnt <= '0;
                            if (tc_cnt == (SEQ_LEN/SA_COLS)-1) begin tc_cnt <= '0;
                                if (row_cnt == SEQ_LEN-1) begin row_cnt <= '0;
                                    if (head_cnt == NUM_HEADS-1) begin
                                        head_cnt <= '0; state <= ST_POS_SCORE;
                                    end else head_cnt <= head_cnt + 1;
                                end else row_cnt <= row_cnt + 1;
                            end else tc_cnt <= tc_cnt + 1;
                        end else tr_cnt <= tr_cnt + 1;
                    end
                end

                // ── pos_score_h = QV_h · POS_h^T ─────────────────────────────
                ST_POS_SCORE: begin
                    if (sa_cvalid) begin
                        for (int c2 = 0; c2 < SA_COLS; c2++)
                            buf_pos[(ihead*SEQ_LEN*SEQ_LEN +
                                      irow*SEQ_LEN + itc*SA_COLS + c2 + 1)*DATA_WIDTH - 1
                                    -: DATA_WIDTH]
                                <= DATA_WIDTH'(sa_colflat[(c2+1)*ACCUM_WIDTH-1 -: ACCUM_WIDTH]
                                               >>> FRAC_WIDTH);
                        if (tr_cnt == (HEAD_DIM/SA_ROWS)-1) begin tr_cnt <= '0;
                            if (tc_cnt == (SEQ_LEN/SA_COLS)-1) begin tc_cnt <= '0;
                                if (row_cnt == SEQ_LEN-1) begin row_cnt <= '0;
                                    if (head_cnt == NUM_HEADS-1) begin
                                        head_cnt <= '0; state <= ST_REL_SKEW;
                                    end else head_cnt <= head_cnt + 1;
                                end else row_cnt <= row_cnt + 1;
                            end else tc_cnt <= tc_cnt + 1;
                        end else tr_cnt <= tr_cnt + 1;
                    end
                end

                // ── Transformer-XL relative positional skew ───────────────────
                // pos_score[h][i][j] = pre_skew[h][i][(j+i) % SEQ_LEN]
                // Cyclic left-shift of row i by i positions.
                // Single-cycle combinational sweep.
                ST_REL_SKEW: begin
                    for (int h2 = 0; h2 < NUM_HEADS; h2++) begin
                        for (int qi = 0; qi < SEQ_LEN; qi++) begin
                            for (int ki = 0; ki < SEQ_LEN; ki++) begin
                                reg [$clog2(SEQ_LEN)-1:0] src_ki;
                                src_ki = ($clog2(SEQ_LEN))'((ki + qi) % SEQ_LEN);
                                buf_pos[(h2*SEQ_LEN*SEQ_LEN + qi*SEQ_LEN + ki + 1)*DATA_WIDTH-1
                                        -: DATA_WIDTH]
                                    <= buf_pos[(h2*SEQ_LEN*SEQ_LEN + qi*SEQ_LEN +
                                                int'(src_ki) + 1)*DATA_WIDTH-1
                                               -: DATA_WIDTH];
                            end
                        end
                    end
                    state <= ST_SCALE_ADD;
                end

                // ── score = (content + pos) / √D_MODEL + mask ─────────────────
                ST_SCALE_ADD: begin
                    for (int h2 = 0; h2 < NUM_HEADS; h2++) begin
                        for (int qi = 0; qi < SEQ_LEN; qi++) begin
                            for (int ki = 0; ki < SEQ_LEN; ki++) begin
                                reg [DATA_WIDTH-1:0]  cv, pv, sv;
                                reg [ACCUM_WIDTH-1:0] sa_v;
                                cv   = buf_cont[(h2*SEQ_LEN*SEQ_LEN + qi*SEQ_LEN + ki + 1)*DATA_WIDTH-1
                                                -: DATA_WIDTH];
                                pv   = buf_pos [(h2*SEQ_LEN*SEQ_LEN + qi*SEQ_LEN + ki + 1)*DATA_WIDTH-1
                                                -: DATA_WIDTH];
                                sv   = DATA_WIDTH'($signed(cv) + $signed(pv));
                                sa_v = ACCUM_WIDTH'(
                                    ($signed({{(ACCUM_WIDTH-DATA_WIDTH){sv[DATA_WIDTH-1]}},   sv})   *
                                     $signed({{(ACCUM_WIDTH-DATA_WIDTH){SCALE_FP[DATA_WIDTH-1]}}, SCALE_FP}))
                                    >>> FRAC_WIDTH);
                                buf_score[(h2*SEQ_LEN*SEQ_LEN + qi*SEQ_LEN + ki + 1)*DATA_WIDTH-1
                                          -: DATA_WIDTH] <=
                                    use_mask
                                    ? DATA_WIDTH'(sa_v) +
                                      attn_mask_flat[(qi*SEQ_LEN+ki+1)*DATA_WIDTH-1 -: DATA_WIDTH]
                                    : DATA_WIDTH'(sa_v);
                            end
                        end
                    end
                    head_cnt   <= '0;
                    seq_cnt    <= '0;
                    sm_pending <= 1'b1;
                    state      <= ST_SOFTMAX;
                end

                // ── A_h = softmax(score_h) ────────────────────────────────────
                ST_SOFTMAX: begin
                    if (sm_pending) sm_pending <= 1'b0;
                    if (sm_done) begin
                        buf_attn[(ihead*SEQ_LEN*SEQ_LEN + iseq*SEQ_LEN)*DATA_WIDTH
                                 +: SEQ_LEN*DATA_WIDTH] <= sm_attn;
                        if (seq_cnt == SEQ_LEN-1) begin
                            seq_cnt <= '0;
                            if (head_cnt == NUM_HEADS-1) begin
                                head_cnt <= '0; row_cnt <= '0;
                                tr_cnt   <= '0; tc_cnt  <= '0;
                                state    <= ST_CONTEXT;
                            end else begin
                                head_cnt   <= head_cnt + 1;
                                sm_pending <= 1'b1;
                            end
                        end else begin
                            seq_cnt    <= seq_cnt + 1;
                            sm_pending <= 1'b1;
                        end
                    end
                end

                // ── C_h = A_h · V_h ───────────────────────────────────────────
                ST_CONTEXT: begin
                    if (sa_cvalid) begin
                        for (int c2 = 0; c2 < SA_COLS; c2++)
                            buf_C[(irow*D_MODEL + ihead*HEAD_DIM +
                                   itc*SA_COLS + c2 + 1)*DATA_WIDTH - 1
                                  -: DATA_WIDTH]
                                <= DATA_WIDTH'(sa_colflat[(c2+1)*ACCUM_WIDTH-1 -: ACCUM_WIDTH]
                                               >>> FRAC_WIDTH);
                        if (tr_cnt == (SEQ_LEN/SA_ROWS)-1) begin tr_cnt <= '0;
                            if (tc_cnt == (HEAD_DIM/SA_COLS)-1) begin tc_cnt <= '0;
                                if (row_cnt == SEQ_LEN-1) begin row_cnt <= '0;
                                    if (head_cnt == NUM_HEADS-1) begin
                                        head_cnt <= '0; tr_cnt <= '0; tc_cnt <= '0;
                                        state    <= ST_OUT_PROJ;
                                    end else head_cnt <= head_cnt + 1;
                                end else row_cnt <= row_cnt + 1;
                            end else tc_cnt <= tc_cnt + 1;
                        end else tr_cnt <= tr_cnt + 1;
                    end
                end

                // ── Y = concat(C) · W_O ───────────────────────────────────────
                ST_OUT_PROJ: begin
                    if (sa_cvalid) begin
                        for (int c2 = 0; c2 < SA_COLS; c2++)
                            buf_Y[(irow*D_MODEL + itc*SA_COLS + c2 + 1)*DATA_WIDTH - 1
                                  -: DATA_WIDTH]
                                <= DATA_WIDTH'(sa_colflat[(c2+1)*ACCUM_WIDTH-1 -: ACCUM_WIDTH]
                                               >>> FRAC_WIDTH);
                        if (tr_cnt == TILE_R-1) begin tr_cnt <= '0;
                            if (tc_cnt == TILE_C-1) begin tc_cnt <= '0;
                                if (row_cnt == SEQ_LEN-1) begin
                                    row_cnt <= '0; state <= ST_EMIT;
                                end else row_cnt <= row_cnt + 1;
                            end else tc_cnt <= tc_cnt + 1;
                        end else tr_cnt <= tr_cnt + 1;
                    end
                end

                // ── Stream Y row by row ───────────────────────────────────────
                ST_EMIT: begin
                    y_valid    <= 1'b1;
                    y_row_flat <= buf_Y[(irow*D_MODEL + D_MODEL)*DATA_WIDTH - 1 -: ROW_BITS];
                    y_row_idx  <= row_cnt;
                    if (row_cnt == SEQ_LEN-1) begin
                        row_cnt <= '0; y_valid <= 1'b0; state <= ST_DONE;
                    end else row_cnt <= row_cnt + 1;
                end

                ST_DONE: begin
                    done  <= 1'b1;
                    busy  <= 1'b0;
                    state <= ST_IDLE;
                end

                default: state <= ST_IDLE;
            endcase
        end
    end

    // =========================================================================
    // Simulation-only assertions
    // =========================================================================
    // synthesis translate_off
    initial begin
        assert (D_MODEL % NUM_HEADS == 0)
            else $fatal(1,"D_MODEL=%0d not divisible by NUM_HEADS=%0d",D_MODEL,NUM_HEADS);
        assert (D_MODEL % SA_ROWS == 0 && D_MODEL % SA_COLS == 0)
            else $fatal(1,"D_MODEL must be divisible by SA_ROWS and SA_COLS");
        assert (SEQ_LEN % SA_ROWS == 0)
            else $fatal(1,"SEQ_LEN must be divisible by SA_ROWS");
        assert (HEAD_DIM % SA_COLS == 0)
            else $fatal(1,"HEAD_DIM=%0d not divisible by SA_COLS=%0d",HEAD_DIM,SA_COLS);
        assert (ACCUM_WIDTH >= 2*DATA_WIDTH)
            else $fatal(1,"ACCUM_WIDTH must be >= 2*DATA_WIDTH");
    end

    logic done_d;
    always_ff @(posedge clk) begin : blk_dbg
        if (!rst_n) done_d <= 1'b0;
        else        done_d <= done;
        if (rst_n && done_d && done)
            $warning("ASSERT: done held >1 cycle");
        if (rst_n && (state == ST_IDLE) && y_valid)
            $warning("ASSERT: y_valid asserted in IDLE");
    end
    // synthesis translate_on

endmodule : conformer_rel_mhsa_accel

`default_nettype wire
