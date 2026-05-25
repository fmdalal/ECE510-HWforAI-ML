// =============================================================================
// chiplet_head.sv  —  Chiplets 1..8  (one per attention head)
// =============================================================================
// Stage 2 / Stage 4 time-multiplexed head chiplet.
//
// Stage 2 (cfg_mode=0):  scores = Q[h] * Kt[h]   (QKᵀ tiled)
//   - RX Q tile from chiplet 0 via UCIe
//   - RX K tile from chiplet 0 via UCIe, transpose internally
//   - Compute QKᵀ via 64x64 systolic array
//   - TX raw scores tile to Taylor chiplet (ID 9) via UCIe
//
// Stage 4 (cfg_mode=1):  context[h] = softmax_scores[h] * V[h]
//   - RX normalised probability tile from Taylor chiplet (ID 9) via UCIe
//   - RX V tile (re-loaded from on-chiplet SRAM)
//   - Compute scores x V via same systolic array
//   - TX context tile to chiplet 0 (OutProj) via UCIe
//
// Each head chiplet has its own 64x64 systolic array.
// All 8 head chiplets operate in parallel — same cfg_mode broadcast.
//
// Clock domains used in this file
// --------------------------------
//   clk_core (1 GHz chiplet compute clock)
//           : ALL registered logic in this module
//           : systolic_array instance (sa_main)
//           : ucie_tx / ucie_rx FDI instances (u_rxa, u_rxb, u_tx)
//           : K transpose register, flush/clear counters, TX select FF
// =============================================================================

`timescale 1ns/1ps
`default_nettype none

module chiplet_head #(
    parameter int HEAD_ID   = 0,     // 0..7
    parameter int D_HEAD    = 64,
    parameter int TILE      = 64,
    parameter int K_DIM     = 64,
    parameter int SEQ_TILE  = 64     // sequence length tile
)(
    input  wire        clk_core,   // 1 GHz chiplet compute clock
    input  wire        rst_n,

    // -------------------------------------------------------------------------
    // Configuration
    // -------------------------------------------------------------------------
    input  wire        cfg_mode,      // 0=QKt stage2, 1=ScoresxV stage4
    input  wire [7:0]  cfg_num_tiles,
    input  wire        cfg_start,
    output logic       cfg_done,
    output logic [3:0] chiplet_id,    // static: HEAD_ID+1

    // -------------------------------------------------------------------------
    // UCIe RX A  —  Q tile (stage2) or softmax probabilities (stage4)
    //              from chiplet 0 (stage2) or Taylor chiplet 9 (stage4)
    // -------------------------------------------------------------------------
    input  wire [511:0] rxa_bump_data,
    input  wire         rxa_bump_valid,
    output wire         rxa_bump_credit,

    // -------------------------------------------------------------------------
    // UCIe RX B  —  K tile (stage2) or V tile (stage4) from chiplet 0
    // -------------------------------------------------------------------------
    input  wire [511:0] rxb_bump_data,
    input  wire         rxb_bump_valid,
    output wire         rxb_bump_credit,

    // -------------------------------------------------------------------------
    // UCIe TX  —  raw scores (stage2) to Taylor chiplet 9
    //             or context tile (stage4) to chiplet 0
    // -------------------------------------------------------------------------
    output wire [511:0] tx_bump_data,
    output wire         tx_bump_valid,
    input  wire         tx_bump_credit,

    // -------------------------------------------------------------------------
    // Scale factor port: 1/sqrt(D_HEAD) in BF16
    // For D_HEAD=64: 1/8 = 0.125 = BF16 0x3E00
    // -------------------------------------------------------------------------
    input  wire [15:0]  scale_factor
);
    assign chiplet_id = HEAD_ID[3:0] + 4'd1;

    // -------------------------------------------------------------------------
    // UCIe RX A: Q or probs
    // -------------------------------------------------------------------------
    logic        rxa_valid;
    logic [3:0]  rxa_src;
    logic [15:0] rxa_tile [TILE][TILE];

    ucie_rx #(.TILE_DIM(TILE)) u_rxa (
        .clk_core(clk_core), .rst_n(rst_n),
        .bump_data(rxa_bump_data), .bump_valid(rxa_bump_valid),
        .bump_credit(rxa_bump_credit),
        .rx_valid(rxa_valid), .rx_src_id(rxa_src),
        .rx_tile(rxa_tile), .rx_ready(1'b1),
        .rx_crc_err(), .rx_seq_err()
    );

    // -------------------------------------------------------------------------
    // UCIe RX B: K or V
    // -------------------------------------------------------------------------
    logic        rxb_valid;
    logic [3:0]  rxb_src;
    logic [15:0] rxb_tile [TILE][TILE];

    ucie_rx #(.TILE_DIM(TILE)) u_rxb (
        .clk_core(clk_core), .rst_n(rst_n),
        .bump_data(rxb_bump_data), .bump_valid(rxb_bump_valid),
        .bump_credit(rxb_bump_credit),
        .rx_valid(rxb_valid), .rx_src_id(rxb_src),
        .rx_tile(rxb_tile), .rx_ready(1'b1),
        .rx_crc_err(), .rx_seq_err()
    );

    // -------------------------------------------------------------------------
    // Transpose K -> Kt for Stage 2
    // -------------------------------------------------------------------------
    logic [15:0] kt_tile [TILE][TILE];

    always_ff @(posedge clk_core or negedge rst_n) begin : kt_ff
        if (!rst_n) begin
            for (int i = 0; i < TILE; i++)
                for (int j = 0; j < TILE; j++)
                    kt_tile[i][j] <= 16'h0;
        end else if (rxb_valid & ~cfg_mode) begin
            for (int i = 0; i < TILE; i++)
                for (int j = 0; j < TILE; j++)
                    kt_tile[j][i] <= rxb_tile[i][j];
        end
    end

    // -------------------------------------------------------------------------
    // Systolic array: shared for Stage2 (QKt) and Stage4 (scores x V)
    // A input: rxa_tile (Q or softmax probs)
    // B input: kt_tile (stage2) or rxb_tile (stage4 V)
    // -------------------------------------------------------------------------
    wire [15:0] sa_b_col [TILE][TILE];
    genvar gi, gj;
    generate
        for (gi = 0; gi < TILE; gi++)
            for (gj = 0; gj < TILE; gj++)
                assign sa_b_col[gi][gj] = cfg_mode
                                         ? rxb_tile[gi][gj]   // Stage4: V
                                         : kt_tile[gi][gj];   // Stage2: Kt
    endgenerate

    wire [15:0] a_rows [TILE];
    generate
        for (gi = 0; gi < TILE; gi++)
            assign a_rows[gi] = rxa_tile[gi][0];
    endgenerate

    // Flush counter
    logic [7:0]  tile_cnt;
    logic        flush_pulse;
    wire         sa_start = rxa_valid & rxb_valid;

    always_ff @(posedge clk_core or negedge rst_n) begin : flush_ff
        if (!rst_n) begin
            tile_cnt    <= 8'd0;
            flush_pulse <= 1'b0;
        end else if (!cfg_start) begin
            tile_cnt    <= 8'd0;
            flush_pulse <= 1'b0;
        end else if (sa_start) begin
            tile_cnt    <= tile_cnt + 8'd1;
            flush_pulse <= (tile_cnt == cfg_num_tiles - 8'd1);
        end else begin
            flush_pulse <= 1'b0;
        end
    end

    logic [2:0] clear_r;
    wire        clear_pulse = clear_r[2];

    always_ff @(posedge clk_core or negedge rst_n) begin : clear_ff
        if (!rst_n)         clear_r <= 3'b111;
        else if (cfg_start) clear_r <= 3'b111;
        else                clear_r <= {clear_r[1:0], 1'b0};
    end

    wire [15:0] sa_out [TILE][TILE];
    wire        sa_valid;

    systolic_array #(.M(TILE),.N(TILE),.K(K_DIM)) sa_main (
        .clk_core(clk_core), .rst_n(rst_n),
        .en(sa_start | ~clear_pulse),
        .clear(clear_pulse), .flush(flush_pulse),
        .a_row(a_rows), .b_col(sa_b_col),
        .c_out(sa_out), .valid_out(sa_valid)
    );

    // -------------------------------------------------------------------------
    // Stage 2: scale QKt by 1/sqrt(D_HEAD) element-wise
    // Each element: scaled = sa_out[i][j] * scale_factor  (single BF16 mul)
    // -------------------------------------------------------------------------
    logic [15:0] scaled_scores [TILE][TILE];

    genvar si, sj;
    generate
        for (si = 0; si < TILE; si++) begin : sc_row
            for (sj = 0; sj < TILE; sj++) begin : sc_col
                wire [31:0] sc_fp32_nc;
                bf16_mac scale_mac (
                    .clk_core(clk_core), .rst_n(rst_n),
                    .en(sa_valid & ~cfg_mode),
                    .flush(sa_valid & ~cfg_mode),
                    .a(sa_out[si][sj]), .b(scale_factor),
                    .acc_fp32_in(32'h0),
                    .acc_fp32_out(sc_fp32_nc),
                    .acc_bf16_out(scaled_scores[si][sj])
                );
            end
        end
    endgenerate

    // -------------------------------------------------------------------------
    // Output selection:
    //   Stage 2: send scaled_scores to Taylor chiplet (ID 9)
    //   Stage 4: send sa_out (context) to chiplet 0
    // -------------------------------------------------------------------------
    logic        tx_valid_i;
    logic [15:0] tx_tile  [TILE][TILE];
    logic [3:0]  tx_dst;

    always_ff @(posedge clk_core or negedge rst_n) begin : tx_sel_ff
        if (!rst_n) begin
            tx_valid_i <= 1'b0;
            cfg_done   <= 1'b0;
        end else begin
            tx_valid_i <= 1'b0;
            cfg_done   <= 1'b0;
            if (sa_valid & ~cfg_mode) begin
                // Stage 2: scores -> Taylor (ID 9)
                tx_tile    <= scaled_scores;
                tx_dst     <= 4'd9;
                tx_valid_i <= 1'b1;
            end else if (sa_valid & cfg_mode) begin
                // Stage 4: context -> chiplet 0 OutProj
                tx_tile    <= sa_out;
                tx_dst     <= 4'd0;
                tx_valid_i <= 1'b1;
                cfg_done   <= 1'b1;
            end
        end
    end

    ucie_tx #(.TILE_DIM(TILE)) u_tx (
        .clk_core(clk_core), .rst_n(rst_n),
        .tx_valid(tx_valid_i),
        .tx_src_id(HEAD_ID[3:0] + 4'd1),
        .tx_dst_id(tx_dst),
        .tx_tile(tx_tile), .tx_ready(),
        .bump_data(tx_bump_data), .bump_valid(tx_bump_valid),
        .bump_credit(tx_bump_credit)
    );

endmodule

`default_nettype wire
// =============================================================================
// End of chiplet_head.sv
// =============================================================================

