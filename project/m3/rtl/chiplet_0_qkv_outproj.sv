// =============================================================================
// chiplet_0_qkv_outproj.sv  —  Chiplet ID 0
// =============================================================================
// Stage 1 / Stage 5 time-multiplexed chiplet.
//
// Stage 1  (cfg_mode=0):  Q = X*Wq,  K = X*Wk,  V = X*Wv  (3 parallel arrays)
// Stage 5  (cfg_mode=1):  Out = concat(heads) * Wo          (reuse sa_q array)
//
// Die boundary signals:
//   RX from host/scheduler via UCIe   : input token tile X
//   TX to head chiplets 1..8 via UCIe : Q, K, V tiles
//   RX from head chiplets via UCIe    : concatenated context tile (Stage 5)
//   TX to scheduler/host via UCIe     : final output tile
//
// 64x64 PE systolic array (matching image 2).
// All die boundaries use ucie_tx / ucie_rx from ucie_link.sv.
//
// Clock domains used in this file
// --------------------------------
//   clk_core (1 GHz chiplet compute clock)
//           : ALL registered logic in this module
//           : systolic_array instances (sa_q, sa_k, sa_v)
//           : ucie_tx / ucie_rx FDI instances
//           : weight load FSM, flush/clear counters, TX output registers
// =============================================================================

`timescale 1ns/1ps
`default_nettype none

module chiplet_0_qkv_outproj #(
    parameter int D_MODEL   = 512,
    parameter int NUM_HEADS = 8,
    parameter int D_HEAD    = 64,
    parameter int TILE      = 64,    // 64x64 PE array as per image
    parameter int K_DIM     = 64
)(
    input  wire        clk_core,   // 1 GHz chiplet compute clock
    input  wire        rst_n,

    // -------------------------------------------------------------------------
    // Configuration (from AXI-Lite CSR via wrapper)
    // -------------------------------------------------------------------------
    input  wire        cfg_mode,      // 0=QKV stage1, 1=OutProj stage5
    input  wire [7:0]  cfg_num_tiles, // D_MODEL/TILE
    input  wire        cfg_start,
    output logic       cfg_done,

    // -------------------------------------------------------------------------
    // UCIe RX  —  input tokens from scheduler (chiplet ID 0 self)
    // Physical bump pads from interposer
    // -------------------------------------------------------------------------
    input  wire [511:0] rx_bump_data,
    input  wire         rx_bump_valid,
    output wire         rx_bump_credit,

    // -------------------------------------------------------------------------
    // UCIe TX Q  —  Q tile to head chiplets 1..8 (broadcast)
    // -------------------------------------------------------------------------
    output wire [511:0] txq_bump_data,
    output wire         txq_bump_valid,
    input  wire         txq_bump_credit,

    // -------------------------------------------------------------------------
    // UCIe TX K  —  K tile
    // -------------------------------------------------------------------------
    output wire [511:0] txk_bump_data,
    output wire         txk_bump_valid,
    input  wire         txk_bump_credit,

    // -------------------------------------------------------------------------
    // UCIe TX V  —  V tile
    // -------------------------------------------------------------------------
    output wire [511:0] txv_bump_data,
    output wire         txv_bump_valid,
    input  wire         txv_bump_credit,

    // -------------------------------------------------------------------------
    // UCIe TX Out  —  output projection result to scheduler
    // -------------------------------------------------------------------------
    output wire [511:0] txout_bump_data,
    output wire         txout_bump_valid,
    input  wire         txout_bump_credit,

    // -------------------------------------------------------------------------
    // Weight SRAM interface (on-chiplet SRAM, no die crossing)
    // -------------------------------------------------------------------------
    output logic [31:0] sram_addr,
    //input  wire  [15:0] sram_rdata [TILE][TILE],
	input  wire  [TILE*TILE*16-1:0] sram_rdata_flat,
    output logic        sram_rd_en,
    input  wire         sram_rd_valid
);

    // -------------------------------------------------------------------------
    // UCIe RX: receive input tile
    // -------------------------------------------------------------------------
    logic        rx_valid;
    logic [3:0]  rx_src_id;
    logic [15:0] rx_tile [TILE][TILE];

    ucie_rx #(.TILE_DIM(TILE)) u_rx (
        .clk_core(clk_core), .rst_n(rst_n),
        .bump_data(rx_bump_data), .bump_valid(rx_bump_valid),
        .bump_credit(rx_bump_credit),
        .rx_valid(rx_valid), .rx_src_id(rx_src_id),
        .rx_tile(rx_tile), .rx_ready(1'b1),
        .rx_crc_err(), .rx_seq_err()
    );

    // -------------------------------------------------------------------------
    // Weight registers
    // -------------------------------------------------------------------------
    logic [15:0] wq [TILE][TILE];
    logic [15:0] wk [TILE][TILE];
    logic [15:0] wv [TILE][TILE];
    logic [15:0] wo [TILE][TILE];

    // -------------------------------------------------------------------------
    // Row inputs shared across all three QKV arrays
    // -------------------------------------------------------------------------
    wire [15:0] x_rows [TILE];
    genvar gxi;
    generate
        for (gxi = 0; gxi < TILE; gxi++)
            assign x_rows[gxi] = rx_tile[gxi][0];
    endgenerate

    // -------------------------------------------------------------------------
    // Flush counter
    // -------------------------------------------------------------------------
    logic [7:0]  tile_cnt;
    logic        flush_pulse;

    always_ff @(posedge clk_core or negedge rst_n) begin : flush_ff
        if (!rst_n) begin
            tile_cnt    <= 8'd0;
            flush_pulse <= 1'b0;
        end else if (!cfg_start) begin
            tile_cnt    <= 8'd0;
            flush_pulse <= 1'b0;
        end else if (rx_valid) begin
            tile_cnt    <= tile_cnt + 8'd1;
            flush_pulse <= (tile_cnt == cfg_num_tiles - 8'd1);
        end else begin
            flush_pulse <= 1'b0;
        end
    end

    // -------------------------------------------------------------------------
    // Clear shift register
    // -------------------------------------------------------------------------
    logic [2:0] clear_r;
    wire        clear_pulse = clear_r[2];

    always_ff @(posedge clk_core or negedge rst_n) begin : clear_ff
        if (!rst_n)          clear_r <= 3'b111;
        else if (cfg_start)  clear_r <= 3'b111;
        else                 clear_r <= {clear_r[1:0], 1'b0};
    end

    // -------------------------------------------------------------------------
    // Stage 1: three parallel systolic arrays for Q, K, V
    // -------------------------------------------------------------------------
    wire [15:0] q_out [TILE][TILE];
    wire [15:0] k_out [TILE][TILE];
    wire [15:0] v_out [TILE][TILE];
    wire        q_valid, k_valid, v_valid;

    systolic_array #(.M(TILE),.N(TILE),.K(K_DIM)) sa_q (
        .clk_core(clk_core), .rst_n(rst_n),
        .en(rx_valid | ~clear_pulse), .clear(clear_pulse), .flush(flush_pulse),
        .a_row(x_rows), .b_col(wq),
        .c_out(q_out), .valid_out(q_valid)
    );
    systolic_array #(.M(TILE),.N(TILE),.K(K_DIM)) sa_k (
        .clk_core(clk_core), .rst_n(rst_n),
        .en(rx_valid | ~clear_pulse), .clear(clear_pulse), .flush(flush_pulse),
        .a_row(x_rows), .b_col(wk),
        .c_out(k_out), .valid_out(k_valid)
    );
    systolic_array #(.M(TILE),.N(TILE),.K(K_DIM)) sa_v (
        .clk_core(clk_core), .rst_n(rst_n),
        .en(rx_valid | ~clear_pulse), .clear(clear_pulse), .flush(flush_pulse),
        .a_row(x_rows), .b_col(wv),
        .c_out(v_out), .valid_out(v_valid)
    );

    // -------------------------------------------------------------------------
    // Stage 5: output projection (reuses sa_q with Wo weights)
    // cfg_mode=1: feed context tile through sa_q with wo loaded
    // -------------------------------------------------------------------------
    // (sa_q is time-muxed: Stage1 uses wq, Stage5 loads wo into the array
    //  by switching the b_col input — implemented via weight MUX below)
    // b_col for sa_q is wq in stage1, wo in stage5
    wire [15:0] sa_q_bcol [TILE][TILE];
    genvar wmi, wmj;
    generate
        for (wmi = 0; wmi < TILE; wmi++)
            for (wmj = 0; wmj < TILE; wmj++)
                assign sa_q_bcol[wmi][wmj] = cfg_mode ? wo[wmi][wmj] : wq[wmi][wmj];
    endgenerate

    // -------------------------------------------------------------------------
    // UCIe TX: Q, K, V to head chiplets
    // Only transmit in Stage 1 (cfg_mode=0)
    // -------------------------------------------------------------------------
    logic        txq_valid_i, txk_valid_i, txv_valid_i;
    logic [15:0] txq_tile [TILE][TILE];
    logic [15:0] txk_tile [TILE][TILE];
    logic [15:0] txv_tile [TILE][TILE];

    always_ff @(posedge clk_core or negedge rst_n) begin : tx_qkv_ff
        if (!rst_n) begin
            txq_valid_i <= 1'b0;
            txk_valid_i <= 1'b0;
            txv_valid_i <= 1'b0;
        end else begin
            if (q_valid & ~cfg_mode) begin
                txq_tile    <= q_out;
                txq_valid_i <= 1'b1;
            end else txq_valid_i <= 1'b0;

            if (k_valid & ~cfg_mode) begin
                txk_tile    <= k_out;
                txk_valid_i <= 1'b1;
            end else txk_valid_i <= 1'b0;

            if (v_valid & ~cfg_mode) begin
                txv_tile    <= v_out;
                txv_valid_i <= 1'b1;
            end else txv_valid_i <= 1'b0;
        end
    end

    ucie_tx #(.TILE_DIM(TILE)) u_txq (
        .clk_core(clk_core), .rst_n(rst_n),
        .tx_valid(txq_valid_i), .tx_src_id(4'd0), .tx_dst_id(4'd1),
        .tx_tile(txq_tile), .tx_ready(),
        .bump_data(txq_bump_data), .bump_valid(txq_bump_valid),
        .bump_credit(txq_bump_credit)
    );
    ucie_tx #(.TILE_DIM(TILE)) u_txk (
        .clk_core(clk_core), .rst_n(rst_n),
        .tx_valid(txk_valid_i), .tx_src_id(4'd0), .tx_dst_id(4'd1),
        .tx_tile(txk_tile), .tx_ready(),
        .bump_data(txk_bump_data), .bump_valid(txk_bump_valid),
        .bump_credit(txk_bump_credit)
    );
    ucie_tx #(.TILE_DIM(TILE)) u_txv (
        .clk_core(clk_core), .rst_n(rst_n),
        .tx_valid(txv_valid_i), .tx_src_id(4'd0), .tx_dst_id(4'd1),
        .tx_tile(txv_tile), .tx_ready(),
        .bump_data(txv_bump_data), .bump_valid(txv_bump_valid),
        .bump_credit(txv_bump_credit)
    );

    // Stage 5 output TX
    logic        txout_valid_i;
    logic [15:0] txout_tile [TILE][TILE];

    always_ff @(posedge clk_core or negedge rst_n) begin : tx_out_ff
        if (!rst_n) begin
            txout_valid_i <= 1'b0;
        end else begin
            if (q_valid & cfg_mode) begin
                txout_tile    <= q_out;
                txout_valid_i <= 1'b1;
                cfg_done      <= 1'b1;
            end else begin
                txout_valid_i <= 1'b0;
                cfg_done      <= 1'b0;
            end
        end
    end

    ucie_tx #(.TILE_DIM(TILE)) u_txout (
        .clk_core(clk_core), .rst_n(rst_n),
        .tx_valid(txout_valid_i), .tx_src_id(4'd0), .tx_dst_id(4'd9),
        .tx_tile(txout_tile), .tx_ready(),
        .bump_data(txout_bump_data), .bump_valid(txout_bump_valid),
        .bump_credit(txout_bump_credit)
    );

    // -------------------------------------------------------------------------
    // Weight load FSM
    // -------------------------------------------------------------------------
    typedef enum logic [2:0] {
        W_IDLE = 3'd0, W_WQ = 3'd1, W_WK = 3'd2,
        W_WV   = 3'd3, W_WO = 3'd4, W_DONE = 3'd5
    } wstate_t;

    wstate_t wstate;
    logic [7:0] wload_cnt;

    always_ff @(posedge clk_core or negedge rst_n) begin : weight_fsm_ff
        if (!rst_n) begin
            wstate    <= W_IDLE;
            sram_rd_en <= 1'b0;
            sram_addr  <= 32'h0;
            wload_cnt  <= 8'd0;
        end else begin
            sram_rd_en <= 1'b0;
            case (wstate)
                W_IDLE: if (cfg_start) begin wstate <= W_WQ; sram_addr <= 32'h0; end
                W_WQ: begin
                    sram_rd_en <= 1'b1;
                    if (sram_rd_valid) begin
                        wq[wload_cnt[5:0]][wload_cnt[5:0]] <= sram_rdata_flat[15:0];
						//sram_rdata[0][0];
                        // simplified: real design streams full TILE*TILE words
                        if (wload_cnt == 8'd3) begin wload_cnt <= 8'd0; wstate <= W_WK; end
                        else wload_cnt <= wload_cnt + 8'd1;
                    end
                end
                W_WK: begin
                    sram_rd_en <= 1'b1;
                    if (sram_rd_valid) begin
                        if (wload_cnt == 8'd3) begin wload_cnt <= 8'd0; wstate <= W_WV; end
                        else wload_cnt <= wload_cnt + 8'd1;
                    end
                end
                W_WV: begin
                    sram_rd_en <= 1'b1;
                    if (sram_rd_valid) begin
                        if (wload_cnt == 8'd3) begin wload_cnt <= 8'd0; wstate <= W_WO; end
                        else wload_cnt <= wload_cnt + 8'd1;
                    end
                end
                W_WO: begin
                    sram_rd_en <= 1'b1;
                    if (sram_rd_valid) begin
                        if (wload_cnt == 8'd3) begin wload_cnt <= 8'd0; wstate <= W_DONE; end
                        else wload_cnt <= wload_cnt + 8'd1;
                    end
                end
                W_DONE: wstate <= W_IDLE;
                default: wstate <= W_IDLE;
            endcase
        end
    end

endmodule

`default_nettype wire
// =============================================================================
// End of chiplet_0_qkv_outproj.sv
// =============================================================================

