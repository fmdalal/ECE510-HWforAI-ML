// =============================================================================
// chiplet_0_qkv_outproj.sv  —  Chiplet ID 0  [optimised: weight-stationary]
// =============================================================================
// Changes from original
// ----------------------
//  1. Weight-load FSM loads all TILE*TILE words per matrix (was only 4).
//  2. Tile-streaming FSM presents full columns of X and rows of W each cycle.
//  3. Three parallel systolic arrays for Q, K, V; sa_q reused for OutProj.
//  4. acc_clear pulsed at start of each tile multiply.
//
// Load-balance fix (direct tile mux, replaces UCIe arbiter path)
// ---------------------------------------------------------------
//  Two new ports added:
//    ctx_tile_in  [TILE][TILE]  — context tile from compute_core arbiter
//    ctx_tile_valid_in          — pulse when ctx_tile_in is valid
//  In stage 4 (cfg_mode=1) the x_cap block captures ctx_tile_in instead
//  of rx_tile (which comes through ucie_rx from the bump bus).
//  This eliminates the 284-cycle UCIe TX framing overhead per tile that
//  the previous arbiter incurred, saving 728 cycles total (32%).
//  The ucie_rx instance (u_rx) is still present for stage 1 (host tile).
// =============================================================================

`timescale 1ns/1ps
`default_nettype none

module chiplet_0_qkv_outproj #(
    parameter int D_MODEL   = 512,
    parameter int NUM_HEADS = 8,
    parameter int D_HEAD    = 64,
    parameter int TILE      = 64,
    parameter int K_DIM     = 64
)(
    input  wire        clk_core,
    input  wire        clk_link,
    input  wire        rst_n,
    input  wire        cfg_mode,
    input  wire [7:0]  cfg_num_tiles,
    input  wire        cfg_start,
    output logic       cfg_done,

    input  wire [511:0] rx_bump_data,
    input  wire         rx_bump_valid,
    output wire         rx_bump_credit,

    output wire [511:0] txq_bump_data,  output wire txq_bump_valid,  input wire txq_bump_credit,
    output wire [511:0] txk_bump_data,  output wire txk_bump_valid,  input wire txk_bump_credit,
    output wire [511:0] txv_bump_data,  output wire txv_bump_valid,  input wire txv_bump_credit,
    output wire [511:0] txout_bump_data,output wire txout_bump_valid,input wire txout_bump_credit,

    output logic [31:0]             sram_addr,
    input  wire  [TILE*TILE*16-1:0] sram_rdata_flat,
    output logic                    sram_rd_en,
    input  wire                     sram_rd_valid,

    // Direct context tile input (stage 4 only — bypasses UCIe framing)
    // Driven by compute_core arbiter; eliminates 284-cycle UCIe TX per tile.
    input  wire [15:0]              ctx_tile_in   [TILE][TILE],
    input  wire                     ctx_tile_valid_in
);

    // =========================================================================
    // UCIe RX — input token tile X
    // =========================================================================
    logic        rx_valid;
    logic [15:0] rx_tile [TILE][TILE];
    ucie_rx #(.TILE_DIM(TILE)) u_rx (
        .clk_core(clk_core),.rst_n(rst_n),
        .bump_data(rx_bump_data),.bump_valid(rx_bump_valid),.bump_credit(rx_bump_credit),
        .rx_valid(rx_valid),.rx_src_id(),.rx_tile(rx_tile),.rx_ready(1'b1),
        .rx_crc_err(),.rx_seq_err()
    );

    // Capture X tile
    // Stage 1 (cfg_mode=0): capture from ucie_rx (host token tile via bump bus)
    // Stage 4 (cfg_mode=1): capture from ctx_tile_in (direct mux, no UCIe framing)
    logic [15:0] x_tile [TILE][TILE];
    logic        x_cap_done;
    always_ff @(posedge clk_core or negedge rst_n) begin : x_cap
        if (!rst_n) begin
            for (int i=0;i<TILE;i++) for (int j=0;j<TILE;j++) x_tile[i][j]<=16'h0;
            x_cap_done<=1'b0;
        end else begin
            if (!cfg_start) x_cap_done<=1'b0;
            // Stage 1: host token tile from UCIe RX bump bus
            if (rx_valid & ~cfg_mode) begin
                for (int i=0;i<TILE;i++) for (int j=0;j<TILE;j++) x_tile[i][j]<=rx_tile[i][j];
                x_cap_done<=1'b1;
            end
            // Stage 4: context tile direct from arbiter (2-cycle path, no UCIe TX overhead)
            if (ctx_tile_valid_in & cfg_mode) begin
                for (int i=0;i<TILE;i++) for (int j=0;j<TILE;j++) x_tile[i][j]<=ctx_tile_in[i][j];
                x_cap_done<=1'b1;
            end
        end
    end

    // =========================================================================
    // Weight registers — TILE×TILE BF16 each
    // =========================================================================
    logic [15:0] wq [TILE][TILE];
    logic [15:0] wk [TILE][TILE];
    logic [15:0] wv [TILE][TILE];
    logic [15:0] wo [TILE][TILE];

    // =========================================================================
    // Weight-load FSM
    // Loads 4 × TILE×TILE = 16384 words sequentially from SRAM.
    // Each sram_rd_valid beat delivers sram_rdata_flat[15:0].
    // Address map (in BF16 words):
    //   Wq: 0           .. TILE*TILE-1
    //   Wk: TILE*TILE   .. 2*TILE*TILE-1
    //   Wv: 2*TILE*TILE .. 3*TILE*TILE-1
    //   Wo: 3*TILE*TILE .. 4*TILE*TILE-1
    // =========================================================================
    typedef enum logic [2:0] {W_IDLE=0,W_WQ=1,W_WK=2,W_WV=3,W_WO=4,W_DONE=5} wstate_t;
    wstate_t     wstate;
    logic [11:0] wload_cnt;   // 0..4095  (TILE*TILE = 4096)
    logic        weights_ready;

    wire [15:0] sram_word  = sram_rdata_flat[15:0];
    wire  [5:0] wrow       = wload_cnt[11:6];
    wire  [5:0] wcol       = wload_cnt[5:0];
    localparam int TILE_SQ = TILE*TILE;

    always_ff @(posedge clk_core or negedge rst_n) begin : wfsm
        if (!rst_n) begin
            wstate<=W_IDLE; sram_rd_en<=1'b0; sram_addr<=32'h0;
            wload_cnt<=12'd0; weights_ready<=1'b0;
        end else begin
            sram_rd_en<=1'b0;
            case (wstate)
                W_IDLE: begin
                    weights_ready<=1'b0;
                    if (cfg_start) begin wstate<=W_WQ; wload_cnt<=12'd0; sram_addr<=32'h0; end
                end
                W_WQ: begin
                    sram_rd_en<=1'b1;
                    if (sram_rd_valid) begin
                        wq[wrow][wcol]<=sram_word; sram_addr<=sram_addr+32'd1;
                        if (wload_cnt==TILE_SQ-1) begin wload_cnt<=12'd0; wstate<=W_WK; end
                        else wload_cnt<=wload_cnt+12'd1;
                    end
                end
                W_WK: begin
                    sram_rd_en<=1'b1;
                    if (sram_rd_valid) begin
                        wk[wrow][wcol]<=sram_word; sram_addr<=sram_addr+32'd1;
                        if (wload_cnt==TILE_SQ-1) begin wload_cnt<=12'd0; wstate<=W_WV; end
                        else wload_cnt<=wload_cnt+12'd1;
                    end
                end
                W_WV: begin
                    sram_rd_en<=1'b1;
                    if (sram_rd_valid) begin
                        wv[wrow][wcol]<=sram_word; sram_addr<=sram_addr+32'd1;
                        if (wload_cnt==TILE_SQ-1) begin wload_cnt<=12'd0; wstate<=W_WO; end
                        else wload_cnt<=wload_cnt+12'd1;
                    end
                end
                W_WO: begin
                    sram_rd_en<=1'b1;
                    if (sram_rd_valid) begin
                        wo[wrow][wcol]<=sram_word; sram_addr<=sram_addr+32'd1;
                        if (wload_cnt==TILE_SQ-1) begin
                            wload_cnt<=12'd0; wstate<=W_DONE; weights_ready<=1'b1;
                        end else wload_cnt<=wload_cnt+12'd1;
                    end
                end
                W_DONE: begin
                    weights_ready<=1'b1;
                    if (!cfg_start) wstate<=W_IDLE;
                end
                default: wstate<=W_IDLE;
            endcase
        end
    end

    // =========================================================================
    // Compute FSM — streams k=0..TILE-1 into three parallel arrays
    // =========================================================================
    typedef enum logic [2:0] {C_IDLE=0,C_WAIT=1,C_STREAM=2,C_DRAIN=3,C_TX=4} cstate_t;
    cstate_t                    cstate;
    logic [$clog2(TILE)-1:0]    k_cnt;

    logic        sa_data_in, sa_acc_clear, sa_flush;
    logic [15:0] a_col_k  [TILE];  // x_tile column k
    logic [15:0] bq_row_k [TILE];  // wq/wo row k
    logic [15:0] bk_row_k [TILE];  // wk row k
    logic [15:0] bv_row_k [TILE];  // wv row k

    always_comb begin
        for (int i=0;i<TILE;i++) a_col_k[i]  = x_tile[i][k_cnt];
        for (int j=0;j<TILE;j++) bq_row_k[j] = cfg_mode ? wo[k_cnt][j] : wq[k_cnt][j];
        for (int j=0;j<TILE;j++) bk_row_k[j] = wk[k_cnt][j];
        for (int j=0;j<TILE;j++) bv_row_k[j] = wv[k_cnt][j];
    end

    wire q_valid, k_valid, v_valid;

    always_ff @(posedge clk_core or negedge rst_n) begin : cfsm
        if (!rst_n) begin
            cstate<=C_IDLE; k_cnt<='0;
            sa_data_in<=1'b0; sa_acc_clear<=1'b0; sa_flush<=1'b0; cfg_done<=1'b0;
        end else begin
            sa_data_in<=1'b0; sa_acc_clear<=1'b0; sa_flush<=1'b0; cfg_done<=1'b0;
            case (cstate)
                C_IDLE: if (cfg_start) begin k_cnt<='0; cstate<=C_WAIT; end
                C_WAIT: begin
                    if (!cfg_start) cstate<=C_IDLE;
                    else if (weights_ready & x_cap_done) begin
                        sa_acc_clear<=1'b1; k_cnt<='0; cstate<=C_STREAM;
                    end
                end
                C_STREAM: begin
                    sa_data_in<=1'b1;
                    if (k_cnt==(TILE-1)) begin sa_flush<=1'b1; k_cnt<='0; cstate<=C_DRAIN; end
                    else k_cnt<=k_cnt+1'b1;
                end
                C_DRAIN: if (q_valid) cstate<=C_TX;
                C_TX: begin
                    cfg_done   <= cfg_mode;
                    x_cap_done <= 1'b0;  // clear so C_WAIT waits for next arb pulse
                    cstate     <= C_WAIT;
                end
                default: cstate<=C_IDLE;
            endcase
        end
    end

    // =========================================================================
    // Three parallel systolic arrays
    // =========================================================================
    wire [15:0] q_out [TILE][TILE];
    wire [15:0] k_out [TILE][TILE];
    wire [15:0] v_out [TILE][TILE];

    systolic_array #(.M(TILE),.N(TILE),.K(K_DIM)) sa_q (
        .clk_core(clk_core),.rst_n(rst_n),
        .data_in(sa_data_in),.acc_clear(sa_acc_clear),.flush(sa_flush),
        .a_col_in(a_col_k),.b_row_in(bq_row_k),
        .c_out(q_out),.valid_out(q_valid)
    );
    systolic_array #(.M(TILE),.N(TILE),.K(K_DIM)) sa_k (
        .clk_core(clk_core),.rst_n(rst_n),
        .data_in(sa_data_in),.acc_clear(sa_acc_clear),.flush(sa_flush),
        .a_col_in(a_col_k),.b_row_in(bk_row_k),
        .c_out(k_out),.valid_out(k_valid)
    );
    systolic_array #(.M(TILE),.N(TILE),.K(K_DIM)) sa_v (
        .clk_core(clk_core),.rst_n(rst_n),
        .data_in(sa_data_in),.acc_clear(sa_acc_clear),.flush(sa_flush),
        .a_col_in(a_col_k),.b_row_in(bv_row_k),
        .c_out(v_out),.valid_out(v_valid)
    );

    // =========================================================================
    // Output tile registers + UCIe TX
    // =========================================================================
    logic txq_vi,txk_vi,txv_vi,txout_vi;
    logic [15:0] txq_t[TILE][TILE],txk_t[TILE][TILE],txv_t[TILE][TILE],txout_t[TILE][TILE];

    always_ff @(posedge clk_core or negedge rst_n) begin : tx_reg
        if (!rst_n) begin txq_vi<=0;txk_vi<=0;txv_vi<=0;txout_vi<=0; end
        else begin
            txq_vi<=0;txk_vi<=0;txv_vi<=0;txout_vi<=0;
            if (cstate==C_TX) begin
                if (~cfg_mode) begin
                    for (int i=0;i<TILE;i++) for (int j=0;j<TILE;j++) begin
                        txq_t[i][j]<=q_out[i][j]; txk_t[i][j]<=k_out[i][j]; txv_t[i][j]<=v_out[i][j];
                    end
                    txq_vi<=1'b1; txk_vi<=1'b1; txv_vi<=1'b1;
                end else begin
                    for (int i=0;i<TILE;i++) for (int j=0;j<TILE;j++) txout_t[i][j]<=q_out[i][j];
                    txout_vi<=1'b1;
                end
            end
        end
    end

    ucie_tx #(.TILE_DIM(TILE)) u_txq  (.clk_core(clk_core),.clk_link(clk_link),.rst_n(rst_n),.tx_valid(txq_vi),  .tx_src_id(4'd0),.tx_dst_id(4'd1),.tx_tile(txq_t),  .tx_ready(),.bump_data(txq_bump_data),  .bump_valid(txq_bump_valid),  .bump_credit(txq_bump_credit));
    ucie_tx #(.TILE_DIM(TILE)) u_txk  (.clk_core(clk_core),.clk_link(clk_link),.rst_n(rst_n),.tx_valid(txk_vi),  .tx_src_id(4'd0),.tx_dst_id(4'd1),.tx_tile(txk_t),  .tx_ready(),.bump_data(txk_bump_data),  .bump_valid(txk_bump_valid),  .bump_credit(txk_bump_credit));
    ucie_tx #(.TILE_DIM(TILE)) u_txv  (.clk_core(clk_core),.clk_link(clk_link),.rst_n(rst_n),.tx_valid(txv_vi),  .tx_src_id(4'd0),.tx_dst_id(4'd1),.tx_tile(txv_t),  .tx_ready(),.bump_data(txv_bump_data),  .bump_valid(txv_bump_valid),  .bump_credit(txv_bump_credit));
    ucie_tx #(.TILE_DIM(TILE)) u_txout(.clk_core(clk_core),.clk_link(clk_link),.rst_n(rst_n),.tx_valid(txout_vi),.tx_src_id(4'd0),.tx_dst_id(4'd9),.tx_tile(txout_t),.tx_ready(),.bump_data(txout_bump_data),.bump_valid(txout_bump_valid),.bump_credit(txout_bump_credit));

endmodule

`default_nettype wire
// =============================================================================
// End of chiplet_0_qkv_outproj.sv  (optimised)
// =============================================================================
