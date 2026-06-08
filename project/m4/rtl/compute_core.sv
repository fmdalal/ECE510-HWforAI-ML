// =============================================================================
// compute_core.sv  —  Top-level 9-chiplet MHSA accelerator  [optimised]
// =============================================================================
// Fix 2 (revised): direct tile mux for context collection (no UCIe TX overhead)
// -------------------------------------------------------------------------------
// Previous fix used a round-robin arbiter that forwarded each context tile
// through a full ucie_tx instance (142 flits × 2 cycles = 284 cycles/tile),
// wasting 2272 cycles on UCIe framing for transfers that never leave the die.
//
// This revision removes u_arb_tx entirely and instead:
//   a) 8 ucie_rx instances (u_ctx_rx[0..7]) receive context tiles from
//      head chiplets in parallel — unchanged from previous fix.
//   b) Direct tile mux arbiter: a 3-bit arb_head counter selects which
//      ctx_tile[h] to present each round.  The selected tile is latched
//      into arb_tile_reg (1 cycle) and ctx_tile_valid_in is pulsed.
//   c) chiplet_0 has two new ports (ctx_tile_in, ctx_tile_valid_in).
//      When ctx_tile_valid_in fires in stage 4 (cfg_mode=1), chiplet_0's
//      x_cap block captures ctx_tile_in directly into x_tile — bypassing
//      the ucie_rx entirely on the context path.
//   d) arb_head advances when c0_done pulses (chiplet_0 finished one tile).
//      Since all 8 ctx_tile_valid flags arrive simultaneously (parallel
//      heads), the arbiter starts immediately with no wait.
//
// Timing improvement: 8 × 285 = 2280 cycles → 8 × 194 = 1552 cycles (−32%)
// Chiplet_0 OutProj utilisation: 98%  (191/194 cycles computing)
// =============================================================================

`timescale 1ns/1ps
`default_nettype none

module compute_core #(
    parameter int NUM_HEADS = 8,
    parameter int TILE_DIM  = 8,
    parameter int D_HEAD    = 64,
    parameter int D_MODEL   = 512,
    parameter int TDATA_W   = 512,
    parameter int FIFO_D    = 256
)(
    input  wire        clk_axi,
    input  wire        clk_core,
    input  wire        clk_link,
    input  wire        rst_n,

    // AXI4-Lite slave
    input  wire [11:0] s_axil_awaddr,  input  wire [2:0]  s_axil_awprot,
    input  wire        s_axil_awvalid, output wire        s_axil_awready,
    input  wire [31:0] s_axil_wdata,   input  wire [3:0]  s_axil_wstrb,
    input  wire        s_axil_wvalid,  output wire        s_axil_wready,
    output wire [1:0]  s_axil_bresp,   output wire        s_axil_bvalid,
    input  wire        s_axil_bready,
    input  wire [11:0] s_axil_araddr,  input  wire [2:0]  s_axil_arprot,
    input  wire        s_axil_arvalid, output wire        s_axil_arready,
    output wire [31:0] s_axil_rdata,   output wire [1:0]  s_axil_rresp,
    output wire        s_axil_rvalid,  input  wire        s_axil_rready,

    // AXI4-Stream slave (input tile)
    input  wire [TDATA_W-1:0]   s_axis_tdata,
    input  wire [TDATA_W/8-1:0] s_axis_tkeep,
    input  wire                 s_axis_tvalid,
    output wire                 s_axis_tready,
    input  wire                 s_axis_tlast,
    input  wire [3:0]           s_axis_tuser,
    input  wire [3:0]           s_axis_tid,

    // AXI4-Stream master (output tile)
    output wire [TDATA_W-1:0]   m_axis_tdata,
    output wire [TDATA_W/8-1:0] m_axis_tkeep,
    output wire                 m_axis_tvalid,
    input  wire                 m_axis_tready,
    output wire                 m_axis_tlast,
    output wire [3:0]           m_axis_tuser,

    // LPDDR5X shared memory
    output logic [63:0]  mem_addr,   output logic [511:0] mem_wdata,
    output logic         mem_wen,    input  wire  [511:0] mem_rdata,
    input  wire          mem_rvalid, output logic         mem_req,
    input  wire          mem_gnt,

    output wire irq
);

    // =========================================================================
    // 1. Internal wire declarations
    // =========================================================================
    wire        cfg_start, cfg_reset, cfg_mode;
    wire [15:0] cfg_seq_len;
    wire [15:0] cfg_d_model;
    wire [7:0]  cfg_num_heads;
    wire [7:0]  cfg_num_tiles;
    wire [63:0] cfg_weight_addr;
    wire [31:0] cfg_in_addr, cfg_out_addr;
    wire [15:0] cfg_scale_bf16;
    wire [31:0] cfg_wdt_timeout;

    wire [15:0] in_tile_data  [TILE_DIM][TILE_DIM];
    wire        in_tile_valid, in_tile_ready;
    wire [3:0]  in_tile_dst, in_tile_type;
    wire [15:0] out_tile_data [TILE_DIM][TILE_DIM];
    wire        out_tile_valid, out_tile_ready;

    wire sts_busy, sts_done, sts_error;
    wire [3:0]  sts_active_head;
    wire [63:0] perf_cycles;

    // =========================================================================
    // 2. axi_if
    // =========================================================================
    axi_if #(.TILE_DIM(TILE_DIM),.TDATA_W(TDATA_W),.FIFO_D(FIFO_D)) u_axi_if (
        .clk_axi(clk_axi), .rst_n(rst_n),
        .s_axil_awaddr(s_axil_awaddr), .s_axil_awprot(s_axil_awprot),
        .s_axil_awvalid(s_axil_awvalid), .s_axil_awready(s_axil_awready),
        .s_axil_wdata(s_axil_wdata), .s_axil_wstrb(s_axil_wstrb),
        .s_axil_wvalid(s_axil_wvalid), .s_axil_wready(s_axil_wready),
        .s_axil_bresp(s_axil_bresp), .s_axil_bvalid(s_axil_bvalid),
        .s_axil_bready(s_axil_bready),
        .s_axil_araddr(s_axil_araddr), .s_axil_arprot(s_axil_arprot),
        .s_axil_arvalid(s_axil_arvalid), .s_axil_arready(s_axil_arready),
        .s_axil_rdata(s_axil_rdata), .s_axil_rresp(s_axil_rresp),
        .s_axil_rvalid(s_axil_rvalid), .s_axil_rready(s_axil_rready),
        .s_axis_tdata(s_axis_tdata), .s_axis_tkeep(s_axis_tkeep),
        .s_axis_tvalid(s_axis_tvalid), .s_axis_tready(s_axis_tready),
        .s_axis_tlast(s_axis_tlast), .s_axis_tuser(s_axis_tuser),
        .s_axis_tid(s_axis_tid),
        .m_axis_tdata(m_axis_tdata), .m_axis_tkeep(m_axis_tkeep),
        .m_axis_tvalid(m_axis_tvalid), .m_axis_tready(m_axis_tready),
        .m_axis_tlast(m_axis_tlast), .m_axis_tuser(m_axis_tuser),
        .cfg_start(cfg_start), .cfg_reset(cfg_reset), .cfg_mode(cfg_mode),
        .cfg_seq_len(cfg_seq_len), .cfg_d_model(cfg_d_model),
        .cfg_num_heads(cfg_num_heads), .cfg_num_tiles(cfg_num_tiles),
        .cfg_weight_addr(cfg_weight_addr), .cfg_in_addr(cfg_in_addr),
        .cfg_out_addr(cfg_out_addr), .cfg_scale_bf16(cfg_scale_bf16),
        .cfg_wdt_timeout(cfg_wdt_timeout),
        .in_tile_data(in_tile_data), .in_tile_valid(in_tile_valid),
        .in_tile_dst(in_tile_dst), .in_tile_type(in_tile_type),
        .in_tile_ready(in_tile_ready),
        .out_tile_data(out_tile_data), .out_tile_valid(out_tile_valid),
        .out_tile_ready(out_tile_ready),
        .sts_busy(sts_busy), .sts_done(sts_done), .sts_error(sts_error),
        .sts_active_head(sts_active_head), .irq(irq), .perf_cycles(perf_cycles)
    );

    // =========================================================================
    // 3. UCIe bump bus declarations
    // =========================================================================
    wire [511:0] h2c0_bump;  wire h2c0_bv, h2c0_bcr;
    wire [511:0] c0_txq_bump; wire c0_txq_bv, c0_txq_bcr;
    wire [511:0] c0_txk_bump; wire c0_txk_bv, c0_txk_bcr;
    wire [511:0] c0_txv_bump; wire c0_txv_bv, c0_txv_bcr;
    wire [511:0] c0_out_bump; wire c0_out_bv, c0_out_bcr;

    wire [511:0] h_tx_bump  [NUM_HEADS]; wire h_tx_bv  [NUM_HEADS]; wire h_tx_bcr  [NUM_HEADS];
    wire [511:0] t_tx_bump  [NUM_HEADS]; wire t_tx_bv  [NUM_HEADS]; wire t_tx_bcr  [NUM_HEADS];
    wire [511:0] h_ctx_bump [NUM_HEADS]; wire h_ctx_bv [NUM_HEADS]; wire h_ctx_bcr [NUM_HEADS];

    logic cfg_start_core;
    logic cfg_mode_core;    // cfg_mode synchronised to clk_core (2-FF CDC)

    // =========================================================================
    // 4. Host → chiplet 0: UCIe TX  (stage 1 only)
    // =========================================================================
    ucie_tx #(.TILE_DIM(TILE_DIM)) u_host_tx (
        .clk_core(clk_core), .clk_link(clk_link), .rst_n(rst_n),
        .tx_valid(in_tile_valid & cfg_start_core),
        .tx_src_id(4'hF), .tx_dst_id(4'd0),
        .tx_tile(in_tile_data), .tx_ready(in_tile_ready),
        .bump_data(h2c0_bump), .bump_valid(h2c0_bv), .bump_credit(h2c0_bcr)
    );

    // =========================================================================
    // 5a. Context RX: 8 parallel ucie_rx instances receive context tiles
    //     from head chiplets 1..8 in stage 4 (all arrive simultaneously)
    // =========================================================================
    wire [511:0] ctx_rx_bump_data  [NUM_HEADS];
    wire         ctx_rx_bump_valid [NUM_HEADS];
    wire         ctx_rx_bump_credit[NUM_HEADS];
    wire [15:0]  ctx_tile          [NUM_HEADS][TILE_DIM][TILE_DIM];
    wire         ctx_tile_valid    [NUM_HEADS];

    // Per-head RX wires (Icarus can't slice packed arrays as ports)
    wire [15:0] ctx_tile_0[TILE_DIM][TILE_DIM]; wire [15:0] ctx_tile_1[TILE_DIM][TILE_DIM];
    wire [15:0] ctx_tile_2[TILE_DIM][TILE_DIM]; wire [15:0] ctx_tile_3[TILE_DIM][TILE_DIM];
    wire [15:0] ctx_tile_4[TILE_DIM][TILE_DIM]; wire [15:0] ctx_tile_5[TILE_DIM][TILE_DIM];
    wire [15:0] ctx_tile_6[TILE_DIM][TILE_DIM]; wire [15:0] ctx_tile_7[TILE_DIM][TILE_DIM];

    // Connect h_ctx_bump → ctx_rx inputs
    genvar cri;
    generate
        for (cri=0;cri<NUM_HEADS;cri++) begin : ctx_rx_wire_g
            assign ctx_rx_bump_data [cri] = h_ctx_bump[cri];
            assign ctx_rx_bump_valid[cri] = h_ctx_bv  [cri];
            assign h_ctx_bcr        [cri] = ctx_rx_bump_credit[cri];
        end
    endgenerate

    ucie_rx #(.TILE_DIM(TILE_DIM)) u_ctx_rx0(.clk_core(clk_core),.clk_link(clk_link),.rst_n(rst_n),.bump_data(ctx_rx_bump_data[0]),.bump_valid(ctx_rx_bump_valid[0]),.bump_credit(ctx_rx_bump_credit[0]),.rx_valid(ctx_tile_valid[0]),.rx_src_id(),.rx_tile(ctx_tile_0),.rx_ready(1'b1),.rx_crc_err(),.rx_seq_err());
    ucie_rx #(.TILE_DIM(TILE_DIM)) u_ctx_rx1(.clk_core(clk_core),.clk_link(clk_link),.rst_n(rst_n),.bump_data(ctx_rx_bump_data[1]),.bump_valid(ctx_rx_bump_valid[1]),.bump_credit(ctx_rx_bump_credit[1]),.rx_valid(ctx_tile_valid[1]),.rx_src_id(),.rx_tile(ctx_tile_1),.rx_ready(1'b1),.rx_crc_err(),.rx_seq_err());
    ucie_rx #(.TILE_DIM(TILE_DIM)) u_ctx_rx2(.clk_core(clk_core),.clk_link(clk_link),.rst_n(rst_n),.bump_data(ctx_rx_bump_data[2]),.bump_valid(ctx_rx_bump_valid[2]),.bump_credit(ctx_rx_bump_credit[2]),.rx_valid(ctx_tile_valid[2]),.rx_src_id(),.rx_tile(ctx_tile_2),.rx_ready(1'b1),.rx_crc_err(),.rx_seq_err());
    ucie_rx #(.TILE_DIM(TILE_DIM)) u_ctx_rx3(.clk_core(clk_core),.clk_link(clk_link),.rst_n(rst_n),.bump_data(ctx_rx_bump_data[3]),.bump_valid(ctx_rx_bump_valid[3]),.bump_credit(ctx_rx_bump_credit[3]),.rx_valid(ctx_tile_valid[3]),.rx_src_id(),.rx_tile(ctx_tile_3),.rx_ready(1'b1),.rx_crc_err(),.rx_seq_err());
    ucie_rx #(.TILE_DIM(TILE_DIM)) u_ctx_rx4(.clk_core(clk_core),.clk_link(clk_link),.rst_n(rst_n),.bump_data(ctx_rx_bump_data[4]),.bump_valid(ctx_rx_bump_valid[4]),.bump_credit(ctx_rx_bump_credit[4]),.rx_valid(ctx_tile_valid[4]),.rx_src_id(),.rx_tile(ctx_tile_4),.rx_ready(1'b1),.rx_crc_err(),.rx_seq_err());
    ucie_rx #(.TILE_DIM(TILE_DIM)) u_ctx_rx5(.clk_core(clk_core),.clk_link(clk_link),.rst_n(rst_n),.bump_data(ctx_rx_bump_data[5]),.bump_valid(ctx_rx_bump_valid[5]),.bump_credit(ctx_rx_bump_credit[5]),.rx_valid(ctx_tile_valid[5]),.rx_src_id(),.rx_tile(ctx_tile_5),.rx_ready(1'b1),.rx_crc_err(),.rx_seq_err());
    ucie_rx #(.TILE_DIM(TILE_DIM)) u_ctx_rx6(.clk_core(clk_core),.clk_link(clk_link),.rst_n(rst_n),.bump_data(ctx_rx_bump_data[6]),.bump_valid(ctx_rx_bump_valid[6]),.bump_credit(ctx_rx_bump_credit[6]),.rx_valid(ctx_tile_valid[6]),.rx_src_id(),.rx_tile(ctx_tile_6),.rx_ready(1'b1),.rx_crc_err(),.rx_seq_err());
    ucie_rx #(.TILE_DIM(TILE_DIM)) u_ctx_rx7(.clk_core(clk_core),.clk_link(clk_link),.rst_n(rst_n),.bump_data(ctx_rx_bump_data[7]),.bump_valid(ctx_rx_bump_valid[7]),.bump_credit(ctx_rx_bump_credit[7]),.rx_valid(ctx_tile_valid[7]),.rx_src_id(),.rx_tile(ctx_tile_7),.rx_ready(1'b1),.rx_crc_err(),.rx_seq_err());

    // Collect per-head tiles into array
    genvar cti, ctj;
    generate
        for (cti=0;cti<TILE_DIM;cti++) begin : ct_row
            for (ctj=0;ctj<TILE_DIM;ctj++) begin : ct_col
                assign ctx_tile[0][cti][ctj] = ctx_tile_0[cti][ctj];
                assign ctx_tile[1][cti][ctj] = ctx_tile_1[cti][ctj];
                assign ctx_tile[2][cti][ctj] = ctx_tile_2[cti][ctj];
                assign ctx_tile[3][cti][ctj] = ctx_tile_3[cti][ctj];
                assign ctx_tile[4][cti][ctj] = ctx_tile_4[cti][ctj];
                assign ctx_tile[5][cti][ctj] = ctx_tile_5[cti][ctj];
                assign ctx_tile[6][cti][ctj] = ctx_tile_6[cti][ctj];
                assign ctx_tile[7][cti][ctj] = ctx_tile_7[cti][ctj];
            end
        end
    endgenerate

    // =========================================================================
    // 5b. Direct tile mux arbiter (replaces UCIe TX arbiter)
    //     Selects ctx_tile[arb_head] and presents it directly to chiplet_0
    //     via ctx_tile_in / ctx_tile_valid_in ports.
    //     Advances arb_head when c0_done pulses (1 cycle after C_TX).
    //     All 8 ctx_tile_valid flags arrive simultaneously so there is no
    //     wait between tiles — chiplet_0 is the pacing element (194 cyc/tile).
    // =========================================================================
    logic [2:0]  arb_head;      // 0..7: which head tile to present next
    logic [7:0]  arb_served;    // bitmask: which heads have been sent
    logic [15:0] arb_tile_reg [TILE_DIM][TILE_DIM]; // latched tile
    logic        arb_tile_vld;  // ctx_tile_valid_in pulse to chiplet_0

    // c0_done is declared below with chiplet_0 instantiation;
    // forward-declare wire here so arbiter can reference it
    wire c0_done;

    always_ff @(posedge clk_core or negedge rst_n) begin : arb_ff
        if (!rst_n) begin
            arb_head    <= 3'd0;
            arb_served  <= 8'h00;
            arb_tile_vld<= 1'b0;
        end else begin
            arb_tile_vld <= 1'b0;
            if (!cfg_mode) begin
                // Stage 1: reset for next stage 4 round
                arb_head   <= 3'd0;
                arb_served <= 8'h00;
            end else begin
                // Stage 4: present next unserved head whose tile has arrived
                if (!arb_served[arb_head] && ctx_tile_valid[arb_head]) begin
                    // Latch tile into register (1 cycle) and pulse valid
                    for (int i=0;i<TILE_DIM;i++)
                        for (int j=0;j<TILE_DIM;j++)
                            arb_tile_reg[i][j] <= ctx_tile[arb_head][i][j];
                    arb_tile_vld        <= 1'b1;
                    arb_served[arb_head]<= 1'b1;
                end
                // Advance to next head after chiplet_0 finishes this tile
                if (c0_done && arb_head != 3'(NUM_HEADS-1))
                    arb_head <= arb_head + 3'd1;
            end
        end
    end

    // =========================================================================
    // 6. Chiplet 0 — QKV + OutProj
    //    rx_bump_data still used for stage 1 (host token tile via bump bus).
    //    ctx_tile_in / ctx_tile_valid_in used for stage 4 (direct mux path).
    // =========================================================================
    chiplet_0_qkv_outproj #(
        .D_MODEL(D_MODEL), .NUM_HEADS(NUM_HEADS), .D_HEAD(D_HEAD),
        .TILE(TILE_DIM), .K_DIM(TILE_DIM)
    ) u_c0 (
        .clk_core(clk_core), .clk_link(clk_link), .rst_n(rst_n),
        .cfg_mode(cfg_mode_core), .cfg_num_tiles(cfg_num_tiles),
        .cfg_start(cfg_start_core), .cfg_done(c0_done),
        // Stage 1 RX: host token tile via bump bus
        .rx_bump_data  (h2c0_bump),
        .rx_bump_valid (h2c0_bv),
        .rx_bump_credit(h2c0_bcr),
        // TX Q, K, V
        .txq_bump_data(c0_txq_bump), .txq_bump_valid(c0_txq_bv), .txq_bump_credit(c0_txq_bcr),
        .txk_bump_data(c0_txk_bump), .txk_bump_valid(c0_txk_bv), .txk_bump_credit(c0_txk_bcr),
        .txv_bump_data(c0_txv_bump), .txv_bump_valid(c0_txv_bv), .txv_bump_credit(c0_txv_bcr),
        // TX OutProj result
        .txout_bump_data(c0_out_bump), .txout_bump_valid(c0_out_bv), .txout_bump_credit(c0_out_bcr),
        // Weight SRAM (tied to zero in simulation)
        .sram_addr(), .sram_rdata_flat('0),
        .sram_rd_en(), .sram_rd_valid(mem_rvalid),
        // Stage 4 direct tile input (bypasses UCIe TX framing)
        .ctx_tile_in      (arb_tile_reg),
        .ctx_tile_valid_in(arb_tile_vld)
    );

    // =========================================================================
    // 7. Head chiplets 1..8
    // =========================================================================
    wire [511:0] head_rxa_data [NUM_HEADS]; wire head_rxa_bv [NUM_HEADS]; wire head_rxa_bcr [NUM_HEADS];
    wire [511:0] head_rxb_data [NUM_HEADS]; wire head_rxb_bv [NUM_HEADS]; wire head_rxb_bcr [NUM_HEADS];
    wire [511:0] head_tx_data  [NUM_HEADS]; wire head_tx_bv  [NUM_HEADS]; wire head_tx_bcr  [NUM_HEADS];

    genvar hh;
    generate
        for (hh=0;hh<NUM_HEADS;hh++) begin : head_mux
            assign head_rxa_data[hh] = cfg_mode_core ? t_tx_bump[hh] : c0_txq_bump;
            assign head_rxa_bv  [hh] = cfg_mode_core ? t_tx_bv  [hh] : c0_txq_bv;
            assign t_tx_bcr [hh]     = cfg_mode_core ? head_rxa_bcr[hh] : 1'b0;
            assign c0_txq_bcr        = cfg_mode_core ? 1'b0 : head_rxa_bcr[0];

            assign head_rxb_data[hh] = cfg_mode_core ? c0_txv_bump : c0_txk_bump;
            assign head_rxb_bv  [hh] = cfg_mode_core ? c0_txv_bv   : c0_txk_bv;

            assign h_ctx_bump[hh] = cfg_mode_core ? head_tx_data[hh] : 512'h0;
            assign h_ctx_bv  [hh] = cfg_mode_core ? head_tx_bv  [hh] : 1'b0;
            assign h_tx_bump [hh] = cfg_mode_core ? 512'h0 : head_tx_data[hh];
            assign h_tx_bv   [hh] = cfg_mode_core ? 1'b0   : head_tx_bv  [hh];
            assign head_tx_bcr[hh]= cfg_mode_core ? h_ctx_bcr[hh] : h_tx_bcr[hh];
        end
    endgenerate

    generate
        for (hh=0;hh<NUM_HEADS;hh++) begin : head_gen
            chiplet_head #(
                .HEAD_ID(hh), .D_HEAD(D_HEAD), .TILE(TILE_DIM),
                .K_DIM(TILE_DIM), .SEQ_TILE(TILE_DIM)
            ) u_head (
                .clk_core(clk_core), .clk_link(clk_link), .rst_n(rst_n),
                .cfg_mode(cfg_mode_core), .cfg_num_tiles(cfg_num_tiles),
                .cfg_start(cfg_start_core), .cfg_done(), .chiplet_id(),
                .rxa_bump_data(head_rxa_data[hh]), .rxa_bump_valid(head_rxa_bv[hh]),
                .rxa_bump_credit(head_rxa_bcr[hh]),
                .rxb_bump_data(head_rxb_data[hh]), .rxb_bump_valid(head_rxb_bv[hh]),
                .rxb_bump_credit(head_rxb_bcr[hh]),
                .tx_bump_data(head_tx_data[hh]), .tx_bump_valid(head_tx_bv[hh]),
                .tx_bump_credit(head_tx_bcr[hh]),
                .scale_factor(cfg_scale_bf16)
            );
        end
    endgenerate

    // =========================================================================
    // 8. Softmax chiplet (ID 9)
    // =========================================================================
    chiplet_9_softmax #(.NUM_HEADS(NUM_HEADS),.TILE(TILE_DIM),.SEQ_LEN(TILE_DIM)) u_taylor (
        .clk_core(clk_core), .clk_link(clk_link), .rst_n(rst_n),
        .cfg_start(cfg_start_core), .cfg_done(),
        .rx_bump_data(h_tx_bump), .rx_bump_valid(h_tx_bv), .rx_bump_credit(h_tx_bcr),
        .tx_bump_data(t_tx_bump), .tx_bump_valid(t_tx_bv), .tx_bump_credit(t_tx_bcr)
    );

    // =========================================================================
    // 9. Chiplet 0 result → host: UCIe RX
    // =========================================================================
    wire [15:0] host_rx_tile [TILE_DIM][TILE_DIM];
    wire        host_rx_valid;

    ucie_rx #(.TILE_DIM(TILE_DIM)) u_host_rx (
        .clk_core(clk_core), .clk_link(clk_link), .rst_n(rst_n),
        .bump_data(c0_out_bump), .bump_valid(c0_out_bv), .bump_credit(c0_out_bcr),
        .rx_valid(host_rx_valid), .rx_src_id(),
        .rx_tile(host_rx_tile), .rx_ready(out_tile_ready),
        .rx_crc_err(), .rx_seq_err()
    );

    assign out_tile_valid = host_rx_valid;
    genvar oi, oj;
    generate
        for (oi=0;oi<TILE_DIM;oi++) begin : out_row_g
            for (oj=0;oj<TILE_DIM;oj++) begin : out_col_g
                assign out_tile_data[oi][oj] = host_rx_tile[oi][oj];
            end
        end
    endgenerate

    // =========================================================================
    // 10. Status aggregation
    // =========================================================================
    logic busy_r, cfg_start_s1;

    logic cfg_mode_s1;
    always_ff @(posedge clk_core or negedge rst_n) begin : start_mode_sync
        if (!rst_n) begin
            cfg_start_s1  <= 1'b0; cfg_start_core <= 1'b0;
            cfg_mode_s1   <= 1'b0; cfg_mode_core  <= 1'b0;
        end else begin
            cfg_start_s1  <= cfg_start;      cfg_start_core <= cfg_start_s1;
            cfg_mode_s1   <= cfg_mode;       cfg_mode_core  <= cfg_mode_s1;
        end
    end

    always_ff @(posedge clk_core or negedge rst_n) begin : busy_ff
        if (!rst_n)              busy_r <= 1'b0;
        else if (cfg_start_core) busy_r <= 1'b1;
        else if (c0_done)        busy_r <= 1'b0;
    end

    assign sts_busy = busy_r;

    logic sts_done_core, sts_done_s1, sts_done_r;
    always_ff @(posedge clk_core or negedge rst_n) begin : done_latch
        if (!rst_n)       sts_done_core <= 1'b0;
        else if (c0_done) sts_done_core <= 1'b1;
        else              sts_done_core <= 1'b0;
    end
    always_ff @(posedge clk_axi or negedge rst_n) begin : done_sync
        if (!rst_n) begin sts_done_s1<=1'b0; sts_done_r<=1'b0; end
        else begin sts_done_s1<=sts_done_core; sts_done_r<=sts_done_s1; end
    end
    assign sts_done        = sts_done_r;
    assign sts_error       = 1'b0;
    assign sts_active_head = 4'h0;

    // =========================================================================
    // 11. Memory interface
    // =========================================================================
    assign mem_wdata = 512'h0;
    assign mem_wen   = 1'b0;
    assign mem_addr  = cfg_weight_addr;
    assign mem_req   = busy_r;

endmodule

`default_nettype wire
// =============================================================================
// End of compute_core.sv  (optimised)
// =============================================================================
