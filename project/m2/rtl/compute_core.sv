// =============================================================================
// compute_core.sv  —  SoC top-level wrapper  (PURE WIRING — no AXI logic inside)
// =============================================================================
//
// This file contains NO AXI logic, NO CSR registers, NO FIFOs.
// Its only responsibility is connecting axi_if.sv to the 10 chiplets
// through UCIe bump buses.
//
// Block view
// ----------
//
//   CPU Wishbone
//       │
//   ┌───┴──────────────────────────────────────────┐
//   │  axi_if                                      │
//   │   ├── wb2axip bridge                         │
//   │   ├── axi_lite_csr  (18 CSR registers)       │
//   │   ├── axis_input_fifo  (512-bit → tile)      │
//   │   ├── axis_output_fifo (tile  → 512-bit)     │
//   │   ├── watchdog                               │
//   │   └── perf_counter                           │
//   └───┬──────────────────────────────────────────┘
//       │  cfg_* / in_tile_* / out_tile_* / sts_*
//       │
//   ┌───┴──────────────────────────────────────────┐
//   │  soc_top  (THIS FILE)                        │
//   │                                              │
//   │  ucie_tx ──► chiplet_0_qkv_outproj ──►       │
//   │              (UCIe Q/K/V broadcast)          │
//   │              ▼                               │
//   │         chiplet_head × 8 ──► chiplet_9_taylor│
//   │         ◄────────────────────────────────    │
//   │              ▼ context                       │
//   │         chiplet_0_qkv_outproj (Stage 5)      │
//   │              ▼                               │
//   │         ucie_rx ──► axi_if out_tile          │
//   └──────────────────────────────────────────────┘
//
// Chiplet IDs
//   0  QKV + OutProj (Stage 1 / Stage 5 time-mux)
//   1  Head 0   2  Head 1   3  Head 2   4  Head 3
//   5  Head 4   6  Head 5   7  Head 6   8  Head 7
//   9  Taylor chiplet
// =============================================================================

`timescale 1ns/1ps
`default_nettype none

module soc_top #(
    parameter int NUM_HEADS = 8,
    parameter int TILE_DIM  = 64,
    parameter int D_HEAD    = 64,
    parameter int D_MODEL   = 512,
    parameter int TDATA_W   = 512,
    parameter int FIFO_D    = 256
)(
    // -----------------------------------------------------------------------
    // Clock ports  (three separate clock domains)
    // -----------------------------------------------------------------------
    input  wire        clk_axi,    // 250 MHz — AXI/host interface domain
                                   //   drives: axi_if (wb2axip, CSR, FIFOs)
    input  wire        clk_core,   // 1 GHz  — chiplet compute domain
                                   //   drives: all chiplets, systolic arrays,
                                   //           UCIe FDI TX/RX logic
    input  wire        clk_link,   // 2 GHz  — UCIe PHY bump pad domain
                                   //   drives: UCIe PHY (physical layer,
                                   //           not yet modelled in RTL)
                                   //   NOTE: tie to clk_core in simulation
    input  wire        rst_n,      // async active-low reset (all domains)

    // Wishbone B4 slave (from CPU → axi_if → wb2axip)
    input  wire        wb_cyc,
    input  wire        wb_stb,
    input  wire        wb_we,
    input  wire [31:0] wb_addr,
    input  wire [31:0] wb_wdata,
    input  wire [3:0]  wb_sel,
    output wire        wb_stall,
    output wire        wb_ack,
    output wire [31:0] wb_rdata,
    output wire        wb_err,

    // LPDDR5X-8533 shared memory (137 GB/s)
    output logic [63:0]  mem_addr,
    output logic [511:0] mem_wdata,
    output logic         mem_wen,
    input  wire  [511:0] mem_rdata,
    input  wire          mem_rvalid,
    output logic         mem_req,
    input  wire          mem_gnt,

    // Interrupt to CPU
    output wire        irq
);

    // =========================================================================
    // 1. Internal wire declarations — axi_if ↔ chiplets
    // =========================================================================

    // Config (CSR → chiplets)
    wire        cfg_start;
    wire        cfg_reset;
    wire        cfg_mode;
    wire [15:0] cfg_seq_len;
    wire [15:0] cfg_d_model;
    wire [7:0]  cfg_num_heads;
    wire [7:0]  cfg_num_tiles;
    wire [63:0] cfg_weight_addr;
    wire [31:0] cfg_in_addr;
    wire [31:0] cfg_out_addr;
    wire [15:0] cfg_scale_bf16;
    wire [31:0] cfg_wdt_timeout;

    // Input tile  (axi_if → host UCIe TX → chiplet 0)
    wire [15:0] in_tile_data  [TILE_DIM][TILE_DIM];
    wire        in_tile_valid;
    wire [3:0]  in_tile_dst;
    wire [3:0]  in_tile_type;
    wire        in_tile_ready;

    // Output tile (chiplet 0 UCIe RX → axi_if)
    wire [15:0] out_tile_data [TILE_DIM][TILE_DIM];
    wire        out_tile_valid;
    wire        out_tile_ready;

    // Status (chiplets → axi_if)
    wire        sts_busy;
    wire        sts_done;
    wire        sts_error;
    wire [3:0]  sts_active_head;
    wire [63:0] perf_cycles;

    // =========================================================================
    // 2. axi_if instantiation
    // =========================================================================
    axi_if #(
        .TILE_DIM(TILE_DIM),
        .TDATA_W (TDATA_W),
        .FIFO_D  (FIFO_D)
    ) u_axi_if (
        .clk_axi         (clk_axi),   // 250 MHz AXI domain
        .rst_n           (rst_n),
        // Wishbone
        .wb_cyc          (wb_cyc),
        .wb_stb          (wb_stb),
        .wb_we           (wb_we),
        .wb_addr         (wb_addr),
        .wb_wdata        (wb_wdata),
        .wb_sel          (wb_sel),
        .wb_stall        (wb_stall),
        .wb_ack          (wb_ack),
        .wb_rdata        (wb_rdata),
        .wb_err          (wb_err),
        // Config
        .cfg_start       (cfg_start),
        .cfg_reset       (cfg_reset),
        .cfg_mode        (cfg_mode),
        .cfg_seq_len     (cfg_seq_len),
        .cfg_d_model     (cfg_d_model),
        .cfg_num_heads   (cfg_num_heads),
        .cfg_num_tiles   (cfg_num_tiles),
        .cfg_weight_addr (cfg_weight_addr),
        .cfg_in_addr     (cfg_in_addr),
        .cfg_out_addr    (cfg_out_addr),
        .cfg_scale_bf16  (cfg_scale_bf16),
        .cfg_wdt_timeout (cfg_wdt_timeout),
        // Input tile
        .in_tile_data    (in_tile_data),
        .in_tile_valid   (in_tile_valid),
        .in_tile_dst     (in_tile_dst),
        .in_tile_type    (in_tile_type),
        .in_tile_ready   (in_tile_ready),
        // Output tile
        .out_tile_data   (out_tile_data),
        .out_tile_valid  (out_tile_valid),
        .out_tile_ready  (out_tile_ready),
        // Status
        .sts_busy        (sts_busy),
        .sts_done        (sts_done),
        .sts_error       (sts_error),
        .sts_active_head (sts_active_head),
        .irq             (irq),
        .perf_cycles     (perf_cycles)
    );

    // =========================================================================
    // 3. UCIe bump bus declarations
    // =========================================================================

    // Host → chiplet 0
    wire [511:0] h2c0_bump;
    wire         h2c0_bv;
    wire         h2c0_bcr;

    // Chiplet 0 → all heads: Q (broadcast)
    wire [511:0] c0_txq_bump;
    wire         c0_txq_bv;
    wire         c0_txq_bcr;

    // Chiplet 0 → all heads: K (broadcast)
    wire [511:0] c0_txk_bump;
    wire         c0_txk_bv;
    wire         c0_txk_bcr;

    // Chiplet 0 → all heads: V (broadcast, also reused in stage 4)
    wire [511:0] c0_txv_bump;
    wire         c0_txv_bv;
    wire         c0_txv_bcr;

    // Chiplet 0 output projection → host
    wire [511:0] c0_out_bump;
    wire         c0_out_bv;
    wire         c0_out_bcr;

    // Head h → Taylor: scaled scores  (stage 2)
    wire [511:0] h_tx_bump  [NUM_HEADS];
    wire         h_tx_bv    [NUM_HEADS];
    wire         h_tx_bcr   [NUM_HEADS];

    // Taylor → head h: probabilities  (stage 4)
    wire [511:0] t_tx_bump  [NUM_HEADS];
    wire         t_tx_bv    [NUM_HEADS];
    wire         t_tx_bcr   [NUM_HEADS];

    // Head h → chiplet 0: context      (stage 4)
    wire [511:0] h_ctx_bump [NUM_HEADS];
    wire         h_ctx_bv   [NUM_HEADS];
    wire         h_ctx_bcr  [NUM_HEADS];

    // =========================================================================
    // 4. Host → chiplet 0: UCIe TX
    // =========================================================================
    // CDC NOTE: in_tile_valid crosses clk_axi → clk_core here.
    // Add a 2-FF synchroniser on tx_valid before tapeout.
    ucie_tx #(.TILE_DIM(TILE_DIM)) u_host_tx (
        .clk_core    (clk_core),   // 1 GHz compute domain
        .rst_n       (rst_n),
        .tx_valid    (in_tile_valid & cfg_start_core),
        .tx_src_id   (4'hF),
        .tx_dst_id   (4'd0),
        .tx_tile     (in_tile_data),
        .tx_ready    (in_tile_ready),
        .bump_data   (h2c0_bump),
        .bump_valid  (h2c0_bv),
        .bump_credit (h2c0_bcr)
    );

    // =========================================================================
    // 5. Chiplet 0  —  QKV projection (stage 1) + OutProj (stage 5)
    // =========================================================================
    wire c0_done;

    chiplet_0_qkv_outproj #(
        .D_MODEL  (D_MODEL),
        .NUM_HEADS(NUM_HEADS),
        .D_HEAD   (D_HEAD),
        .TILE     (TILE_DIM),
        .K_DIM    (TILE_DIM)
    ) u_c0 (
        .clk              (clk_core),
        .rst_n            (rst_n),
        .cfg_mode         (cfg_mode),
        .cfg_num_tiles    (cfg_num_tiles),
        .cfg_start        (cfg_start_core),
        .cfg_done         (c0_done),
        // RX: input tokens from host
        .rx_bump_data     (h2c0_bump),
        .rx_bump_valid    (h2c0_bv),
        .rx_bump_credit   (h2c0_bcr),
        // TX Q
        .txq_bump_data    (c0_txq_bump),
        .txq_bump_valid   (c0_txq_bv),
        .txq_bump_credit  (c0_txq_bcr),
        // TX K
        .txk_bump_data    (c0_txk_bump),
        .txk_bump_valid   (c0_txk_bv),
        .txk_bump_credit  (c0_txk_bcr),
        // TX V
        .txv_bump_data    (c0_txv_bump),
        .txv_bump_valid   (c0_txv_bv),
        .txv_bump_credit  (c0_txv_bcr),
        // TX output projection result
        .txout_bump_data   (c0_out_bump),
        .txout_bump_valid  (c0_out_bv),
        .txout_bump_credit (c0_out_bcr),
        // Weight SRAM (sourced from LPDDR5X via mem port)
        .sram_addr        (),
        .sram_rdata       (512'h0),
        .sram_rd_en       (),
        .sram_rd_valid    (mem_rvalid)
    );

    // =========================================================================
    // 6. Head chiplets 1..8
    // RX A mux:
    //   stage 2 (cfg_mode=0): Q from chiplet 0  broadcast
    //   stage 4 (cfg_mode=1): probs from Taylor (per-head)
    // RX B mux:
    //   stage 2 (cfg_mode=0): K from chiplet 0  broadcast
    //   stage 4 (cfg_mode=1): V from chiplet 0  broadcast
    // TX mux:
    //   stage 2: scores → Taylor
    //   stage 4: context → chiplet 0
    // =========================================================================
    // Mux wires for head RX/TX port selection (ternary not legal as l-value)
    wire [511:0] head_rxa_data  [NUM_HEADS];
    wire         head_rxa_bv    [NUM_HEADS];
    wire         head_rxa_bcr   [NUM_HEADS];
    wire [511:0] head_rxb_data  [NUM_HEADS];
    wire         head_rxb_bv    [NUM_HEADS];
    wire         head_rxb_bcr   [NUM_HEADS];
    wire [511:0] head_tx_data   [NUM_HEADS];
    wire         head_tx_bv     [NUM_HEADS];
    wire         head_tx_bcr    [NUM_HEADS];

    genvar hh;
    generate
        for (hh = 0; hh < NUM_HEADS; hh++) begin : head_mux
            // RX A mux
            assign head_rxa_data[hh] = cfg_mode ? t_tx_bump[hh] : c0_txq_bump;
            assign head_rxa_bv  [hh] = cfg_mode ? t_tx_bv  [hh] : c0_txq_bv;
            // RX A credit: output from head — connect to correct target
            assign t_tx_bcr [hh] = cfg_mode ? head_rxa_bcr[hh] : 1'b0;
            assign c0_txq_bcr    = cfg_mode ? 1'b0 : head_rxa_bcr[0]; // shared broadcast

            // RX B mux
            assign head_rxb_data[hh] = cfg_mode ? c0_txv_bump : c0_txk_bump;
            assign head_rxb_bv  [hh] = cfg_mode ? c0_txv_bv   : c0_txk_bv;

            // TX mux: route head TX to correct destination
            assign h_ctx_bump[hh] = cfg_mode ? head_tx_data[hh] : 512'h0;
            assign h_ctx_bv  [hh] = cfg_mode ? head_tx_bv  [hh] : 1'b0;
            assign h_tx_bump [hh] = cfg_mode ? 512'h0 : head_tx_data[hh];
            assign h_tx_bv   [hh] = cfg_mode ? 1'b0   : head_tx_bv  [hh];
            assign head_tx_bcr[hh]= cfg_mode ? h_ctx_bcr[hh] : h_tx_bcr[hh];
        end
    endgenerate

    generate
        for (hh = 0; hh < NUM_HEADS; hh++) begin : head_gen
            chiplet_head #(
                .HEAD_ID  (hh),
                .D_HEAD   (D_HEAD),
                .TILE     (TILE_DIM),
                .K_DIM    (TILE_DIM),
                .SEQ_TILE (TILE_DIM)
            ) u_head (
                .clk_core        (clk_core),
                .rst_n           (rst_n),
                .cfg_mode        (cfg_mode),
                .cfg_num_tiles   (cfg_num_tiles),
                .cfg_start       (cfg_start),
                .cfg_done        (),
                .chiplet_id      (),
                .rxa_bump_data   (head_rxa_data [hh]),
                .rxa_bump_valid  (head_rxa_bv   [hh]),
                .rxa_bump_credit (head_rxa_bcr  [hh]),
                .rxb_bump_data   (head_rxb_data [hh]),
                .rxb_bump_valid  (head_rxb_bv   [hh]),
                .rxb_bump_credit (head_rxb_bcr  [hh]),
                .tx_bump_data    (head_tx_data  [hh]),
                .tx_bump_valid   (head_tx_bv    [hh]),
                .tx_bump_credit  (head_tx_bcr   [hh]),
                .scale_factor    (cfg_scale_bf16)
            );
        end
    endgenerate

    // =========================================================================
    // 7. Taylor chiplet (ID 9)
    // =========================================================================
    chiplet_9_taylor #(
        .NUM_HEADS(NUM_HEADS),
        .TILE     (TILE_DIM),
        .SEQ_LEN  (TILE_DIM)
    ) u_taylor (
        .clk_core        (clk_core),   // 1 GHz compute domain
        .rst_n           (rst_n),
        .cfg_start       (cfg_start),
        .cfg_done        (),
        .rx_bump_data    (h_tx_bump),
        .rx_bump_valid   (h_tx_bv),
        .rx_bump_credit  (h_tx_bcr),
        .tx_bump_data    (t_tx_bump),
        .tx_bump_valid   (t_tx_bv),
        .tx_bump_credit  (t_tx_bcr)
    );

    // =========================================================================
    // 8. Chiplet 0 result → host: UCIe RX
    // =========================================================================
    wire [15:0] host_rx_tile [TILE_DIM][TILE_DIM];
    wire        host_rx_valid;

    // CDC NOTE: rx_valid crosses clk_core → clk_axi here.
    // Add a 2-FF synchroniser on rx_valid before tapeout.
    ucie_rx #(.TILE_DIM(TILE_DIM)) u_host_rx (
        .clk_core    (clk_core),   // 1 GHz compute domain
        .rst_n       (rst_n),
        .bump_data   (c0_out_bump),
        .bump_valid  (c0_out_bv),
        .bump_credit (c0_out_bcr),
        .rx_valid    (host_rx_valid),
        .rx_src_id   (),
        .rx_tile     (host_rx_tile),
        .rx_ready    (out_tile_ready),
        .rx_crc_err  (),
        .rx_seq_err  ()
    );

    // Connect to axi_if output port
    assign out_tile_valid = host_rx_valid;
    genvar oi, oj;
    generate
        for (oi = 0; oi < TILE_DIM; oi++) begin : out_row_g
            for (oj = 0; oj < TILE_DIM; oj++) begin : out_col_g
                assign out_tile_data[oi][oj] = host_rx_tile[oi][oj];
            end
        end
    endgenerate

    // =========================================================================
    // 9. Status aggregation → axi_if
    // =========================================================================
    logic busy_r;
    // 2-FF synchroniser: cfg_start (clk_axi) → clk_core domain
    logic cfg_start_s1, cfg_start_core;
    always_ff @(posedge clk_core or negedge rst_n) begin : start_sync
        if (!rst_n) begin
            cfg_start_s1   <= 1'b0;
            cfg_start_core <= 1'b0;
        end else begin
            cfg_start_s1   <= cfg_start;
            cfg_start_core <= cfg_start_s1;
        end
    end

    // busy_r lives in clk_core domain
    always_ff @(posedge clk_core or negedge rst_n) begin : busy_ff
        if (!rst_n)              busy_r <= 1'b0;
        else if (cfg_start_core) busy_r <= 1'b1;
        else if (c0_done)        busy_r <= 1'b0;
    end

    assign sts_busy        = busy_r;

    // Latch c0_done in clk_core domain, then sync to clk_axi for CSR
    logic sts_done_core;
    always_ff @(posedge clk_core or negedge rst_n) begin : done_latch_core
        if (!rst_n)        sts_done_core <= 1'b0;
        else if (c0_done)  sts_done_core <= 1'b1;
        else               sts_done_core <= 1'b0;
    end

    // 2-FF sync: clk_core → clk_axi
    logic sts_done_s1, sts_done_r;
    always_ff @(posedge clk_axi or negedge rst_n) begin : done_sync
        if (!rst_n) begin
            sts_done_s1 <= 1'b0;
            sts_done_r  <= 1'b0;
        end else begin
            sts_done_s1 <= sts_done_core;
            sts_done_r  <= sts_done_s1;
        end
    end
    assign sts_done = sts_done_r;
    assign sts_error       = 1'b0;   // extend: collect UCIe CRC errors
    assign sts_active_head = 4'h0;   // extend: mux from head chiplets

    // =========================================================================
    // 10. Memory interface
    // =========================================================================
    assign mem_wdata = 512'h0;
    assign mem_wen   = 1'b0;
    assign mem_addr  = cfg_weight_addr;
    // mem_req stays high while busy (not just on the start pulse)
    assign mem_req   = busy_r;

endmodule

`default_nettype wire
// =============================================================================
// End of soc_top.sv
// =============================================================================
