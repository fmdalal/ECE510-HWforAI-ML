// =============================================================================
// axi_if.sv  —  AXI4-Lite + AXI4-Stream Interface Block
// =============================================================================
//
// Purpose
// -------
// This file contains ONLY the host-facing interface logic.
// It has no knowledge of chiplets, UCIe, or systolic arrays.
// It presents clean, tiled BF16 data outward to soc_top.sv.
//
// Block diagram
// -------------
//
//   CPU cores
//      │  Wishbone B4 master
//      ▼
//   ┌─────────────────────────────────┐
//   │         wb2axip bridge          │  (ZipCPU open-source)
//   │  wb2axilite ──► AXI4-Lite       │  single r/w, no burst
//   │  wb2axi + axis wrapper ──► AXI4-Stream │  512-bit tensor flow
//   └─────────────────────────────────┘
//          │                    │
//          ▼                    ▼
//   ┌─────────────┐    ┌──────────────────┐
//   │  axi_lite_  │    │  axis_input_fifo │  512-bit → BF16 tile
//   │  csr        │    │  axis_output_    │  BF16 tile → 512-bit
//   └─────────────┘    │  fifo            │
//          │           └──────────────────┘
//          │                    │
//    cfg_* signals         tile_* signals
//          │                    │
//          └────────┬───────────┘
//                   ▼
//             to soc_top.sv
//
// AXI4-Lite CSR Register Map  (byte addresses, 32-bit words)
// ----------------------------------------------------------
//  0x000  CTRL          [0]=start  [1]=reset  [2]=mode(0=stage1,1=stage5)
//  0x004  STATUS        [0]=busy   [1]=done   [2]=error  [7:4]=active_head
//  0x008  SEQ_LEN       sequence length (max 4096)
//  0x00C  D_MODEL       model dimension (default 512)
//  0x010  NUM_HEADS     number of attention heads (default 8)
//  0x014  NUM_TILES     D_MODEL / TILE_DIM  (default 8 for 512/64)
//  0x018  WEIGHT_ADDR_L HBM weight base address [31:0]
//  0x01C  WEIGHT_ADDR_H HBM weight base address [63:32]
//  0x020  IN_ADDR       input token buffer address [31:0]
//  0x024  OUT_ADDR      output buffer address [31:0]
//  0x028  INTR_EN       interrupt enable mask
//  0x02C  INTR_STAT     interrupt status (write-1-to-clear)
//  0x030  PERF_CYCLE_L  cycle counter [31:0]
//  0x034  PERF_CYCLE_H  cycle counter [63:32]
//  0x038  SCALE_BF16    1/sqrt(d_head) in BF16 in [31:16] (default 0x3E00)
//  0x03C  VERSION       read-only: 0x0002_0000
//  0x040  TILE_DIM      tile dimension (read-only, reflects parameter)
//  0x044  WDT_TIMEOUT   watchdog timeout in cycles (default 0x00FFFFFF)
//
// AXI4-Stream
// -----------
//  TDATA  : 512 bits = 32 BF16 words per beat
//  TKEEP  : 64 bytes, byte-enables
//  TUSER  : 4 bits  — 0=token, 1=weight, 2=ctrl, 3=result
//  TID    : 4 bits  — destination chiplet ID
//  TDEST  : 4 bits  — stream type
//  TLAST  : end-of-tile marker
//
// Tile assembly
// -------------
//  TILE_DIM x TILE_DIM x 16-bit = TILE_DIM² x 16 bits
//  For TILE_DIM=64: 64x64x16 = 65536 bits = 128 AXI-Stream beats (512-bit)
//
// Coding rules
// ------------
//  always_ff @(posedge clk_axi or negedge rst_n)   — all registered logic
//  always_comb begin                            — all combinational logic
//  No always @(*)  anywhere
//  No real / $bitstoreal / DW_* cells
// =============================================================================

`timescale 1ns/1ps
`default_nettype none

// =============================================================================
// wb2axip  —  ZipCPU Wishbone-to-AXI bridge (black-box)
// Source: https://github.com/ZipCPU/wb2axip
// Provide wb2axip.v from the repo when synthesising.
// =============================================================================
module wb2axip #(
    parameter int AW     = 32,
    parameter int DW     = 32,
    parameter int AXIDW  = 512
)(
    input  wire              i_clk,      // clk_axi  250 MHz (Wishbone/AXI clock)
    input  wire              i_reset,

    // Wishbone B4 slave
    input  wire              i_wb_cyc,
    input  wire              i_wb_stb,
    input  wire              i_wb_we,
    input  wire [AW-1:0]     i_wb_addr,
    input  wire [DW-1:0]     i_wb_data,
    input  wire [DW/8-1:0]   i_wb_sel,
    output wire              o_wb_stall,
    output wire              o_wb_ack,
    output wire [DW-1:0]     o_wb_data,
    output wire              o_wb_err,

    // AXI4-Lite master  (→ axi_lite_csr)
    output wire [11:0]       o_axil_awaddr,
    output wire [2:0]        o_axil_awprot,
    output wire              o_axil_awvalid,
    input  wire              i_axil_awready,
    output wire [DW-1:0]     o_axil_wdata,
    output wire [DW/8-1:0]   o_axil_wstrb,
    output wire              o_axil_wvalid,
    input  wire              i_axil_wready,
    input  wire [1:0]        i_axil_bresp,
    input  wire              i_axil_bvalid,
    output wire              o_axil_bready,
    output wire [11:0]       o_axil_araddr,
    output wire [2:0]        o_axil_arprot,
    output wire              o_axil_arvalid,
    input  wire              i_axil_arready,
    input  wire [DW-1:0]     i_axil_rdata,
    input  wire [1:0]        i_axil_rresp,
    input  wire              i_axil_rvalid,
    output wire              o_axil_rready,

    // AXI4-Stream master  (→ axis_input_fifo)
    output wire [AXIDW-1:0]    o_axis_tdata,
    output wire [AXIDW/8-1:0]  o_axis_tkeep,
    output wire                o_axis_tvalid,
    input  wire                i_axis_tready,
    output wire                o_axis_tlast,
    output wire [3:0]          o_axis_tuser,
    output wire [3:0]          o_axis_tid
);
endmodule


// =============================================================================
// axi_lite_csr
// AXI4-Lite slave.  Owns all CSR registers.
// Drives cfg_* outputs consumed by soc_top.
// Accepts status inputs from soc_top.
// =============================================================================
module axi_lite_csr #(
    parameter int TILE_DIM = 64
)(
    input  wire        clk_axi,   // 250 MHz AXI host interface clock
    input  wire        rst_n,

    // AXI4-Lite slave port
    input  wire [11:0] s_awaddr,
    input  wire [2:0]  s_awprot,
    input  wire        s_awvalid,
    output logic       s_awready,
    input  wire [31:0] s_wdata,
    input  wire [3:0]  s_wstrb,
    input  wire        s_wvalid,
    output logic       s_wready,
    output logic [1:0] s_bresp,
    output logic       s_bvalid,
    input  wire        s_bready,
    input  wire [11:0] s_araddr,
    input  wire [2:0]  s_arprot,
    input  wire        s_arvalid,
    output logic       s_arready,
    output logic [31:0] s_rdata,
    output logic [1:0]  s_rresp,
    output logic        s_rvalid,
    input  wire         s_rready,

    // Config outputs  →  soc_top / chiplets
    output logic        cfg_start,        // 1-cycle pulse on write
    output logic        cfg_reset,        // 1-cycle pulse on write
    output logic        cfg_mode,         // 0=stage1 QKV, 1=stage5 OutProj
    output logic [15:0] cfg_seq_len,
    output logic [15:0] cfg_d_model,
    output logic [7:0]  cfg_num_heads,
    output logic [7:0]  cfg_num_tiles,
    output logic [63:0] cfg_weight_addr,
    output logic [31:0] cfg_in_addr,
    output logic [31:0] cfg_out_addr,
    output logic [15:0] cfg_scale_bf16,   // BF16 in [15:0]
    output logic [31:0] cfg_wdt_timeout,

    // Status inputs  ←  soc_top
    input  wire         sts_busy,
    input  wire         sts_done,
    input  wire         sts_error,
    input  wire [3:0]   sts_active_head,
    input  wire [63:0]  sts_perf_cycles,

    // Interrupt output
    output logic        irq
);
    // -----------------------------------------------------------------------
    // Register storage
    // -----------------------------------------------------------------------
    logic [31:0] r_ctrl;
    logic [31:0] r_seq_len;
    logic [31:0] r_d_model;
    logic [31:0] r_num_heads;
    logic [31:0] r_num_tiles;
    logic [31:0] r_weight_l;
    logic [31:0] r_weight_h;
    logic [31:0] r_in_addr;
    logic [31:0] r_out_addr;
    logic [31:0] r_intr_en;
    logic [31:0] r_intr_stat;
    logic [31:0] r_scale;
    logic [31:0] r_wdt_timeout;

    localparam logic [31:0] VERSION  = 32'h0002_0000;
    localparam logic [31:0] TILE_REG = TILE_DIM;

    // -----------------------------------------------------------------------
    // Write channel FSM
    // -----------------------------------------------------------------------
    typedef enum logic [1:0] {
        WR_IDLE = 2'd0,
        WR_DATA = 2'd1,
        WR_RESP = 2'd2
    } wr_state_t;

    wr_state_t   wr_state;
    logic [11:0] wr_addr_lat;

    always_ff @(posedge clk_axi or negedge rst_n) begin : wr_ff
        if (!rst_n) begin
            wr_state      <= WR_IDLE;
            s_awready     <= 1'b0;
            s_wready      <= 1'b0;
            s_bvalid      <= 1'b0;
            s_bresp       <= 2'b00;
            cfg_start     <= 1'b0;
            cfg_reset     <= 1'b0;
            r_ctrl        <= 32'h0;
            r_seq_len     <= 32'd256;
            r_d_model     <= 32'd512;
            r_num_heads   <= 32'd8;
            r_num_tiles   <= 32'd8;
            r_weight_l    <= 32'h0;
            r_weight_h    <= 32'h0;
            r_in_addr     <= 32'h0;
            r_out_addr    <= 32'h0;
            r_intr_en     <= 32'h0;
            r_intr_stat   <= 32'h0;
            r_scale       <= {16'h3E00, 16'h0};  // 0.125 in BF16
            r_wdt_timeout <= 32'h00FF_FFFF;
        end else begin
            cfg_start <= 1'b0;   // auto-clear every cycle
            cfg_reset <= 1'b0;

            // interrupt set on rising status edges
            if (sts_done  & r_intr_en[1]) r_intr_stat <= r_intr_stat | 32'h2;
            if (sts_error & r_intr_en[2]) r_intr_stat <= r_intr_stat | 32'h4;

            case (wr_state)
                WR_IDLE: begin
                    s_awready <= 1'b1;
                    if (s_awvalid) begin
                        s_awready   <= 1'b0;
                        wr_addr_lat <= s_awaddr;
                        wr_state    <= WR_DATA;
                    end
                end

                WR_DATA: begin
                    s_wready <= 1'b1;
                    if (s_wvalid) begin
                        s_wready <= 1'b0;
                        casez (wr_addr_lat)
                            12'h000: begin
                                r_ctrl    <= s_wdata;
                                cfg_start <= s_wdata[0];
                                cfg_reset <= s_wdata[1];
                            end
                            12'h008: r_seq_len     <= s_wdata;
                            12'h00C: r_d_model     <= s_wdata;
                            12'h010: r_num_heads   <= s_wdata;
                            12'h014: r_num_tiles   <= s_wdata;
                            12'h018: r_weight_l    <= s_wdata;
                            12'h01C: r_weight_h    <= s_wdata;
                            12'h020: r_in_addr     <= s_wdata;
                            12'h024: r_out_addr    <= s_wdata;
                            12'h028: r_intr_en     <= s_wdata;
                            12'h02C: r_intr_stat   <= r_intr_stat & ~s_wdata; // W1C
                            12'h038: r_scale       <= s_wdata;
                            12'h044: r_wdt_timeout <= s_wdata;
                            default: ; // read-only or reserved: ignore
                        endcase
                        wr_state <= WR_RESP;
                    end
                end

                WR_RESP: begin
                    s_bvalid <= 1'b1;
                    s_bresp  <= 2'b00; // OKAY
                    if (s_bready) begin
                        s_bvalid <= 1'b0;
                        wr_state <= WR_IDLE;
                    end
                end

                default: wr_state <= WR_IDLE;
            endcase
        end
    end

    // -----------------------------------------------------------------------
    // Config output assignments  (combinational from registers)
    // -----------------------------------------------------------------------
    assign cfg_mode        = r_ctrl[2];
    assign cfg_seq_len     = r_seq_len[15:0];
    assign cfg_d_model     = r_d_model[15:0];
    assign cfg_num_heads   = r_num_heads[7:0];
    assign cfg_num_tiles   = r_num_tiles[7:0];
    assign cfg_weight_addr = {r_weight_h, r_weight_l};
    assign cfg_in_addr     = r_in_addr;
    assign cfg_out_addr    = r_out_addr;
    assign cfg_scale_bf16  = r_scale[31:16];
    assign cfg_wdt_timeout = r_wdt_timeout;
    assign irq             = |(r_intr_stat & r_intr_en);

    // -----------------------------------------------------------------------
    // Read channel FSM
    // -----------------------------------------------------------------------
    typedef enum logic [1:0] {
        RD_IDLE = 2'd0,
        RD_LATCH = 2'd1,
        RD_WAIT = 2'd2
    } rd_state_t;

    rd_state_t rd_state;

    always_ff @(posedge clk_axi or negedge rst_n) begin : rd_ff
        if (!rst_n) begin
            rd_state  <= RD_IDLE;
            s_arready <= 1'b0;
            s_rvalid  <= 1'b0;
            s_rdata   <= 32'h0;
            s_rresp   <= 2'b00;
        end else begin
            case (rd_state)
                RD_IDLE: begin
                    s_arready <= 1'b1;
                    s_rvalid  <= 1'b0;
                    if (s_arvalid) begin
                        s_arready <= 1'b0;
                        rd_state  <= RD_LATCH;
                    end
                end

                RD_LATCH: begin
                    s_rvalid <= 1'b1;
                    s_rresp  <= 2'b00;
                    casez (s_araddr)
                        12'h000: s_rdata <= r_ctrl;
                        12'h004: s_rdata <= {20'h0,
                                             sts_active_head,
                                             5'h0,
                                             sts_error,
                                             sts_done,
                                             sts_busy};
                        12'h008: s_rdata <= r_seq_len;
                        12'h00C: s_rdata <= r_d_model;
                        12'h010: s_rdata <= r_num_heads;
                        12'h014: s_rdata <= r_num_tiles;
                        12'h018: s_rdata <= r_weight_l;
                        12'h01C: s_rdata <= r_weight_h;
                        12'h020: s_rdata <= r_in_addr;
                        12'h024: s_rdata <= r_out_addr;
                        12'h028: s_rdata <= r_intr_en;
                        12'h02C: s_rdata <= r_intr_stat;
                        12'h030: s_rdata <= sts_perf_cycles[31:0];
                        12'h034: s_rdata <= sts_perf_cycles[63:32];
                        12'h038: s_rdata <= r_scale;
                        12'h03C: s_rdata <= VERSION;
                        12'h040: s_rdata <= TILE_REG;
                        12'h044: s_rdata <= r_wdt_timeout;
                        default: s_rdata <= 32'hDEAD_BEEF;
                    endcase
                    rd_state <= RD_WAIT;
                end

                RD_WAIT: begin
                    if (s_rready) begin
                        s_rvalid <= 1'b0;
                        rd_state <= RD_IDLE;
                    end
                end

                default: rd_state <= RD_IDLE;
            endcase
        end
    end
endmodule


// =============================================================================
// axis_input_fifo
// AXI4-Stream slave.  512-bit beats → BF16 TILE_DIM×TILE_DIM tile.
//
// Depth:  FIFO_D entries of 512 bits.
// Tiles:  TILE_DIM² × 16-bit words / 32 words-per-beat = BEATS_PER_TILE beats.
//         For TILE_DIM=64: 4096 words / 32 = 128 beats per tile.
//
// Outputs:
//   tile_out       — assembled BF16 tile (2-D unpacked array)
//   tile_valid     — tile_out is valid this cycle, accepted when tile_ready=1
//   tile_dst       — TID from the stream (destination chiplet)
//   tile_type      — TUSER from the stream (0=token, 1=weight)
// =============================================================================
module axis_input_fifo #(
    parameter int TDATA_W  = 512,
    parameter int TILE_DIM = 64,
    parameter int FIFO_D   = 256    // beats; must hold at least BEATS_PER_TILE
)(
    input  wire        clk_axi,   // 250 MHz AXI host interface clock
    input  wire        rst_n,

    // AXI4-Stream slave
    input  wire [TDATA_W-1:0]    s_tdata,
    input  wire [TDATA_W/8-1:0]  s_tkeep,
    input  wire                  s_tvalid,
    output logic                 s_tready,
    input  wire                  s_tlast,
    input  wire [3:0]            s_tuser,
    input  wire [3:0]            s_tid,

    // Tile output
    output logic [15:0]  tile_out   [TILE_DIM][TILE_DIM],
    output logic         tile_valid,
    output logic [3:0]   tile_dst,
    output logic [3:0]   tile_type,
    input  wire          tile_ready
);
    localparam int WORDS_PER_BEAT = TDATA_W / 16;                    // 32
    localparam int BEATS_PER_TILE = (TILE_DIM * TILE_DIM) / WORDS_PER_BEAT; // 128

    // FIFO storage
    logic [TDATA_W-1:0] f_data  [FIFO_D];
    logic [3:0]         f_tid   [FIFO_D];
    logic [3:0]         f_tuser [FIFO_D];
    logic               f_tlast [FIFO_D];

    logic [$clog2(FIFO_D)-1:0] wr_ptr, rd_ptr;
    logic [$clog2(FIFO_D):0]   count;

    wire fifo_full  = (count == FIFO_D[$clog2(FIFO_D):0]);
    wire fifo_empty = (count == '0);

    // ------------------------------------------------------------------
    // Write side: accept AXI-Stream beats into FIFO
    // ------------------------------------------------------------------
    always_ff @(posedge clk_axi or negedge rst_n) begin : fifo_wr_ff
        if (!rst_n) begin
            wr_ptr   <= '0;
            count    <= '0;
            s_tready <= 1'b1;
        end else begin
            s_tready <= ~fifo_full;
            if (s_tvalid & s_tready & ~fifo_full) begin
                f_data [wr_ptr] <= s_tdata;
                f_tid  [wr_ptr] <= s_tid;
                f_tuser[wr_ptr] <= s_tuser;
                f_tlast[wr_ptr] <= s_tlast;
                wr_ptr          <= wr_ptr + 1;
                count           <= count  + 1;
            end
        end
    end

    // ------------------------------------------------------------------
    // Read side: assemble BEATS_PER_TILE beats into one tile, then assert
    // tile_valid for one cycle (held until tile_ready).
    // ------------------------------------------------------------------
    logic [$clog2(BEATS_PER_TILE):0] beat_cnt;
    logic [15:0] flat [TILE_DIM * TILE_DIM];

    typedef enum logic [1:0] {
        TA_FILL  = 2'd0,
        TA_VALID = 2'd1,
        TA_WAIT  = 2'd2
    } ta_state_t;

    ta_state_t ta_state;

    always_ff @(posedge clk_axi or negedge rst_n) begin : tile_asm_ff
        if (!rst_n) begin
            ta_state   <= TA_FILL;
            rd_ptr     <= '0;
            beat_cnt   <= '0;
            tile_valid <= 1'b0;
            tile_dst   <= 4'h0;
            tile_type  <= 4'h0;
            for (int i = 0; i < TILE_DIM; i++)
                for (int j = 0; j < TILE_DIM; j++)
                    tile_out[i][j] <= 16'h0;
        end else begin
            case (ta_state)
                TA_FILL: begin
                    tile_valid <= 1'b0;
                    if (!fifo_empty) begin
                        // Unpack one beat into flat array
                        for (int w = 0; w < WORDS_PER_BEAT; w++) begin
                            int idx;
                            idx = beat_cnt * WORDS_PER_BEAT + w;
                            flat[idx] <= f_data[rd_ptr][w*16 +: 16];
                        end
                        tile_dst  <= f_tid  [rd_ptr];
                        tile_type <= f_tuser[rd_ptr];
                        rd_ptr    <= rd_ptr + 1;
                        count     <= count  - 1;

                        if (beat_cnt == BEATS_PER_TILE - 1) begin
                            beat_cnt <= '0;
                            ta_state <= TA_VALID;
                        end else begin
                            beat_cnt <= beat_cnt + 1;
                        end
                    end
                end

                TA_VALID: begin
                    // Unflatten into 2D tile
                    for (int i = 0; i < TILE_DIM; i++)
                        for (int j = 0; j < TILE_DIM; j++)
                            tile_out[i][j] <= flat[i * TILE_DIM + j];
                    tile_valid <= 1'b1;
                    ta_state   <= TA_WAIT;
                end

                TA_WAIT: begin
                    if (tile_ready) begin
                        tile_valid <= 1'b0;
                        ta_state   <= TA_FILL;
                    end
                end

                default: ta_state <= TA_FILL;
            endcase
        end
    end
endmodule


// =============================================================================
// axis_output_fifo
// BF16 TILE_DIM×TILE_DIM tile → AXI4-Stream master.
// Serialises result tiles back to the host DMA engine.
// =============================================================================
module axis_output_fifo #(
    parameter int TDATA_W  = 512,
    parameter int TILE_DIM = 64,
    parameter int FIFO_D   = 256
)(
    input  wire        clk_axi,   // 250 MHz AXI host interface clock
    input  wire        rst_n,

    // Tile input from soc_top
    input  wire  [15:0] tile_in    [TILE_DIM][TILE_DIM],
    input  wire         tile_valid,
    output logic        tile_ready,

    // AXI4-Stream master
    output logic [TDATA_W-1:0]    m_tdata,
    output logic [TDATA_W/8-1:0]  m_tkeep,
    output logic                  m_tvalid,
    input  wire                   m_tready,
    output logic                  m_tlast,
    output logic [3:0]            m_tuser   // always 3 = RESULT
);
    localparam int WORDS_PER_BEAT = TDATA_W / 16;
    localparam int BEATS_PER_TILE = (TILE_DIM * TILE_DIM) / WORDS_PER_BEAT;

    // Beat-level FIFO
    logic [TDATA_W-1:0] f_data [FIFO_D];
    logic               f_last [FIFO_D];

    logic [$clog2(FIFO_D)-1:0] wr_ptr, rd_ptr;
    logic [$clog2(FIFO_D):0]   count;

    wire fifo_full  = (count == FIFO_D[$clog2(FIFO_D):0]);
    wire fifo_empty = (count == '0);

    assign tile_ready = ~fifo_full;

    // ------------------------------------------------------------------
    // Write side: flatten tile → beats → FIFO
    // ------------------------------------------------------------------
    logic [$clog2(BEATS_PER_TILE):0] in_beat;
    logic [15:0] flat_in [TILE_DIM * TILE_DIM];

    always_ff @(posedge clk_axi or negedge rst_n) begin : out_wr_ff
        if (!rst_n) begin
            wr_ptr  <= '0;
            count   <= '0;
            in_beat <= '0;
            for (int i = 0; i < TILE_DIM * TILE_DIM; i++)
                flat_in[i] <= 16'h0;
        end else begin
            if (tile_valid & tile_ready) begin
                // Flatten on first beat of tile
                if (in_beat == 0)
                    for (int i = 0; i < TILE_DIM; i++)
                        for (int j = 0; j < TILE_DIM; j++)
                            flat_in[i * TILE_DIM + j] <= tile_in[i][j];

                // Pack one beat worth of words
                if (!fifo_full) begin
                    for (int w = 0; w < WORDS_PER_BEAT; w++) begin
                        int idx;
                        idx = in_beat * WORDS_PER_BEAT + w;
                        f_data[wr_ptr][w*16 +: 16] <= flat_in[idx];
                    end
                    f_last[wr_ptr] <= (in_beat == BEATS_PER_TILE - 1);
                    wr_ptr         <= wr_ptr + 1;
                    count          <= count  + 1;
                    in_beat        <= (in_beat == BEATS_PER_TILE - 1)
                                      ? '0 : in_beat + 1;
                end
            end
        end
    end

    // ------------------------------------------------------------------
    // Read side: drain FIFO → AXI-Stream master
    // ------------------------------------------------------------------
    always_ff @(posedge clk_axi or negedge rst_n) begin : out_rd_ff
        if (!rst_n) begin
            rd_ptr   <= '0;
            m_tvalid <= 1'b0;
            m_tdata  <= '0;
            m_tkeep  <= '1;
            m_tlast  <= 1'b0;
            m_tuser  <= 4'd3;  // RESULT
        end else begin
            if (!fifo_empty & (~m_tvalid | m_tready)) begin
                m_tdata  <= f_data[rd_ptr];
                m_tlast  <= f_last[rd_ptr];
                m_tkeep  <= {(TDATA_W/8){1'b1}};
                m_tuser  <= 4'd3;
                m_tvalid <= 1'b1;
                rd_ptr   <= rd_ptr + 1;
                count    <= count  - 1;
            end else if (fifo_empty & m_tready) begin
                m_tvalid <= 1'b0;
            end
        end
    end
endmodule


// =============================================================================
// watchdog
// Counts up while en=1.  Resets to 0 on kick.
// Asserts timeout when counter reaches limit.
// =============================================================================
module watchdog #(
    parameter int W = 32
)(
    input  wire         clk_axi,   // 250 MHz AXI host interface clock
    input  wire         rst_n,
    input  wire         en,
    input  wire         kick,
    input  wire [W-1:0] limit,
    output logic        timeout
);
    logic [W-1:0] cnt;

    always_ff @(posedge clk_axi or negedge rst_n) begin : wdt_ff
        if (!rst_n) begin
            cnt     <= '0;
            timeout <= 1'b0;
        end else if (!en) begin
            cnt     <= '0;
            timeout <= 1'b0;
        end else if (kick) begin
            cnt     <= '0;
            timeout <= 1'b0;
        end else begin
            if (cnt >= limit)
                timeout <= 1'b1;
            else
                cnt <= cnt + 1;
        end
    end
endmodule


// =============================================================================
// perf_counter
// 64-bit active cycle counter.  Resets on each cfg_start pulse.
// =============================================================================
module perf_counter (
    input  wire        clk_axi,   // 250 MHz AXI host interface clock
    input  wire        rst_n,
    input  wire        start,   // reset counter
    input  wire        en,      // count while high
    output logic [63:0] cycles
);
    always_ff @(posedge clk_axi or negedge rst_n) begin : perf_ff
        if (!rst_n)      cycles <= 64'h0;
        else if (start)  cycles <= 64'h0;
        else if (en)     cycles <= cycles + 64'd1;
    end
endmodule


// =============================================================================
// axi_if  —  top-level AXI interface block
//
// This is the ONLY module instantiated by soc_top.sv on the host side.
// It exposes:
//   - Wishbone B4 slave inward (from CPU)
//   - cfg_* outputs (config to chiplets)
//   - tile_* outputs (input data to chiplet 0)
//   - tile_* inputs  (output data from chiplet 0)
//   - sts_* inputs   (status from soc_top)
//   - irq output     (interrupt to CPU)
// =============================================================================
module axi_if #(
    parameter int TILE_DIM = 64,
    parameter int TDATA_W  = 512,
    parameter int FIFO_D   = 256
)(
    input  wire        clk_axi,   // 250 MHz AXI / host interface clock
    input  wire        rst_n,

    // ------------------------------------------------------------------
    // Wishbone B4 slave  (from CPU cores)
    // ------------------------------------------------------------------
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

    // ------------------------------------------------------------------
    // Config outputs  →  soc_top
    // ------------------------------------------------------------------
    output wire        cfg_start,
    output wire        cfg_reset,
    output wire        cfg_mode,
    output wire [15:0] cfg_seq_len,
    output wire [15:0] cfg_d_model,
    output wire [7:0]  cfg_num_heads,
    output wire [7:0]  cfg_num_tiles,
    output wire [63:0] cfg_weight_addr,
    output wire [31:0] cfg_in_addr,
    output wire [31:0] cfg_out_addr,
    output wire [15:0] cfg_scale_bf16,
    output wire [31:0] cfg_wdt_timeout,

    // ------------------------------------------------------------------
    // Input tile  →  soc_top  (token / weight data from DMA)
    // ------------------------------------------------------------------
    output wire [15:0] in_tile_data  [TILE_DIM][TILE_DIM],
    output wire        in_tile_valid,
    output wire [3:0]  in_tile_dst,    // destination chiplet
    output wire [3:0]  in_tile_type,   // 0=token, 1=weight
    input  wire        in_tile_ready,

    // ------------------------------------------------------------------
    // Output tile  ←  soc_top  (results back to DMA)
    // ------------------------------------------------------------------
    input  wire [15:0] out_tile_data  [TILE_DIM][TILE_DIM],
    input  wire        out_tile_valid,
    output wire        out_tile_ready,

    // ------------------------------------------------------------------
    // Status inputs  ←  soc_top
    // ------------------------------------------------------------------
    input  wire        sts_busy,
    input  wire        sts_done,
    input  wire        sts_error,
    input  wire [3:0]  sts_active_head,

    // ------------------------------------------------------------------
    // Interrupt output  →  CPU
    // ------------------------------------------------------------------
    output wire        irq,

    // ------------------------------------------------------------------
    // Performance counter output  (readable via CSR)
    // ------------------------------------------------------------------
    output wire [63:0] perf_cycles
);

    // -----------------------------------------------------------------------
    // wb2axip bridge:  Wishbone → AXI4-Lite + AXI4-Stream
    // -----------------------------------------------------------------------
    wire [11:0]  axil_awaddr;
    wire [2:0]   axil_awprot;
    wire         axil_awvalid, axil_awready;
    wire [31:0]  axil_wdata;
    wire [3:0]   axil_wstrb;
    wire         axil_wvalid,  axil_wready;
    wire [1:0]   axil_bresp;
    wire         axil_bvalid,  axil_bready;
    wire [11:0]  axil_araddr;
    wire [2:0]   axil_arprot;
    wire         axil_arvalid, axil_arready;
    wire [31:0]  axil_rdata;
    wire [1:0]   axil_rresp;
    wire         axil_rvalid,  axil_rready;

    wire [TDATA_W-1:0]   axis_tdata;
    wire [TDATA_W/8-1:0] axis_tkeep;
    wire                 axis_tvalid, axis_tready;
    wire                 axis_tlast;
    wire [3:0]           axis_tuser;
    wire [3:0]           axis_tid;

    wb2axip #(
        .AW(32), .DW(32), .AXIDW(TDATA_W)
    ) u_bridge (
        .i_clk           (clk_axi),   // 250 MHz AXI clock
        .i_reset         (~rst_n),
        .i_wb_cyc        (wb_cyc),
        .i_wb_stb        (wb_stb),
        .i_wb_we         (wb_we),
        .i_wb_addr       (wb_addr),
        .i_wb_data       (wb_wdata),
        .i_wb_sel        (wb_sel),
        .o_wb_stall      (wb_stall),
        .o_wb_ack        (wb_ack),
        .o_wb_data       (wb_rdata),
        .o_wb_err        (wb_err),
        // AXI4-Lite
        .o_axil_awaddr   (axil_awaddr),
        .o_axil_awprot   (axil_awprot),
        .o_axil_awvalid  (axil_awvalid),
        .i_axil_awready  (axil_awready),
        .o_axil_wdata    (axil_wdata),
        .o_axil_wstrb    (axil_wstrb),
        .o_axil_wvalid   (axil_wvalid),
        .i_axil_wready   (axil_wready),
        .i_axil_bresp    (axil_bresp),
        .i_axil_bvalid   (axil_bvalid),
        .o_axil_bready   (axil_bready),
        .o_axil_araddr   (axil_araddr),
        .o_axil_arprot   (axil_arprot),
        .o_axil_arvalid  (axil_arvalid),
        .i_axil_arready  (axil_arready),
        .i_axil_rdata    (axil_rdata),
        .i_axil_rresp    (axil_rresp),
        .i_axil_rvalid   (axil_rvalid),
        .o_axil_rready   (axil_rready),
        // AXI4-Stream
        .o_axis_tdata    (axis_tdata),
        .o_axis_tkeep    (axis_tkeep),
        .o_axis_tvalid   (axis_tvalid),
        .i_axis_tready   (axis_tready),
        .o_axis_tlast    (axis_tlast),
        .o_axis_tuser    (axis_tuser),
        .o_axis_tid      (axis_tid)
    );

    // -----------------------------------------------------------------------
    // AXI4-Lite CSR
    // -----------------------------------------------------------------------
    wire        csr_cfg_start;
    wire        csr_cfg_reset;
    wire        csr_cfg_mode;
    wire [15:0] csr_seq_len;
    wire [15:0] csr_d_model;
    wire [7:0]  csr_num_heads;
    wire [7:0]  csr_num_tiles;
    wire [63:0] csr_weight_addr;
    wire [31:0] csr_in_addr;
    wire [31:0] csr_out_addr;
    wire [15:0] csr_scale_bf16;
    wire [31:0] csr_wdt_timeout;

    axi_lite_csr #(.TILE_DIM(TILE_DIM)) u_csr (
        .clk_axi(clk_axi),
        .rst_n            (rst_n),
        // AXI4-Lite slave
        .s_awaddr         (axil_awaddr),
        .s_awprot         (axil_awprot),
        .s_awvalid        (axil_awvalid),
        .s_awready        (axil_awready),
        .s_wdata          (axil_wdata),
        .s_wstrb          (axil_wstrb),
        .s_wvalid         (axil_wvalid),
        .s_wready         (axil_wready),
        .s_bresp          (axil_bresp),
        .s_bvalid         (axil_bvalid),
        .s_bready         (axil_bready),
        .s_araddr         (axil_araddr),
        .s_arprot         (axil_arprot),
        .s_arvalid        (axil_arvalid),
        .s_arready        (axil_arready),
        .s_rdata          (axil_rdata),
        .s_rresp          (axil_rresp),
        .s_rvalid         (axil_rvalid),
        .s_rready         (axil_rready),
        // Config outputs
        .cfg_start        (csr_cfg_start),
        .cfg_reset        (csr_cfg_reset),
        .cfg_mode         (csr_cfg_mode),
        .cfg_seq_len      (csr_seq_len),
        .cfg_d_model      (csr_d_model),
        .cfg_num_heads    (csr_num_heads),
        .cfg_num_tiles    (csr_num_tiles),
        .cfg_weight_addr  (csr_weight_addr),
        .cfg_in_addr      (csr_in_addr),
        .cfg_out_addr     (csr_out_addr),
        .cfg_scale_bf16   (csr_scale_bf16),
        .cfg_wdt_timeout  (csr_wdt_timeout),
        // Status
        .sts_busy         (sts_busy),
        .sts_done         (sts_done),
        .sts_error        (sts_error),
        .sts_active_head  (sts_active_head),
        .sts_perf_cycles  (perf_cycles),
        .irq              (irq)
    );

    // Pass config outputs to top-level ports
    assign cfg_start       = csr_cfg_start;
    assign cfg_reset       = csr_cfg_reset;
    assign cfg_mode        = csr_cfg_mode;
    assign cfg_seq_len     = csr_seq_len;
    assign cfg_d_model     = csr_d_model;
    assign cfg_num_heads   = csr_num_heads;
    assign cfg_num_tiles   = csr_num_tiles;
    assign cfg_weight_addr = csr_weight_addr;
    assign cfg_in_addr     = csr_in_addr;
    assign cfg_out_addr    = csr_out_addr;
    assign cfg_scale_bf16  = csr_scale_bf16;
    assign cfg_wdt_timeout = csr_wdt_timeout;

    // -----------------------------------------------------------------------
    // AXI4-Stream input FIFO  (stream → tile)
    // -----------------------------------------------------------------------
    axis_input_fifo #(
        .TDATA_W (TDATA_W),
        .TILE_DIM(TILE_DIM),
        .FIFO_D  (FIFO_D)
    ) u_in_fifo (
        .clk_axi(clk_axi),
        .rst_n      (rst_n),
        .s_tdata    (axis_tdata),
        .s_tkeep    (axis_tkeep),
        .s_tvalid   (axis_tvalid),
        .s_tready   (axis_tready),
        .s_tlast    (axis_tlast),
        .s_tuser    (axis_tuser),
        .s_tid      (axis_tid),
        .tile_out   (in_tile_data),
        .tile_valid (in_tile_valid),
        .tile_dst   (in_tile_dst),
        .tile_type  (in_tile_type),
        .tile_ready (in_tile_ready)
    );

    // -----------------------------------------------------------------------
    // AXI4-Stream output FIFO  (tile → stream)
    // -----------------------------------------------------------------------
    wire [TDATA_W-1:0]   m_tdata;
    wire [TDATA_W/8-1:0] m_tkeep;
    wire                 m_tvalid;
    wire                 m_tlast;
    wire [3:0]           m_tuser;

    axis_output_fifo #(
        .TDATA_W (TDATA_W),
        .TILE_DIM(TILE_DIM),
        .FIFO_D  (FIFO_D)
    ) u_out_fifo (
        .clk_axi(clk_axi),
        .rst_n      (rst_n),
        .tile_in    (out_tile_data),
        .tile_valid (out_tile_valid),
        .tile_ready (out_tile_ready),
        .m_tdata    (m_tdata),
        .m_tkeep    (m_tkeep),
        .m_tvalid   (m_tvalid),
        .m_tready   (1'b1),    // host always ready; for backpressure wire to
                               // a DMA-ready signal from the CPU subsystem
        .m_tlast    (m_tlast),
        .m_tuser    (m_tuser)
    );

    // -----------------------------------------------------------------------
    // Watchdog
    // -----------------------------------------------------------------------
    watchdog #(.W(32)) u_wdt (
        .clk_axi(clk_axi),
        .rst_n   (rst_n),
        .en      (sts_busy),
        .kick    (out_tile_valid),   // any output = progress
        .limit   (csr_wdt_timeout),
        .timeout (/* connected in soc_top */)
    );

    // -----------------------------------------------------------------------
    // Performance counter
    // -----------------------------------------------------------------------
    perf_counter u_perf (
        .clk_axi(clk_axi),
        .rst_n  (rst_n),
        .start  (csr_cfg_start),
        .en     (sts_busy),
        .cycles (perf_cycles)
    );

endmodule

`default_nettype wire
// =============================================================================
// End of axi_if.sv
// =============================================================================
