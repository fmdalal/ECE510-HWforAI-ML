// =============================================================================
// soc_top_stubs.sv  —  Compile-time stubs for tb_soc_top cocotb testbench
// =============================================================================
// Compile BEFORE axi_if.sv and compute_core.sv:
//   iverilog -g2012 soc_top_stubs.sv axi_if.sv compute_core.sv
//
// The chiplet_0 stub auto-fires cfg_done ~20 clk_core cycles after cfg_start
// so the soc_top FSM (busy_r) can be exercised without real chiplet RTL.
// =============================================================================
`timescale 1ns/1ps
`default_nettype none



// ---------------------------------------------------------------------------
// ucie_tx stub
// ---------------------------------------------------------------------------
module ucie_tx #(parameter int TILE_DIM = 64)(
    input  wire        clk_core,
    input  wire        rst_n,
    input  wire        tx_valid,
    input  wire [3:0]  tx_src_id,
    input  wire [3:0]  tx_dst_id,
    input  wire [15:0] tx_tile [TILE_DIM][TILE_DIM],
    output wire        tx_ready,
    output wire [511:0] bump_data,
    output wire         bump_valid,
    input  wire         bump_credit
);
    assign tx_ready   = 1'b1;
    assign bump_data  = '0;
    assign bump_valid = 1'b0;
endmodule


// ---------------------------------------------------------------------------
// ucie_rx stub
// ---------------------------------------------------------------------------
module ucie_rx #(parameter int TILE_DIM = 64)(
    input  wire         clk_core,
    input  wire         rst_n,
    input  wire [511:0] bump_data,
    input  wire         bump_valid,
    output wire         bump_credit,
    output wire         rx_valid,
    output wire [3:0]   rx_src_id,
    output wire [15:0]  rx_tile [TILE_DIM][TILE_DIM],
    input  wire         rx_ready,
    output wire         rx_crc_err,
    output wire         rx_seq_err
);
    assign bump_credit = 1'b1;
    assign rx_valid    = 1'b0;
    assign rx_src_id   = '0;
    assign rx_crc_err  = 1'b0;
    assign rx_seq_err  = 1'b0;
    genvar i, j;
    generate
        for (i = 0; i < TILE_DIM; i++)
            for (j = 0; j < TILE_DIM; j++)
                assign rx_tile[i][j] = '0;
    endgenerate
endmodule


// ---------------------------------------------------------------------------
// chiplet_0_qkv_outproj stub
// Fires cfg_done ~20 clk_core cycles after cfg_start to exercise busy_r FSM.
// ---------------------------------------------------------------------------
module chiplet_0_qkv_outproj #(
    parameter int D_MODEL   = 512,
    parameter int NUM_HEADS = 8,
    parameter int D_HEAD    = 64,
    parameter int TILE      = 64,
    parameter int K_DIM     = 64
)(
    input  wire        clk,
    input  wire        rst_n,
    input  wire        cfg_mode,
    input  wire [7:0]  cfg_num_tiles,
    input  wire        cfg_start,
    output wire        cfg_done,
    input  wire [511:0] rx_bump_data,
    input  wire         rx_bump_valid,
    output wire         rx_bump_credit,
    output wire [511:0] txq_bump_data,
    output wire         txq_bump_valid,
    input  wire         txq_bump_credit,
    output wire [511:0] txk_bump_data,
    output wire         txk_bump_valid,
    input  wire         txk_bump_credit,
    output wire [511:0] txv_bump_data,
    output wire         txv_bump_valid,
    input  wire         txv_bump_credit,
    output wire [511:0] txout_bump_data,
    output wire         txout_bump_valid,
    input  wire         txout_bump_credit,
    output wire [63:0]  sram_addr,
    input  wire [511:0] sram_rdata,
    output wire         sram_rd_en,
    input  wire         sram_rd_valid
);
    logic [7:0] cnt;
    logic done_r;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            cnt    <= '0;
            done_r <= 1'b0;
        end else if (cfg_start) begin
            cnt    <= '0;
            done_r <= 1'b0;
        end else begin
            if (cnt < 8'd200) cnt <= cnt + 1;
            done_r <= (cnt == 8'd199);
        end
    end

    assign cfg_done         = done_r;
    assign rx_bump_credit   = 1'b1;
    assign txq_bump_data    = '0;  assign txq_bump_valid    = 1'b0;
    assign txk_bump_data    = '0;  assign txk_bump_valid    = 1'b0;
    assign txv_bump_data    = '0;  assign txv_bump_valid    = 1'b0;
    assign txout_bump_data  = '0;  assign txout_bump_valid  = 1'b0;
    assign sram_addr        = '0;  assign sram_rd_en        = 1'b0;
endmodule


// ---------------------------------------------------------------------------
// chiplet_head stub
// ---------------------------------------------------------------------------
module chiplet_head #(
    parameter int HEAD_ID  = 0,
    parameter int D_HEAD   = 64,
    parameter int TILE     = 64,
    parameter int K_DIM    = 64,
    parameter int SEQ_TILE = 64
)(
    input  wire        clk_core,
    input  wire        rst_n,
    input  wire        cfg_mode,
    input  wire [7:0]  cfg_num_tiles,
    input  wire        cfg_start,
    output wire        cfg_done,
    output wire [3:0]  chiplet_id,
    input  wire [511:0] rxa_bump_data,
    input  wire         rxa_bump_valid,
    output wire         rxa_bump_credit,
    input  wire [511:0] rxb_bump_data,
    input  wire         rxb_bump_valid,
    output wire         rxb_bump_credit,
    output wire [511:0] tx_bump_data,
    output wire         tx_bump_valid,
    input  wire         tx_bump_credit,
    input  wire [15:0]  scale_factor
);
    assign cfg_done        = 1'b0;
    assign chiplet_id      = HEAD_ID[3:0];
    assign rxa_bump_credit = 1'b1;
    assign rxb_bump_credit = 1'b1;
    assign tx_bump_data    = '0;
    assign tx_bump_valid   = 1'b0;
endmodule


// ---------------------------------------------------------------------------
// chiplet_9_taylor stub
// ---------------------------------------------------------------------------
module chiplet_9_taylor #(
    parameter int NUM_HEADS = 8,
    parameter int TILE      = 64,
    parameter int SEQ_LEN   = 64
)(
    input  wire        clk_core,
    input  wire        rst_n,
    input  wire        cfg_start,
    output wire        cfg_done,
    input  wire [511:0] rx_bump_data  [NUM_HEADS],
    input  wire         rx_bump_valid [NUM_HEADS],
    output wire         rx_bump_credit[NUM_HEADS],
    output wire [511:0] tx_bump_data  [NUM_HEADS],
    output wire         tx_bump_valid [NUM_HEADS],
    input  wire         tx_bump_credit[NUM_HEADS]
);
    assign cfg_done = 1'b0;
    genvar k;
    generate for (k = 0; k < NUM_HEADS; k++) begin : t_stub
        assign rx_bump_credit[k] = 1'b1;
        assign tx_bump_data[k]   = '0;
        assign tx_bump_valid[k]  = 1'b0;
    end endgenerate
endmodule









`default_nettype wire
