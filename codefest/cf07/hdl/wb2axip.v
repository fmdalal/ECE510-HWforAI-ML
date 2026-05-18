`default_nettype none
module wb2axip #(
    parameter signed [31:0] AW = 32,
    parameter signed [31:0] DW = 32,
    parameter signed [31:0] AXIDW = 512
)(
    input  wire              i_clk,
    input  wire              i_reset,
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
    output wire [11:0]       o_axil_awaddr,
    output wire [2:0]        o_axil_awprot,
    output wire              o_axil_awvalid,
    input  wire              i_axil_awready,
    output wire [AXIDW-1:0]  o_axil_wdata,
    output wire [AXIDW/8-1:0] o_axil_wstrb,
    output wire              o_axil_wvalid,
    input  wire              i_axil_wready,
    input  wire [1:0]        i_axil_bresp,
    input  wire              i_axil_bvalid,
    output wire              o_axil_bready,
    output wire [11:0]       o_axil_araddr,
    output wire [2:0]        o_axil_arprot,
    output wire              o_axil_arvalid,
    input  wire              i_axil_arready,
    input  wire [AXIDW-1:0]  i_axil_rdata,
    input  wire [1:0]        i_axil_rresp,
    input  wire              i_axil_rvalid,
    output wire              o_axil_rready,
    output wire [AXIDW-1:0]  o_axis_tdata,
    output wire [AXIDW/8-1:0] o_axis_tkeep,
    output wire              o_axis_tvalid,
    input  wire              i_axis_tready,
    output wire              o_axis_tlast,
    output wire              o_axis_tuser,
    output wire              o_axis_tid
);
    assign o_wb_stall     = 1'b0;
    assign o_wb_ack       = i_wb_stb & i_wb_cyc;
    assign o_wb_data      = {DW{1'b0}};
    assign o_wb_err       = 1'b0;
    assign o_axil_awaddr  = 12'h0;
    assign o_axil_awprot  = 3'h0;
    assign o_axil_awvalid = 1'b0;
    assign o_axil_wdata   = {AXIDW{1'b0}};
    assign o_axil_wstrb   = {(AXIDW/8){1'b0}};
    assign o_axil_wvalid  = 1'b0;
    assign o_axil_bready  = 1'b1;
    assign o_axil_araddr  = 12'h0;
    assign o_axil_arprot  = 3'h0;
    assign o_axil_arvalid = 1'b0;
    assign o_axil_rready  = 1'b1;
    assign o_axis_tdata   = {AXIDW{1'b0}};
    assign o_axis_tkeep   = {(AXIDW/8){1'b0}};
    assign o_axis_tvalid  = 1'b0;
    assign o_axis_tlast   = 1'b0;
    assign o_axis_tuser   = 1'b0;
    assign o_axis_tid     = 1'b0;
endmodule