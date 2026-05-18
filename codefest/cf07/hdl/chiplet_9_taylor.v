`default_nettype none
module chiplet_9_taylor (
	clk_core,
	rst_n,
	cfg_start,
	cfg_done,
	rx_bump_data,
	rx_bump_valid,
	rx_bump_credit,
	tx_bump_data,
	tx_bump_valid,
	tx_bump_credit
);
	parameter signed [31:0] NUM_HEADS = 8;
	parameter signed [31:0] TILE = 64;
	parameter signed [31:0] SEQ_LEN = 64;
	input wire clk_core;
	input wire rst_n;
	input wire cfg_start;
	output wire cfg_done;
	input wire [(NUM_HEADS * 512) - 1:0] rx_bump_data;
	input wire [0:NUM_HEADS - 1] rx_bump_valid;
	output wire [0:NUM_HEADS - 1] rx_bump_credit;
	output wire [(NUM_HEADS * 512) - 1:0] tx_bump_data;
	output wire [0:NUM_HEADS - 1] tx_bump_valid;
	input wire [0:NUM_HEADS - 1] tx_bump_credit;
	assign cfg_done = 1'b0;
	genvar _gv_k_1;
	generate
		for (_gv_k_1 = 0; _gv_k_1 < NUM_HEADS; _gv_k_1 = _gv_k_1 + 1) begin : t_stub
			localparam k = _gv_k_1;
			assign rx_bump_credit[k] = 1'b1;
			assign tx_bump_data[((NUM_HEADS - 1) - k) * 512+:512] = 1'sb0;
			assign tx_bump_valid[k] = 1'b0;
		end
	endgenerate
endmodule
