`default_nettype none
module chiplet_head (
	clk_core,
	rst_n,
	cfg_mode,
	cfg_num_tiles,
	cfg_start,
	cfg_done,
	chiplet_id,
	rxa_bump_data,
	rxa_bump_valid,
	rxa_bump_credit,
	rxb_bump_data,
	rxb_bump_valid,
	rxb_bump_credit,
	tx_bump_data,
	tx_bump_valid,
	tx_bump_credit,
	scale_factor
);
	parameter signed [31:0] HEAD_ID = 0;
	parameter signed [31:0] D_HEAD = 64;
	parameter signed [31:0] TILE = 64;
	parameter signed [31:0] K_DIM = 64;
	parameter signed [31:0] SEQ_TILE = 64;
	input wire clk_core;
	input wire rst_n;
	input wire cfg_mode;
	input wire [7:0] cfg_num_tiles;
	input wire cfg_start;
	output wire cfg_done;
	output wire [3:0] chiplet_id;
	input wire [511:0] rxa_bump_data;
	input wire rxa_bump_valid;
	output wire rxa_bump_credit;
	input wire [511:0] rxb_bump_data;
	input wire rxb_bump_valid;
	output wire rxb_bump_credit;
	output wire [511:0] tx_bump_data;
	output wire tx_bump_valid;
	input wire tx_bump_credit;
	input wire [15:0] scale_factor;
	assign cfg_done = 1'b0;
	assign chiplet_id = HEAD_ID[3:0];
	assign rxa_bump_credit = 1'b1;
	assign rxb_bump_credit = 1'b1;
	assign tx_bump_data = 1'sb0;
	assign tx_bump_valid = 1'b0;
endmodule
