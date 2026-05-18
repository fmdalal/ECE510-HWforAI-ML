`default_nettype none
module chiplet_0_qkv_outproj (
	clk,
	rst_n,
	cfg_mode,
	cfg_num_tiles,
	cfg_start,
	cfg_done,
	rx_bump_data,
	rx_bump_valid,
	rx_bump_credit,
	txq_bump_data,
	txq_bump_valid,
	txq_bump_credit,
	txk_bump_data,
	txk_bump_valid,
	txk_bump_credit,
	txv_bump_data,
	txv_bump_valid,
	txv_bump_credit,
	txout_bump_data,
	txout_bump_valid,
	txout_bump_credit,
	sram_addr,
	sram_rdata,
	sram_rd_en,
	sram_rd_valid
);
	parameter signed [31:0] D_MODEL = 512;
	parameter signed [31:0] NUM_HEADS = 8;
	parameter signed [31:0] D_HEAD = 64;
	parameter signed [31:0] TILE = 64;
	parameter signed [31:0] K_DIM = 64;
	input wire clk;
	input wire rst_n;
	input wire cfg_mode;
	input wire [7:0] cfg_num_tiles;
	input wire cfg_start;
	output wire cfg_done;
	input wire [511:0] rx_bump_data;
	input wire rx_bump_valid;
	output wire rx_bump_credit;
	output wire [511:0] txq_bump_data;
	output wire txq_bump_valid;
	input wire txq_bump_credit;
	output wire [511:0] txk_bump_data;
	output wire txk_bump_valid;
	input wire txk_bump_credit;
	output wire [511:0] txv_bump_data;
	output wire txv_bump_valid;
	input wire txv_bump_credit;
	output wire [511:0] txout_bump_data;
	output wire txout_bump_valid;
	input wire txout_bump_credit;
	output wire [63:0] sram_addr;
	input wire [511:0] sram_rdata;
	output wire sram_rd_en;
	input wire sram_rd_valid;
	reg [7:0] cnt;
	reg done_r;
	always @(posedge clk or negedge rst_n)
		if (!rst_n) begin
			cnt <= 1'sb0;
			done_r <= 1'b0;
		end
		else if (cfg_start) begin
			cnt <= 1'sb0;
			done_r <= 1'b0;
		end
		else begin
			if (cnt < 8'd200)
				cnt <= cnt + 1;
			done_r <= cnt == 8'd199;
		end
	assign cfg_done = done_r;
	assign rx_bump_credit = 1'b1;
	assign txq_bump_data = 1'sb0;
	assign txq_bump_valid = 1'b0;
	assign txk_bump_data = 1'sb0;
	assign txk_bump_valid = 1'b0;
	assign txv_bump_data = 1'sb0;
	assign txv_bump_valid = 1'b0;
	assign txout_bump_data = 1'sb0;
	assign txout_bump_valid = 1'b0;
	assign sram_addr = 1'sb0;
	assign sram_rd_en = 1'b0;
endmodule
