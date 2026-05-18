`default_nettype none
module bf16_mac (
	clk_core,
	rst_n,
	en,
	flush,
	a,
	b,
	acc_fp32_in,
	acc_fp32_out,
	acc_bf16_out
);
	input wire clk_core;
	input wire rst_n;
	input wire en;
	input wire flush;
	input wire [15:0] a;
	input wire [15:0] b;
	input wire [31:0] acc_fp32_in;
	output reg [31:0] acc_fp32_out;
	output reg [15:0] acc_bf16_out;
	wire [31:0] fp32_prod;
	fp32_mul u_mul(
		.a(a),
		.b(b),
		.result(fp32_prod)
	);
	wire [31:0] fp32_sum;
	fp32_add u_add(
		.a(fp32_prod),
		.b(acc_fp32_in),
		.result(fp32_sum)
	);
	always @(posedge clk_core or negedge rst_n) begin : acc_fp32_ff
		if (!rst_n)
			acc_fp32_out <= 32'h00000000;
		else if (en)
			acc_fp32_out <= fp32_sum;
	end
	wire rnd_bit = fp32_sum[15];
	wire sticky = |fp32_sum[14:0];
	wire lsb_bit = fp32_sum[16];
	wire round_up = rnd_bit & (sticky | lsb_bit);
	wire [15:0] bf16_rne = fp32_sum[31:16] + {15'h0000, round_up};
	always @(posedge clk_core or negedge rst_n) begin : acc_bf16_ff
		if (!rst_n)
			acc_bf16_out <= 16'h0000;
		else if (flush)
			acc_bf16_out <= bf16_rne;
	end
endmodule
