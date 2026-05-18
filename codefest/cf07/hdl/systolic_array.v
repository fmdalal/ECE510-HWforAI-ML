`default_nettype none
module systolic_array (
	clk_core,
	rst_n,
	en,
	clear,
	flush,
	a_row,
	b_col,
	c_out,
	valid_out
);
	parameter signed [31:0] M = 64;
	parameter signed [31:0] N = 64;
	parameter signed [31:0] K = 64;
	input wire clk_core;
	input wire rst_n;
	input wire en;
	input wire clear;
	input wire flush;
	input wire [(M * 16) - 1:0] a_row;
	input wire [((N * K) * 16) - 1:0] b_col;
	output wire [((M * N) * 16) - 1:0] c_out;
	output reg valid_out;
	reg [31:0] acc_fp32 [0:M - 1][0:N - 1];
	reg [15:0] a_reg [0:M - 1][0:K - 1];
	always @(posedge clk_core or negedge rst_n) begin : skew_ff
		if (!rst_n) begin : sv2v_autoblock_1
			reg signed [31:0] i;
			for (i = 0; i < M; i = i + 1)
				begin : sv2v_autoblock_2
					reg signed [31:0] k;
					for (k = 0; k < K; k = k + 1)
						a_reg[i][k] <= 16'h0000;
				end
		end
		else if (en) begin : sv2v_autoblock_3
			reg signed [31:0] i;
			for (i = 0; i < M; i = i + 1)
				begin
					a_reg[i][0] <= a_row[((M - 1) - i) * 16+:16];
					begin : sv2v_autoblock_4
						reg signed [31:0] k;
						for (k = 1; k < K; k = k + 1)
							a_reg[i][k] <= a_reg[i][k - 1];
					end
				end
		end
	end
	genvar _gv_gi_1;
	genvar _gv_gj_1;
	generate
		for (_gv_gi_1 = 0; _gv_gi_1 < M; _gv_gi_1 = _gv_gi_1 + 1) begin : row_gen
			localparam gi = _gv_gi_1;
			for (_gv_gj_1 = 0; _gv_gj_1 < N; _gv_gj_1 = _gv_gj_1 + 1) begin : col_gen
				localparam gj = _gv_gj_1;
				wire [31:0] fp32_out_w;
				bf16_mac pe_inst(
					.clk_core(clk_core),
					.rst_n(rst_n & ~clear),
					.en(en),
					.flush(flush),
					.a(a_reg[gi][(gj < K ? gj : K - 1)]),
					.b(b_col[((((N - 1) - gj) * K) + ((K - 1) - (gi < K ? gi : K - 1))) * 16+:16]),
					.acc_fp32_in(acc_fp32[gi][gj]),
					.acc_fp32_out(fp32_out_w),
					.acc_bf16_out(c_out[((((M - 1) - gi) * N) + ((N - 1) - gj)) * 16+:16])
				);
				always @(posedge clk_core or negedge rst_n) begin : acc_fb_ff
					if (!rst_n || clear)
						acc_fp32[gi][gj] <= 32'h00000000;
					else if (en)
						acc_fp32[gi][gj] <= fp32_out_w;
				end
			end
		end
	endgenerate
	localparam signed [31:0] DRAIN = (((M + N) - 1) + K) + 1;
	reg [7:0] cycle_cnt;
	always @(posedge clk_core or negedge rst_n) begin : drain_ff
		if (!rst_n || clear) begin
			cycle_cnt <= 8'd0;
			valid_out <= 1'b0;
		end
		else if (en) begin
			if (cycle_cnt < DRAIN[7:0])
				cycle_cnt <= cycle_cnt + 8'd1;
			valid_out <= flush & (cycle_cnt >= (DRAIN[7:0] - 8'd1));
		end
		else
			valid_out <= 1'b0;
	end
endmodule
