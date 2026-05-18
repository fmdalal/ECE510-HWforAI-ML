`default_nettype none
module ucie_rx (
	clk_core,
	rst_n,
	bump_data,
	bump_valid,
	bump_credit,
	rx_valid,
	rx_src_id,
	rx_tile,
	rx_ready,
	rx_crc_err,
	rx_seq_err
);
	reg _sv2v_0;
	parameter signed [31:0] TILE_DIM = 16;
	parameter signed [31:0] WORDS = 256;
	parameter signed [31:0] WORDS_PER_FL = 29;
	parameter signed [31:0] FLITS = 9;
	input wire clk_core;
	input wire rst_n;
	input wire [511:0] bump_data;
	input wire bump_valid;
	output reg bump_credit;
	output reg rx_valid;
	output reg [3:0] rx_src_id;
	output reg [((TILE_DIM * TILE_DIM) * 16) - 1:0] rx_tile;
	input wire rx_ready;
	output reg rx_crc_err;
	output reg rx_seq_err;
	wire [3:0] src_id = bump_data[511:508];
	wire [3:0] dst_id = bump_data[507:504];
	wire [7:0] seq_num = bump_data[503:496];
	wire [7:0] flit_num = bump_data[495:488];
	wire [463:0] payload = bump_data[479:16];
	wire [7:0] rx_crc = bump_data[15:8];
	reg [495:0] crc_chk_in;
	wire [7:0] crc_chk_val;
	always @(*) begin
		if (_sv2v_0)
			;
		crc_chk_in = bump_data[511:16];
	end
	ucie_crc8 #(.W(496)) crc_chk_inst(
		.data_in(crc_chk_in),
		.crc_out(crc_chk_val)
	);
	reg [15:0] buf_words [0:WORDS - 1];
	reg [3:0] expected_flit;
	reg [7:0] expected_seq;
	always @(*) begin
		if (_sv2v_0)
			;
		begin : sv2v_autoblock_1
			reg signed [31:0] i;
			for (i = 0; i < TILE_DIM; i = i + 1)
				begin : sv2v_autoblock_2
					reg signed [31:0] j;
					for (j = 0; j < TILE_DIM; j = j + 1)
						rx_tile[((((TILE_DIM - 1) - i) * TILE_DIM) + ((TILE_DIM - 1) - j)) * 16+:16] = buf_words[(i * TILE_DIM) + j];
				end
		end
	end
	always @(posedge clk_core or negedge rst_n) begin : rx_ff
		if (!rst_n) begin
			rx_valid <= 1'b0;
			rx_src_id <= 4'h0;
			rx_crc_err <= 1'b0;
			rx_seq_err <= 1'b0;
			bump_credit <= 1'b0;
			expected_flit <= 4'd0;
			expected_seq <= 8'd0;
			begin : sv2v_autoblock_3
				reg signed [31:0] w;
				for (w = 0; w < WORDS; w = w + 1)
					buf_words[w] <= 16'h0000;
			end
		end
		else begin
			rx_valid <= 1'b0;
			rx_crc_err <= 1'b0;
			rx_seq_err <= 1'b0;
			bump_credit <= 1'b0;
			if (bump_valid) begin
				if (crc_chk_val != rx_crc)
					rx_crc_err <= 1'b1;
				else if (flit_num != {4'h0, expected_flit}) begin
					rx_seq_err <= 1'b1;
					expected_flit <= 4'd0;
				end
				else begin
					begin : sv2v_autoblock_4
						reg signed [31:0] w;
						for (w = 0; w < WORDS_PER_FL; w = w + 1)
							begin : sv2v_autoblock_5
								reg signed [31:0] idx;
								idx = (flit_num[3:0] * WORDS_PER_FL) + w;
								if (idx < WORDS)
									buf_words[idx] <= payload[w * 16+:16];
							end
					end
					bump_credit <= 1'b1;
					rx_src_id <= src_id;
					if (flit_num == (FLITS - 1)) begin
						expected_flit <= 4'd0;
						expected_seq <= expected_seq + 8'd1;
						rx_valid <= 1'b1;
					end
					else
						expected_flit <= expected_flit + 4'd1;
				end
			end
		end
	end
	initial _sv2v_0 = 0;
endmodule
