`default_nettype none
module ucie_tx (
	clk_core,
	rst_n,
	tx_valid,
	tx_src_id,
	tx_dst_id,
	tx_tile,
	tx_ready,
	bump_data,
	bump_valid,
	bump_credit
);
	reg _sv2v_0;
	parameter signed [31:0] TILE_DIM = 16;
	parameter signed [31:0] WORDS = 256;
	parameter signed [31:0] WORDS_PER_FL = 29;
	parameter signed [31:0] FLITS = 9;
	input wire clk_core;
	input wire rst_n;
	input wire tx_valid;
	input wire [3:0] tx_src_id;
	input wire [3:0] tx_dst_id;
	input wire [((TILE_DIM * TILE_DIM) * 16) - 1:0] tx_tile;
	output reg tx_ready;
	output reg [511:0] bump_data;
	output reg bump_valid;
	input wire bump_credit;
	reg [15:0] flat [0:WORDS - 1];
	always @(*) begin
		if (_sv2v_0)
			;
		begin : sv2v_autoblock_1
			reg signed [31:0] i;
			for (i = 0; i < TILE_DIM; i = i + 1)
				begin : sv2v_autoblock_2
					reg signed [31:0] j;
					for (j = 0; j < TILE_DIM; j = j + 1)
						flat[(i * TILE_DIM) + j] = tx_tile[((((TILE_DIM - 1) - i) * TILE_DIM) + ((TILE_DIM - 1) - j)) * 16+:16];
				end
		end
	end
	reg [1:0] tx_state;
	reg [3:0] flit_cnt;
	reg [7:0] seq_cnt;
	reg [3:0] credits;
	reg [15:0] tile_buf [0:WORDS - 1];
	reg [495:0] crc_in;
	wire [7:0] crc_val;
	reg [463:0] payload;
	always @(*) begin
		if (_sv2v_0)
			;
		payload = 464'h0;
		begin : sv2v_autoblock_3
			reg signed [31:0] w;
			for (w = 0; w < WORDS_PER_FL; w = w + 1)
				begin : sv2v_autoblock_4
					reg signed [31:0] idx;
					idx = (flit_cnt * WORDS_PER_FL) + w;
					if (idx < WORDS)
						payload[w * 16+:16] = tile_buf[idx];
					else
						payload[w * 16+:16] = 16'h0000;
				end
		end
	end
	always @(*) begin
		if (_sv2v_0)
			;
		crc_in = {tx_src_id, tx_dst_id, seq_cnt, flit_cnt[3:0], 12'h009, payload};
	end
	ucie_crc8 #(.W(496)) crc_inst(
		.data_in(crc_in),
		.crc_out(crc_val)
	);
	always @(posedge clk_core or negedge rst_n) begin : credit_ff
		if (!rst_n)
			credits <= 4'd8;
		else if (bump_credit & ~bump_valid)
			credits <= credits + 4'd1;
		else if ((~bump_credit & bump_valid) & (credits > 0))
			credits <= credits - 4'd1;
	end
	always @(posedge clk_core or negedge rst_n) begin : tx_fsm_ff
		if (!rst_n) begin
			tx_state <= 2'd0;
			tx_ready <= 1'b1;
			bump_valid <= 1'b0;
			flit_cnt <= 4'd0;
			seq_cnt <= 8'd0;
			bump_data <= 512'h0;
			begin : sv2v_autoblock_5
				reg signed [31:0] w;
				for (w = 0; w < WORDS; w = w + 1)
					tile_buf[w] <= 16'h0000;
			end
		end
		else
			case (tx_state)
				2'd0: begin
					bump_valid <= 1'b0;
					tx_ready <= 1'b1;
					if (tx_valid) begin
						begin : sv2v_autoblock_6
							reg signed [31:0] w;
							for (w = 0; w < WORDS; w = w + 1)
								tile_buf[w] <= flat[w];
						end
						flit_cnt <= 4'd0;
						tx_ready <= 1'b0;
						tx_state <= 2'd1;
					end
				end
				2'd1:
					if (credits > 0) begin
						bump_data <= {tx_src_id, tx_dst_id, seq_cnt, flit_cnt[3:0], 12'h009, payload, crc_val, 8'h00};
						bump_valid <= 1'b1;
						if (flit_cnt == (FLITS - 1)) begin
							seq_cnt <= seq_cnt + 8'd1;
							flit_cnt <= 4'd0;
							tx_state <= 2'd0;
						end
						else begin
							flit_cnt <= flit_cnt + 4'd1;
							tx_state <= 2'd2;
						end
					end
					else
						bump_valid <= 1'b0;
				2'd2: begin
					bump_valid <= 1'b0;
					tx_state <= 2'd1;
				end
				default: tx_state <= 2'd0;
			endcase
	end
	initial _sv2v_0 = 0;
endmodule
