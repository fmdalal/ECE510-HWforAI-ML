`default_nettype none
module axis_input_fifo (
	clk_axi,
	rst_n,
	s_tdata,
	s_tkeep,
	s_tvalid,
	s_tready,
	s_tlast,
	s_tuser,
	s_tid,
	tile_out,
	tile_valid,
	tile_dst,
	tile_type,
	tile_ready
);
	parameter signed [31:0] TDATA_W = 512;
	parameter signed [31:0] TILE_DIM = 64;
	parameter signed [31:0] FIFO_D = 256;
	input wire clk_axi;
	input wire rst_n;
	input wire [TDATA_W - 1:0] s_tdata;
	input wire [(TDATA_W / 8) - 1:0] s_tkeep;
	input wire s_tvalid;
	output reg s_tready;
	input wire s_tlast;
	input wire [3:0] s_tuser;
	input wire [3:0] s_tid;
	output reg [((TILE_DIM * TILE_DIM) * 16) - 1:0] tile_out;
	output reg tile_valid;
	output reg [3:0] tile_dst;
	output reg [3:0] tile_type;
	input wire tile_ready;
	localparam signed [31:0] WORDS_PER_BEAT = TDATA_W / 16;
	localparam signed [31:0] BEATS_PER_TILE = (TILE_DIM * TILE_DIM) / WORDS_PER_BEAT;
	reg [TDATA_W - 1:0] f_data [0:FIFO_D - 1];
	reg [3:0] f_tid [0:FIFO_D - 1];
	reg [3:0] f_tuser [0:FIFO_D - 1];
	reg f_tlast [0:FIFO_D - 1];
	reg [$clog2(FIFO_D) - 1:0] wr_ptr;
	reg [$clog2(FIFO_D) - 1:0] rd_ptr;
	reg [$clog2(FIFO_D):0] count;
	wire fifo_full = count == FIFO_D[$clog2(FIFO_D):0];
	wire fifo_empty = count == {($clog2(FIFO_D) >= 0 ? $clog2(FIFO_D) + 1 : 1 - $clog2(FIFO_D)) {1'sb0}};
	always @(posedge clk_axi or negedge rst_n) begin : fifo_wr_ff
		if (!rst_n) begin
			wr_ptr <= 1'sb0;
			count <= 1'sb0;
			s_tready <= 1'b1;
		end
		else begin
			s_tready <= ~fifo_full;
			if ((s_tvalid & s_tready) & ~fifo_full) begin
				f_data[wr_ptr] <= s_tdata;
				f_tid[wr_ptr] <= s_tid;
				f_tuser[wr_ptr] <= s_tuser;
				f_tlast[wr_ptr] <= s_tlast;
				wr_ptr <= wr_ptr + 1;
				count <= count + 1;
			end
		end
	end
	reg [$clog2(BEATS_PER_TILE):0] beat_cnt;
	reg [15:0] flat [0:(TILE_DIM * TILE_DIM) - 1];
	reg [1:0] ta_state;
	always @(posedge clk_axi or negedge rst_n) begin : tile_asm_ff
		if (!rst_n) begin
			ta_state <= 2'd0;
			rd_ptr <= 1'sb0;
			beat_cnt <= 1'sb0;
			tile_valid <= 1'b0;
			tile_dst <= 4'h0;
			tile_type <= 4'h0;
			begin : sv2v_autoblock_1
				reg signed [31:0] i;
				for (i = 0; i < TILE_DIM; i = i + 1)
					begin : sv2v_autoblock_2
						reg signed [31:0] j;
						for (j = 0; j < TILE_DIM; j = j + 1)
							tile_out[((((TILE_DIM - 1) - i) * TILE_DIM) + ((TILE_DIM - 1) - j)) * 16+:16] <= 16'h0000;
					end
			end
		end
		else
			case (ta_state)
				2'd0: begin
					tile_valid <= 1'b0;
					if (!fifo_empty) begin
						begin : sv2v_autoblock_3
							reg signed [31:0] w;
							for (w = 0; w < WORDS_PER_BEAT; w = w + 1)
								begin : sv2v_autoblock_4
									reg signed [31:0] idx;
									idx = (beat_cnt * WORDS_PER_BEAT) + w;
									flat[idx] <= f_data[rd_ptr][w * 16+:16];
								end
						end
						tile_dst <= f_tid[rd_ptr];
						tile_type <= f_tuser[rd_ptr];
						rd_ptr <= rd_ptr + 1;
						count <= count - 1;
						if (beat_cnt == (BEATS_PER_TILE - 1)) begin
							beat_cnt <= 1'sb0;
							ta_state <= 2'd1;
						end
						else
							beat_cnt <= beat_cnt + 1;
					end
				end
				2'd1: begin
					begin : sv2v_autoblock_5
						reg signed [31:0] i;
						for (i = 0; i < TILE_DIM; i = i + 1)
							begin : sv2v_autoblock_6
								reg signed [31:0] j;
								for (j = 0; j < TILE_DIM; j = j + 1)
									tile_out[((((TILE_DIM - 1) - i) * TILE_DIM) + ((TILE_DIM - 1) - j)) * 16+:16] <= flat[(i * TILE_DIM) + j];
							end
					end
					tile_valid <= 1'b1;
					ta_state <= 2'd2;
				end
				2'd2:
					if (tile_ready) begin
						tile_valid <= 1'b0;
						ta_state <= 2'd0;
					end
				default: ta_state <= 2'd0;
			endcase
	end
endmodule
