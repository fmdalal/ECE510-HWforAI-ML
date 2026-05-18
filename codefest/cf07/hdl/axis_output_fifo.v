`default_nettype none
module axis_output_fifo (
	clk_axi,
	rst_n,
	tile_in,
	tile_valid,
	tile_ready,
	m_tdata,
	m_tkeep,
	m_tvalid,
	m_tready,
	m_tlast,
	m_tuser
);
	parameter signed [31:0] TDATA_W = 512;
	parameter signed [31:0] TILE_DIM = 64;
	parameter signed [31:0] FIFO_D = 256;
	input wire clk_axi;
	input wire rst_n;
	input wire [((TILE_DIM * TILE_DIM) * 16) - 1:0] tile_in;
	input wire tile_valid;
	output wire tile_ready;
	output reg [TDATA_W - 1:0] m_tdata;
	output reg [(TDATA_W / 8) - 1:0] m_tkeep;
	output reg m_tvalid;
	input wire m_tready;
	output reg m_tlast;
	output reg [3:0] m_tuser;
	localparam signed [31:0] WORDS_PER_BEAT = TDATA_W / 16;
	localparam signed [31:0] BEATS_PER_TILE = (TILE_DIM * TILE_DIM) / WORDS_PER_BEAT;
	reg [TDATA_W - 1:0] f_data [0:FIFO_D - 1];
	reg f_last [0:FIFO_D - 1];
	reg [$clog2(FIFO_D) - 1:0] wr_ptr;
	reg [$clog2(FIFO_D) - 1:0] rd_ptr;
	reg [$clog2(FIFO_D):0] count;
	wire fifo_full = count == FIFO_D[$clog2(FIFO_D):0];
	wire fifo_empty = count == {($clog2(FIFO_D) >= 0 ? $clog2(FIFO_D) + 1 : 1 - $clog2(FIFO_D)) {1'sb0}};
	assign tile_ready = ~fifo_full;
	reg [$clog2(BEATS_PER_TILE):0] in_beat;
	reg [15:0] flat_in [0:(TILE_DIM * TILE_DIM) - 1];
	always @(posedge clk_axi or negedge rst_n) begin : out_wr_ff
		if (!rst_n) begin
			wr_ptr <= 1'sb0;
			count <= 1'sb0;
			in_beat <= 1'sb0;
			begin : sv2v_autoblock_1
				reg signed [31:0] i;
				for (i = 0; i < (TILE_DIM * TILE_DIM); i = i + 1)
					flat_in[i] <= 16'h0000;
			end
		end
		else if (tile_valid & tile_ready) begin
			if (in_beat == 0) begin : sv2v_autoblock_2
				reg signed [31:0] i;
				for (i = 0; i < TILE_DIM; i = i + 1)
					begin : sv2v_autoblock_3
						reg signed [31:0] j;
						for (j = 0; j < TILE_DIM; j = j + 1)
							flat_in[(i * TILE_DIM) + j] <= tile_in[((((TILE_DIM - 1) - i) * TILE_DIM) + ((TILE_DIM - 1) - j)) * 16+:16];
					end
			end
			if (!fifo_full) begin
				begin : sv2v_autoblock_4
					reg signed [31:0] w;
					for (w = 0; w < WORDS_PER_BEAT; w = w + 1)
						begin : sv2v_autoblock_5
							reg signed [31:0] idx;
							idx = (in_beat * WORDS_PER_BEAT) + w;
							f_data[wr_ptr][w * 16+:16] <= flat_in[idx];
						end
				end
				f_last[wr_ptr] <= in_beat == (BEATS_PER_TILE - 1);
				wr_ptr <= wr_ptr + 1;
				count <= count + 1;
				in_beat <= (in_beat == (BEATS_PER_TILE - 1) ? {($clog2(BEATS_PER_TILE) >= 0 ? $clog2(BEATS_PER_TILE) + 1 : 1 - $clog2(BEATS_PER_TILE)) {1'sb0}} : in_beat + 1);
			end
		end
	end
	always @(posedge clk_axi or negedge rst_n) begin : out_rd_ff
		if (!rst_n) begin
			rd_ptr <= 1'sb0;
			m_tvalid <= 1'b0;
			m_tdata <= 1'sb0;
			m_tkeep <= 1'sb1;
			m_tlast <= 1'b0;
			m_tuser <= 4'd3;
		end
		else if (!fifo_empty & (~m_tvalid | m_tready)) begin
			m_tdata <= f_data[rd_ptr];
			m_tlast <= f_last[rd_ptr];
			m_tkeep <= {TDATA_W / 8 {1'b1}};
			m_tuser <= 4'd3;
			m_tvalid <= 1'b1;
			rd_ptr <= rd_ptr + 1;
			count <= count - 1;
		end
		else if (fifo_empty & m_tready)
			m_tvalid <= 1'b0;
	end
endmodule
