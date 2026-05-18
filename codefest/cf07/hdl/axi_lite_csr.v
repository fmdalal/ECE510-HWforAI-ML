`default_nettype none
module axi_lite_csr (
	clk_axi,
	rst_n,
	s_awaddr,
	s_awprot,
	s_awvalid,
	s_awready,
	s_wdata,
	s_wstrb,
	s_wvalid,
	s_wready,
	s_bresp,
	s_bvalid,
	s_bready,
	s_araddr,
	s_arprot,
	s_arvalid,
	s_arready,
	s_rdata,
	s_rresp,
	s_rvalid,
	s_rready,
	cfg_start,
	cfg_reset,
	cfg_mode,
	cfg_seq_len,
	cfg_d_model,
	cfg_num_heads,
	cfg_num_tiles,
	cfg_weight_addr,
	cfg_in_addr,
	cfg_out_addr,
	cfg_scale_bf16,
	cfg_wdt_timeout,
	sts_busy,
	sts_done,
	sts_error,
	sts_active_head,
	sts_perf_cycles,
	irq
);
	parameter signed [31:0] TILE_DIM = 64;
	input wire clk_axi;
	input wire rst_n;
	input wire [11:0] s_awaddr;
	input wire [2:0] s_awprot;
	input wire s_awvalid;
	output reg s_awready;
	input wire [31:0] s_wdata;
	input wire [3:0] s_wstrb;
	input wire s_wvalid;
	output reg s_wready;
	output reg [1:0] s_bresp;
	output reg s_bvalid;
	input wire s_bready;
	input wire [11:0] s_araddr;
	input wire [2:0] s_arprot;
	input wire s_arvalid;
	output reg s_arready;
	output reg [31:0] s_rdata;
	output reg [1:0] s_rresp;
	output reg s_rvalid;
	input wire s_rready;
	output reg cfg_start;
	output reg cfg_reset;
	output wire cfg_mode;
	output wire [15:0] cfg_seq_len;
	output wire [15:0] cfg_d_model;
	output wire [7:0] cfg_num_heads;
	output wire [7:0] cfg_num_tiles;
	output wire [63:0] cfg_weight_addr;
	output wire [31:0] cfg_in_addr;
	output wire [31:0] cfg_out_addr;
	output wire [15:0] cfg_scale_bf16;
	output wire [31:0] cfg_wdt_timeout;
	input wire sts_busy;
	input wire sts_done;
	input wire sts_error;
	input wire [3:0] sts_active_head;
	input wire [63:0] sts_perf_cycles;
	output wire irq;
	reg [31:0] r_ctrl;
	reg [31:0] r_seq_len;
	reg [31:0] r_d_model;
	reg [31:0] r_num_heads;
	reg [31:0] r_num_tiles;
	reg [31:0] r_weight_l;
	reg [31:0] r_weight_h;
	reg [31:0] r_in_addr;
	reg [31:0] r_out_addr;
	reg [31:0] r_intr_en;
	reg [31:0] r_intr_stat;
	reg [31:0] r_scale;
	reg [31:0] r_wdt_timeout;
	localparam [31:0] VERSION = 32'h00020000;
	localparam [31:0] TILE_REG = TILE_DIM;
	reg [1:0] wr_state;
	reg [11:0] wr_addr_lat;
	always @(posedge clk_axi or negedge rst_n) begin : wr_ff
		if (!rst_n) begin
			wr_state <= 2'd0;
			s_awready <= 1'b0;
			s_wready <= 1'b0;
			s_bvalid <= 1'b0;
			s_bresp <= 2'b00;
			cfg_start <= 1'b0;
			cfg_reset <= 1'b0;
			r_ctrl <= 32'h00000000;
			r_seq_len <= 32'd256;
			r_d_model <= 32'd512;
			r_num_heads <= 32'd8;
			r_num_tiles <= 32'd8;
			r_weight_l <= 32'h00000000;
			r_weight_h <= 32'h00000000;
			r_in_addr <= 32'h00000000;
			r_out_addr <= 32'h00000000;
			r_intr_en <= 32'h00000000;
			r_intr_stat <= 32'h00000000;
			r_scale <= 32'h3e000000;
			r_wdt_timeout <= 32'h00ffffff;
		end
		else begin
			cfg_start <= 1'b0;
			cfg_reset <= 1'b0;
			if (sts_done & r_intr_en[1])
				r_intr_stat <= r_intr_stat | 32'h00000002;
			if (sts_error & r_intr_en[2])
				r_intr_stat <= r_intr_stat | 32'h00000004;
			case (wr_state)
				2'd0: begin
					s_awready <= 1'b1;
					if (s_awvalid) begin
						s_awready <= 1'b0;
						wr_addr_lat <= s_awaddr;
						wr_state <= 2'd1;
					end
				end
				2'd1: begin
					s_wready <= 1'b1;
					if (s_wvalid) begin
						s_wready <= 1'b0;
						casez (wr_addr_lat)
							12'h000: begin
								r_ctrl <= s_wdata;
								cfg_start <= s_wdata[0];
								cfg_reset <= s_wdata[1];
							end
							12'h008: r_seq_len <= s_wdata;
							12'h00c: r_d_model <= s_wdata;
							12'h010: r_num_heads <= s_wdata;
							12'h014: r_num_tiles <= s_wdata;
							12'h018: r_weight_l <= s_wdata;
							12'h01c: r_weight_h <= s_wdata;
							12'h020: r_in_addr <= s_wdata;
							12'h024: r_out_addr <= s_wdata;
							12'h028: r_intr_en <= s_wdata;
							12'h02c: r_intr_stat <= r_intr_stat & ~s_wdata;
							12'h038: r_scale <= s_wdata;
							12'h044: r_wdt_timeout <= s_wdata;
							default:
								;
						endcase
						wr_state <= 2'd2;
					end
				end
				2'd2: begin
					s_bvalid <= 1'b1;
					s_bresp <= 2'b00;
					if (s_bready) begin
						s_bvalid <= 1'b0;
						wr_state <= 2'd0;
					end
				end
				default: wr_state <= 2'd0;
			endcase
		end
	end
	assign cfg_mode = r_ctrl[2];
	assign cfg_seq_len = r_seq_len[15:0];
	assign cfg_d_model = r_d_model[15:0];
	assign cfg_num_heads = r_num_heads[7:0];
	assign cfg_num_tiles = r_num_tiles[7:0];
	assign cfg_weight_addr = {r_weight_h, r_weight_l};
	assign cfg_in_addr = r_in_addr;
	assign cfg_out_addr = r_out_addr;
	assign cfg_scale_bf16 = r_scale[31:16];
	assign cfg_wdt_timeout = r_wdt_timeout;
	assign irq = |(r_intr_stat & r_intr_en);
	reg [1:0] rd_state;
	always @(posedge clk_axi or negedge rst_n) begin : rd_ff
		if (!rst_n) begin
			rd_state <= 2'd0;
			s_arready <= 1'b0;
			s_rvalid <= 1'b0;
			s_rdata <= 32'h00000000;
			s_rresp <= 2'b00;
		end
		else
			case (rd_state)
				2'd0: begin
					s_arready <= 1'b1;
					s_rvalid <= 1'b0;
					if (s_arvalid) begin
						s_arready <= 1'b0;
						rd_state <= 2'd1;
					end
				end
				2'd1: begin
					s_rvalid <= 1'b1;
					s_rresp <= 2'b00;
					casez (s_araddr)
						12'h000: s_rdata <= r_ctrl;
						12'h004: s_rdata <= {20'h00000, sts_active_head, 5'h00, sts_error, sts_done, sts_busy};
						12'h008: s_rdata <= r_seq_len;
						12'h00c: s_rdata <= r_d_model;
						12'h010: s_rdata <= r_num_heads;
						12'h014: s_rdata <= r_num_tiles;
						12'h018: s_rdata <= r_weight_l;
						12'h01c: s_rdata <= r_weight_h;
						12'h020: s_rdata <= r_in_addr;
						12'h024: s_rdata <= r_out_addr;
						12'h028: s_rdata <= r_intr_en;
						12'h02c: s_rdata <= r_intr_stat;
						12'h030: s_rdata <= sts_perf_cycles[31:0];
						12'h034: s_rdata <= sts_perf_cycles[63:32];
						12'h038: s_rdata <= r_scale;
						12'h03c: s_rdata <= VERSION;
						12'h040: s_rdata <= TILE_REG;
						12'h044: s_rdata <= r_wdt_timeout;
						default: s_rdata <= 32'hdeadbeef;
					endcase
					rd_state <= 2'd2;
				end
				2'd2:
					if (s_rready) begin
						s_rvalid <= 1'b0;
						rd_state <= 2'd0;
					end
				default: rd_state <= 2'd0;
			endcase
	end
endmodule
