`default_nettype none
module synth_top (
	clk_axi,
	clk_core,
	clk_link,
	rst_n,
	wb_cyc,
	wb_stb,
	wb_we,
	wb_addr,
	wb_wdata,
	wb_sel,
	wb_stall,
	wb_ack,
	wb_rdata,
	wb_err,
	mem_addr,
	mem_wdata,
	mem_wen,
	mem_rdata,
	mem_rvalid,
	mem_req,
	mem_gnt,
	out_tile_data,
	out_tile_valid,
	irq
);
	parameter signed [31:0] NUM_HEADS = 8;
	parameter signed [31:0] TILE_DIM = 64;
	parameter signed [31:0] D_HEAD = 64;
	parameter signed [31:0] D_MODEL = 512;
	parameter signed [31:0] TDATA_W = 512;
	parameter signed [31:0] FIFO_D = 256;
	input wire clk_axi;
	input wire clk_core;
	input wire clk_link;
	input wire rst_n;
	input wire wb_cyc;
	input wire wb_stb;
	input wire wb_we;
	input wire [31:0] wb_addr;
	input wire [31:0] wb_wdata;
	input wire [3:0] wb_sel;
	output wire wb_stall;
	output wire wb_ack;
	output wire [31:0] wb_rdata;
	output wire wb_err;
	output wire [63:0] mem_addr;
	output wire [511:0] mem_wdata;
	output wire mem_wen;
	input wire [511:0] mem_rdata;
	input wire mem_rvalid;
	output wire mem_req;
	input wire mem_gnt;
	output wire [((TILE_DIM * TILE_DIM) * 16) - 1:0] out_tile_data;
	output wire out_tile_valid;
	output wire irq;
	wire cfg_start;
	wire cfg_reset;
	wire cfg_mode;
	wire [15:0] cfg_seq_len;
	wire [15:0] cfg_d_model;
	wire [7:0] cfg_num_heads;
	wire [7:0] cfg_num_tiles;
	wire [63:0] cfg_weight_addr;
	wire [31:0] cfg_in_addr;
	wire [31:0] cfg_out_addr;
	wire [15:0] cfg_scale_bf16;
	wire [31:0] cfg_wdt_timeout;
	wire [((TILE_DIM * TILE_DIM) * 16) - 1:0] in_tile_data;
	wire in_tile_valid;
	wire [3:0] in_tile_dst;
	wire [3:0] in_tile_type;
	wire in_tile_ready;
	wire out_tile_ready;
	wire sts_busy;
	wire sts_done;
	wire sts_error;
	wire [3:0] sts_active_head;
	wire [63:0] perf_cycles;
	axi_if #(
		.TILE_DIM(TILE_DIM),
		.TDATA_W(TDATA_W),
		.FIFO_D(FIFO_D)
	) u_axi_if(
		.clk_axi(clk_axi),
		.rst_n(rst_n),
		.wb_cyc(wb_cyc),
		.wb_stb(wb_stb),
		.wb_we(wb_we),
		.wb_addr(wb_addr),
		.wb_wdata(wb_wdata),
		.wb_sel(wb_sel),
		.wb_stall(wb_stall),
		.wb_ack(wb_ack),
		.wb_rdata(wb_rdata),
		.wb_err(wb_err),
		.cfg_start(cfg_start),
		.cfg_reset(cfg_reset),
		.cfg_mode(cfg_mode),
		.cfg_seq_len(cfg_seq_len),
		.cfg_d_model(cfg_d_model),
		.cfg_num_heads(cfg_num_heads),
		.cfg_num_tiles(cfg_num_tiles),
		.cfg_weight_addr(cfg_weight_addr),
		.cfg_in_addr(cfg_in_addr),
		.cfg_out_addr(cfg_out_addr),
		.cfg_scale_bf16(cfg_scale_bf16),
		.cfg_wdt_timeout(cfg_wdt_timeout),
		.in_tile_data(in_tile_data),
		.in_tile_valid(in_tile_valid),
		.in_tile_dst(in_tile_dst),
		.in_tile_type(in_tile_type),
		.in_tile_ready(in_tile_ready),
		.out_tile_data(out_tile_data),
		.out_tile_valid(out_tile_valid),
		.out_tile_ready(out_tile_ready),
		.sts_busy(sts_busy),
		.sts_done(sts_done),
		.sts_error(sts_error),
		.sts_active_head(sts_active_head),
		.irq(irq),
		.perf_cycles(perf_cycles)
	);
	wire [511:0] h2c0_bump;
	wire h2c0_bv;
	wire h2c0_bcr;
	wire [511:0] c0_txq_bump;
	wire c0_txq_bv;
	wire c0_txq_bcr;
	wire [511:0] c0_txk_bump;
	wire c0_txk_bv;
	wire c0_txk_bcr;
	wire [511:0] c0_txv_bump;
	wire c0_txv_bv;
	wire c0_txv_bcr;
	wire [511:0] c0_out_bump;
	wire c0_out_bv;
	wire c0_out_bcr;
	wire [(NUM_HEADS * 512) - 1:0] h_tx_bump;
	wire [0:NUM_HEADS - 1] h_tx_bv;
	wire [0:NUM_HEADS - 1] h_tx_bcr;
	wire [(NUM_HEADS * 512) - 1:0] t_tx_bump;
	wire [0:NUM_HEADS - 1] t_tx_bv;
	wire [0:NUM_HEADS - 1] t_tx_bcr;
	wire [511:0] h_ctx_bump [0:NUM_HEADS - 1];
	wire h_ctx_bv [0:NUM_HEADS - 1];
	wire h_ctx_bcr [0:NUM_HEADS - 1];
	reg cfg_start_core;
	ucie_tx #(.TILE_DIM(TILE_DIM)) u_host_tx(
		.clk_core(clk_core),
		.rst_n(rst_n),
		.tx_valid(in_tile_valid & cfg_start_core),
		.tx_src_id(4'hf),
		.tx_dst_id(4'd0),
		.tx_tile(in_tile_data),
		.tx_ready(in_tile_ready),
		.bump_data(h2c0_bump),
		.bump_valid(h2c0_bv),
		.bump_credit(h2c0_bcr)
	);
	wire c0_done;
	chiplet_0_qkv_outproj #(
		.D_MODEL(D_MODEL),
		.NUM_HEADS(NUM_HEADS),
		.D_HEAD(D_HEAD),
		.TILE(TILE_DIM),
		.K_DIM(TILE_DIM)
	) u_c0(
		.clk(clk_core),
		.rst_n(rst_n),
		.cfg_mode(cfg_mode),
		.cfg_num_tiles(cfg_num_tiles),
		.cfg_start(cfg_start_core),
		.cfg_done(c0_done),
		.rx_bump_data(h2c0_bump),
		.rx_bump_valid(h2c0_bv),
		.rx_bump_credit(h2c0_bcr),
		.txq_bump_data(c0_txq_bump),
		.txq_bump_valid(c0_txq_bv),
		.txq_bump_credit(c0_txq_bcr),
		.txk_bump_data(c0_txk_bump),
		.txk_bump_valid(c0_txk_bv),
		.txk_bump_credit(c0_txk_bcr),
		.txv_bump_data(c0_txv_bump),
		.txv_bump_valid(c0_txv_bv),
		.txv_bump_credit(c0_txv_bcr),
		.txout_bump_data(c0_out_bump),
		.txout_bump_valid(c0_out_bv),
		.txout_bump_credit(c0_out_bcr),
		.sram_addr(),
		.sram_rdata({65536{1'b0}}),
		.sram_rd_en(),
		.sram_rd_valid(mem_rvalid)
	);
	wire [511:0] head_rxa_data [0:NUM_HEADS - 1];
	wire head_rxa_bv [0:NUM_HEADS - 1];
	wire head_rxa_bcr [0:NUM_HEADS - 1];
	wire [511:0] head_rxb_data [0:NUM_HEADS - 1];
	wire head_rxb_bv [0:NUM_HEADS - 1];
	wire head_rxb_bcr [0:NUM_HEADS - 1];
	wire [511:0] head_tx_data [0:NUM_HEADS - 1];
	wire head_tx_bv [0:NUM_HEADS - 1];
	wire head_tx_bcr [0:NUM_HEADS - 1];
	genvar _gv_hh_1;
	generate
		for (_gv_hh_1 = 0; _gv_hh_1 < NUM_HEADS; _gv_hh_1 = _gv_hh_1 + 1) begin : head_mux
			localparam hh = _gv_hh_1;
			assign head_rxa_data[hh] = (cfg_mode ? t_tx_bump[((NUM_HEADS - 1) - hh) * 512+:512] : c0_txq_bump);
			assign head_rxa_bv[hh] = (cfg_mode ? t_tx_bv[hh] : c0_txq_bv);
			assign t_tx_bcr[hh] = (cfg_mode ? head_rxa_bcr[hh] : 1'b0);
			assign c0_txq_bcr = (cfg_mode ? 1'b0 : head_rxa_bcr[0]);
			assign head_rxb_data[hh] = (cfg_mode ? c0_txv_bump : c0_txk_bump);
			assign head_rxb_bv[hh] = (cfg_mode ? c0_txv_bv : c0_txk_bv);
			assign h_ctx_bump[hh] = (cfg_mode ? head_tx_data[hh] : 512'h0);
			assign h_ctx_bv[hh] = (cfg_mode ? head_tx_bv[hh] : 1'b0);
			assign h_tx_bump[((NUM_HEADS - 1) - hh) * 512+:512] = (cfg_mode ? 512'h0 : head_tx_data[hh]);
			assign h_tx_bv[hh] = (cfg_mode ? 1'b0 : head_tx_bv[hh]);
			assign head_tx_bcr[hh] = (cfg_mode ? h_ctx_bcr[hh] : h_tx_bcr[hh]);
		end
		for (_gv_hh_1 = 0; _gv_hh_1 < NUM_HEADS; _gv_hh_1 = _gv_hh_1 + 1) begin : head_gen
			localparam hh = _gv_hh_1;
			chiplet_head #(
				.HEAD_ID(hh),
				.D_HEAD(D_HEAD),
				.TILE(TILE_DIM),
				.K_DIM(TILE_DIM),
				.SEQ_TILE(TILE_DIM)
			) u_head(
				.clk_core(clk_core),
				.rst_n(rst_n),
				.cfg_mode(cfg_mode),
				.cfg_num_tiles(cfg_num_tiles),
				.cfg_start(cfg_start),
				.cfg_done(),
				.chiplet_id(),
				.rxa_bump_data(head_rxa_data[hh]),
				.rxa_bump_valid(head_rxa_bv[hh]),
				.rxa_bump_credit(head_rxa_bcr[hh]),
				.rxb_bump_data(head_rxb_data[hh]),
				.rxb_bump_valid(head_rxb_bv[hh]),
				.rxb_bump_credit(head_rxb_bcr[hh]),
				.tx_bump_data(head_tx_data[hh]),
				.tx_bump_valid(head_tx_bv[hh]),
				.tx_bump_credit(head_tx_bcr[hh]),
				.scale_factor(cfg_scale_bf16)
			);
		end
	endgenerate
	chiplet_9_taylor #(
		.NUM_HEADS(NUM_HEADS),
		.TILE(TILE_DIM),
		.SEQ_LEN(TILE_DIM)
	) u_taylor(
		.clk_core(clk_core),
		.rst_n(rst_n),
		.cfg_start(cfg_start),
		.cfg_done(),
		.rx_bump_data(h_tx_bump),
		.rx_bump_valid(h_tx_bv),
		.rx_bump_credit(h_tx_bcr),
		.tx_bump_data(t_tx_bump),
		.tx_bump_valid(t_tx_bv),
		.tx_bump_credit(t_tx_bcr)
	);
	wire [((TILE_DIM * TILE_DIM) * 16) - 1:0] host_rx_tile;
	wire host_rx_valid;
	ucie_rx #(.TILE_DIM(TILE_DIM)) u_host_rx(
		.clk_core(clk_core),
		.rst_n(rst_n),
		.bump_data(c0_out_bump),
		.bump_valid(c0_out_bv),
		.bump_credit(c0_out_bcr),
		.rx_valid(host_rx_valid),
		.rx_src_id(),
		.rx_tile(host_rx_tile),
		.rx_ready(out_tile_ready),
		.rx_crc_err(),
		.rx_seq_err()
	);
	assign out_tile_valid = host_rx_valid;
	genvar _gv_oi_1;
	genvar _gv_oj_1;
	generate
		for (_gv_oi_1 = 0; _gv_oi_1 < TILE_DIM; _gv_oi_1 = _gv_oi_1 + 1) begin : out_row_g
			localparam oi = _gv_oi_1;
			for (_gv_oj_1 = 0; _gv_oj_1 < TILE_DIM; _gv_oj_1 = _gv_oj_1 + 1) begin : out_col_g
				localparam oj = _gv_oj_1;
				assign out_tile_data[((((TILE_DIM - 1) - oi) * TILE_DIM) + ((TILE_DIM - 1) - oj)) * 16+:16] = host_rx_tile[((((TILE_DIM - 1) - oi) * TILE_DIM) + ((TILE_DIM - 1) - oj)) * 16+:16];
			end
		end
	endgenerate
	reg busy_r;
	reg cfg_start_s1;
	always @(posedge clk_core or negedge rst_n) begin : start_sync
		if (!rst_n) begin
			cfg_start_s1 <= 1'b0;
			cfg_start_core <= 1'b0;
		end
		else begin
			cfg_start_s1 <= cfg_start;
			cfg_start_core <= cfg_start_s1;
		end
	end
	always @(posedge clk_core or negedge rst_n) begin : busy_ff
		if (!rst_n)
			busy_r <= 1'b0;
		else if (cfg_start_core)
			busy_r <= 1'b1;
		else if (c0_done)
			busy_r <= 1'b0;
	end
	assign sts_busy = busy_r;
	reg sts_done_core;
	always @(posedge clk_core or negedge rst_n) begin : done_latch_core
		if (!rst_n)
			sts_done_core <= 1'b0;
		else if (c0_done)
			sts_done_core <= 1'b1;
		else
			sts_done_core <= 1'b0;
	end
	reg sts_done_s1;
	reg sts_done_r;
	always @(posedge clk_axi or negedge rst_n) begin : done_sync
		if (!rst_n) begin
			sts_done_s1 <= 1'b0;
			sts_done_r <= 1'b0;
		end
		else begin
			sts_done_s1 <= sts_done_core;
			sts_done_r <= sts_done_s1;
		end
	end
	assign sts_done = sts_done_r;
	assign sts_error = 1'b0;
	assign sts_active_head = 4'h0;
	assign mem_wdata = 512'h0;
	assign mem_wen = 1'b0;
	assign mem_addr = cfg_weight_addr;
	assign mem_req = busy_r;
endmodule
