`default_nettype none
module axi_if (
	clk_axi,
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
	in_tile_data,
	in_tile_valid,
	in_tile_dst,
	in_tile_type,
	in_tile_ready,
	out_tile_data,
	out_tile_valid,
	out_tile_ready,
	sts_busy,
	sts_done,
	sts_error,
	sts_active_head,
	irq,
	perf_cycles
);
	parameter signed [31:0] TILE_DIM = 64;
	parameter signed [31:0] TDATA_W = 512;
	parameter signed [31:0] FIFO_D = 256;
	input wire clk_axi;
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
	output wire cfg_start;
	output wire cfg_reset;
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
	output wire [((TILE_DIM * TILE_DIM) * 16) - 1:0] in_tile_data;
	output wire in_tile_valid;
	output wire [3:0] in_tile_dst;
	output wire [3:0] in_tile_type;
	input wire in_tile_ready;
	input wire [((TILE_DIM * TILE_DIM) * 16) - 1:0] out_tile_data;
	input wire out_tile_valid;
	output wire out_tile_ready;
	input wire sts_busy;
	input wire sts_done;
	input wire sts_error;
	input wire [3:0] sts_active_head;
	output wire irq;
	output wire [63:0] perf_cycles;
	wire [11:0] axil_awaddr;
	wire [2:0] axil_awprot;
	wire axil_awvalid;
	wire axil_awready;
	wire [31:0] axil_wdata;
	wire [3:0] axil_wstrb;
	wire axil_wvalid;
	wire axil_wready;
	wire [1:0] axil_bresp;
	wire axil_bvalid;
	wire axil_bready;
	wire [11:0] axil_araddr;
	wire [2:0] axil_arprot;
	wire axil_arvalid;
	wire axil_arready;
	wire [31:0] axil_rdata;
	wire [1:0] axil_rresp;
	wire axil_rvalid;
	wire axil_rready;
	wire [TDATA_W - 1:0] axis_tdata;
	wire [(TDATA_W / 8) - 1:0] axis_tkeep;
	wire axis_tvalid;
	wire axis_tready;
	wire axis_tlast;
	wire [3:0] axis_tuser;
	wire [3:0] axis_tid;
	wb2axip #(
		.AW(32),
		.DW(32),
		.AXIDW(TDATA_W)
	) u_bridge(
		.i_clk(clk_axi),
		.i_reset(~rst_n),
		.i_wb_cyc(wb_cyc),
		.i_wb_stb(wb_stb),
		.i_wb_we(wb_we),
		.i_wb_addr(wb_addr),
		.i_wb_data(wb_wdata),
		.i_wb_sel(wb_sel),
		.o_wb_stall(wb_stall),
		.o_wb_ack(wb_ack),
		.o_wb_data(wb_rdata),
		.o_wb_err(wb_err),
		.o_axil_awaddr(axil_awaddr),
		.o_axil_awprot(axil_awprot),
		.o_axil_awvalid(axil_awvalid),
		.i_axil_awready(axil_awready),
		.o_axil_wdata(axil_wdata),
		.o_axil_wstrb(axil_wstrb),
		.o_axil_wvalid(axil_wvalid),
		.i_axil_wready(axil_wready),
		.i_axil_bresp(axil_bresp),
		.i_axil_bvalid(axil_bvalid),
		.o_axil_bready(axil_bready),
		.o_axil_araddr(axil_araddr),
		.o_axil_arprot(axil_arprot),
		.o_axil_arvalid(axil_arvalid),
		.i_axil_arready(axil_arready),
		.i_axil_rdata(axil_rdata),
		.i_axil_rresp(axil_rresp),
		.i_axil_rvalid(axil_rvalid),
		.o_axil_rready(axil_rready),
		.o_axis_tdata(axis_tdata),
		.o_axis_tkeep(axis_tkeep),
		.o_axis_tvalid(axis_tvalid),
		.i_axis_tready(axis_tready),
		.o_axis_tlast(axis_tlast),
		.o_axis_tuser(axis_tuser),
		.o_axis_tid(axis_tid)
	);
	wire csr_cfg_start;
	wire csr_cfg_reset;
	wire csr_cfg_mode;
	wire [15:0] csr_seq_len;
	wire [15:0] csr_d_model;
	wire [7:0] csr_num_heads;
	wire [7:0] csr_num_tiles;
	wire [63:0] csr_weight_addr;
	wire [31:0] csr_in_addr;
	wire [31:0] csr_out_addr;
	wire [15:0] csr_scale_bf16;
	wire [31:0] csr_wdt_timeout;
	axi_lite_csr #(.TILE_DIM(TILE_DIM)) u_csr(
		.clk_axi(clk_axi),
		.rst_n(rst_n),
		.s_awaddr(axil_awaddr),
		.s_awprot(axil_awprot),
		.s_awvalid(axil_awvalid),
		.s_awready(axil_awready),
		.s_wdata(axil_wdata),
		.s_wstrb(axil_wstrb),
		.s_wvalid(axil_wvalid),
		.s_wready(axil_wready),
		.s_bresp(axil_bresp),
		.s_bvalid(axil_bvalid),
		.s_bready(axil_bready),
		.s_araddr(axil_araddr),
		.s_arprot(axil_arprot),
		.s_arvalid(axil_arvalid),
		.s_arready(axil_arready),
		.s_rdata(axil_rdata),
		.s_rresp(axil_rresp),
		.s_rvalid(axil_rvalid),
		.s_rready(axil_rready),
		.cfg_start(csr_cfg_start),
		.cfg_reset(csr_cfg_reset),
		.cfg_mode(csr_cfg_mode),
		.cfg_seq_len(csr_seq_len),
		.cfg_d_model(csr_d_model),
		.cfg_num_heads(csr_num_heads),
		.cfg_num_tiles(csr_num_tiles),
		.cfg_weight_addr(csr_weight_addr),
		.cfg_in_addr(csr_in_addr),
		.cfg_out_addr(csr_out_addr),
		.cfg_scale_bf16(csr_scale_bf16),
		.cfg_wdt_timeout(csr_wdt_timeout),
		.sts_busy(sts_busy),
		.sts_done(sts_done),
		.sts_error(sts_error),
		.sts_active_head(sts_active_head),
		.sts_perf_cycles(perf_cycles),
		.irq(irq)
	);
	assign cfg_start = csr_cfg_start;
	assign cfg_reset = csr_cfg_reset;
	assign cfg_mode = csr_cfg_mode;
	assign cfg_seq_len = csr_seq_len;
	assign cfg_d_model = csr_d_model;
	assign cfg_num_heads = csr_num_heads;
	assign cfg_num_tiles = csr_num_tiles;
	assign cfg_weight_addr = csr_weight_addr;
	assign cfg_in_addr = csr_in_addr;
	assign cfg_out_addr = csr_out_addr;
	assign cfg_scale_bf16 = csr_scale_bf16;
	assign cfg_wdt_timeout = csr_wdt_timeout;
	axis_input_fifo #(
		.TDATA_W(TDATA_W),
		.TILE_DIM(TILE_DIM),
		.FIFO_D(FIFO_D)
	) u_in_fifo(
		.clk_axi(clk_axi),
		.rst_n(rst_n),
		.s_tdata(axis_tdata),
		.s_tkeep(axis_tkeep),
		.s_tvalid(axis_tvalid),
		.s_tready(axis_tready),
		.s_tlast(axis_tlast),
		.s_tuser(axis_tuser),
		.s_tid(axis_tid),
		.tile_out(in_tile_data),
		.tile_valid(in_tile_valid),
		.tile_dst(in_tile_dst),
		.tile_type(in_tile_type),
		.tile_ready(in_tile_ready)
	);
	wire [TDATA_W - 1:0] m_tdata;
	wire [(TDATA_W / 8) - 1:0] m_tkeep;
	wire m_tvalid;
	wire m_tlast;
	wire [3:0] m_tuser;
	axis_output_fifo #(
		.TDATA_W(TDATA_W),
		.TILE_DIM(TILE_DIM),
		.FIFO_D(FIFO_D)
	) u_out_fifo(
		.clk_axi(clk_axi),
		.rst_n(rst_n),
		.tile_in(out_tile_data),
		.tile_valid(out_tile_valid),
		.tile_ready(out_tile_ready),
		.m_tdata(m_tdata),
		.m_tkeep(m_tkeep),
		.m_tvalid(m_tvalid),
		.m_tready(1'b1),
		.m_tlast(m_tlast),
		.m_tuser(m_tuser)
	);
	watchdog #(.W(32)) u_wdt(
		.clk_axi(clk_axi),
		.rst_n(rst_n),
		.en(sts_busy),
		.kick(out_tile_valid),
		.limit(csr_wdt_timeout),
		.timeout()
	);
	perf_counter u_perf(
		.clk_axi(clk_axi),
		.rst_n(rst_n),
		.start(csr_cfg_start),
		.en(sts_busy),
		.cycles(perf_cycles)
	);
endmodule
