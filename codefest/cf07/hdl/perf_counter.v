`default_nettype none
module perf_counter (
	clk_axi,
	rst_n,
	start,
	en,
	cycles
);
	input wire clk_axi;
	input wire rst_n;
	input wire start;
	input wire en;
	output reg [63:0] cycles;
	always @(posedge clk_axi or negedge rst_n) begin : perf_ff
		if (!rst_n)
			cycles <= 64'h0000000000000000;
		else if (start)
			cycles <= 64'h0000000000000000;
		else if (en)
			cycles <= cycles + 64'd1;
	end
endmodule
