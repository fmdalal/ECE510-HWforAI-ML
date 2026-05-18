`default_nettype none
module watchdog (
	clk_axi,
	rst_n,
	en,
	kick,
	limit,
	timeout
);
	parameter signed [31:0] W = 32;
	input wire clk_axi;
	input wire rst_n;
	input wire en;
	input wire kick;
	input wire [W - 1:0] limit;
	output reg timeout;
	reg [W - 1:0] cnt;
	always @(posedge clk_axi or negedge rst_n) begin : wdt_ff
		if (!rst_n) begin
			cnt <= 1'sb0;
			timeout <= 1'b0;
		end
		else if (!en) begin
			cnt <= 1'sb0;
			timeout <= 1'b0;
		end
		else if (kick) begin
			cnt <= 1'sb0;
			timeout <= 1'b0;
		end
		else if (cnt >= limit)
			timeout <= 1'b1;
		else
			cnt <= cnt + 1;
	end
endmodule
