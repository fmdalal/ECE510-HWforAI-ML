`default_nettype none
module bf16_add (
	a,
	b,
	result
);
	reg _sv2v_0;
	input wire [15:0] a;
	input wire [15:0] b;
	output reg [15:0] result;
	wire [31:0] fp32_a = {a, 16'h0000};
	wire [31:0] fp32_b = {b, 16'h0000};
	wire [31:0] fp32_sum;
	fp32_add u_add(
		.a(fp32_a),
		.b(fp32_b),
		.result(fp32_sum)
	);
	wire rup = fp32_sum[15] & (|fp32_sum[14:0] | fp32_sum[16]);
	always @(*) begin
		if (_sv2v_0)
			;
		result = fp32_sum[31:16] + {15'h0000, rup};
	end
	initial _sv2v_0 = 0;
endmodule
