`default_nettype none
module fp32_mul (
	a,
	b,
	result
);
	reg _sv2v_0;
	input wire [15:0] a;
	input wire [15:0] b;
	output reg [31:0] result;
	wire [31:0] fp32_a = {a, 16'h0000};
	wire [31:0] fp32_b = {b, 16'h0000};
	wire sign_a = fp32_a[31];
	wire [7:0] exp_a = fp32_a[30:23];
	wire [23:0] mant_a = {1'b1, fp32_a[22:0]};
	wire sign_b = fp32_b[31];
	wire [7:0] exp_b = fp32_b[30:23];
	wire [23:0] mant_b = {1'b1, fp32_b[22:0]};
	wire sign_r = sign_a ^ sign_b;
	wire [8:0] exp_sum = ({1'b0, exp_a} + {1'b0, exp_b}) - 9'd127;
	wire [47:0] mant_prod = mant_a * mant_b;
	wire norm_sh = mant_prod[47];
	wire [8:0] exp_n = (norm_sh ? exp_sum + 9'd1 : exp_sum);
	wire [22:0] mant_n = (norm_sh ? mant_prod[46:24] : mant_prod[45:23]);
	wire a_zero = exp_a == 8'h00;
	wire b_zero = exp_b == 8'h00;
	wire a_inf = (exp_a == 8'hff) & (fp32_a[22:0] == 23'h000000);
	wire b_inf = (exp_b == 8'hff) & (fp32_b[22:0] == 23'h000000);
	wire a_nan = (exp_a == 8'hff) & (fp32_a[22:0] != 23'h000000);
	wire b_nan = (exp_b == 8'hff) & (fp32_b[22:0] != 23'h000000);
	wire res_nan = ((a_nan | b_nan) | (a_inf & b_zero)) | (b_inf & a_zero);
	wire res_inf = (a_inf | b_inf) & ~res_nan;
	wire res_zero = (a_zero | b_zero) & ~res_nan;
	wire res_ovf = ((exp_n >= 9'd255) & ~res_nan) & ~res_zero;
	wire res_udf = (exp_n[8] & ~res_nan) & ~res_zero;
	always @(*) begin
		if (_sv2v_0)
			;
		if (res_nan)
			result = 32'h7fc00000;
		else if (res_inf | res_ovf)
			result = {sign_r, 31'h7f800000};
		else if (res_zero | res_udf)
			result = {sign_r, 31'h00000000};
		else
			result = {sign_r, exp_n[7:0], mant_n};
	end
	initial _sv2v_0 = 0;
endmodule
