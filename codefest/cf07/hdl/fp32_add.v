`default_nettype none
module fp32_add (
	a,
	b,
	result
);
	reg _sv2v_0;
	input wire [31:0] a;
	input wire [31:0] b;
	output reg [31:0] result;
	wire sign_a = a[31];
	wire [7:0] exp_a = a[30:23];
	wire [23:0] mant_a = {1'b1, a[22:0]};
	wire sign_b = b[31];
	wire [7:0] exp_b = b[30:23];
	wire [23:0] mant_b = {1'b1, b[22:0]};
	wire a_ge = exp_a >= exp_b;
	wire [7:0] exp_big = (a_ge ? exp_a : exp_b);
	wire sign_big = (a_ge ? sign_a : sign_b);
	wire sign_sml = (a_ge ? sign_b : sign_a);
	wire [23:0] mant_big = (a_ge ? mant_a : mant_b);
	wire [23:0] mant_sml = (a_ge ? mant_b : mant_a);
	wire [7:0] exp_diff = exp_big - (a_ge ? exp_b : exp_a);
	wire [4:0] sh = (exp_diff > 8'd27 ? 5'd27 : exp_diff[4:0]);
	wire [26:0] mbig_e = {mant_big, 3'b000};
	wire [26:0] msml_e = {mant_sml, 3'b000} >> sh;
	wire same_sign = sign_big == sign_sml;
	wire [27:0] msum = (same_sign ? {1'b0, mbig_e} + {1'b0, msml_e} : {1'b0, mbig_e} - {1'b0, msml_e});
	wire sign_r = sign_big;
	reg [4:0] lzc;
	always @(*) begin
		if (_sv2v_0)
			;
		casez (msum[27:0])
			28'b1zzzzzzzzzzzzzzzzzzzzzzzzzzz: lzc = 5'd0;
			28'b01zzzzzzzzzzzzzzzzzzzzzzzzzz: lzc = 5'd1;
			28'b001zzzzzzzzzzzzzzzzzzzzzzzzz: lzc = 5'd2;
			28'b0001zzzzzzzzzzzzzzzzzzzzzzzz: lzc = 5'd3;
			28'b00001zzzzzzzzzzzzzzzzzzzzzzz: lzc = 5'd4;
			28'b000001zzzzzzzzzzzzzzzzzzzzzz: lzc = 5'd5;
			28'b0000001zzzzzzzzzzzzzzzzzzzzz: lzc = 5'd6;
			28'b00000001zzzzzzzzzzzzzzzzzzzz: lzc = 5'd7;
			28'b000000001zzzzzzzzzzzzzzzzzzz: lzc = 5'd8;
			28'b0000000001zzzzzzzzzzzzzzzzzz: lzc = 5'd9;
			28'b00000000001zzzzzzzzzzzzzzzzz: lzc = 5'd10;
			28'b000000000001zzzzzzzzzzzzzzzz: lzc = 5'd11;
			28'b0000000000001zzzzzzzzzzzzzzz: lzc = 5'd12;
			28'b00000000000001zzzzzzzzzzzzzz: lzc = 5'd13;
			28'b000000000000001zzzzzzzzzzzzz: lzc = 5'd14;
			28'b0000000000000001zzzzzzzzzzzz: lzc = 5'd15;
			28'b00000000000000001zzzzzzzzzzz: lzc = 5'd16;
			28'b000000000000000001zzzzzzzzzz: lzc = 5'd17;
			28'b0000000000000000001zzzzzzzzz: lzc = 5'd18;
			28'b00000000000000000001zzzzzzzz: lzc = 5'd19;
			28'b000000000000000000001zzzzzzz: lzc = 5'd20;
			28'b0000000000000000000001zzzzzz: lzc = 5'd21;
			28'b00000000000000000000001zzzzz: lzc = 5'd22;
			28'b000000000000000000000001zzzz: lzc = 5'd23;
			28'b0000000000000000000000001zzz: lzc = 5'd24;
			28'b00000000000000000000000001zz: lzc = 5'd25;
			28'b000000000000000000000000001z: lzc = 5'd26;
			default: lzc = 5'd27;
		endcase
	end
	wire [27:0] mnorm = msum << lzc;
	wire [8:0] exp_norm = ({1'b0, exp_big} - {4'b0000, lzc}) + 9'd3;
	wire rnd_bit = mnorm[2];
	wire sticky = |mnorm[1:0];
	wire lsb_bit = mnorm[3];
	wire do_rnd = rnd_bit & (sticky | lsb_bit);
	wire [24:0] mant_rnd = {1'b0, mnorm[26:3]} + {24'b000000000000000000000000, do_rnd};
	wire [22:0] mant_f = (mant_rnd[24] ? mant_rnd[23:1] : mant_rnd[22:0]);
	wire [8:0] exp_f = (mant_rnd[24] ? exp_norm + 9'd1 : exp_norm);
	wire a_zero = exp_a == 8'h00;
	wire b_zero = exp_b == 8'h00;
	wire a_inf = exp_a == 8'hff;
	wire b_inf = exp_b == 8'hff;
	wire res_zer = msum == 28'h0000000;
	always @(*) begin
		if (_sv2v_0)
			;
		if (a_inf | b_inf)
			result = {sign_r, 31'h7f800000};
		else if (a_zero & b_zero)
			result = 32'h00000000;
		else if (res_zer)
			result = 32'h00000000;
		else if (exp_f[8])
			result = {sign_r, 31'h7f800000};
		else
			result = {sign_r, exp_f[7:0], mant_f};
	end
	initial _sv2v_0 = 0;
endmodule
