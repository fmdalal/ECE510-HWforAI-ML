`default_nettype none
module ucie_crc8 (
	data_in,
	crc_out
);
	reg _sv2v_0;
	parameter signed [31:0] W = 496;
	input wire [W - 1:0] data_in;
	output reg [7:0] crc_out;
	always @(*) begin : sv2v_autoblock_1
		reg [7:0] c;
		if (_sv2v_0)
			;
		c = 8'hff;
		begin : sv2v_autoblock_2
			reg signed [31:0] i;
			for (i = W - 1; i >= 0; i = i - 1)
				if (c[7] ^ data_in[i])
					c = {c[6:0], 1'b0} ^ 8'h31;
				else
					c = {c[6:0], 1'b0};
		end
		crc_out = c ^ 8'hff;
	end
	initial _sv2v_0 = 0;
endmodule
