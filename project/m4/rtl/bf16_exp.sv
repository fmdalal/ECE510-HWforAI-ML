`timescale 1ns/1ps

`default_nettype none

module bf16_exp (

    input  wire  [15:0] a,

    output logic [15:0] result

);

    wire        sign_a = a[15];

    wire [7:0]  exp_a  = a[14:7];

    wire [6:0]  mant_a = a[6:0];

    wire a_is_nan     = (exp_a == 8'hFF) && (mant_a != 7'h00);

    wire a_is_pos_inf = (exp_a == 8'hFF) && (mant_a == 7'h00) && (sign_a == 1'b0);

    wire a_is_neg_inf = (exp_a == 8'hFF) && (mant_a == 7'h00) && (sign_a == 1'b1);

    wire a_is_zero    = (exp_a == 8'h00);

    wire a_large_pos  = (sign_a == 1'b0) && (exp_a >= 8'h85);

    wire a_large_neg  = (sign_a == 1'b1) && (exp_a >= 8'h85);

    localparam [14:0] LOG2E_Q14 = 15'd23637;

    wire [7:0]  mantissa8 = {1'b1, mant_a};

    wire [22:0] product   = {15'b0, mantissa8} * {8'b0, LOG2E_Q14};

    wire [7:0]  shift_amt = 8'd141 - exp_a;

    wire [22:0] p_shifted = product >> shift_amt;

    wire [14:0] p_fixed7  = p_shifted[14:0];

    wire [7:0]  k         = p_fixed7[14:7];

    wire [6:0]  f_bits    = p_fixed7[6:0];

    wire        f_nz      = (f_bits != 7'h00);

    logic [6:0] lut_neg;

    always_comb begin

        case (f_bits)

            7'h00: lut_neg = 7'h00;

            7'h01: lut_neg = 7'h7f;

            7'h02: lut_neg = 7'h7d;

            7'h03: lut_neg = 7'h7c;

            7'h04: lut_neg = 7'h7b;

            7'h05: lut_neg = 7'h79;

            7'h06: lut_neg = 7'h78;

            7'h07: lut_neg = 7'h76;

            7'h08: lut_neg = 7'h75;

            7'h09: lut_neg = 7'h74;

            7'h0a: lut_neg = 7'h73;

            7'h0b: lut_neg = 7'h71;

            7'h0c: lut_neg = 7'h70;

            7'h0d: lut_neg = 7'h6f;

            7'h0e: lut_neg = 7'h6d;

            7'h0f: lut_neg = 7'h6c;

            7'h10: lut_neg = 7'h6b;

            7'h11: lut_neg = 7'h69;

            7'h12: lut_neg = 7'h68;

            7'h13: lut_neg = 7'h67;

            7'h14: lut_neg = 7'h66;

            7'h15: lut_neg = 7'h64;

            7'h16: lut_neg = 7'h63;

            7'h17: lut_neg = 7'h62;

            7'h18: lut_neg = 7'h61;

            7'h19: lut_neg = 7'h60;

            7'h1a: lut_neg = 7'h5e;

            7'h1b: lut_neg = 7'h5d;

            7'h1c: lut_neg = 7'h5c;

            7'h1d: lut_neg = 7'h5b;

            7'h1e: lut_neg = 7'h5a;

            7'h1f: lut_neg = 7'h58;

            7'h20: lut_neg = 7'h57;

            7'h21: lut_neg = 7'h56;

            7'h22: lut_neg = 7'h55;

            7'h23: lut_neg = 7'h54;

            7'h24: lut_neg = 7'h53;

            7'h25: lut_neg = 7'h52;

            7'h26: lut_neg = 7'h50;

            7'h27: lut_neg = 7'h4f;

            7'h28: lut_neg = 7'h4e;

            7'h29: lut_neg = 7'h4d;

            7'h2a: lut_neg = 7'h4c;

            7'h2b: lut_neg = 7'h4b;

            7'h2c: lut_neg = 7'h4a;

            7'h2d: lut_neg = 7'h49;

            7'h2e: lut_neg = 7'h48;

            7'h2f: lut_neg = 7'h46;

            7'h30: lut_neg = 7'h45;

            7'h31: lut_neg = 7'h44;

            7'h32: lut_neg = 7'h43;

            7'h33: lut_neg = 7'h42;

            7'h34: lut_neg = 7'h41;

            7'h35: lut_neg = 7'h40;

            7'h36: lut_neg = 7'h3f;

            7'h37: lut_neg = 7'h3e;

            7'h38: lut_neg = 7'h3d;

            7'h39: lut_neg = 7'h3c;

            7'h3a: lut_neg = 7'h3b;

            7'h3b: lut_neg = 7'h3a;

            7'h3c: lut_neg = 7'h39;

            7'h3d: lut_neg = 7'h38;

            7'h3e: lut_neg = 7'h37;

            7'h3f: lut_neg = 7'h36;

            7'h40: lut_neg = 7'h35;

            7'h41: lut_neg = 7'h34;

            7'h42: lut_neg = 7'h33;

            7'h43: lut_neg = 7'h32;

            7'h44: lut_neg = 7'h31;

            7'h45: lut_neg = 7'h30;

            7'h46: lut_neg = 7'h2f;

            7'h47: lut_neg = 7'h2e;

            7'h48: lut_neg = 7'h2d;

            7'h49: lut_neg = 7'h2c;

            7'h4a: lut_neg = 7'h2b;

            7'h4b: lut_neg = 7'h2b;

            7'h4c: lut_neg = 7'h2a;

            7'h4d: lut_neg = 7'h29;

            7'h4e: lut_neg = 7'h28;

            7'h4f: lut_neg = 7'h27;

            7'h50: lut_neg = 7'h26;

            7'h51: lut_neg = 7'h25;

            7'h52: lut_neg = 7'h24;

            7'h53: lut_neg = 7'h23;

            7'h54: lut_neg = 7'h22;

            7'h55: lut_neg = 7'h22;

            7'h56: lut_neg = 7'h21;

            7'h57: lut_neg = 7'h20;

            7'h58: lut_neg = 7'h1f;

            7'h59: lut_neg = 7'h1e;

            7'h5a: lut_neg = 7'h1d;

            7'h5b: lut_neg = 7'h1c;

            7'h5c: lut_neg = 7'h1c;

            7'h5d: lut_neg = 7'h1b;

            7'h5e: lut_neg = 7'h1a;

            7'h5f: lut_neg = 7'h19;

            7'h60: lut_neg = 7'h18;

            7'h61: lut_neg = 7'h17;

            7'h62: lut_neg = 7'h17;

            7'h63: lut_neg = 7'h16;

            7'h64: lut_neg = 7'h15;

            7'h65: lut_neg = 7'h14;

            7'h66: lut_neg = 7'h13;

            7'h67: lut_neg = 7'h13;

            7'h68: lut_neg = 7'h12;

            7'h69: lut_neg = 7'h11;

            7'h6a: lut_neg = 7'h10;

            7'h6b: lut_neg = 7'h0f;

            7'h6c: lut_neg = 7'h0f;

            7'h6d: lut_neg = 7'h0e;

            7'h6e: lut_neg = 7'h0d;

            7'h6f: lut_neg = 7'h0c;

            7'h70: lut_neg = 7'h0c;

            7'h71: lut_neg = 7'h0b;

            7'h72: lut_neg = 7'h0a;

            7'h73: lut_neg = 7'h09;

            7'h74: lut_neg = 7'h09;

            7'h75: lut_neg = 7'h08;

            7'h76: lut_neg = 7'h07;

            7'h77: lut_neg = 7'h06;

            7'h78: lut_neg = 7'h06;

            7'h79: lut_neg = 7'h05;

            7'h7a: lut_neg = 7'h04;

            7'h7b: lut_neg = 7'h04;

            7'h7c: lut_neg = 7'h03;

            7'h7d: lut_neg = 7'h02;

            7'h7e: lut_neg = 7'h01;

            7'h7f: lut_neg = 7'h01;

            default: lut_neg = 7'h00;

        endcase

    end

    logic [6:0] lut_pos;

    always_comb begin

        case (f_bits)

            7'h00: lut_pos = 7'h00;

            7'h01: lut_pos = 7'h01;

            7'h02: lut_pos = 7'h01;

            7'h03: lut_pos = 7'h02;

            7'h04: lut_pos = 7'h03;

            7'h05: lut_pos = 7'h04;

            7'h06: lut_pos = 7'h04;

            7'h07: lut_pos = 7'h05;

            7'h08: lut_pos = 7'h06;

            7'h09: lut_pos = 7'h06;

            7'h0a: lut_pos = 7'h07;

            7'h0b: lut_pos = 7'h08;

            7'h0c: lut_pos = 7'h09;

            7'h0d: lut_pos = 7'h09;

            7'h0e: lut_pos = 7'h0a;

            7'h0f: lut_pos = 7'h0b;

            7'h10: lut_pos = 7'h0c;

            7'h11: lut_pos = 7'h0c;

            7'h12: lut_pos = 7'h0d;

            7'h13: lut_pos = 7'h0e;

            7'h14: lut_pos = 7'h0f;

            7'h15: lut_pos = 7'h0f;

            7'h16: lut_pos = 7'h10;

            7'h17: lut_pos = 7'h11;

            7'h18: lut_pos = 7'h12;

            7'h19: lut_pos = 7'h13;

            7'h1a: lut_pos = 7'h13;

            7'h1b: lut_pos = 7'h14;

            7'h1c: lut_pos = 7'h15;

            7'h1d: lut_pos = 7'h16;

            7'h1e: lut_pos = 7'h17;

            7'h1f: lut_pos = 7'h17;

            7'h20: lut_pos = 7'h18;

            7'h21: lut_pos = 7'h19;

            7'h22: lut_pos = 7'h1a;

            7'h23: lut_pos = 7'h1b;

            7'h24: lut_pos = 7'h1c;

            7'h25: lut_pos = 7'h1c;

            7'h26: lut_pos = 7'h1d;

            7'h27: lut_pos = 7'h1e;

            7'h28: lut_pos = 7'h1f;

            7'h29: lut_pos = 7'h20;

            7'h2a: lut_pos = 7'h21;

            7'h2b: lut_pos = 7'h22;

            7'h2c: lut_pos = 7'h22;

            7'h2d: lut_pos = 7'h23;

            7'h2e: lut_pos = 7'h24;

            7'h2f: lut_pos = 7'h25;

            7'h30: lut_pos = 7'h26;

            7'h31: lut_pos = 7'h27;

            7'h32: lut_pos = 7'h28;

            7'h33: lut_pos = 7'h29;

            7'h34: lut_pos = 7'h2a;

            7'h35: lut_pos = 7'h2b;

            7'h36: lut_pos = 7'h2b;

            7'h37: lut_pos = 7'h2c;

            7'h38: lut_pos = 7'h2d;

            7'h39: lut_pos = 7'h2e;

            7'h3a: lut_pos = 7'h2f;

            7'h3b: lut_pos = 7'h30;

            7'h3c: lut_pos = 7'h31;

            7'h3d: lut_pos = 7'h32;

            7'h3e: lut_pos = 7'h33;

            7'h3f: lut_pos = 7'h34;

            7'h40: lut_pos = 7'h35;

            7'h41: lut_pos = 7'h36;

            7'h42: lut_pos = 7'h37;

            7'h43: lut_pos = 7'h38;

            7'h44: lut_pos = 7'h39;

            7'h45: lut_pos = 7'h3a;

            7'h46: lut_pos = 7'h3b;

            7'h47: lut_pos = 7'h3c;

            7'h48: lut_pos = 7'h3d;

            7'h49: lut_pos = 7'h3e;

            7'h4a: lut_pos = 7'h3f;

            7'h4b: lut_pos = 7'h40;

            7'h4c: lut_pos = 7'h41;

            7'h4d: lut_pos = 7'h42;

            7'h4e: lut_pos = 7'h43;

            7'h4f: lut_pos = 7'h44;

            7'h50: lut_pos = 7'h45;

            7'h51: lut_pos = 7'h46;

            7'h52: lut_pos = 7'h48;

            7'h53: lut_pos = 7'h49;

            7'h54: lut_pos = 7'h4a;

            7'h55: lut_pos = 7'h4b;

            7'h56: lut_pos = 7'h4c;

            7'h57: lut_pos = 7'h4d;

            7'h58: lut_pos = 7'h4e;

            7'h59: lut_pos = 7'h4f;

            7'h5a: lut_pos = 7'h50;

            7'h5b: lut_pos = 7'h52;

            7'h5c: lut_pos = 7'h53;

            7'h5d: lut_pos = 7'h54;

            7'h5e: lut_pos = 7'h55;

            7'h5f: lut_pos = 7'h56;

            7'h60: lut_pos = 7'h57;

            7'h61: lut_pos = 7'h58;

            7'h62: lut_pos = 7'h5a;

            7'h63: lut_pos = 7'h5b;

            7'h64: lut_pos = 7'h5c;

            7'h65: lut_pos = 7'h5d;

            7'h66: lut_pos = 7'h5e;

            7'h67: lut_pos = 7'h60;

            7'h68: lut_pos = 7'h61;

            7'h69: lut_pos = 7'h62;

            7'h6a: lut_pos = 7'h63;

            7'h6b: lut_pos = 7'h64;

            7'h6c: lut_pos = 7'h66;

            7'h6d: lut_pos = 7'h67;

            7'h6e: lut_pos = 7'h68;

            7'h6f: lut_pos = 7'h69;

            7'h70: lut_pos = 7'h6b;

            7'h71: lut_pos = 7'h6c;

            7'h72: lut_pos = 7'h6d;

            7'h73: lut_pos = 7'h6f;

            7'h74: lut_pos = 7'h70;

            7'h75: lut_pos = 7'h71;

            7'h76: lut_pos = 7'h73;

            7'h77: lut_pos = 7'h74;

            7'h78: lut_pos = 7'h75;

            7'h79: lut_pos = 7'h76;

            7'h7a: lut_pos = 7'h78;

            7'h7b: lut_pos = 7'h79;

            7'h7c: lut_pos = 7'h7b;

            7'h7d: lut_pos = 7'h7c;

            7'h7e: lut_pos = 7'h7d;

            7'h7f: lut_pos = 7'h7f;

            default: lut_pos = 7'h00;

        endcase

    end

    wire [6:0] out_mant = (sign_a == 1'b0) ? (f_nz ? lut_pos : 7'h00)

                                            : (f_nz ? lut_neg  : 7'h00);

    wire [8:0] out_exp = (sign_a == 1'b0) ? (9'd127 + {1'b0, k})

                       : (f_nz == 1'b0)   ? (9'd127 - {1'b0, k})

                       :                    (9'd127 - {1'b0, k} - 9'd1);

    wire out_overflow  = (out_exp >= 9'd255);

    wire out_underflow = out_exp[8] || (out_exp == 9'd0);

    assign result = a_is_nan                          ? 16'h7FC0 :
                    (a_is_pos_inf || a_large_pos)       ? 16'h7F80 :
                    (a_is_neg_inf || a_large_neg)       ? 16'h0000 :
                    a_is_zero                           ? 16'h3F80 :
                    out_overflow                        ? 16'h7F80 :
                    out_underflow                       ? 16'h0000 :
                                                          {1'b0, out_exp[7:0], out_mant};

endmodule

`default_nettype wire
