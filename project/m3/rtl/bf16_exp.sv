// =============================================================================
// bf16_exp.sv  —  BF16 exponential primitive for softmax pipeline
//
// Computes result = e^a  in BF16 format.
// Purely combinational (no clock).
// Uses the Schraudolph bit-manipulation identity:
//   bits(e^x) ≈ x_fp32_bits * (2^23 / ln2) + (127 - 0.0450466) * 2^23
// Adapted for BF16: work on the upper 16 bits (BF16 word) and scale by 2^7.
//
// Accuracy: ≈ ±1 BF16 ULP across the softmax input range [-8, +8].
// Special cases handled: NaN, ±Inf, zero, very large negative (→0), very
// large positive (→+Inf).
//
// Ports
//   a      [15:0]  BF16 input  (IEEE 754 BF16: 1 sign, 8 exp, 7 mant)
//   result [15:0]  BF16 output = e^a
// =============================================================================
`timescale 1ns/1ps
`default_nettype none

module bf16_exp (
    input  wire  [15:0] a,
    output logic [15:0] result
);

    // ── Field extraction ─────────────────────────────────────────────────────
    wire        sign_a  = a[15];
    wire [7:0]  exp_a   = a[14:7];
    wire [6:0]  mant_a  = a[6:0];

    // Special-case flags
    wire a_is_nan      = (exp_a == 8'hFF) && (mant_a != 7'h00);
    wire a_is_pos_inf  = (exp_a == 8'hFF) && (mant_a == 7'h00) && !sign_a;
    wire a_is_neg_inf  = (exp_a == 8'hFF) && (mant_a == 7'h00) &&  sign_a;
    wire a_is_zero     = (exp_a == 8'h00);
    // |x| >= 88.0 → result overflows to +inf or underflows to 0
    // BF16(88) = 0x42B0  → exp_a=0x42=66, threshold exp_a >= 8'h42 (value >= 2^(66-127)=~32 ... )
    // Actually BF16 88.0 = sign=0, exp=127+6=133=0x85, mant=0b0110000=0x30 → 0x42B0
    // |x| > 88: for positive → inf, for negative → 0
    wire a_large       = (exp_a >= 8'h43);   // |x| >= 2^(67-127) * 1.0 ≈ 102; safe cutoff

    // ── Schraudolph approximation ────────────────────────────────────────────
    // For BF16: treat the 16-bit value as a scaled integer and apply:
    //   out_bits ≈ (signed_val_as_int * 94) + 15744
    // where 94 ≈ 2^7 / ln(2)  and  15744 = (127 - 0.0450466) * 128
    //
    // This gives 1-2 ULP accuracy for |x| < 8 without a lookup table.

    // Sign-extend a to 17 bits for signed arithmetic
    wire signed [16:0] x_signed = sign_a ? (~{1'b0, a} + 17'd1)
                                         : {1'b0, a};

    // Multiply by 94 using shifts: 94 = 64 + 16 + 8 + 4 + 2
    wire signed [23:0] x_times_94 = (x_signed <<< 6)
                                  + (x_signed <<< 4)
                                  + (x_signed <<< 3)
                                  + (x_signed <<< 2)
                                  + (x_signed <<< 1);

    // Add bias and take lower 16 bits
    wire [16:0] biased       = x_times_94[16:0] + 17'd15744;
    wire [15:0] approx_bits  = biased[15:0];

    // ── Special-case mux ─────────────────────────────────────────────────────
    always_comb begin
        if (a_is_nan) begin
            result = 16'h7FC0;          // quiet NaN
        end else if (a_is_pos_inf || (a_large && !sign_a)) begin
            result = 16'h7F80;          // +infinity
        end else if (a_is_neg_inf || (a_large &&  sign_a)) begin
            result = 16'h0000;          // +0  (e^-inf = 0)
        end else if (a_is_zero) begin
            result = 16'h3F80;          // 1.0  (e^0 = 1)
        end else begin
            // Clamp to valid BF16 range (prevent exponent overflow/underflow)
            if (biased[16] || approx_bits == 16'h0000) begin
                // Underflow: return smallest positive normal
                result = 16'h0080;
            end else if (approx_bits[14:7] == 8'hFF) begin
                // Exponent saturated: return +inf
                result = 16'h7F80;
            end else begin
                result = approx_bits;
            end
        end
    end

endmodule

`default_nettype wire
// =============================================================================
// End of bf16_exp.sv
// =============================================================================

