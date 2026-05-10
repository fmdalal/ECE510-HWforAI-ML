// ============================================================
// 4x4 Binary-Weight Crossbar MAC Unit
// Inputs:  in_vec  — 4x 8-bit signed, packed as [31:0]
// Weights: weight  — 4x4 bits packed as [15:0] (weight[i][j] = bit [i*4+j])
// Outputs: out_vec — 4x 10-bit signed, packed as [39:0]
// out[j] = Σ_i weight[i][j] * in[i]
// ============================================================
module crossbar_mac (
    input  logic        clk,
    input  logic        rst_n,
    input  logic [31:0] in_vec,     // {in[3],in[2],in[1],in[0]} each 8-bit signed
    input  logic [15:0] weight,     // weight[i*4+j]
    output logic [39:0] out_vec     // {out[3],out[2],out[1],out[0]} each 10-bit signed
);
    logic signed [9:0] mac [0:3];
    logic signed [7:0] inv [0:3];
    assign inv[0] = in_vec[7:0];
    assign inv[1] = in_vec[15:8];
    assign inv[2] = in_vec[23:16];
    assign inv[3] = in_vec[31:24];

    integer ii, jj;
    always_comb begin
        for (jj=0; jj<4; jj++) begin
            mac[jj] = 10'sd0;
            for (ii=0; ii<4; ii++) begin
                if (weight[ii*4+jj])
                    mac[jj] = mac[jj] + 10'(signed'(inv[ii]));
                else
                    mac[jj] = mac[jj] - 10'(signed'(inv[ii]));
            end
        end
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) out_vec <= 40'b0;
        else begin
            out_vec[9:0]   <= mac[0];
            out_vec[19:10] <= mac[1];
            out_vec[29:20] <= mac[2];
            out_vec[39:30] <= mac[3];
        end
    end
endmodule
