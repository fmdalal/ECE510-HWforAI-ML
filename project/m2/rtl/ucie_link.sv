// =============================================================================
// ucie_link.sv  —  UCIe FDI link primitives (shared by all 10 chiplets)
//
// Clock domains used in this file
// --------------------------------
//   clk_core (1 GHz chiplet compute clock)
//           : ucie_tx  — flit packetisation FSM, credit counter, tile latch
//           : ucie_rx  — flit reassembly FSM, CRC check, tile output register
//
//   NOTE: The physical bump pad drivers (UCIe PHY layer) operate on clk_link
//         (2 GHz). That layer is not modelled in this RTL file — it would wrap
//         ucie_tx/ucie_rx in a PHY module that re-times on clk_link.
//         clk_link is therefore not present as a port here; it lives in the
//         ucie_phy module instantiated in soc_top.sv.
// =============================================================================
// Every chiplet instantiates ucie_tx and/or ucie_rx at its boundary.
// Signals that cross a chiplet boundary go through these modules ONLY.
// No raw wires cross die boundaries anywhere in this design.
//
// Flit format (64 bytes = 512 bits):
//   [511:508] src_id   [3:0]
//   [507:504] dst_id   [3:0]
//   [503:496] seq_num  [7:0]
//   [495:488] flit_num [7:0]    0..FLITS_PER_TILE-1
//   [487:480] total    [7:0]    FLITS_PER_TILE
//   [479:16]  payload  [463:0]  29 BF16 words per flit
//   [15:8]    crc8     [7:0]
//   [7:0]     reserved [7:0]
//
// 16x16 BF16 tile = 256 words = 4096 bits
// ceil(256/29) = 9 flits per tile
//
// Chiplet ID map
//   0  QKV + OutProj  (Stage 1 / Stage 5 time-mux)
//   1  Head 0  (QKt Stage2 / ScoresxV Stage4 time-mux)
//   2  Head 1
//   3  Head 2
//   4  Head 3
//   5  Head 4
//   6  Head 5
//   7  Head 6
//   8  Head 7
//   9  Taylor (softmax replacement)
// =============================================================================

`timescale 1ns/1ps
`default_nettype none

// ---------------------------------------------------------------------------
// crc8  —  CRC-8/MAXIM polynomial 0x31, purely combinational
// ---------------------------------------------------------------------------
module ucie_crc8 #(
    parameter int W = 496
)(
    input  wire [W-1:0]  data_in,
    output logic [7:0]   crc_out
);
    always_comb begin
        logic [7:0] c;
        c = 8'hFF;
        for (int i = W-1; i >= 0; i--) begin
            if (c[7] ^ data_in[i])
                c = {c[6:0], 1'b0} ^ 8'h31;
            else
                c = {c[6:0], 1'b0};
        end
        crc_out = c ^ 8'hFF;
    end
endmodule

// ---------------------------------------------------------------------------
// ucie_tx  —  tile -> flits -> bump pads
//
// Caller drives:
//   tx_valid   — tile data ready
//   tx_src_id  — this chiplet's ID
//   tx_dst_id  — destination chiplet ID
//   tx_tile    — BF16 16x16 tile (256 words)
//
// Module produces:
//   bump_data  — 512-bit flit (one per cycle for 9 cycles)
//   bump_valid — flit is valid on bump pads
//   tx_ready   — module ready for next tile (goes low during transmission)
// ---------------------------------------------------------------------------
module ucie_tx #(
    parameter int TILE_DIM     = 16,
    parameter int WORDS        = 256,   // TILE_DIM*TILE_DIM
    parameter int WORDS_PER_FL = 29,    // BF16 words per flit payload
    parameter int FLITS        = 9      // ceil(256/29)
)(
    input  wire        clk_core,   // 1 GHz — FDI flit packetisation
    input  wire        rst_n,

    // from chiplet logic
    input  wire        tx_valid,
    input  wire [3:0]  tx_src_id,
    input  wire [3:0]  tx_dst_id,
    input  wire [15:0] tx_tile [TILE_DIM][TILE_DIM],
    output logic       tx_ready,

    // to bump pads / interposer
    output logic [511:0] bump_data,
    output logic         bump_valid,
    input  wire          bump_credit    // credit return from receiver
);
    // flatten tile to 256 BF16 words
    logic [15:0] flat [WORDS];
    always_comb begin
        for (int i = 0; i < TILE_DIM; i++)
            for (int j = 0; j < TILE_DIM; j++)
                flat[i*TILE_DIM + j] = tx_tile[i][j];
    end

    typedef enum logic [1:0] {
        TX_IDLE = 2'd0,
        TX_SEND = 2'd1,
        TX_WAIT = 2'd2
    } tx_state_t;

    tx_state_t tx_state;
    logic [3:0]  flit_cnt;
    logic [7:0]  seq_cnt;
    logic [3:0]  credits;
    logic [15:0] tile_buf [WORDS];

    // CRC over header + payload (496 bits = bits[511:16])
    logic [495:0] crc_in;
    logic [7:0]   crc_val;

    // build current flit payload
    logic [463:0] payload;
    always_comb begin
        payload = 464'h0;
        for (int w = 0; w < WORDS_PER_FL; w++) begin
            int idx;
            idx = flit_cnt * WORDS_PER_FL + w;
            if (idx < WORDS)
                payload[w*16 +: 16] = tile_buf[idx];
            else
                payload[w*16 +: 16] = 16'h0;
        end
    end

    always_comb begin
        crc_in = {tx_src_id, tx_dst_id, seq_cnt, flit_cnt[3:0], 4'h0,
                  8'd9, payload};
    end

    ucie_crc8 #(.W(496)) crc_inst (
        .data_in(crc_in),
        .crc_out(crc_val)
    );

    // credit counter
    always_ff @(posedge clk_core or negedge rst_n) begin : credit_ff
        if (!rst_n)
            credits <= 4'd8;
        else begin
            if (bump_credit & ~bump_valid)
                credits <= credits + 4'd1;
            else if (~bump_credit & bump_valid & (credits > 0))
                credits <= credits - 4'd1;
        end
    end

    // TX FSM
    always_ff @(posedge clk_core or negedge rst_n) begin : tx_fsm_ff
        if (!rst_n) begin
            tx_state   <= TX_IDLE;
            tx_ready   <= 1'b1;
            bump_valid <= 1'b0;
            flit_cnt   <= 4'd0;
            seq_cnt    <= 8'd0;
            bump_data  <= 512'h0;
            for (int w = 0; w < WORDS; w++) tile_buf[w] <= 16'h0;
        end else begin
            case (tx_state)
                TX_IDLE: begin
                    bump_valid <= 1'b0;
                    tx_ready   <= 1'b1;
                    if (tx_valid) begin
                        // latch tile
                        for (int w = 0; w < WORDS; w++)
                            tile_buf[w] <= flat[w];
                        flit_cnt <= 4'd0;
                        tx_ready <= 1'b0;
                        tx_state <= TX_SEND;
                    end
                end
                TX_SEND: begin
                    if (credits > 0) begin
                        bump_data  <= {tx_src_id, tx_dst_id,
                                       seq_cnt, flit_cnt[3:0], 4'h0,
                                       8'd9, payload,
                                       crc_val, 8'h00};
                        bump_valid <= 1'b1;
                        if (flit_cnt == FLITS - 1) begin
                            seq_cnt  <= seq_cnt + 8'd1;
                            flit_cnt <= 4'd0;
                            tx_state <= TX_IDLE;
                        end else begin
                            flit_cnt <= flit_cnt + 4'd1;
                            tx_state <= TX_WAIT;
                        end
                    end else begin
                        bump_valid <= 1'b0;
                    end
                end
                TX_WAIT: begin
                    bump_valid <= 1'b0;
                    tx_state   <= TX_SEND;
                end
                default: tx_state <= TX_IDLE;
            endcase
        end
    end
endmodule


// ---------------------------------------------------------------------------
// ucie_rx  —  bump pads -> flits -> tile
// ---------------------------------------------------------------------------
module ucie_rx #(
    parameter int TILE_DIM     = 16,
    parameter int WORDS        = 256,
    parameter int WORDS_PER_FL = 29,
    parameter int FLITS        = 9
)(
    input  wire        clk_core,   // 1 GHz — FDI flit reassembly
    input  wire        rst_n,

    // from bump pads
    input  wire [511:0] bump_data,
    input  wire         bump_valid,
    output logic        bump_credit,   // credit return to transmitter

    // to chiplet logic
    output logic        rx_valid,
    output logic [3:0]  rx_src_id,
    output logic [15:0] rx_tile [TILE_DIM][TILE_DIM],
    input  wire         rx_ready,

    // errors
    output logic        rx_crc_err,
    output logic        rx_seq_err
);
    // field extraction
    wire [3:0]   src_id    = bump_data[511:508];
    wire [3:0]   dst_id    = bump_data[507:504];
    wire [7:0]   seq_num   = bump_data[503:496];
    wire [7:0]   flit_num  = bump_data[495:488];
    wire [463:0] payload   = bump_data[479:16];
    wire [7:0]   rx_crc    = bump_data[15:8];

    // CRC check
    logic [495:0] crc_chk_in;
    logic [7:0]   crc_chk_val;
    always_comb begin
        crc_chk_in = bump_data[511:16];
    end
    ucie_crc8 #(.W(496)) crc_chk_inst (
        .data_in(crc_chk_in),
        .crc_out(crc_chk_val)
    );

    // reassembly buffer
    logic [15:0] buf_words [WORDS];
    logic [3:0]  expected_flit;
    logic [7:0]  expected_seq;

    // unflatten buffer to tile
    always_comb begin
        for (int i = 0; i < TILE_DIM; i++)
            for (int j = 0; j < TILE_DIM; j++)
                rx_tile[i][j] = buf_words[i*TILE_DIM + j];
    end

    always_ff @(posedge clk_core or negedge rst_n) begin : rx_ff
        if (!rst_n) begin
            rx_valid      <= 1'b0;
            rx_src_id     <= 4'h0;
            rx_crc_err    <= 1'b0;
            rx_seq_err    <= 1'b0;
            bump_credit   <= 1'b0;
            expected_flit <= 4'd0;
            expected_seq  <= 8'd0;
            for (int w = 0; w < WORDS; w++) buf_words[w] <= 16'h0;
        end else begin
            rx_valid    <= 1'b0;
            rx_crc_err  <= 1'b0;
            rx_seq_err  <= 1'b0;
            bump_credit <= 1'b0;

            if (bump_valid) begin
                if (crc_chk_val != rx_crc) begin
                    rx_crc_err <= 1'b1;
                end else begin
                    if (flit_num != {4'h0, expected_flit}) begin
                        rx_seq_err    <= 1'b1;
                        expected_flit <= 4'd0;
                    end else begin
                        // unpack payload into buffer
                        for (int w = 0; w < WORDS_PER_FL; w++) begin
                            int idx;
                            idx = flit_num[3:0] * WORDS_PER_FL + w;
                            if (idx < WORDS)
                                buf_words[idx] <= payload[w*16 +: 16];
                        end
                        bump_credit   <= 1'b1;
                        rx_src_id     <= src_id;

                        if (flit_num == FLITS - 1) begin
                            expected_flit <= 4'd0;
                            expected_seq  <= expected_seq + 8'd1;
                            rx_valid      <= 1'b1;
                        end else begin
                            expected_flit <= expected_flit + 4'd1;
                        end
                    end
                end
            end
        end
    end
endmodule

`default_nettype wire
// =============================================================================
// End of ucie_link.sv
// =============================================================================
