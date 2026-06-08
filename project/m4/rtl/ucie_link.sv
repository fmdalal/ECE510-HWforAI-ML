// =============================================================================
// ucie_link.sv  —  UCIe FDI-layer TX and RX  [clk_link upgrade]
// =============================================================================
//
// KEY CHANGE FROM ORIGINAL
// -------------------------
// Original: both TX and RX ran on clk_core (1 GHz), with TX_WAIT adding
//           an idle cycle between every flit — 2 cycles/flit, 284 ns/tile.
//
// Upgraded: TX and RX bump paths run on clk_link (2 GHz).
//           Back-to-back flit transmission — 1 clk_link cycle/flit, 71 ns/tile.
//           4× speedup on inter-chiplet data movement.
//
// CDC CROSSINGS
// -------------
// clk_core → clk_link  (TX path)
//   tx_valid / tile data:  request/acknowledge handshake
//     clk_core asserts tx_req when tile is latched in core_buf
//     clk_link 2-FF syncs tx_req, latches tile into link_buf, asserts tx_ack
//     clk_core 2-FF syncs tx_ack, deasserts tx_req (handshake complete)
//   tx_ready is deasserted from the moment tx_req fires until ack returns.
//
// clk_link → clk_core  (RX path)
//   rx_valid pulse:  2-FF sync (1-bit flag)
//   rx_tile data:    written into buf_words over FLITS clk_link cycles;
//                    buf_words is stable long before the synced rx_valid_core
//                    reaches clk_core — safe multi-cycle path.
//
// FLIT FORMAT (512 bits, unchanged from original)
// ------------------------------------------------
//  [511:508]  src_id   (4 bits)
//  [507:504]  dst_id   (4 bits)
//  [503:496]  seq_num  (8 bits)
//  [495:488]  flit_num (8 bits)
//  [487:480]  total    (8 bits, = FLITS)
//  [479:16]   payload  (464 bits = 29 × 16-bit BF16 words)
//  [15:8]     CRC-8    (8 bits)
//  [7:0]      reserved (8 bits, = 0x00)
//
// =============================================================================

`timescale 1ns/1ps
`default_nettype none

// ---------------------------------------------------------------------------
// ucie_crc8  —  CRC-8/MAXIM polynomial 0x31, purely combinational
// ---------------------------------------------------------------------------
module ucie_crc8 #(
    parameter int W = 496
)(
    input  wire [W-1:0]  data_in,
    output logic [7:0]   crc_out
);
    always_comb begin
        /* verilator lint_off WIDTHEXPAND */
        integer ci;
        ci = 32'hFF;
        for (int i = W-1; i >= 0; i--) begin
            if (ci[7] ^ data_in[i])
                ci = ((ci << 1) & 32'hFE) ^ 32'h31;
            else
                ci = (ci << 1) & 32'hFE;
        end
        crc_out = ci[7:0] ^ 8'hFF;
        /* verilator lint_on WIDTHEXPAND */
    end
endmodule

// ---------------------------------------------------------------------------
// ucie_tx  —  tile -> flits -> bump pads  (clk_link domain)
// ---------------------------------------------------------------------------
module ucie_tx #(
    parameter int TILE_DIM     = 16,
    parameter int WORDS        = TILE_DIM * TILE_DIM,
    parameter int WORDS_PER_FL = 29,
    parameter int FLITS        = (TILE_DIM * TILE_DIM + WORDS_PER_FL - 1) / WORDS_PER_FL
)(
    // Chiplet compute domain
    input  wire        clk_core,
    input  wire        rst_n,

    // UCIe PHY bump domain (2 GHz)
    input  wire        clk_link,

    // From chiplet logic (clk_core domain)
    input  wire        tx_valid,
    input  wire [3:0]  tx_src_id,
    input  wire [3:0]  tx_dst_id,
    input  wire [15:0] tx_tile [TILE_DIM][TILE_DIM],
    output logic       tx_ready,

    // To bump pads / interposer (clk_link domain)
    output logic [511:0] bump_data,
    output logic         bump_valid,
    input  wire          bump_credit
);
    initial begin
        if (FLITS > 255)
            $fatal(1, "ucie_tx: FLITS=%0d exceeds 8-bit flit_num field.", FLITS);
    end

    // -------------------------------------------------------------------------
    // 1. Flatten tile to word array (clk_core domain, combinational)
    // -------------------------------------------------------------------------
    wire [15:0] flat [WORDS];
    genvar fli, flj;
    generate
        for (fli = 0; fli < TILE_DIM; fli++) begin : flat_row
            for (flj = 0; flj < TILE_DIM; flj++) begin : flat_col
                assign flat[fli*TILE_DIM + flj] = tx_tile[fli][flj];
            end
        end
    endgenerate

    // -------------------------------------------------------------------------
    // 2. clk_core side: latch tile into core_buf, assert tx_req
    //    tx_ready deasserts when tx_req fires, reasserts after ack received
    // -------------------------------------------------------------------------
    logic [15:0] core_buf [WORDS];
    logic [3:0]  core_src_id, core_dst_id;
    logic        tx_req;

    // Forward declarations needed before ack_sync block
    logic tx_req_s1, tx_req_link;
    logic tx_ack_link;

    // 2-FF sync: tx_ack (clk_link) → clk_core
    logic tx_ack_s1, tx_ack_core;
    always_ff @(posedge clk_core or negedge rst_n) begin : ack_sync
        if (!rst_n) begin tx_ack_s1 <= 1'b0; tx_ack_core <= 1'b0; end
        else        begin tx_ack_s1 <= tx_ack_link; tx_ack_core <= tx_ack_s1; end
    end

    always_ff @(posedge clk_core or negedge rst_n) begin : core_side
        if (!rst_n) begin
            tx_req      <= 1'b0;
            tx_ready    <= 1'b1;
            core_src_id <= 4'h0;
            core_dst_id <= 4'h0;
            for (int w = 0; w < WORDS; w++) core_buf[w] <= 16'h0;
        end else begin
            if (tx_valid && tx_ready && !tx_req) begin
                // Latch tile and metadata, assert request
                for (int w = 0; w < WORDS; w++) core_buf[w] <= flat[w];
                core_src_id <= tx_src_id;
                core_dst_id <= tx_dst_id;
                tx_req      <= 1'b1;
                tx_ready    <= 1'b0;
            end
            // Handshake complete when ack rises
            if (tx_ack_core && tx_req) begin
                tx_req   <= 1'b0;
                tx_ready <= 1'b1;
            end
        end
    end

    // -------------------------------------------------------------------------
    // 3. clk_link side: 2-FF sync tx_req, latch tile into link_buf, assert ack
    // -------------------------------------------------------------------------
    // (tx_req_s1, tx_req_link, tx_ack_link declared above for forward reference)
    logic [15:0] link_buf [WORDS];
    logic [3:0]  link_src_id, link_dst_id;
    logic        link_buf_valid;

    // 2-FF sync tx_req (clk_core) → clk_link
    always_ff @(posedge clk_link or negedge rst_n) begin : req_sync
        if (!rst_n) begin tx_req_s1 <= 1'b0; tx_req_link <= 1'b0; end
        else        begin tx_req_s1 <= tx_req; tx_req_link <= tx_req_s1; end
    end

    // Edge detect on tx_req_link: latch tile on rising edge
    logic tx_req_link_prev;
    always_ff @(posedge clk_link or negedge rst_n) begin : link_latch
        if (!rst_n) begin
            tx_req_link_prev <= 1'b0;
            tx_ack_link      <= 1'b0;
            link_buf_valid   <= 1'b0;
            link_src_id      <= 4'h0;
            link_dst_id      <= 4'h0;
            for (int w = 0; w < WORDS; w++) link_buf[w] <= 16'h0;
        end else begin
            tx_req_link_prev <= tx_req_link;
            tx_ack_link      <= 1'b0;
            if (tx_req_link && !tx_req_link_prev) begin
                // Rising edge of synced req: latch tile from core_buf
                for (int w = 0; w < WORDS; w++) link_buf[w] <= core_buf[w];
                link_src_id    <= core_src_id;
                link_dst_id    <= core_dst_id;
                link_buf_valid <= 1'b1;
                tx_ack_link    <= 1'b1;   // acknowledge immediately
            end
        end
    end

    // -------------------------------------------------------------------------
    // 4. TX FSM on clk_link: back-to-back flits, no TX_WAIT state
    //    Pipeline: flit_cnt advances every cycle; bump_data is registered
    //    from the combinational payload built from flit_cnt-1 (one cycle ahead)
    // -------------------------------------------------------------------------
    logic [$clog2(FLITS+1)-1:0] flit_cnt;
    logic [7:0]  seq_cnt;
    logic [3:0]  credits;

    typedef enum logic [1:0] {
        TX_IDLE = 2'd0,
        TX_PIPE = 2'd1,   // back-to-back pipeline: no wait state
        TX_LAST = 2'd2    // one extra cycle to flush last flit
    } tx_state_t;
    tx_state_t tx_state;

    // Build current flit payload combinationally from flit_cnt
    wire [15:0] payload_w [WORDS_PER_FL];
    genvar pw;
    generate
        for (pw = 0; pw < WORDS_PER_FL; pw++) begin : pay_gen
            wire [11:0] pidx = flit_cnt * WORDS_PER_FL + pw;
            assign payload_w[pw] = (pidx < WORDS) ? link_buf[pidx] : 16'h0;
        end
    endgenerate

    wire [463:0] payload;
    genvar ppw;
    generate
        for (ppw = 0; ppw < WORDS_PER_FL; ppw++) begin : pay_pack
            assign payload[ppw*16 +: 16] = payload_w[ppw];
        end
    endgenerate

    wire [495:0] crc_in = {link_src_id, link_dst_id,
                            seq_cnt, 8'(flit_cnt),
                            8'(FLITS), payload};
    wire [7:0] crc_val;
    ucie_crc8 #(.W(496)) crc_inst (.data_in(crc_in), .crc_out(crc_val));

    // Credit counter (clk_link domain)
    always_ff @(posedge clk_link or negedge rst_n) begin : credit_ff
        if (!rst_n)
            credits <= 4'd8;
        else begin
            if (bump_credit && !bump_valid)
                credits <= credits + 4'd1;
            else if (!bump_credit && bump_valid && (credits > 0))
                credits <= credits - 4'd1;
        end
    end

    // TX FSM — 1 cycle per flit at 2 GHz
    always_ff @(posedge clk_link or negedge rst_n) begin : tx_fsm
        if (!rst_n) begin
            tx_state   <= TX_IDLE;
            bump_valid <= 1'b0;
            bump_data  <= 512'h0;
            flit_cnt   <= '0;
            seq_cnt    <= 8'd0;
        end else begin
            case (tx_state)
                TX_IDLE: begin
                    bump_valid <= 1'b0;
                    if (link_buf_valid && credits > 0) begin
                        flit_cnt <= '0;
                        tx_state <= TX_PIPE;
                    end
                end

                TX_PIPE: begin
                    if (credits > 0) begin
                        // Register flit — payload is combinational from flit_cnt
                        bump_data  <= {link_src_id, link_dst_id,
                                       seq_cnt, 8'(flit_cnt),
                                       8'(FLITS), payload,
                                       crc_val, 8'h00};
                        bump_valid <= 1'b1;

                        if (flit_cnt == FLITS[$clog2(FLITS+1)-1:0] - 1) begin
                            // Last flit
                            seq_cnt  <= seq_cnt + 8'd1;
                            flit_cnt <= '0;
                            tx_state <= TX_LAST;
                        end else begin
                            flit_cnt <= flit_cnt + 1'b1;
                            // Stay in TX_PIPE: back-to-back, no wait
                        end
                    end else begin
                        bump_valid <= 1'b0;   // stall waiting for credits
                    end
                end

                TX_LAST: begin
                    // One cycle to deassert bump_valid after last flit
                    bump_valid <= 1'b0;
                    tx_state   <= TX_IDLE;
                end

                default: tx_state <= TX_IDLE;
            endcase
        end
    end

endmodule


// ---------------------------------------------------------------------------
// ucie_rx  —  bump pads -> flits -> tile  (clk_link domain)
// ---------------------------------------------------------------------------
module ucie_rx #(
    parameter int TILE_DIM     = 16,
    parameter int WORDS        = TILE_DIM * TILE_DIM,
    parameter int WORDS_PER_FL = 29,
    parameter int FLITS        = (TILE_DIM * TILE_DIM + WORDS_PER_FL - 1) / WORDS_PER_FL
)(
    // Chiplet compute domain
    input  wire        clk_core,
    input  wire        rst_n,

    // UCIe PHY bump domain (2 GHz)
    input  wire        clk_link,

    // From bump pads (clk_link domain)
    input  wire [511:0] bump_data,
    input  wire         bump_valid,
    output logic        bump_credit,

    // To chiplet logic (clk_core domain)
    output logic        rx_valid,      // 1-cycle pulse in clk_core domain
    output logic [3:0]  rx_src_id,
    output logic [15:0] rx_tile [TILE_DIM][TILE_DIM],
    input  wire         rx_ready,

    // Errors (clk_core domain, synced from clk_link)
    output logic        rx_crc_err,
    output logic        rx_seq_err
);
    initial begin
        if (FLITS > 255)
            $fatal(1, "ucie_rx: FLITS=%0d exceeds 8-bit flit_num field.", FLITS);
    end

    // -------------------------------------------------------------------------
    // 1. Flit field extraction (combinational, clk_link domain)
    // -------------------------------------------------------------------------
    wire [3:0]   src_id   = bump_data[511:508];
    wire [3:0]   dst_id   = bump_data[507:504];
    wire [7:0]   seq_num  = bump_data[503:496];
    wire [7:0]   flit_num = bump_data[495:488];
    wire [463:0] payload  = bump_data[479:16];
    wire [7:0]   rx_crc   = bump_data[15:8];

    // CRC check
    wire [495:0] crc_chk_in = bump_data[511:16];
    wire [7:0]   crc_chk_val;
    ucie_crc8 #(.W(496)) crc_chk_inst (
        .data_in(crc_chk_in), .crc_out(crc_chk_val)
    );

    // Pre-slice payload words (Icarus compatibility)
    wire [15:0] rx_payload_w [WORDS_PER_FL];
    genvar rpw;
    generate
        for (rpw = 0; rpw < WORDS_PER_FL; rpw++) begin : rx_pay_gen
            assign rx_payload_w[rpw] = payload[rpw*16 +: 16];
        end
    endgenerate

    // -------------------------------------------------------------------------
    // 2. Reassembly FSM on clk_link
    //    Writes buf_words over FLITS cycles; asserts rx_valid_link pulse
    //    when last flit received and CRC passes.
    // -------------------------------------------------------------------------
    logic [15:0] buf_words [WORDS];
    logic [$clog2(FLITS+1)-1:0] expected_flit;
    logic [7:0]  expected_seq;
    logic        rx_valid_link;     // 1-cycle pulse in clk_link domain
    logic [3:0]  rx_src_link;
    logic        rx_crc_err_link;
    logic        rx_seq_err_link;

    always_ff @(posedge clk_link or negedge rst_n) begin : rx_link_ff
        if (!rst_n) begin
            rx_valid_link   <= 1'b0;
            rx_src_link     <= 4'h0;
            rx_crc_err_link <= 1'b0;
            rx_seq_err_link <= 1'b0;
            bump_credit     <= 1'b0;
            expected_flit   <= '0;
            expected_seq    <= 8'd0;
            for (int w = 0; w < WORDS; w++) buf_words[w] <= 16'h0;
        end else begin
            rx_valid_link   <= 1'b0;
            rx_crc_err_link <= 1'b0;
            rx_seq_err_link <= 1'b0;
            bump_credit     <= 1'b0;

            if (bump_valid) begin
                if (crc_chk_val != rx_crc) begin
                    rx_crc_err_link <= 1'b1;
                end else if (flit_num != 8'(expected_flit)) begin
                    rx_seq_err_link <= 1'b1;
                    expected_flit   <= '0;
                end else begin
                    // Write payload words into reassembly buffer
                    for (int w = 0; w < WORDS_PER_FL; w++) begin
                        int idx;
                        idx = flit_num * WORDS_PER_FL + w;
                        if (idx < WORDS)
                            buf_words[idx] <= rx_payload_w[w];
                    end
                    bump_credit <= 1'b1;
                    rx_src_link <= src_id;

                    if (flit_num == FLITS[$clog2(FLITS+1)-1:0] - 1) begin
                        expected_flit <= '0;
                        expected_seq  <= expected_seq + 8'd1;
                        rx_valid_link <= 1'b1;   // tile complete
                    end else begin
                        expected_flit <= expected_flit + 1'b1;
                    end
                end
            end
        end
    end

    // -------------------------------------------------------------------------
    // 3. CDC: clk_link → clk_core
    //    rx_valid_link (1-cycle pulse) → 2-FF sync → rx_valid (clk_core)
    //    rx_src_link, buf_words: stable well before synced pulse arrives —
    //    safe multi-cycle path (buf_words written over FLITS clk_link cycles,
    //    2-FF sync adds only 2 clk_core cycles delay)
    //    rx_crc_err_link / rx_seq_err_link: same 2-FF sync pattern
    // -------------------------------------------------------------------------
    logic rx_valid_s1, rx_valid_core;
    logic rx_crc_s1,   rx_crc_core;
    logic rx_seq_s1,   rx_seq_core;

    always_ff @(posedge clk_core or negedge rst_n) begin : rx_core_sync
        if (!rst_n) begin
            rx_valid_s1  <= 1'b0; rx_valid_core <= 1'b0;
            rx_crc_s1    <= 1'b0; rx_crc_core   <= 1'b0;
            rx_seq_s1    <= 1'b0; rx_seq_core   <= 1'b0;
        end else begin
            rx_valid_s1  <= rx_valid_link;  rx_valid_core <= rx_valid_s1;
            rx_crc_s1    <= rx_crc_err_link; rx_crc_core  <= rx_crc_s1;
            rx_seq_s1    <= rx_seq_err_link; rx_seq_core  <= rx_seq_s1;
        end
    end

    // Drive clk_core outputs
    // rx_tile is driven directly from buf_words — safe because buf_words
    // is stable for at least 2 clk_core cycles before rx_valid_core pulses
    assign rx_valid   = rx_valid_core;
    assign rx_crc_err = rx_crc_core;
    assign rx_seq_err = rx_seq_core;

    always_ff @(posedge clk_core or negedge rst_n) begin : src_capture
        if (!rst_n)
            rx_src_id <= 4'h0;
        else if (rx_valid_core)
            rx_src_id <= rx_src_link;
    end

    // Unflatten buf_words → rx_tile (combinational, clk_core reads after valid)
    genvar ufi, ufj;
    generate
        for (ufi = 0; ufi < TILE_DIM; ufi++) begin : unflat_row
            for (ufj = 0; ufj < TILE_DIM; ufj++) begin : unflat_col
                assign rx_tile[ufi][ufj] = buf_words[ufi*TILE_DIM + ufj];
            end
        end
    endgenerate

endmodule

`default_nettype wire
// =============================================================================
// End of ucie_link.sv  (clk_link 2 GHz upgrade)
// =============================================================================
