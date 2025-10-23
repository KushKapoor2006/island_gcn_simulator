//=====================================================================
// igcn_accelerator.sv  -- Icarus-compatible, fixed (final)
//=====================================================================
`timescale 1ns / 1ps

module igcn_accelerator #(
    parameter integer NUM_PES = 4,
    parameter integer C_MAX = 32,
    parameter integer PE_COMPUTE_CYCLES = 100,
    parameter integer FRAGMENTATION_PENALTY_CYCLES = 50
)(
    input  wire clk,
    input  wire rst_n,
    input  wire start_processing,         // 1-cycle pulse
    input  wire [15:0] island_size,
    input  wire strategy_is_enhanced,
    input  wire island_needs_penalty,
    output wire island_accepted,          // registered 1-cycle pulse
    output wire accelerator_busy,
    output wire processing_done
);

    // ----------------------------
    // Registers / state (module scope)
    // ----------------------------
    // PE state: bit per PE (0 = idle, 1 = busy)
    reg [NUM_PES-1:0] pe_state_q;
    reg [NUM_PES-1:0] pe_state_d;
    reg [15:0] pe_timer_q [0:NUM_PES-1];
    reg [15:0] pe_timer_d [0:NUM_PES-1];

    // controller FSM: 0 = idle, 1 = waiting
    reg controller_state_q;
    reg controller_state_d;

    // latched request info
    reg [15:0] latched_island_size;
    reg        latched_strategy;
    reg        latched_needs_penalty;

    // island accepted pulse (registered)
    reg island_accepted_q;
    reg island_accepted_d;

    // helper (module-scope temporaries to avoid mid-block declarations)
    integer i;
    integer pe_idx1;
    integer pe_idx2;
    integer pe_idx;
    integer cycles_to_run;
    integer cycles_to_run_l;

    // ----------------------------
    // Sequential: update PEs & controller latch/accept
    // ----------------------------
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            pe_state_q <= {NUM_PES{1'b0}}; // all idle
            for (i = 0; i < NUM_PES; i = i + 1) pe_timer_q[i] <= 16'h0;
            controller_state_q <= 1'b0;
            latched_island_size <= 16'h0;
            latched_strategy <= 1'b0;
            latched_needs_penalty <= 1'b0;
            island_accepted_q <= 1'b0;
        end else begin
            // update PE registers
            for (i = 0; i < NUM_PES; i = i + 1) begin
                pe_state_q[i] <= pe_state_d[i];
                pe_timer_q[i] <= pe_timer_d[i];
            end

            // controller state and registered accept
            controller_state_q <= controller_state_d;
            island_accepted_q <= island_accepted_d;

            // latch request when controller idle and TB asserts start_processing
            if (controller_state_q == 1'b0 && start_processing) begin
                latched_island_size <= island_size;
                latched_strategy <= strategy_is_enhanced;
                latched_needs_penalty <= island_needs_penalty;
            end
        end
    end

    // ----------------------------
    // Combinational next-state generation (single driver for pe_state_d / pe_timer_d)
    // ----------------------------
    always @(*) begin
        // default: copy current states / timers (with countdown behavior)
        for (i = 0; i < NUM_PES; i = i + 1) begin
            if (pe_state_q[i] == 1'b1) begin
                if (pe_timer_q[i] <= 1) begin
                    pe_state_d[i] = 1'b0; // idle next
                    pe_timer_d[i] = 16'h0;
                end else begin
                    pe_state_d[i] = 1'b1;
                    pe_timer_d[i] = pe_timer_q[i] - 1;
                end
            end else begin
                pe_state_d[i] = 1'b0;
                pe_timer_d[i] = 16'h0;
            end
        end

        // controller defaults
        controller_state_d = controller_state_q;
        island_accepted_d  = 1'b0;

        // choose active request: new if controller idle and start_processing, else latched
        if (controller_state_q == 1'b0 && start_processing) begin
            // compute cycles_to_run for live request
            cycles_to_run = PE_COMPUTE_CYCLES + (island_needs_penalty ? FRAGMENTATION_PENALTY_CYCLES : 0);

            // initialize sentinel indices
            pe_idx1 = -1;
            pe_idx2 = -1;
            pe_idx = -1;

            // try allocate pair if enhanced and fits 2*C_MAX
            if (strategy_is_enhanced && (island_size > C_MAX) && (island_size <= 2*C_MAX) && (NUM_PES >= 2)) begin
                // find contiguous idle pair
                pe_idx1 = -1;
                for (i = 0; i < NUM_PES - 1; i = i + 1) begin
                    if ((pe_state_q[i] == 1'b0) && (pe_state_q[i+1] == 1'b0) && (pe_idx1 == -1)) begin
                        pe_idx1 = i;
                        pe_idx2 = i + 1;
                    end
                end
                if (pe_idx1 != -1) begin
                    pe_state_d[pe_idx1] = 1'b1;
                    pe_timer_d[pe_idx1] = cycles_to_run;
                    pe_state_d[pe_idx2] = 1'b1;
                    pe_timer_d[pe_idx2] = cycles_to_run;
                    island_accepted_d = 1'b1; // allocated successfully
                    $display("[%0t] Controller: Allocated island size %0d to PE pair %0d-%0d (Enhanced). Penalty: %0d",
                             $time, island_size, pe_idx1, pe_idx2, island_needs_penalty);
                end
            end

            // fallback: single PE allocate if fits
            if (!island_accepted_d && (island_size <= C_MAX)) begin
                pe_idx = -1;
                for (i = 0; i < NUM_PES; i = i + 1) begin
                    if (pe_state_q[i] == 1'b0 && pe_idx == -1) begin
                        pe_idx = i;
                    end
                end
                if (pe_idx != -1) begin
                    pe_state_d[pe_idx] = 1'b1;
                    pe_timer_d[pe_idx] = cycles_to_run;
                    island_accepted_d = 1'b1;
                    $display("[%0t] Controller: Allocated island size %0d to PE %0d. Penalty: %0d",
                             $time, island_size, pe_idx, island_needs_penalty);
                end
            end

            // if still not allocated: either no free PE (wait) or too large (error -> unblock)
            if (!island_accepted_d) begin
                if ((strategy_is_enhanced && (island_size > 2*C_MAX)) || (!strategy_is_enhanced && (island_size > C_MAX))) begin
                    $display("[%0t] ERROR: island_size %0d too large for strategy. TB should partition it.", $time, island_size);
                    island_accepted_d = 1'b1; // unblock TB defensively
                end else begin
                    controller_state_d = 1'b1; // WAITING_FOR_PE
                end
            end else begin
                controller_state_d = 1'b0; // allocated -> stay idle
            end

        end else if (controller_state_q == 1'b1) begin
            // attempt allocation for latched request
            cycles_to_run_l = PE_COMPUTE_CYCLES + (latched_needs_penalty ? FRAGMENTATION_PENALTY_CYCLES : 0);

            pe_idx1 = -1; pe_idx2 = -1; pe_idx = -1;

            if (latched_strategy && (latched_island_size > C_MAX) && (latched_island_size <= 2*C_MAX) && (NUM_PES >= 2)) begin
                // find contiguous idle pair for latched request
                pe_idx1 = -1;
                for (i = 0; i < NUM_PES - 1; i = i + 1) begin
                    if ((pe_state_q[i] == 1'b0) && (pe_state_q[i+1] == 1'b0) && (pe_idx1 == -1)) begin
                        pe_idx1 = i;
                        pe_idx2 = i + 1;
                    end
                end
                if (pe_idx1 != -1) begin
                    pe_state_d[pe_idx1] = 1'b1;
                    pe_timer_d[pe_idx1] = cycles_to_run_l;
                    pe_state_d[pe_idx2] = 1'b1;
                    pe_timer_d[pe_idx2] = cycles_to_run_l;
                    island_accepted_d = 1'b1;
                    $display("[%0t] Controller: Allocated latched island size %0d to PE pair %0d-%0d (Enhanced). Penalty: %0d",
                             $time, latched_island_size, pe_idx1, pe_idx2, latched_needs_penalty);
                end
            end

            if (!island_accepted_d && (latched_island_size <= C_MAX)) begin
                pe_idx = -1;
                for (i = 0; i < NUM_PES; i = i + 1) begin
                    if (pe_state_q[i] == 1'b0 && pe_idx == -1) begin
                        pe_idx = i;
                    end
                end
                if (pe_idx != -1) begin
                    pe_state_d[pe_idx] = 1'b1;
                    pe_timer_d[pe_idx] = cycles_to_run_l;
                    island_accepted_d = 1'b1;
                    $display("[%0t] Controller: Allocated latched island size %0d to PE %0d. Penalty: %0d",
                             $time, latched_island_size, pe_idx, latched_needs_penalty);
                end
            end

            if (island_accepted_d) begin
                controller_state_d = 1'b0;
            end else begin
                controller_state_d = 1'b1;
            end

        end else begin
            // no request: stay idle
            controller_state_d = controller_state_q;
            island_accepted_d = 1'b0;
        end
    end

    // register and expose island_accepted
    assign island_accepted = island_accepted_q;

    // processing_done: detect transition busy->idle with controller idle
    reg busy_last_cycle_q;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) busy_last_cycle_q <= 1'b0;
        else busy_last_cycle_q <= (|pe_state_q);
    end

    assign processing_done = busy_last_cycle_q && !(|pe_state_q) && (controller_state_q == 1'b0);
    assign accelerator_busy = (|pe_state_q);

endmodule
