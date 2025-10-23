//=====================================================================
// tb_igcn_accelerator.sv  -- Icarus-compatible testbench
//=====================================================================
`timescale 1ns / 1ps

module tb_igcn_accelerator;

    // Parameters (match DUT)
    localparam integer NUM_PES = 4;
    localparam integer C_MAX = 32;
    localparam integer PE_COMPUTE_CYCLES = 100;
    localparam integer FRAGMENTATION_PENALTY_CYCLES = 50;

    // Signals
    reg clk = 0;
    reg rst_n;
    reg start_processing;
    reg [15:0] island_size;
    reg strategy_is_enhanced;
    reg island_needs_penalty;
    wire island_accepted;
    wire accelerator_busy;
    wire processing_done;

    // Instantiate DUT (explicit port mapping)
    igcn_accelerator #(
        .NUM_PES(NUM_PES),
        .C_MAX(C_MAX),
        .PE_COMPUTE_CYCLES(PE_COMPUTE_CYCLES),
        .FRAGMENTATION_PENALTY_CYCLES(FRAGMENTATION_PENALTY_CYCLES)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .start_processing(start_processing),
        .island_size(island_size),
        .strategy_is_enhanced(strategy_is_enhanced),
        .island_needs_penalty(island_needs_penalty),
        .island_accepted(island_accepted),
        .accelerator_busy(accelerator_busy),
        .processing_done(processing_done)
    );

    // Clock generation: 10ns period
    always #5 clk = ~clk;

    // Helper tasks
    task reset_dut();
        begin
            rst_n = 1'b0;
            start_processing = 1'b0;
            island_size = 16'h0;
            strategy_is_enhanced = 1'b0;
            island_needs_penalty = 1'b0;
            #20;
            rst_n = 1'b1;
            #10;
            $display("[%0t] TB: Reset complete.", $time);
        end
    endtask

    task submit_island_chunk(input integer size, input bit strategy_enh, input bit needs_penalty);
        begin
            @(posedge clk);
            strategy_is_enhanced = strategy_enh;
            island_size = size;
            island_needs_penalty = needs_penalty;
            start_processing = 1'b1;
            @(posedge clk); // hold one cycle
            start_processing = 1'b0;

            // wait for registered acceptance pulse
            wait (island_accepted == 1'b1);
            $display("[%0t] TB: Accelerator accepted island size=%0d (penalty=%0d, strategy=%s)",
                     $time, size, needs_penalty, strategy_enh ? "Enhanced" : "Baseline");
            @(posedge clk); // ensure acceptance pulse cleared
        end
    endtask

    // Main test
    initial begin
        integer island_sizes[0:4];
        integer i;
        longint baseline_start, baseline_end, enhanced_start, enhanced_end;

        // populate island sizes (example set)
        island_sizes[0] = C_MAX / 2;
        island_sizes[1] = C_MAX - 5;
        island_sizes[2] = C_MAX + 10;
        island_sizes[3] = (C_MAX * 18) / 10; // ~1.8 * C_MAX
        island_sizes[4] = (C_MAX * 25) / 10; // ~2.5 * C_MAX

        $display("CONFIG: NUM_PES=%0d C_MAX=%0d", NUM_PES, C_MAX);

        $dumpfile("igcn_tb.vcd");
        $dumpvars(0, tb_igcn_accelerator);

        // Baseline test
        reset_dut();
        $display("\n--- Baseline Strategy ---");
        strategy_is_enhanced = 1'b0;
        baseline_start = $time;

        for (i = 0; i < 5; i = i + 1) begin
            integer raw;
            raw = island_sizes[i];
            if (raw <= C_MAX) begin
                submit_island_chunk(raw, 1'b0, 1'b0);
            end else begin
                integer rem;
                rem = raw;
                while (rem > 0) begin
                    integer chunk;
                    if (rem > C_MAX) chunk = C_MAX; else chunk = rem;
                    submit_island_chunk(chunk, 1'b0, 1'b1);
                    rem = rem - chunk;
                end
            end
        end

        // wait for completion
        wait (processing_done == 1'b1);
        baseline_end = $time;
        $display("[%0t] TB: Baseline finished. cycles=%0d", $time, (baseline_end - baseline_start)/10);

        // Enhanced test
        #50;
        reset_dut();
        $display("\n--- Enhanced Strategy ---");
        strategy_is_enhanced = 1'b1;
        enhanced_start = $time;

        for (i = 0; i < 5; i = i + 1) begin
            integer raw;
            raw = island_sizes[i];
            if (raw <= (2*C_MAX) && NUM_PES >= 2) begin
                submit_island_chunk(raw, 1'b1, 1'b0);
            end else begin
                integer rem;
                rem = raw;
                while (rem > 0) begin
                    integer chunk;
                    if (rem > (2*C_MAX)) chunk = 2*C_MAX; else chunk = rem;
                    submit_island_chunk(chunk, 1'b1, 1'b1);
                    rem = rem - chunk;
                end
            end
        end

        wait (processing_done == 1'b1);
        enhanced_end = $time;
        $display("[%0t] TB: Enhanced finished. cycles=%0d", $time, (enhanced_end - enhanced_start)/10);

        // summary
        $display("\n--- Results ---");
        $display("Baseline cycles: %0d", (baseline_end - baseline_start)/10);
        $display("Enhanced cycles: %0d", (enhanced_end - enhanced_start)/10);
        if ((enhanced_end - enhanced_start) > 0) begin
            real speedup;
            speedup = real'(baseline_end - baseline_start) / real'(enhanced_end - enhanced_start);
            $display("Speedup: %.2fx", speedup);
        end

        #50;
        $finish;
    end

endmodule
