typedef enum logic [1:0] {
    STATE_IDLE = 2'h0,
    STATE_LOAD = 2'h1,
    STATE_EXEC = 2'h2,
    STATE_DONE = 2'h3
} fsm_state_e;

module test_3_case (
    input  fsm_state_e state_i,
    input  logic       valid_i,
    input  logic [7:0] d0_i,
    input  logic [7:0] d1_i,
    output logic [7:0] result_o
);
  always_comb begin
    unique case (state_i)
      STATE_IDLE: begin                       // Path 1: state_i == STATE_IDLE
        result_o = d0_i;
      end
      STATE_LOAD, STATE_EXEC: begin           // Path 2: state_i == STATE_LOAD || state_i == STATE_EXEC
        if (valid_i) begin
            result_o = d1_i;                 // Path 2.1: Path 2 && valid_i
        end else begin
            result_o = 8'hFF;                // Path 2.2: Path 2 && !valid_i
        end
      end
      default: begin                          // Path 3: state_i == STATE_DONE
        result_o = d0_i & d1_i;
      end
    endcase
  end
endmodule