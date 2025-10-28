module test_7_multicycle (
    input  logic clk,
    input  logic reset_n,
    input  logic instr_valid_i,
    input  logic [7:0] data_in_i,
    output logic [7:0] state_out_o 
);
  logic [7:0] pipeline_reg; 
  logic [7:0] next_pipeline_reg;

  // 时序逻辑 (Always_ff): 用于测试 sequential logicType 和三级条件路径
  always_ff @(posedge clk or negedge reset_n) begin
    if (!reset_n) begin                           // Path 1: !reset_n
      pipeline_reg <= 8'h00; 
    end else if (instr_valid_i) begin             // Path 2: reset_n && instr_valid_i
      pipeline_reg <= next_pipeline_reg; 
    end else begin                                // Path 3: reset_n && !instr_valid_i
      pipeline_reg <= pipeline_reg; 
    end
  end

  // 组合逻辑: pipeline_reg 是控制信号
  always_comb begin
    if (pipeline_reg == 8'hFF) begin              // Condition: pipeline_reg == 0xFF
        next_pipeline_reg = data_in_i;
    end else begin
        next_pipeline_reg = pipeline_reg + 8'h1;
    end
    state_out_o = pipeline_reg;
  end
endmodule