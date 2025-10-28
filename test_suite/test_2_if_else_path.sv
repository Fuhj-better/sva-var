module test_2_if_else (
    input  logic [3:0] sel_i,
    input  logic       en_i,
    input  logic [7:0] data0_i,
    input  logic [7:0] data1_i,
    input  logic [7:0] data2_i,
    output logic [7:0] result_o
);
  logic [7:0] mid_result;

  always_comb begin
    mid_result = 8'h00; 

    if (sel_i == 4'h1 && en_i) begin         // Path 1: (sel_i == 1) && en_i
      mid_result = data0_i;
    end else if (sel_i == 4'h2 || sel_i == 4'h3) begin // Path 2: !(Path 1) && (sel_i == 2 || sel_i == 3)
      mid_result = data1_i;
    end else begin                           // Path 3: !(Path 1) && !(Path 2)
      mid_result = data2_i;
    end
    
    result_o = mid_result;
  end
endmodule