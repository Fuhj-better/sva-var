module test_4_generate (
    input  logic [3:0] in_vector_i,
    output logic [3:0] out_vector_o
);
  logic [3:0] mid_vector;

  // Generate loop for inversion (4 separate assignments to mid_vector[i])
  for (genvar i = 0; i < 4; i++) begin : gen_bit_invert
    assign mid_vector[i] = ~in_vector_i[i]; 
  end

  // Another generate block (4 separate assignments to out_vector_o[j])
  for (genvar j = 0; j < 4; j++) begin : gen_bit_pass
    assign out_vector_o[j] = mid_vector[j]; 
  end
endmodule