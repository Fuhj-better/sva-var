module test_sequential (
    input  logic clk,
    input  logic rst_n,
    input  logic sel,
    input  logic [7:0] d_in1,
    input  logic [7:0] d_in2,
    output logic [7:0] d_out
);

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            d_out <= 8'b0;
        else if (sel)
            d_out <= d_in1;
        else
            d_out <= d_in2;
    end

endmodule