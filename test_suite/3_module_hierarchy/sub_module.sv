module sub_module (
    input  logic in1,
    input  logic in2,
    output logic out1
);
    assign out1 = in1 & in2;
endmodule