module test_basic (
    input  logic a,
    output logic c
);
    logic b;

    assign b = a;
    assign c = ~b;

endmodule