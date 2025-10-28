module top_module (
    input  logic a,
    input  logic b,
    output logic c
);
    sub_module i_sub (
        .in1(a),
        .in2(b),
        .out1(c)
    );
endmodule