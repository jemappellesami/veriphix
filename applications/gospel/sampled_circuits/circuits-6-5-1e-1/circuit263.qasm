OPENQASM 2.0;
include "qelib1.inc";
qreg q264[6];
rx(pi) q264[0];
cx q264[1],q264[2];
cx q264[2],q264[3];
cx q264[1],q264[2];
cx q264[1],q264[0];
