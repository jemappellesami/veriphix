OPENQASM 2.0;
include "qelib1.inc";
qreg q528[3];
cx q528[2],q528[1];
cx q528[1],q528[2];
cx q528[1],q528[0];
rx(pi/4) q528[1];
