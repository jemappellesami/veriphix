OPENQASM 2.0;
include "qelib1.inc";
qreg q528[3];
rx(5*pi/4) q528[2];
cx q528[2],q528[1];
cx q528[0],q528[1];
