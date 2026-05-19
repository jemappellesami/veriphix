OPENQASM 2.0;
include "qelib1.inc";
qreg q877[4];
rx(pi/2) q877[3];
cx q877[3],q877[2];
cx q877[2],q877[1];
cx q877[0],q877[1];
rx(pi/4) q877[1];
