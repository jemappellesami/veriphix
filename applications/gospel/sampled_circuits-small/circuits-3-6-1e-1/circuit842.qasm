OPENQASM 2.0;
include "qelib1.inc";
qreg q843[3];
rx(3*pi/4) q843[2];
cx q843[2],q843[1];
cx q843[1],q843[0];
rx(pi/4) q843[1];
