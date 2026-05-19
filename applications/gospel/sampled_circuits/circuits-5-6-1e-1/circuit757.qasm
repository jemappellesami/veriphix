OPENQASM 2.0;
include "qelib1.inc";
qreg q758[5];
cx q758[4],q758[3];
cx q758[2],q758[3];
cx q758[2],q758[1];
cx q758[0],q758[1];
rx(pi/4) q758[1];
