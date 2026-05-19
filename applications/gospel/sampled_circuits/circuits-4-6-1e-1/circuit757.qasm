OPENQASM 2.0;
include "qelib1.inc";
qreg q758[4];
rx(pi/4) q758[3];
cx q758[3],q758[2];
cx q758[2],q758[1];
cx q758[1],q758[0];
rx(pi/4) q758[1];
