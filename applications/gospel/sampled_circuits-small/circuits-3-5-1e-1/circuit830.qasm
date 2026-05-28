OPENQASM 2.0;
include "qelib1.inc";
qreg q831[3];
rz(pi/4) q831[2];
cx q831[1],q831[2];
cx q831[1],q831[0];
