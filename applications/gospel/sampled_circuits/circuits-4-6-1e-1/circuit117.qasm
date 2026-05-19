OPENQASM 2.0;
include "qelib1.inc";
qreg q118[4];
rz(pi/2) q118[3];
cx q118[3],q118[2];
cx q118[1],q118[2];
cx q118[1],q118[0];
rx(pi/4) q118[1];
