OPENQASM 2.0;
include "qelib1.inc";
qreg q118[3];
rx(5*pi/4) q118[0];
cx q118[1],q118[0];
rx(pi/2) q118[1];
cx q118[2],q118[1];
cx q118[1],q118[0];
