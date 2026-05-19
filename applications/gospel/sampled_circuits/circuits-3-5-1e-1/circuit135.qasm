OPENQASM 2.0;
include "qelib1.inc";
qreg q136[3];
rx(pi/4) q136[2];
cx q136[1],q136[2];
cx q136[1],q136[0];
