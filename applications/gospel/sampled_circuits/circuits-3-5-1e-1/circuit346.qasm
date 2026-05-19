OPENQASM 2.0;
include "qelib1.inc";
qreg q347[3];
cx q347[0],q347[1];
rx(7*pi/4) q347[1];
rx(pi) q347[0];
cx q347[1],q347[2];
cx q347[0],q347[1];
