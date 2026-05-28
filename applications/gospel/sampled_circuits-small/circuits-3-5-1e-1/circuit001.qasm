OPENQASM 2.0;
include "qelib1.inc";
qreg q2[3];
rx(pi) q2[1];
rx(pi/4) q2[2];
cx q2[2],q2[1];
cx q2[1],q2[0];
