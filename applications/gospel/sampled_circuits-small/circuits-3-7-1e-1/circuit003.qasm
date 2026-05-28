OPENQASM 2.0;
include "qelib1.inc";
qreg q4[3];
cx q4[1],q4[2];
rx(pi/2) q4[2];
cx q4[2],q4[1];
cx q4[1],q4[0];
