OPENQASM 2.0;
include "qelib1.inc";
qreg q401[3];
rx(pi) q401[2];
cx q401[1],q401[2];
cx q401[0],q401[1];
rx(pi/4) q401[1];
