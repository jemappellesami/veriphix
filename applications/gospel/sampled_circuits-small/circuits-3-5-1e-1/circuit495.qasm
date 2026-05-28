OPENQASM 2.0;
include "qelib1.inc";
qreg q496[3];
rx(pi/4) q496[2];
cx q496[1],q496[2];
cx q496[1],q496[0];
