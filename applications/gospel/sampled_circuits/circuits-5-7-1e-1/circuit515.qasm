OPENQASM 2.0;
include "qelib1.inc";
qreg q516[5];
rx(pi/4) q516[4];
cx q516[3],q516[4];
cx q516[3],q516[2];
cx q516[1],q516[2];
cx q516[1],q516[0];
