OPENQASM 2.0;
include "qelib1.inc";
qreg q979[6];
rx(pi) q979[0];
rz(pi) q979[2];
cx q979[3],q979[2];
cx q979[1],q979[2];
cx q979[0],q979[1];
