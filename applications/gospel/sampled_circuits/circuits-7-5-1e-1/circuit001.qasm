OPENQASM 2.0;
include "qelib1.inc";
qreg q2[7];
cx q2[5],q2[4];
cx q2[3],q2[4];
cx q2[3],q2[2];
cx q2[1],q2[2];
cx q2[0],q2[1];
