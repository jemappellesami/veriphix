OPENQASM 2.0;
include "qelib1.inc";
qreg q723[5];
cx q723[3],q723[4];
cx q723[2],q723[3];
cx q723[2],q723[1];
cx q723[1],q723[0];
