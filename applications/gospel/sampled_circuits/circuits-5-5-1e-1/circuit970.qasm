OPENQASM 2.0;
include "qelib1.inc";
qreg q971[5];
cx q971[4],q971[3];
cx q971[2],q971[3];
cx q971[1],q971[2];
cx q971[1],q971[0];
