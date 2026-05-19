OPENQASM 2.0;
include "qelib1.inc";
qreg q47[6];
cx q47[4],q47[5];
cx q47[3],q47[4];
cx q47[2],q47[3];
cx q47[1],q47[2];
cx q47[1],q47[0];
