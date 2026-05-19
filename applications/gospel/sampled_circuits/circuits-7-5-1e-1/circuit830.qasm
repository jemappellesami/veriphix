OPENQASM 2.0;
include "qelib1.inc";
qreg q831[7];
cx q831[4],q831[5];
cx q831[4],q831[3];
cx q831[3],q831[2];
cx q831[1],q831[2];
cx q831[0],q831[1];
