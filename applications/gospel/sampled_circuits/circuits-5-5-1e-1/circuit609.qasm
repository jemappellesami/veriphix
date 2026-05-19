OPENQASM 2.0;
include "qelib1.inc";
qreg q610[5];
cx q610[4],q610[3];
cx q610[3],q610[2];
cx q610[2],q610[1];
cx q610[0],q610[1];
