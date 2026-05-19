OPENQASM 2.0;
include "qelib1.inc";
qreg q607[6];
cx q607[4],q607[5];
cx q607[3],q607[4];
cx q607[2],q607[3];
cx q607[2],q607[1];
cx q607[0],q607[1];
