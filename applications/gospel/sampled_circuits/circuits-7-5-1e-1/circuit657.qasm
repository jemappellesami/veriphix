OPENQASM 2.0;
include "qelib1.inc";
qreg q658[7];
cx q658[5],q658[4];
cx q658[4],q658[3];
cx q658[3],q658[2];
cx q658[1],q658[2];
cx q658[1],q658[0];
