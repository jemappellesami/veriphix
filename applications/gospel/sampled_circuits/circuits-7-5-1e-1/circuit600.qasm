OPENQASM 2.0;
include "qelib1.inc";
qreg q601[7];
cx q601[4],q601[5];
cx q601[4],q601[3];
cx q601[3],q601[2];
cx q601[2],q601[1];
cx q601[1],q601[0];
