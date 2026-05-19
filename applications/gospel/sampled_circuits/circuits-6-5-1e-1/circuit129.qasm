OPENQASM 2.0;
include "qelib1.inc";
qreg q130[6];
cx q130[5],q130[4];
cx q130[3],q130[4];
cx q130[2],q130[3];
cx q130[2],q130[1];
cx q130[0],q130[1];
