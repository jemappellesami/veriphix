OPENQASM 2.0;
include "qelib1.inc";
qreg q241[6];
cx q241[4],q241[5];
cx q241[3],q241[4];
cx q241[3],q241[2];
cx q241[1],q241[2];
cx q241[1],q241[0];
