OPENQASM 2.0;
include "qelib1.inc";
qreg q802[6];
cx q802[3],q802[4];
cx q802[3],q802[2];
cx q802[4],q802[5];
cx q802[1],q802[2];
cx q802[0],q802[1];
