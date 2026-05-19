OPENQASM 2.0;
include "qelib1.inc";
qreg q156[4];
rz(pi) q156[3];
cx q156[3],q156[2];
cx q156[2],q156[1];
cx q156[0],q156[1];
