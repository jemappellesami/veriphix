OPENQASM 2.0;
include "qelib1.inc";
qreg q950[5];
cx q950[3],q950[4];
cx q950[2],q950[3];
cx q950[1],q950[2];
cx q950[0],q950[1];
