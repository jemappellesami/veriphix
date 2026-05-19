OPENQASM 2.0;
include "qelib1.inc";
qreg q73[3];
cx q73[1],q73[0];
rz(pi) q73[1];
cx q73[1],q73[2];
cx q73[0],q73[1];
