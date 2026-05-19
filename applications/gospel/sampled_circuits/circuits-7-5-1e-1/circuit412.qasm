OPENQASM 2.0;
include "qelib1.inc";
qreg q413[7];
cx q413[4],q413[5];
cx q413[4],q413[3];
cx q413[2],q413[3];
cx q413[2],q413[1];
cx q413[0],q413[1];
