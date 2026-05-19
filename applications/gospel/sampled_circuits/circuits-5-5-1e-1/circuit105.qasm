OPENQASM 2.0;
include "qelib1.inc";
qreg q106[5];
cx q106[3],q106[4];
cx q106[3],q106[2];
cx q106[2],q106[1];
cx q106[0],q106[1];
