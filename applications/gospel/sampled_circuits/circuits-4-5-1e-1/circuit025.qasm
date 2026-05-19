OPENQASM 2.0;
include "qelib1.inc";
qreg q26[4];
cx q26[1],q26[2];
cx q26[3],q26[2];
cx q26[2],q26[1];
cx q26[0],q26[1];
