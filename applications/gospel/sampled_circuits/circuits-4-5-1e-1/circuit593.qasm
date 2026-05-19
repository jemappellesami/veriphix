OPENQASM 2.0;
include "qelib1.inc";
qreg q594[4];
cx q594[3],q594[2];
cx q594[2],q594[3];
cx q594[2],q594[1];
cx q594[0],q594[1];
