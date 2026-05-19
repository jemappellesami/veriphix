OPENQASM 2.0;
include "qelib1.inc";
qreg q296[5];
cx q296[4],q296[3];
cx q296[3],q296[2];
cx q296[1],q296[2];
cx q296[1],q296[0];
