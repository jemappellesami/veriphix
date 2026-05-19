OPENQASM 2.0;
include "qelib1.inc";
qreg q760[5];
cx q760[3],q760[4];
cx q760[3],q760[2];
cx q760[2],q760[1];
cx q760[0],q760[1];
