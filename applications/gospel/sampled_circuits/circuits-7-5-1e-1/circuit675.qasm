OPENQASM 2.0;
include "qelib1.inc";
qreg q676[7];
cx q676[4],q676[5];
cx q676[4],q676[3];
cx q676[2],q676[3];
cx q676[2],q676[1];
cx q676[1],q676[0];
