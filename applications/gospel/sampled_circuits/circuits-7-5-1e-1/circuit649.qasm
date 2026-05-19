OPENQASM 2.0;
include "qelib1.inc";
qreg q650[7];
cx q650[4],q650[5];
cx q650[4],q650[3];
cx q650[3],q650[2];
cx q650[1],q650[2];
cx q650[0],q650[1];
