OPENQASM 2.0;
include "qelib1.inc";
qreg q83[6];
cx q83[3],q83[4];
cx q83[3],q83[2];
cx q83[2],q83[1];
cx q83[0],q83[1];
