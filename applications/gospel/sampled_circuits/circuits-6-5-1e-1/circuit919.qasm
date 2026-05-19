OPENQASM 2.0;
include "qelib1.inc";
qreg q920[6];
cx q920[5],q920[4];
cx q920[3],q920[4];
cx q920[3],q920[2];
cx q920[2],q920[1];
cx q920[1],q920[0];
