OPENQASM 2.0;
include "qelib1.inc";
qreg q832[5];
cx q832[4],q832[3];
cx q832[2],q832[3];
cx q832[1],q832[2];
cx q832[0],q832[1];
