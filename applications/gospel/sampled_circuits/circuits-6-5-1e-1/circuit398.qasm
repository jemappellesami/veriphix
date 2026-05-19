OPENQASM 2.0;
include "qelib1.inc";
qreg q399[6];
cx q399[4],q399[5];
cx q399[3],q399[4];
cx q399[3],q399[2];
cx q399[1],q399[2];
cx q399[0],q399[1];
