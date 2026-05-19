OPENQASM 2.0;
include "qelib1.inc";
qreg q880[7];
cx q880[4],q880[5];
cx q880[4],q880[3];
cx q880[2],q880[3];
cx q880[2],q880[1];
cx q880[0],q880[1];
