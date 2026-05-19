OPENQASM 2.0;
include "qelib1.inc";
qreg q754[5];
cx q754[4],q754[3];
cx q754[3],q754[2];
cx q754[2],q754[1];
cx q754[0],q754[1];
