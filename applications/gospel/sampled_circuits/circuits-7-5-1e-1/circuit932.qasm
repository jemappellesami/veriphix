OPENQASM 2.0;
include "qelib1.inc";
qreg q933[7];
cx q933[4],q933[5];
cx q933[4],q933[3];
cx q933[2],q933[3];
cx q933[2],q933[1];
cx q933[1],q933[0];
