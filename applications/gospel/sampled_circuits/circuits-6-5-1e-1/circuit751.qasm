OPENQASM 2.0;
include "qelib1.inc";
qreg q752[6];
cx q752[4],q752[5];
cx q752[4],q752[3];
cx q752[2],q752[3];
cx q752[1],q752[2];
cx q752[1],q752[0];
