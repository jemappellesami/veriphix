OPENQASM 2.0;
include "qelib1.inc";
qreg q752[3];
cx q752[1],q752[0];
cx q752[0],q752[1];
cx q752[2],q752[1];
cx q752[1],q752[0];
