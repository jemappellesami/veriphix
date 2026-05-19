OPENQASM 2.0;
include "qelib1.inc";
qreg q882[5];
cx q882[4],q882[3];
cx q882[3],q882[2];
cx q882[1],q882[2];
cx q882[0],q882[1];
