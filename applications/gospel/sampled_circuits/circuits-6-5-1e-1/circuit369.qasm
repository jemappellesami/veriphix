OPENQASM 2.0;
include "qelib1.inc";
qreg q370[6];
cx q370[1],q370[0];
cx q370[4],q370[3];
cx q370[2],q370[3];
cx q370[1],q370[2];
cx q370[1],q370[0];
