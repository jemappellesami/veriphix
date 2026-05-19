OPENQASM 2.0;
include "qelib1.inc";
qreg q124[5];
cx q124[3],q124[4];
cx q124[2],q124[3];
cx q124[2],q124[1];
cx q124[1],q124[0];
