OPENQASM 2.0;
include "qelib1.inc";
qreg q124[3];
rx(pi) q124[2];
cx q124[1],q124[2];
cx q124[1],q124[0];
