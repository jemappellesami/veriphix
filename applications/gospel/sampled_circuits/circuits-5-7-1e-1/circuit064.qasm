OPENQASM 2.0;
include "qelib1.inc";
qreg q65[5];
rx(pi) q65[4];
cx q65[3],q65[4];
cx q65[3],q65[2];
cx q65[1],q65[2];
cx q65[1],q65[0];
