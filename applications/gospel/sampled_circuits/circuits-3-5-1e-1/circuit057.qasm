OPENQASM 2.0;
include "qelib1.inc";
qreg q58[3];
cx q58[0],q58[1];
rx(pi) q58[2];
cx q58[1],q58[2];
cx q58[0],q58[1];
