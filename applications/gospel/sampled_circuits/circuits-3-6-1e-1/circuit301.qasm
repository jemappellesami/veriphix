OPENQASM 2.0;
include "qelib1.inc";
qreg q302[3];
cx q302[1],q302[2];
cx q302[2],q302[1];
cx q302[0],q302[1];
rx(pi/4) q302[1];
