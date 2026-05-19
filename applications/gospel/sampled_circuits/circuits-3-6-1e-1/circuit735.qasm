OPENQASM 2.0;
include "qelib1.inc";
qreg q736[3];
cx q736[1],q736[0];
rz(pi) q736[1];
cx q736[2],q736[1];
cx q736[0],q736[1];
rx(pi/4) q736[1];
