OPENQASM 2.0;
include "qelib1.inc";
qreg q547[3];
cx q547[0],q547[1];
rx(pi/2) q547[1];
rz(pi) q547[0];
cx q547[2],q547[1];
cx q547[1],q547[0];
