OPENQASM 2.0;
include "qelib1.inc";
qreg q547[3];
rx(7*pi/4) q547[2];
cx q547[2],q547[1];
cx q547[0],q547[1];
rx(pi/4) q547[1];
