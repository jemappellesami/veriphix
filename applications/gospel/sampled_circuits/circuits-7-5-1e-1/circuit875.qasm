OPENQASM 2.0;
include "qelib1.inc";
qreg q876[7];
rx(pi/2) q876[0];
rx(3*pi/4) q876[2];
cx q876[2],q876[3];
cx q876[1],q876[2];
cx q876[1],q876[0];
