OPENQASM 2.0;
include "qelib1.inc";
qreg q166[5];
rx(3*pi/2) q166[0];
cx q166[4],q166[3];
cx q166[2],q166[3];
cx q166[2],q166[1];
cx q166[1],q166[0];
