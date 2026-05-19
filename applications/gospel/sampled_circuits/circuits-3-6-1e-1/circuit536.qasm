OPENQASM 2.0;
include "qelib1.inc";
qreg q537[3];
rz(3*pi/2) q537[2];
cx q537[1],q537[2];
cx q537[1],q537[0];
rx(pi/4) q537[1];
