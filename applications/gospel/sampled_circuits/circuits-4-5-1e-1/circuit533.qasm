OPENQASM 2.0;
include "qelib1.inc";
qreg q534[4];
rx(3*pi/4) q534[3];
cx q534[2],q534[3];
cx q534[1],q534[2];
cx q534[0],q534[1];
