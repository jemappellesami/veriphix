OPENQASM 2.0;
include "qelib1.inc";
qreg q188[3];
rx(pi) q188[1];
rx(5*pi/4) q188[2];
cx q188[1],q188[2];
cx q188[1],q188[0];
rx(pi/4) q188[1];
