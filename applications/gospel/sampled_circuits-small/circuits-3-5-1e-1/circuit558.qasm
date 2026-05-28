OPENQASM 2.0;
include "qelib1.inc";
qreg q559[3];
rx(3*pi/2) q559[2];
cx q559[2],q559[1];
cx q559[1],q559[0];
