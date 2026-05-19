OPENQASM 2.0;
include "qelib1.inc";
qreg q174[3];
rx(3*pi/4) q174[2];
cx q174[2],q174[1];
cx q174[0],q174[1];
