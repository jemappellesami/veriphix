OPENQASM 2.0;
include "qelib1.inc";
qreg q475[4];
rx(3*pi/2) q475[3];
cx q475[2],q475[3];
cx q475[1],q475[2];
cx q475[0],q475[1];
rx(pi/4) q475[1];
