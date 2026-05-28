OPENQASM 2.0;
include "qelib1.inc";
qreg q646[3];
cx q646[1],q646[0];
cx q646[2],q646[1];
rx(3*pi/2) q646[0];
cx q646[0],q646[1];
rx(pi/4) q646[1];
