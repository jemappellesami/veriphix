OPENQASM 2.0;
include "qelib1.inc";
qreg q182[3];
rx(3*pi/4) q182[2];
cx q182[1],q182[2];
cx q182[0],q182[1];
