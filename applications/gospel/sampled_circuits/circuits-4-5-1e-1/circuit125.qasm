OPENQASM 2.0;
include "qelib1.inc";
qreg q126[4];
rx(pi) q126[1];
rx(3*pi/4) q126[2];
cx q126[3],q126[2];
cx q126[1],q126[2];
cx q126[1],q126[0];
