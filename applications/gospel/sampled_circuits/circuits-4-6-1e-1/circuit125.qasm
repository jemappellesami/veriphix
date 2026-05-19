OPENQASM 2.0;
include "qelib1.inc";
qreg q126[4];
cx q126[3],q126[2];
cx q126[2],q126[3];
cx q126[2],q126[1];
cx q126[1],q126[0];
rx(pi/4) q126[1];
