OPENQASM 2.0;
include "qelib1.inc";
qreg q227[4];
rz(pi/2) q227[0];
cx q227[1],q227[0];
cx q227[2],q227[1];
cx q227[0],q227[1];
