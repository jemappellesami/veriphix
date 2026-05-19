OPENQASM 2.0;
include "qelib1.inc";
qreg q998[4];
rz(pi) q998[3];
cx q998[3],q998[2];
cx q998[2],q998[1];
cx q998[1],q998[0];
