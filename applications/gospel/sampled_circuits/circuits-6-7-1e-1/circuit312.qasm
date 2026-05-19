OPENQASM 2.0;
include "qelib1.inc";
qreg q313[6];
rx(pi) q313[0];
cx q313[0],q313[1];
cx q313[1],q313[2];
rz(pi) q313[0];
cx q313[1],q313[0];
