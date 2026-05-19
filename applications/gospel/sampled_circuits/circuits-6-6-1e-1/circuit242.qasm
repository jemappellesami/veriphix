OPENQASM 2.0;
include "qelib1.inc";
qreg q243[6];
cx q243[3],q243[4];
cx q243[2],q243[3];
cx q243[2],q243[1];
cx q243[1],q243[0];
rx(pi/4) q243[1];
