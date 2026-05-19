OPENQASM 2.0;
include "qelib1.inc";
qreg q632[5];
cx q632[4],q632[3];
cx q632[2],q632[3];
cx q632[2],q632[1];
cx q632[1],q632[0];
rx(pi/4) q632[1];
