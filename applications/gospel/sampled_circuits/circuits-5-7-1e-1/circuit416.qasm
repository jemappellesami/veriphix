OPENQASM 2.0;
include "qelib1.inc";
qreg q417[5];
rx(pi/2) q417[4];
cx q417[4],q417[3];
cx q417[3],q417[2];
cx q417[1],q417[2];
cx q417[1],q417[0];
