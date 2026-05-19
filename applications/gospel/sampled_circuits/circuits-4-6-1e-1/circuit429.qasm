OPENQASM 2.0;
include "qelib1.inc";
qreg q430[4];
rx(3*pi/2) q430[3];
cx q430[3],q430[2];
cx q430[1],q430[2];
cx q430[0],q430[1];
rx(pi/4) q430[1];
