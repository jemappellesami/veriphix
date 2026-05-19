OPENQASM 2.0;
include "qelib1.inc";
qreg q946[4];
rz(3*pi/2) q946[3];
cx q946[2],q946[3];
cx q946[1],q946[2];
cx q946[1],q946[0];
rx(pi/4) q946[1];
