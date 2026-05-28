OPENQASM 2.0;
include "qelib1.inc";
qreg q946[3];
rx(3*pi/2) q946[0];
cx q946[0],q946[1];
cx q946[2],q946[1];
cx q946[1],q946[0];
