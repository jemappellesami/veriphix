OPENQASM 2.0;
include "qelib1.inc";
qreg q318[5];
cx q318[4],q318[3];
cx q318[3],q318[2];
cx q318[2],q318[1];
cx q318[1],q318[0];
rx(pi/4) q318[1];
