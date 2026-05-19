OPENQASM 2.0;
include "qelib1.inc";
qreg q88[3];
rx(3*pi/4) q88[0];
rx(3*pi/4) q88[2];
cx q88[2],q88[1];
cx q88[0],q88[1];
rx(pi/4) q88[1];
