OPENQASM 2.0;
include "qelib1.inc";
qreg q531[4];
rx(7*pi/4) q531[0];
cx q531[3],q531[2];
cx q531[0],q531[1];
cx q531[1],q531[2];
rx(pi/4) q531[0];
