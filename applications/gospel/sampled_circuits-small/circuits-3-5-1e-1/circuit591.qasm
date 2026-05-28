OPENQASM 2.0;
include "qelib1.inc";
qreg q592[3];
rx(pi) q592[0];
cx q592[2],q592[1];
rz(7*pi/4) q592[1];
rx(pi) q592[1];
cx q592[1],q592[0];
