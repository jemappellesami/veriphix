OPENQASM 2.0;
include "qelib1.inc";
qreg q592[4];
rx(pi/2) q592[3];
cx q592[3],q592[2];
cx q592[2],q592[1];
cx q592[1],q592[0];
