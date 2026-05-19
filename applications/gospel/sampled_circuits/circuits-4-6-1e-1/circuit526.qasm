OPENQASM 2.0;
include "qelib1.inc";
qreg q527[4];
rx(pi/2) q527[3];
cx q527[2],q527[3];
cx q527[2],q527[1];
cx q527[1],q527[0];
rx(pi/4) q527[1];
