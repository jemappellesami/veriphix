OPENQASM 2.0;
include "qelib1.inc";
qreg q1[3];
cx q1[1],q1[0];
cx q1[0],q1[1];
cx q1[2],q1[1];
cx q1[1],q1[0];
rx(pi/4) q1[1];
