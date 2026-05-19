OPENQASM 2.0;
include "qelib1.inc";
qreg q682[5];
cx q682[4],q682[3];
cx q682[2],q682[3];
cx q682[2],q682[1];
cx q682[1],q682[0];
rx(pi/4) q682[1];
