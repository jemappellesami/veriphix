OPENQASM 2.0;
include "qelib1.inc";
qreg q910[5];
cx q910[3],q910[4];
cx q910[3],q910[2];
cx q910[1],q910[2];
cx q910[0],q910[1];
rx(pi/4) q910[1];
